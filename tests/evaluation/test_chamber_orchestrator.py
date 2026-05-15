"""Tests for the chamber-pillar orchestrator + RunRecord results layer.

Covers `evaluation.chamber_pipeline.orchestrator` (AgentSpec, registry,
run_cell, run_sweep) and `.results` (RunRecord, Parquet/CSV IO).

The orchestrator is the M4-load-bearing surface — every chamber sweep
in M5+ dispatches through it. These tests pin its observable behavior
so M4b's CLI runs (and M5's sweep) can trust the contract:

  - Per-cell isolation (one bad cell doesn't lose the surrounding sweep)
  - Compatibility-filter skip semantics (skipped cells produce
    well-formed RunRecords, never NotImplementedError mid-sweep)
  - PC-degeneracy capture per cell via the inference logger
  - LLM-call counting via FakeLLM's .calls attribute
  - RunRecord round-trip through Parquet/CSV
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline.orchestrator import (
    AGENT_REGISTRY,
    MENU_SIZES,
    AgentSpec,
    SweepSpec,
    _budget_k_for,
    _build_agent_kwargs,
    _CountingLLM,
    _invoke_with_timeout,
    _PcDegeneracyHandler,
    _read_llm_metrics,
    count_cells,
    get_spec,
    iter_sweep_cells,
    run_cell,
    run_sweep,
)
from evaluation.chamber_pipeline.results import (
    RunRecord,
    write_records_csv,
    write_records_parquet,
)

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# FakeLLM — same fixture pattern as test_chamber_llm_agents.py
# ---------------------------------------------------------------------------


class FakeLLM:
    """Synthetic LiteLLM-shaped completion callable with `.calls` accessor."""

    def __init__(self, responder: Any = None) -> None:
        self._responder = responder or self._default_responder
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx})
        content = self._responder(idx, messages)
        return {"choices": [{"message": {"content": content}}]}

    @staticmethod
    def _default_responder(idx: int, messages: list[dict[str, str]]) -> str:
        """Pick the idx-th menu entry; emit '{}' for adjacency-emission prompts."""
        user_text = messages[-1]["content"]
        menu_entries = [
            line.strip()
            for line in user_text.splitlines()
            if line.strip().startswith(("uniform_", "exp_"))
        ]
        if menu_entries:
            return menu_entries[idx % len(menu_entries)]
        return "{}"


# ---------------------------------------------------------------------------
# AgentSpec + registry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    """Inventory of the registered agents, plus AgentSpec.is_compatible."""

    def test_registry_has_five_agents(self) -> None:
        assert len(AGENT_REGISTRY) == 5

    def test_registry_names_are_unique(self) -> None:
        names = [s.name for s in AGENT_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate agent names: {names}"

    def test_registry_matches_plan_5_1(self) -> None:
        """Names match plan §5.1's five variants exactly."""
        actual = sorted(s.name for s in AGENT_REGISTRY)
        expected = sorted(["random", "greedy_ig_lite", "llm_only", "llm_pc", "planner_reasoner"])
        assert actual == expected

    def test_greedy_ig_lite_is_lt_only(self) -> None:
        """Plan §5.1 row 2 footnote: GIG-lite is LT-only."""
        spec = get_spec("greedy_ig_lite")
        assert spec.chambers == ("lt",)
        assert spec.is_compatible("lt") is True
        assert spec.is_compatible("wt") is False  # type: ignore[arg-type]

    def test_other_agents_run_on_both_chambers(self) -> None:
        """All other agents support both LT and WT."""
        for name in ("random", "llm_only", "llm_pc", "planner_reasoner"):
            spec = get_spec(name)
            assert "lt" in spec.chambers
            assert "wt" in spec.chambers

    def test_llm_acceptance_flags_are_correct(self) -> None:
        """LLM-bearing variants accept_llm=True; non-LLM don't."""
        assert get_spec("random").accepts_llm is False
        assert get_spec("greedy_ig_lite").accepts_llm is False
        assert get_spec("llm_only").accepts_llm is True
        assert get_spec("llm_pc").accepts_llm is True
        assert get_spec("planner_reasoner").accepts_llm is True

    def test_planner_reasoner_extra_kwargs(self) -> None:
        spec = get_spec("planner_reasoner")
        assert "planner_budget" in spec.extra_kwargs
        assert "reasoner_budget" in spec.extra_kwargs

    def test_get_spec_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown agent name"):
            get_spec("nonexistent_variant")


# ---------------------------------------------------------------------------
# Kwargs builder
# ---------------------------------------------------------------------------


class TestBuildAgentKwargs:
    """Per-variant kwargs assembly. Centralizes the variant-specific dispatch."""

    def test_random_kwargs(self) -> None:
        spec = get_spec("random")
        kw = _build_agent_kwargs(spec, budget_k=10, seed=7, pc_alpha=0.1, llm=None)
        assert kw == {"seed": 7, "pc_alpha": 0.1}

    def test_llm_only_omits_pc_alpha(self) -> None:
        """llm_only doesn't have a PC inference step → no pc_alpha kwarg."""
        spec = get_spec("llm_only")
        kw = _build_agent_kwargs(spec, budget_k=10, seed=0, pc_alpha=0.05, llm=None)
        assert "pc_alpha" not in kw
        assert kw == {"seed": 0}

    def test_llm_kwargs_include_llm_when_provided(self) -> None:
        spec = get_spec("llm_pc")
        llm = FakeLLM()
        kw = _build_agent_kwargs(spec, budget_k=5, seed=1, pc_alpha=0.05, llm=llm)
        assert kw["llm"] is llm

    def test_llm_kwargs_omit_llm_when_none(self) -> None:
        """None llm → don't pass the kwarg; agents default-import litellm."""
        spec = get_spec("llm_pc")
        kw = _build_agent_kwargs(spec, budget_k=5, seed=1, pc_alpha=0.05, llm=None)
        assert "llm" not in kw

    def test_planner_reasoner_splits_budget_evenly(self) -> None:
        """Even split: even budget → equal halves; odd → planner gets the extra."""
        spec = get_spec("planner_reasoner")
        kw_even = _build_agent_kwargs(spec, budget_k=10, seed=0, pc_alpha=0.05, llm=None)
        assert kw_even["planner_budget"] == 5
        assert kw_even["reasoner_budget"] == 5
        assert kw_even["planner_budget"] + kw_even["reasoner_budget"] == 10

        kw_odd = _build_agent_kwargs(spec, budget_k=11, seed=0, pc_alpha=0.05, llm=None)
        assert kw_odd["planner_budget"] == 6  # the extra goes to planner
        assert kw_odd["reasoner_budget"] == 5
        assert kw_odd["planner_budget"] + kw_odd["reasoner_budget"] == 11


# ---------------------------------------------------------------------------
# Budget-k math
# ---------------------------------------------------------------------------


class TestBudgetK:
    """Fractional → integer budget conversion respects menu sizes."""

    def test_lt_full_budget(self) -> None:
        assert _budget_k_for("lt", 1.0) == MENU_SIZES["lt"]
        assert _budget_k_for("lt", 1.0) == 59

    def test_wt_full_budget(self) -> None:
        assert _budget_k_for("wt", 1.0) == MENU_SIZES["wt"]
        assert _budget_k_for("wt", 1.0) == 28

    def test_lt_half_budget(self) -> None:
        # 59 * 0.5 = 29.5 → round → 30
        assert _budget_k_for("lt", 0.5) == 30

    def test_lt_minimum_budget_clamps_to_one(self) -> None:
        """Budget fractions like 0.001 must still produce k >= 1."""
        assert _budget_k_for("lt", 0.001) == 1
        assert _budget_k_for("lt", 0.0) == 1

    def test_wt_ten_percent(self) -> None:
        # 28 * 0.10 = 2.8 → round → 3
        assert _budget_k_for("wt", 0.10) == 3

    def test_above_one_clamps_to_menu_size(self) -> None:
        """Defensively cap at menu size if a caller passes >1.0."""
        assert _budget_k_for("lt", 2.0) == 59


# ---------------------------------------------------------------------------
# Cell iteration
# ---------------------------------------------------------------------------


class TestIterSweepCells:
    """Sweep iteration is pure — doesn't load chambers or invoke agents."""

    def test_pilot_count(self) -> None:
        """M4 pilot: LT x 3 budgets x 5 variants x 30 seeds = 450 cells."""
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10, 0.50, 1.00),
            seeds=tuple(range(30)),
        )
        assert count_cells(sweep) == 1 * 3 * 5 * 30
        assert count_cells(sweep, exclude_skipped=True) == 450

    def test_m5_count(self) -> None:
        """Plan §6.1: LT 5x5x30 + WT 5x4x30 = 1350 after compat filter."""
        sweep = SweepSpec(
            chambers=("lt", "wt"),
            budget_fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
            seeds=tuple(range(30)),
        )
        # Iterates all 1500 cells (no early skip in iteration).
        assert count_cells(sweep) == 2 * 5 * 5 * 30
        # After compat filter: WT x variant 2 skipped = 5 x 1 x 30 = 150 fewer.
        assert count_cells(sweep, exclude_skipped=True) == 1350

    def test_filtered_agent_names(self) -> None:
        """SweepSpec.agent_names filters the registry."""
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.5,),
            agent_names=("random", "llm_pc"),
            seeds=(0, 1),
        )
        # 1 chamber x 1 budget x 2 agents x 2 seeds = 4
        assert count_cells(sweep) == 4

    def test_unknown_agent_name_silently_filtered(self) -> None:
        """Names not in registry are dropped (don't error)."""
        sweep = SweepSpec(agent_names=("random", "fake_variant"))
        assert all(s.name == "random" for s in sweep.selected_specs())

    def test_iter_yields_tuple_shape(self) -> None:
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.5,),
            agent_names=("random",),
            seeds=(0,),
        )
        cells = list(iter_sweep_cells(sweep))
        assert len(cells) == 1
        spec, chamber, budget_k, fraction, seed = cells[0]
        assert spec.name == "random"
        assert chamber == "lt"
        assert isinstance(budget_k, int)
        assert fraction == 0.5
        assert seed == 0


# ---------------------------------------------------------------------------
# PC-degeneracy capture
# ---------------------------------------------------------------------------


class TestPcDegeneracyHandler:
    """The per-cell logging handler counts singular-matrix warnings."""

    def test_counts_fell_back_messages(self) -> None:
        h = _PcDegeneracyHandler()
        record = logging.LogRecord(
            name="evaluation.chamber_pipeline.inference",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="PC inference fell back to all-zeros adjacency",
            args=None,
            exc_info=None,
        )
        h.handle(record)
        assert h.count == 1

    def test_ignores_unrelated_warnings(self) -> None:
        h = _PcDegeneracyHandler()
        record = logging.LogRecord(
            name="evaluation.chamber_pipeline.inference",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="some other inference warning",
            args=None,
            exc_info=None,
        )
        h.handle(record)
        assert h.count == 0


# ---------------------------------------------------------------------------
# run_cell — per-cell behavior
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestRunCellHappyPath:
    """run_cell with a working agent on a compatible chamber."""

    def test_random_on_lt_produces_ok_record(self) -> None:
        spec = get_spec("random")
        record = run_cell(
            spec=spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
        )
        assert record.status == "ok"
        assert record.chamber == "lt"
        assert record.agent_name == "random"
        assert record.budget_k == 2
        assert record.seed == 0
        # Scoring fields populated on ok.
        assert record.shd is not None and record.shd >= 0
        assert record.f1 is not None and 0 <= record.f1 <= 1
        assert record.n_edges_predicted is not None and record.n_edges_predicted >= 0
        assert record.n_edges_truth is not None and record.n_edges_truth > 0
        assert record.wall_time_seconds is not None and record.wall_time_seconds > 0
        # Non-LLM variants have no LLM-call count.
        assert record.n_llm_calls is None
        # PC variants do have a degeneracy count (probably 0 on this small case).
        assert record.n_pc_degeneracies is not None
        # Failure fields are None on ok.
        assert record.error_type is None
        assert record.skip_reason is None

    def test_llm_pc_with_mock_llm_produces_ok_record(self) -> None:
        spec = get_spec("llm_pc")
        llm = FakeLLM()
        record = run_cell(
            spec=spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
            llm=llm,
        )
        assert record.status == "ok"
        # LLM variants populate n_llm_calls.
        assert record.n_llm_calls == 2  # one per intervention selection
        # llm_pc still runs PC, so n_pc_degeneracies populated.
        assert record.n_pc_degeneracies is not None

    def test_llm_only_skips_pc_metadata(self) -> None:
        """llm_only doesn't run PC → n_pc_degeneracies is None."""
        spec = get_spec("llm_only")
        llm = FakeLLM()
        record = run_cell(
            spec=spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
            llm=llm,
        )
        assert record.status == "ok"
        assert record.n_pc_degeneracies is None  # llm_only doesn't run PC
        # 2 selections + 1 adjacency emission = 3 LLM calls.
        assert record.n_llm_calls == 3

    def test_planner_reasoner_with_mock_llm(self) -> None:
        """Conservation A + B = budget_k. Even split for budget_k=4 → A=B=2."""
        spec = get_spec("planner_reasoner")
        llm = FakeLLM()
        record = run_cell(
            spec=spec,
            chamber="lt",
            configuration="standard",
            budget_k=4,
            seed=0,
            llm=llm,
        )
        assert record.status == "ok"
        # 2 planner LLM calls + 2 reasoner LLM calls = 4 total.
        assert record.n_llm_calls == 4

    def test_budget_fraction_is_recorded(self) -> None:
        """budget_fraction = budget_k / menu_size, populated on ok."""
        record = run_cell(
            spec=get_spec("random"),
            chamber="lt",
            configuration="standard",
            budget_k=30,
            seed=0,
        )
        assert record.status == "ok"
        # LT has 59 experiments → 30/59 ≈ 0.508
        assert record.budget_fraction == pytest.approx(30 / 59)


@requires_causalchamber
class TestRunCellSkipBehavior:
    """Incompatible chambers produce skipped records, not errors."""

    def test_greedy_ig_lite_on_wt_is_skipped_via_registry(self) -> None:
        """The registry filter catches this BEFORE invoking the agent."""
        spec = get_spec("greedy_ig_lite")
        record = run_cell(
            spec=spec,
            chamber="wt",
            configuration="standard",
            budget_k=2,
            seed=0,
        )
        assert record.status == "skipped"
        assert record.skip_reason is not None
        assert "not compatible" in record.skip_reason.lower()
        # Skipped records still have well-typed timestamps.
        assert record.started_at == record.finished_at
        # No scoring on skipped.
        assert record.shd is None and record.f1 is None


@requires_causalchamber
class TestRunCellIsolation:
    """Per-cell exception isolation — run_cell never raises."""

    def test_unexpected_exception_becomes_error_record(self) -> None:
        """An agent that raises a non-NotImplementedError → status='error'."""

        def crashing_agent(_adapter, **_kwargs):
            raise RuntimeError("simulated agent crash")

        broken_spec = AgentSpec(
            name="broken",
            run=crashing_agent,
            chambers=("lt",),
            kind="non_llm",
        )

        record = run_cell(
            spec=broken_spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
        )
        assert record.status == "error"
        assert record.error_type == "RuntimeError"
        assert record.error_message is not None
        assert "simulated agent crash" in record.error_message
        # Traceback captured into extra for debugging.
        assert "traceback" in record.extra
        assert "RuntimeError" in record.extra["traceback"]

    def test_agent_notimplementederror_becomes_skip(self) -> None:
        """An agent's defensive NotImplementedError is treated as a skip,
        not an error — so the §6.5 figure doesn't show this as a failure."""

        def picky_agent(_adapter, **_kwargs):
            raise NotImplementedError("my own compatibility check failed")

        picky_spec = AgentSpec(
            name="picky",
            run=picky_agent,
            chambers=("lt",),  # registry says compatible
            kind="non_llm",
        )

        record = run_cell(
            spec=picky_spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
        )
        assert record.status == "skipped"
        assert record.skip_reason is not None
        assert "compatibility check failed" in record.skip_reason


# ---------------------------------------------------------------------------
# run_sweep — full-grid behavior
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestRunSweep:
    """Full-grid orchestration."""

    def test_tiny_sweep_returns_one_record_per_cell(self) -> None:
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10,),
            agent_names=("random",),
            seeds=(0, 1),
        )
        records = run_sweep(sweep)
        assert len(records) == 2
        assert all(r.agent_name == "random" for r in records)
        assert all(r.status == "ok" for r in records)

    def test_on_cell_callback_fires(self) -> None:
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10,),
            agent_names=("random",),
            seeds=(0, 1, 2),
        )
        callback_calls: list[tuple[str, int, int]] = []

        def cb(record: RunRecord, idx: int, total: int) -> None:
            callback_calls.append((record.status, idx, total))

        run_sweep(sweep, on_cell=cb)
        assert len(callback_calls) == 3
        # idx 0..2, total always 3.
        assert [c[1] for c in callback_calls] == [0, 1, 2]
        assert all(c[2] == 3 for c in callback_calls)

    def test_sweep_with_mocked_llm_runs_llm_variants(self) -> None:
        """End-to-end smoke: LLM agents through the orchestrator with FakeLLM."""
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10,),
            agent_names=("llm_pc",),
            seeds=(0,),
        )
        llm = FakeLLM()
        records = run_sweep(sweep, llm=llm)
        assert len(records) == 1
        assert records[0].status == "ok"
        # The shared FakeLLM was used by the cell.
        assert len(llm.calls) >= 1


# ---------------------------------------------------------------------------
# RunRecord serialization
# ---------------------------------------------------------------------------


class TestRunRecord:
    """RunRecord shape, defaults, and dict serialization."""

    def _basic(self, **overrides: Any) -> RunRecord:
        defaults = {
            "chamber": "lt",
            "configuration": "standard",
            "agent_name": "random",
            "budget_k": 5,
            "budget_fraction": 0.5,
            "seed": 0,
            "status": "ok",
            "started_at": "2026-05-09T00:00:00",
            "finished_at": "2026-05-09T00:00:01",
        }
        return RunRecord(**{**defaults, **overrides})

    def test_required_fields_only(self) -> None:
        r = self._basic()
        assert r.shd is None
        assert r.error_type is None
        assert r.extra == {}

    def test_to_dict_includes_extra_json(self) -> None:
        r = self._basic(extra={"foo": "bar"})
        d = r.to_dict()
        assert d["extra_json"] == '{"foo": "bar"}'
        assert "extra" not in d  # original key replaced

    def test_to_dict_empty_extra_yields_none(self) -> None:
        r = self._basic()
        d = r.to_dict()
        assert d["extra_json"] is None

    def test_frozen_disallows_mutation(self) -> None:
        r = self._basic()
        with pytest.raises((AttributeError, Exception)):
            r.shd = 1.0  # type: ignore[misc]


class TestRecordsIO:
    """Parquet + CSV writers round-trip correctly."""

    def _records(self) -> list[RunRecord]:
        return [
            RunRecord(
                chamber="lt",
                configuration="standard",
                agent_name="random",
                budget_k=5,
                budget_fraction=0.085,
                seed=0,
                status="ok",
                started_at="2026-05-09T00:00:00",
                finished_at="2026-05-09T00:00:01",
                shd=12.0,
                f1=0.5,
                n_edges_predicted=10,
                n_edges_truth=20,
                wall_time_seconds=1.5,
                n_pc_degeneracies=0,
            ),
            RunRecord(
                chamber="wt",
                configuration="standard",
                agent_name="greedy_ig_lite",
                budget_k=3,
                budget_fraction=0.107,
                seed=0,
                status="skipped",
                started_at="2026-05-09T00:00:00",
                finished_at="2026-05-09T00:00:00",
                skip_reason="agent 'greedy_ig_lite' is not compatible with chamber 'wt'",
            ),
        ]

    def test_write_parquet_roundtrip(self) -> None:
        records = self._records()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.parquet")
            write_records_parquet(records, path)
            assert os.path.exists(path)
            df = pd.read_parquet(path)
        assert len(df) == 2
        assert set(df["status"]) == {"ok", "skipped"}
        # NaN-vs-None handling: shd is NaN for skipped (pandas converts None
        # in numeric columns to NaN).
        ok_row = df[df["status"] == "ok"].iloc[0]
        assert ok_row["shd"] == 12.0
        assert ok_row["f1"] == 0.5

    def test_write_csv_roundtrip(self) -> None:
        records = self._records()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            write_records_csv(records, path)
            assert os.path.exists(path)
            df = pd.read_csv(path)
        assert len(df) == 2

    def test_empty_records_writes_schema_only(self) -> None:
        """Empty list still produces a valid file with column headers."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, "out.parquet")
            write_records_parquet([], parquet_path)
            df = pd.read_parquet(parquet_path)
        assert len(df) == 0
        # Schema preserved — required fields present as columns.
        for col in ("chamber", "agent_name", "status", "shd", "extra_json"):
            assert col in df.columns

    def test_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested/dir/out.parquet")
            write_records_parquet([], path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# numpy/pandas sanity (defensive against editor venv issues)
# ---------------------------------------------------------------------------


def test_numpy_pandas_smoke() -> None:
    assert isinstance(np.zeros(2), np.ndarray)
    assert isinstance(pd.DataFrame({"x": [1]}), pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests added in M4a.1 (post-review polish)
# ---------------------------------------------------------------------------


class TestCountingLLM:
    """Per-cell LLM proxy that counts calls + accumulates token / cost."""

    def test_proxies_to_target(self) -> None:
        captured: list[dict[str, Any]] = []

        def target(*, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
            captured.append({"model": model, "messages": messages})
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        result = wrapper(model="m", messages=[{"role": "user", "content": "hi"}])

        assert len(captured) == 1
        assert result["choices"][0]["message"]["content"] == "ok"
        assert len(wrapper.calls) == 1

    def test_extracts_dict_shape_usage(self) -> None:
        def target(**_: Any) -> dict:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25},
                "_hidden_params": {"response_cost": 0.0042},
            }

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        wrapper(model="m", messages=[])

        assert wrapper.total_input_tokens == 200
        assert wrapper.total_output_tokens == 50
        assert wrapper.total_cost_usd == pytest.approx(0.0084)

    def test_extracts_attr_shape_usage(self) -> None:
        """LiteLLM's Pydantic-shape responses also work."""

        class _Usage:
            prompt_tokens = 30
            completion_tokens = 10

        class _Hidden:
            response_cost = 0.001

        class _Message:
            content = "x"

        class _Choice:
            message = _Message()

        from typing import ClassVar

        class _Resp:
            choices: ClassVar[list[_Choice]] = [_Choice()]
            usage = _Usage()
            _hidden_params = _Hidden()

        def target(**_: Any) -> Any:
            return _Resp()

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])

        assert wrapper.total_input_tokens == 30
        assert wrapper.total_output_tokens == 10
        assert wrapper.total_cost_usd == pytest.approx(0.001)

    def test_missing_usage_does_not_raise(self) -> None:
        """FakeLLM-style responses (no usage field) → counts stay at 0, no crash."""

        def target(**_: Any) -> dict:
            return {"choices": [{"message": {"content": "x"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        assert len(wrapper.calls) == 1
        assert wrapper.total_input_tokens == 0
        assert wrapper.total_output_tokens == 0
        assert wrapper.total_cost_usd == 0.0

    def test_records_call_before_target_invocation(self) -> None:
        """A target that raises must still leave the call recorded — useful
        for cost-attribution audits ('I tried to call, even if it failed')."""

        def target(**_: Any) -> dict:
            raise RuntimeError("simulated API failure")

        wrapper = _CountingLLM(target=target)
        with pytest.raises(RuntimeError):
            wrapper(model="m", messages=[])
        assert len(wrapper.calls) == 1

    def test_injects_default_num_retries(self) -> None:
        """OpenRouter rate-limit fix: by default, every LLM call gets
        num_retries=3 so litellm's exponential backoff catches transient
        429s. Without this, the M4b smoke saw ~30-50% cell error rate
        from sustained-load throttling."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        assert captured[0].get("num_retries") == _CountingLLM.DEFAULT_NUM_RETRIES
        assert captured[0]["num_retries"] == 3

    def test_caller_can_override_num_retries(self) -> None:
        """Caller-supplied num_retries (e.g., 0 to disable for one call) wins."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[], num_retries=0)
        assert captured[0]["num_retries"] == 0

    def test_injects_default_provider_order(self) -> None:
        """OpenRouter provider routing: pinned to Parasail (fastest) +
        fp8 fallbacks. Without this, OpenRouter routes by its own
        cost-or-latency heuristic — which empirically chose AtlasCloud
        and throttled k=30 cells to a 10-min timeout."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        extra = captured[0].get("extra_body", {})
        assert "provider" in extra
        assert extra["provider"]["order"] == list(_CountingLLM.DEFAULT_PROVIDER_ORDER)
        assert extra["provider"]["order"][0] == "Parasail"
        assert extra["provider"]["allow_fallbacks"] is True

    def test_caller_can_override_provider(self) -> None:
        """Caller-supplied extra_body.provider wins (e.g., for ablation)."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(
            model="m",
            messages=[],
            extra_body={"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}},
        )
        assert captured[0]["extra_body"]["provider"]["order"] == ["DeepInfra"]

    def test_injects_default_request_timeout(self) -> None:
        """Per-request timeout is critical: without it, a stuck SSL read
        blocks forever (discovered via M4b smoke root-cause debugging).
        num_retries handles exceptions but never fires on infinite hangs —
        the timeout is what *creates* the exception that retry handles."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        assert captured[0].get("timeout") == _CountingLLM.DEFAULT_REQUEST_TIMEOUT_SECONDS
        assert captured[0]["timeout"] == 30.0

    def test_caller_can_override_request_timeout(self) -> None:
        """Caller-supplied timeout wins (e.g., longer for k=59 cells)."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[], timeout=120.0)
        assert captured[0]["timeout"] == 120.0

    def test_finish_reason_error_triggers_provider_rotation(self) -> None:
        """M4b re-smoke (2026-05-14): Parasail returned HTTP 200 with
        `finish_reason: 'error'` in the body — a soft failure mode that
        OpenRouter's HTTP-level fallback does NOT cycle past. Our wrapper
        must detect this and retry with the next provider in the list.
        """
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            # First call fails (Parasail body-error), second succeeds.
            if len(captured) == 1:
                return {
                    "choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                }
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        wrapper = _CountingLLM(target=target)
        response = wrapper(model="m", messages=[])

        # Two HTTP calls were made: first to Parasail (failed), second
        # rotated to the next provider.
        assert len(captured) == 2
        assert captured[0]["extra_body"]["provider"]["order"][0] == "Parasail"
        assert captured[1]["extra_body"]["provider"]["order"][0] != "Parasail"
        # The bumped (Parasail) provider should still appear, just at the end.
        assert "Parasail" in captured[1]["extra_body"]["provider"]["order"]
        # The successful response is what's returned.
        assert response["choices"][0]["finish_reason"] == "stop"
        # Both attempts are tracked in `calls` for cost-attribution honesty.
        assert len(wrapper.calls) == 2
        assert wrapper.calls[0]["primary_provider"] == "Parasail"
        assert wrapper.calls[1]["attempt"] == 1

    def test_all_providers_fail_returns_last_response(self) -> None:
        """If every provider returns finish_reason='error', the wrapper
        gives up and returns the last (still-bad) response. The caller's
        parser then falls back to its own empty-content path (random
        selection for `parse_selection_response`, empty graph for
        `parse_adjacency_response`)."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {
                "choices": [{"message": {"content": ""}, "finish_reason": "error"}],
            }

        wrapper = _CountingLLM(target=target)
        response = wrapper(model="m", messages=[])

        # Exactly one attempt per provider in the default order.
        assert len(captured) == len(_CountingLLM.DEFAULT_PROVIDER_ORDER)
        # Last response is returned (still bad — caller will fallback).
        assert response["choices"][0]["finish_reason"] == "error"

    def test_caller_provider_override_disables_rotation(self) -> None:
        """If the caller supplies their own `provider` config, our
        rotation logic stays out of the way — single attempt regardless
        of finish_reason. Lets ablation experiments / unit tests pin a
        specific provider without our retry-around-them behavior."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {
                "choices": [{"message": {"content": ""}, "finish_reason": "error"}],
            }

        wrapper = _CountingLLM(target=target)
        wrapper(
            model="m",
            messages=[],
            extra_body={"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}},
        )
        assert len(captured) == 1  # no rotation
        assert captured[0]["extra_body"]["provider"]["order"] == ["DeepInfra"]

    def test_usage_accumulates_across_rotation_attempts(self) -> None:
        """Each retry attempt costs real tokens; we must track them all
        for honest cost attribution. The total should be the sum across
        all attempts, not just the successful one."""
        n_calls = 0

        def target(**kwargs: Any) -> dict:
            nonlocal n_calls
            n_calls += 1
            usage = {"prompt_tokens": 10, "completion_tokens": 20}
            if n_calls == 1:
                # First attempt fails but still uses tokens.
                return {
                    "choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                    "usage": usage,
                }
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": usage,
            }

        wrapper = _CountingLLM(target=target)
        wrapper(model="m", messages=[])
        # Two attempts, each consumed 10 in + 20 out.
        assert wrapper.total_input_tokens == 20
        assert wrapper.total_output_tokens == 40


class TestReadLlmMetrics:
    """The (n_llm_calls, tokens_in, tokens_out, cost_usd) extractor."""

    def test_none_wrapper_yields_all_none(self) -> None:
        n, ti, to, c = _read_llm_metrics(None)
        assert (n, ti, to, c) == (None, None, None, None)

    def test_wrapper_with_calls_populates_all(self) -> None:
        wrapper = _CountingLLM(
            target=lambda **_: {
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        wrapper(model="m", messages=[])
        n, ti, to, c = _read_llm_metrics(wrapper)
        assert n == 1
        assert ti == 10
        assert to == 5
        assert c == 0.0  # no cost reported but tracked

    def test_wrapper_with_zero_calls_yields_n_zero_tokens_none(self) -> None:
        """LLM variant ran a budget=0 short-circuit — n_llm_calls=0 but
        token / cost fields stay None to distinguish 'tracked zero' from
        'no measurement'."""
        wrapper = _CountingLLM()
        n, ti, to, c = _read_llm_metrics(wrapper)
        assert n == 0
        assert ti is None
        assert to is None
        assert c is None


class TestInvokeWithTimeout:
    """Per-cell timeout wrapper around the agent invocation."""

    def test_no_timeout_calls_directly(self) -> None:
        result = _invoke_with_timeout(lambda _adapter, **kw: 42, None, {}, None)
        assert result == 42

    def test_within_timeout_returns_result(self) -> None:
        def fast(_adapter, **_kwargs):
            return "done"

        result = _invoke_with_timeout(fast, None, {}, timeout=5.0)
        assert result == "done"

    def test_exceeds_timeout_raises_timeout_error(self) -> None:
        import time as _time

        def slow(_adapter, **_kwargs):
            _time.sleep(2.0)
            return "never"

        with pytest.raises(TimeoutError, match="timeout"):
            _invoke_with_timeout(slow, None, {}, timeout=0.2)

    def test_unresponsive_target_still_times_out(self) -> None:
        # Production hangs (stuck SSL socket reads) never release the worker
        # thread. A `with ThreadPoolExecutor as ...` context manager would
        # block on shutdown(wait=True) forever in that scenario. Simulate
        # by passing a target that waits on an unset Event.
        import threading as _threading

        hang_event = _threading.Event()

        def hang(_adapter, **_kwargs):
            hang_event.wait()

        outcome: list[str] = []

        def call() -> None:
            try:
                _invoke_with_timeout(hang, None, {}, timeout=0.3)
            except TimeoutError:
                outcome.append("timed_out")
            except Exception as exc:  # pragma: no cover - debug aid only
                outcome.append(f"other:{type(exc).__name__}")

        caller = _threading.Thread(target=call, daemon=True)
        caller.start()
        caller.join(timeout=3.0)

        try:
            assert not caller.is_alive(), (
                "_invoke_with_timeout did not return within 3s — main thread "
                "is stuck in shutdown(wait=True) waiting for uncancellable worker"
            )
            assert outcome == ["timed_out"]
        finally:
            # Release the leaked worker so it can exit (it's daemon=True so
            # even if we forgot this, it wouldn't block test process exit).
            hang_event.set()


class TestRegistryFrozen:
    """AGENT_REGISTRY is a tuple — can't be mutated by tests or callers."""

    def test_registry_is_tuple(self) -> None:
        assert isinstance(AGENT_REGISTRY, tuple)

    def test_registry_cannot_be_appended(self) -> None:
        with pytest.raises(AttributeError):
            AGENT_REGISTRY.append(  # type: ignore[attr-defined]
                AgentSpec(name="rogue", run=lambda *a, **kw: None, chambers=("lt",))
            )


@requires_causalchamber
class TestRunCellNewMetrics:
    """run_cell now populates n_llm_calls + tokens + cost via _CountingLLM."""

    def test_llm_pc_with_real_token_reporting(self) -> None:
        """A FakeLLM that reports usage → tokens populated on the RunRecord."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        del create_contracted_chamber_agent

        class _UsageReportingLLM:
            calls: list[dict[str, Any]]

            def __init__(self) -> None:
                self.calls = []

            def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
                idx = len(self.calls)
                self.calls.append({"model": model, "idx": idx})
                user_text = messages[-1]["content"]
                menu = [
                    line.strip()
                    for line in user_text.splitlines()
                    if line.strip().startswith("uniform_")
                ]
                content = menu[idx % len(menu)] if menu else "{}"
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    "_hidden_params": {"response_cost": 0.001},
                }

        record = run_cell(
            spec=get_spec("llm_pc"),
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
            llm=_UsageReportingLLM(),
        )
        assert record.status == "ok"
        # 2 selection LLM calls expected → 2 x 100 = 200 input, 2 x 20 = 40 output, 2 x 0.001 = 0.002 cost
        assert record.n_llm_calls == 2
        assert record.tokens_in == 200
        assert record.tokens_out == 40
        assert record.cost_usd == pytest.approx(0.002)

    def test_random_agent_has_no_llm_metrics(self) -> None:
        record = run_cell(
            spec=get_spec("random"),
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
        )
        assert record.status == "ok"
        assert record.n_llm_calls is None
        assert record.tokens_in is None
        assert record.tokens_out is None
        assert record.cost_usd is None

    def test_cell_timeout_records_error(self) -> None:
        """A slow agent + tight timeout → status='error', error_type='TimeoutError'."""
        import time as _time

        def slow_agent(_adapter, **_kwargs):
            _time.sleep(2.0)
            import pandas as _pd

            return _pd.DataFrame()

        slow_spec = AgentSpec(name="slow", run=slow_agent, chambers=("lt",), kind="non_llm")
        record = run_cell(
            spec=slow_spec,
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
            cell_timeout_seconds=0.2,
        )
        assert record.status == "error"
        assert record.error_type == "TimeoutError"
        assert record.error_message is not None
        assert "timeout" in record.error_message.lower()


class TestRunRecordNewSchema:
    """RunRecord has tokens_in, tokens_out, cost_usd fields (M5-stable schema)."""

    def test_default_none_for_new_fields(self) -> None:
        r = RunRecord(
            chamber="lt",
            configuration="standard",
            agent_name="random",
            budget_k=1,
            budget_fraction=0.0,
            seed=0,
            status="ok",
            started_at="2026-05-09T00:00:00",
            finished_at="2026-05-09T00:00:01",
        )
        assert r.tokens_in is None
        assert r.tokens_out is None
        assert r.cost_usd is None

    def test_to_dict_preserves_new_fields(self) -> None:
        r = RunRecord(
            chamber="lt",
            configuration="standard",
            agent_name="llm_pc",
            budget_k=1,
            budget_fraction=0.0,
            seed=0,
            status="ok",
            started_at="2026-05-09T00:00:00",
            finished_at="2026-05-09T00:00:01",
            tokens_in=100,
            tokens_out=20,
            cost_usd=0.005,
        )
        d = r.to_dict()
        assert d["tokens_in"] == 100
        assert d["tokens_out"] == 20
        assert d["cost_usd"] == 0.005

    def test_to_dict_handles_non_serializable_extra(self) -> None:
        """default=str fallback in json.dumps prevents mid-sweep crash."""
        r = RunRecord(
            chamber="lt",
            configuration="standard",
            agent_name="random",
            budget_k=1,
            budget_fraction=0.0,
            seed=0,
            status="ok",
            started_at="x",
            finished_at="x",
            extra={"obj": np.array([1, 2, 3])},  # not normally JSON-serializable
        )
        # Must not raise.
        d = r.to_dict()
        # The numpy array got string-ified.
        assert d["extra_json"] is not None
