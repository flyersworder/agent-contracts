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
from evaluation.chamber_pipeline.inference import (
    DEFAULT_COLLINEARITY_THRESHOLD,
    DEFAULT_MAX_ROWS,
)
from evaluation.chamber_pipeline.orchestrator import (
    AGENT_REGISTRY,
    MENU_SIZES,
    AgentSpec,
    SweepConfigurationError,
    SweepSpec,
    _budget_k_for,
    _build_agent_kwargs,
    _CountingLLM,
    _invoke_with_timeout,
    _PcCollinearHandler,
    _PcDegeneracyHandler,
    _PcZeroVarianceHandler,
    _read_llm_metrics,
    _read_llm_provenance,
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

    def test_registry_has_twenty_agents(self) -> None:
        """Five M4b variants, three ladder arms, one ablation, one control,
        the two shared-record arms, and the two coverage-manipulation arms."""
        assert len(AGENT_REGISTRY) == 20

    def test_registry_names_are_unique(self) -> None:
        names = [s.name for s in AGENT_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate agent names: {names}"

    def test_registry_matches_plan_5_1_plus_the_ladder(self) -> None:
        """Plan §5.1's five variants, the ladder's three arms, one ablation,
        the two shared-record arms added 2026-08-29, and the four
        coverage-manipulation arms added 2026-08-30 (M7 Phase 1 follow-up): the
        unrestricted pair and the deconfounded `_ms` pair that excludes weak."""
        actual = sorted(s.name for s in AGENT_REGISTRY)
        expected = sorted(
            [
                "random",
                "greedy_ig_lite",
                "llm_only",
                "llm_pc",
                "planner_reasoner",
                "uncontracted",
                "fan_in_homog",
                "fan_in_spec",
                "fan_in_agg",
                "team",
                "one_shot",
                "critique",
                "coverage_max",
                "coverage_min",
                "coverage_max_ms",
                "wt_coverage_max",
                "wt_coverage_min",
                "coverage_min_ms",
                "team_varsplit",
                "shared_blackboard",
            ]
        )
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
        """M4 pilot: LT x 3 budgets x 5 variants x 30 seeds = 450 cells.

        `agent_names` is named explicitly. Leaving it None would make this
        count track the registry's size, so adding an unrelated arm would
        silently redefine "the pilot" -- which is exactly what the M6 arms did
        before PILOT_SPEC was pinned.
        """
        sweep = SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10, 0.50, 1.00),
            agent_names=(
                "random",
                "greedy_ig_lite",
                "llm_only",
                "llm_pc",
                "planner_reasoner",
            ),
            seeds=tuple(range(30)),
        )
        assert count_cells(sweep) == 1 * 3 * 5 * 30
        assert count_cells(sweep, exclude_skipped=True) == 450

    def test_m5_count(self) -> None:
        """Plan §6.1: LT 5x5x30 + WT 5x4x30 = 1350 after compat filter."""
        sweep = SweepSpec(
            chambers=("lt", "wt"),
            budget_fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
            agent_names=(
                "random",
                "greedy_ig_lite",
                "llm_only",
                "llm_pc",
                "planner_reasoner",
            ),
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
        result = wrapper(model=_PINNED_MODEL, messages=[{"role": "user", "content": "hi"}])

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
        wrapper(model=_PINNED_MODEL, messages=[])
        wrapper(model=_PINNED_MODEL, messages=[])

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
        wrapper(model=_PINNED_MODEL, messages=[])

        assert wrapper.total_input_tokens == 30
        assert wrapper.total_output_tokens == 10
        assert wrapper.total_cost_usd == pytest.approx(0.001)

    def test_missing_usage_does_not_raise(self) -> None:
        """FakeLLM-style responses (no usage field) → counts stay at 0, no crash."""

        def target(**_: Any) -> dict:
            return {"choices": [{"message": {"content": "x"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model=_PINNED_MODEL, messages=[])
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
            wrapper(model=_PINNED_MODEL, messages=[])
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
        wrapper(model=_PINNED_MODEL, messages=[])
        assert captured[0].get("num_retries") == _CountingLLM.DEFAULT_NUM_RETRIES
        assert captured[0]["num_retries"] == 3

    def test_caller_can_override_num_retries(self) -> None:
        """Caller-supplied num_retries (e.g., 0 to disable for one call) wins."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model=_PINNED_MODEL, messages=[], num_retries=0)
        assert captured[0]["num_retries"] == 0

    def test_a_pro_model_request_carries_the_pro_provider_order(self) -> None:
        """The resolver must reach the REQUEST, not just exist.

        `test_injects_default_provider_order` drives `model=_PINNED_MODEL`, which falls
        back to the default, so it passes whether or not per-model routing is
        wired in at all. This drives the real tag and asserts what actually
        goes on the wire -- the difference between a constant we declared and
        a request we sent.
        """
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model="openrouter/deepseek/deepseek-v4-pro", messages=[])
        order = captured[0]["extra_body"]["provider"]["order"]
        assert order == ["Baidu", "StreamLake", "SiliconFlow", "Novita"]
        assert order != list(_CountingLLM.DEFAULT_PROVIDER_ORDER)

    def test_injects_default_provider_order(self) -> None:
        """OpenRouter provider routing: pinned to a fp8-only order so
        OpenRouter doesn't fall back to fp4 (DeepInfra) for AAMAS
        reproducibility. The exact ordering is dynamic across days
        (provider speeds vary), so we assert the order matches
        DEFAULT_PROVIDER_ORDER and trust that constant to encode
        today's best-known ordering."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model=_PINNED_MODEL, messages=[])
        extra = captured[0].get("extra_body", {})
        assert "provider" in extra
        assert extra["provider"]["order"] == list(_CountingLLM.DEFAULT_PROVIDER_ORDER)
        assert extra["provider"]["allow_fallbacks"] is False, (
            "OpenRouter must not route past DEFAULT_PROVIDER_ORDER. With "
            "allow_fallbacks=True the 750-cell WT sweep was served in part by "
            "OpenInference, Relace and DigitalOcean -- 22 cells, none of them "
            "pinned, none of them in PROVIDER_PRECISION. The precision table "
            "and its homogeneity test then guarantee only what we REQUEST, "
            "which is not what the paper claims."
        )

    def test_every_declared_provider_order_is_precision_homogeneous(self) -> None:
        """Every pinned provider must serve the SAME inference precision.

        This is the check that `test_injects_default_provider_order` cannot
        make: that test asserts `order == DEFAULT_PROVIDER_ORDER`, which is
        true by construction whatever the constant contains. The fp8 claim
        lived only in a comment, and drifted -- AtlasCloud sat second in the
        order while serving fp4, and 27 of the 450 M6 ladder cells were
        served by it.

        Walks EVERY declared order, not just the default: per-model orders
        exist precisely because the endpoint ranking differs by model, so a
        new one is exactly where an unchecked precision class would enter.
        """
        precision = _CountingLLM.PROVIDER_PRECISION
        orders = {"<default>": _CountingLLM.DEFAULT_PROVIDER_ORDER}
        orders.update(_CountingLLM.PROVIDER_ORDER_BY_MODEL)
        for label, order in orders.items():
            classes = set()
            for provider in order:
                assert provider in precision, (
                    f"{provider} is pinned for {label} but has no recorded "
                    "precision class; add it to PROVIDER_PRECISION from "
                    "GET /models/{id}/endpoints"
                )
                classes.add(precision[provider])
            assert len(classes) == 1, (
                f"order for {label} mixes precision classes: { {p: precision[p] for p in order} }"
            )

    def test_a_dated_snapshot_does_not_inherit_its_family_pin(self) -> None:
        """`deepseek-v4-pro-0813` is not served by Baidu at all, and
        StreamLake serves `deepseek-v4-pro` at fp8 while its `-0813` endpoint
        reports `unknown`. Quantization belongs to the (provider, model) pair,
        so a substring match would hand a snapshot a pin naming an endpoint
        that does not serve it -- and a precision claim never checked for it.
        """
        family = _CountingLLM._provider_order_for("openrouter/deepseek/deepseek-v4-pro")
        assert family == ("Baidu", "StreamLake", "SiliconFlow", "Novita")
        # The snapshot does not merely get a DIFFERENT order -- it gets none.
        # Falling back to a default was still a silent extrapolation: it named
        # four endpoints under `allow_fallbacks: False` without anyone having
        # probed which of them serve this id.
        with pytest.raises(SweepConfigurationError, match="deepseek-v4-pro-0813"):
            _CountingLLM._provider_order_for("openrouter/deepseek/deepseek-v4-pro-0813")

    def test_provider_order_is_cheapest_first_for_the_model_it_names(self) -> None:
        """Measured 2026-08-27 from GET /models/{id}/endpoints: Parasail is
        the CHEAPEST fp8 endpoint for flash-0731 and the MOST EXPENSIVE one
        for v4-pro. A single global order cannot be right for both, and using
        the flash order on pro overpays 2.2x.
        """
        pro = _CountingLLM._provider_order_for("openrouter/deepseek/deepseek-v4-pro")
        assert pro[0] == "Baidu", "Baidu is the cheapest fp8 endpoint for v4-pro"
        assert "Parasail" not in pro, "Parasail is the most expensive fp8 for v4-pro"

    def test_caller_can_override_provider(self) -> None:
        """Caller-supplied extra_body.provider wins (e.g., for ablation)."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(
            model=_PINNED_MODEL,
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
        wrapper(model=_PINNED_MODEL, messages=[])
        assert captured[0].get("timeout") == _CountingLLM.DEFAULT_REQUEST_TIMEOUT_SECONDS
        # Deliberately NOT asserting a literal here. This test is about the
        # kwarg being injected at all; a second assertion pinning the value
        # made a latency-calibration change fail a pass-through test, which
        # says nothing about whether the value is right.
        # `TestRequestTimeoutIsCalibrated` owns the value, against measured p99.

    def test_caller_can_override_request_timeout(self) -> None:
        """Caller-supplied timeout wins (e.g., longer for k=59 cells)."""
        captured: list[dict[str, Any]] = []

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            return {"choices": [{"message": {"content": "ok"}}]}

        wrapper = _CountingLLM(target=target)
        wrapper(model=_PINNED_MODEL, messages=[], timeout=120.0)
        assert captured[0]["timeout"] == 120.0

    def test_finish_reason_error_triggers_provider_rotation(self) -> None:
        """M4b re-smoke (2026-05-14): a provider returned HTTP 200 with
        `finish_reason: 'error'` in the body — a soft failure mode that
        OpenRouter's HTTP-level fallback does NOT cycle past. Our wrapper
        must detect this and retry with the next provider in the list.
        """
        captured: list[dict[str, Any]] = []
        primary = _CountingLLM.DEFAULT_PROVIDER_ORDER[0]

        def target(**kwargs: Any) -> dict:
            captured.append(dict(kwargs))
            # First call fails (body-error from primary), second succeeds.
            if len(captured) == 1:
                return {
                    "choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                }
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        wrapper = _CountingLLM(target=target)
        response = wrapper(model=_PINNED_MODEL, messages=[])

        # Two HTTP calls were made: first to the configured primary (failed),
        # second rotated to the next provider.
        assert len(captured) == 2
        assert captured[0]["extra_body"]["provider"]["order"][0] == primary
        assert captured[1]["extra_body"]["provider"]["order"][0] != primary
        # The bumped primary should still appear, just at the end.
        assert primary in captured[1]["extra_body"]["provider"]["order"]
        # The successful response is what's returned.
        assert response["choices"][0]["finish_reason"] == "stop"
        # Both attempts are tracked in `calls` for cost-attribution honesty.
        assert len(wrapper.calls) == 2
        assert wrapper.calls[0]["primary_provider"] == primary
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
        response = wrapper(model=_PINNED_MODEL, messages=[])

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
            model=_PINNED_MODEL,
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
        wrapper(model=_PINNED_MODEL, messages=[])
        # Two attempts, each consumed 10 in + 20 out.
        assert wrapper.total_input_tokens == 20
        assert wrapper.total_output_tokens == 40


class TestReadLlmMetrics:
    """The (n_llm_calls, tokens_in, tokens_out, cost_usd, fallbacks, attempts)
    extractor."""

    def test_none_wrapper_yields_all_none(self) -> None:
        n, ti, to, c, fb, att = _read_llm_metrics(None)
        assert fb is None
        assert att is None
        assert (n, ti, to, c) == (None, None, None, None)

    def test_wrapper_with_calls_populates_all(self) -> None:
        wrapper = _CountingLLM(
            target=lambda **_: {
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        wrapper(model=_PINNED_MODEL, messages=[])
        n, ti, to, c, fb, att = _read_llm_metrics(wrapper)
        assert n == 1
        assert ti == 10
        assert to == 5
        assert c == 0.0  # no cost reported but tracked
        assert fb == 0  # a healthy call records no selection fallback
        # One logical call, one attempt: they diverge only under rotation.
        assert att == 1

    def test_wrapper_with_zero_calls_yields_n_zero_tokens_none(self) -> None:
        """LLM variant ran a budget=0 short-circuit — n_llm_calls=0 but
        token / cost fields stay None to distinguish 'tracked zero' from
        'no measurement'."""
        wrapper = _CountingLLM()
        n, ti, to, c, fb, att = _read_llm_metrics(wrapper)
        assert n == 0
        assert att == 0
        assert ti is None
        assert to is None
        assert c is None
        # Zero, not None: no calls were made, so no call degraded. This stays
        # distinguishable from a non-LLM variant, which reports None.
        assert fb == 0


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


class TestLlmProvenance:
    """Every behaviour-affecting setting must be recorded with the cell.

    Diagnosing the 2026-08-23 provider regression required inferring May's
    reasoning-effort level from token arithmetic, because M4b recorded neither
    the effort, nor the serving provider, nor the resolved model id. A sweep
    that records its own configuration can be audited; one that does not has
    to be reverse-engineered.
    """

    @staticmethod
    def _response(provider: str = "Novita", model: str = "deepseek/x"):
        return {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "provider": provider,
            "model": model,
        }

    def test_records_model_effort_and_provider(self) -> None:
        wrapper = _CountingLLM(target=lambda **kw: self._response())
        wrapper(
            model="openrouter/deepseek/deepseek-v4-flash",
            messages=[],
            extra_body={"reasoning": {"effort": "low"}},
        )
        model, effort, providers = _read_llm_provenance(wrapper)
        assert model == "openrouter/deepseek/deepseek-v4-flash"
        assert effort == "low"
        assert providers == "Novita"

    def test_joins_multiple_providers_seen_in_one_cell(self) -> None:
        seen = iter(["Novita", "Parasail"])
        wrapper = _CountingLLM(target=lambda **kw: self._response(provider=next(seen)))
        for _ in range(2):
            wrapper(model=_PINNED_MODEL, messages=[], extra_body={"reasoning": {"effort": "low"}})
        _model, _effort, providers = _read_llm_provenance(wrapper)
        assert providers == "Novita,Parasail"  # sorted, deduplicated

    def test_absent_effort_is_recorded_as_unset_not_guessed(self) -> None:
        """An unset parameter is the exact failure mode being guarded against."""
        wrapper = _CountingLLM(target=lambda **kw: self._response())
        wrapper(model=_PINNED_MODEL, messages=[])
        _model, effort, _providers = _read_llm_provenance(wrapper)
        assert effort == "unset"

    def test_no_calls_yields_all_none(self) -> None:
        assert _read_llm_provenance(_CountingLLM()) == (None, None, None)
        assert _read_llm_provenance(None) == (None, None, None)


class TestLadderRegistration:
    """The three new ladder arms and their budget split."""

    def test_new_arms_registered_with_scout_budgets(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import get_spec

        for name in ("fan_in_homog", "fan_in_spec", "team"):
            spec = get_spec(name)
            assert spec.extra_kwargs == ("scout_a_budget", "scout_b_budget")
            assert spec.kind == "llm_multi"
            assert spec.accepts_llm is True
            assert spec.chambers == ("lt", "wt")

    def test_scout_budgets_split_with_remainder_to_a(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        kwargs = _build_agent_kwargs(get_spec("team"), budget_k=45, seed=0, pc_alpha=0.05, llm=None)
        assert kwargs["scout_a_budget"] == 23
        assert kwargs["scout_b_budget"] == 22
        assert kwargs["scout_a_budget"] + kwargs["scout_b_budget"] == 45

    def test_only_the_spec_arm_differentiates(self) -> None:
        """fan_in_homog and fan_in_spec share one function; the flag differs."""
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        def kw(name: str) -> dict:
            return _build_agent_kwargs(get_spec(name), budget_k=30, seed=0, pc_alpha=0.05, llm=None)

        assert kw("fan_in_spec")["differentiate"] is True
        assert "differentiate" not in kw("fan_in_homog")
        assert "differentiate" not in kw("team")
        assert get_spec("fan_in_homog").run is get_spec("fan_in_spec").run

    def test_existing_arms_kwargs_unchanged(self) -> None:
        """Reuse safety: rungs 0 and 3 must dispatch exactly as in M4b."""
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        assert _build_agent_kwargs(
            get_spec("planner_reasoner"), budget_k=30, seed=0, pc_alpha=0.05, llm=None
        ) == {
            "seed": 0,
            "pc_alpha": 0.05,
            "planner_budget": 15,
            "reasoner_budget": 15,
        }
        assert _build_agent_kwargs(
            get_spec("llm_pc"), budget_k=30, seed=0, pc_alpha=0.05, llm=None
        ) == {"seed": 0, "pc_alpha": 0.05}


def test_tokens_do_not_gate_tool_calls():
    """Tokens are certified post-hoc, not enforced.

    A binding token cap would truncate new-arm cells while the reused rungs 0
    and 3 ran uncapped, breaking the matched-budget comparison in a way that
    looks like a result. The baseline assertion is not redundant: without it
    the test passes vacuously if the node is blocked for an unrelated reason,
    which is exactly what a missing `tool_invocations` grant causes.
    """
    from evaluation.chamber_pipeline.coordination import build_fan_in_graph

    monitor = build_fan_in_graph(k=6, c95=1350, a95=21163).monitor_for("scout_a")
    assert monitor.can_use_tool("intervene") is True  # baseline
    monitor.usage.add_tokens(10**9)
    assert monitor.can_use_tool("intervene") is True  # tokens still do not gate


class TestLadderSelfDescription:
    """Ladder arms describe their own wiring instead of being name-matched.

    Four separate string-literal lookups meant a sixth rung could be added,
    import cleanly, and silently receive plain-role budgets -- under-funding a
    targeted scout 6.5x and producing conservation violations that read as
    real overruns.
    """

    def test_only_ladder_arms_declare_scout_roles(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import AGENT_REGISTRY

        ladder = {s.name for s in AGENT_REGISTRY if s.is_ladder_arm}
        # Two of these are NOT rungs and must be excluded by name from
        # anything reporting "the ladder":
        #
        #  * `fan_in_agg` is an ABLATION of rung 1.
        #  * `team_varsplit` is a one-change VARIANT of rung 4 (the pools are
        #    partitioned by variable rather than by experiment name).
        #
        # Both declare scout roles for the same reason: they are only
        # interpretable against a matched-budget control, so they must resolve
        # the identical calibration as the arm they are compared with. That is
        # exactly why `LADDER_ORDER` is a separate tuple from this set.
        assert ladder == {
            "fan_in_homog",
            "fan_in_spec",
            "fan_in_agg",
            "team",
            "team_varsplit",
        }

    def test_calibration_refuses_a_non_ladder_arm(self) -> None:
        """A silent plain-role fallthrough is what this replaces."""
        from evaluation.chamber_pipeline.orchestrator import (
            _ladder_calibration,
            get_spec,
        )

        with pytest.raises(ValueError, match="not a ladder arm"):
            _ladder_calibration(get_spec("llm_pc"), 45)

    def test_roles_drive_the_budget_not_the_arm_name(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            _ladder_calibration,
            get_spec,
        )

        homog_a, homog_b, _, homog_oh = _ladder_calibration(get_spec("fan_in_homog"), 45)
        spec_a, spec_b, _, _ = _ladder_calibration(get_spec("fan_in_spec"), 45)
        _, _, _, team_oh = _ladder_calibration(get_spec("team"), 45)

        assert homog_a == homog_b  # both plain
        assert spec_b > spec_a  # targeted costs more than broad
        assert homog_oh == 0  # no negotiation rounds
        assert team_oh > 0  # two rounds, at the same provisioning multiple

    def test_every_declared_role_has_a_calibration_figure(self) -> None:
        """Per role AND per chamber the arm claims to support.

        `_ROLE_C95` was keyed by role alone with every figure measured on LT,
        so an arm declaring `chambers=("lt", "wt")` silently used LT's numbers
        on WT. Measured on the WT gate, `targeted` costs 2,868 there against
        LT's 10,379 -- a 3.6x over-provision that would have turned H-C into a
        statement about provisioning. The registry declares the chambers; the
        calibration table must cover what it declares.
        """
        from evaluation.chamber_pipeline.orchestrator import _ROLE_C95, AGENT_REGISTRY

        missing = []
        for spec in AGENT_REGISTRY:
            if spec.scout_roles is None:
                continue
            for chamber in spec.chambers:
                for role in spec.scout_roles:
                    if (chamber, role) not in _ROLE_C95:
                        missing.append(f"{spec.name}: ({chamber}, {role})")
        assert not missing, f"uncalibrated (chamber, role) pairs: {missing}"


class TestProvisioningIsWired:
    """Reverting the provisioning fix must not leave the suite green.

    Both `multiple=_PROVISION_MULTIPLE` at the call site and the constant
    itself were previously unverified: deleting the argument (falling back to
    the `multiple=2` default, halving every scout grant) and setting the
    constant to 1 each left all tests passing. Under-provisioning reads as a
    coordination result, so a regression here is self-disguising.
    """

    def test_run_cell_passes_the_provision_multiple(self) -> None:
        import inspect

        from evaluation.chamber_pipeline import orchestrator

        src = inspect.getsource(orchestrator.run_cell)
        assert "multiple=_PROVISION_MULTIPLE" in src

    def test_the_multiple_actually_scales_the_grant(self) -> None:
        from evaluation.chamber_pipeline.coordination import build_fan_in_graph
        from evaluation.chamber_pipeline.orchestrator import _PROVISION_MULTIPLE

        # Pinned exactly. `>= 3` let a 4 -> 3 mutation through, a 25% cut in
        # every scout grant, and under-provisioning reads as a coordination
        # result rather than as a budgeting error.
        assert _PROVISION_MULTIPLE == 4
        base = build_fan_in_graph(k=30, c95=2205, a95=8557, multiple=1)
        scaled = build_fan_in_graph(k=30, c95=2205, a95=8557, multiple=_PROVISION_MULTIPLE)
        forward = scaled.in_flow("aggregator").tokens // 2
        spendable_base = base.in_flow("scout_a").tokens - forward
        spendable_scaled = scaled.in_flow("scout_a").tokens - forward
        assert spendable_scaled == _PROVISION_MULTIPLE * spendable_base


def test_m6_runs_the_specified_budgets_not_the_pilots() -> None:
    """k=59 equals the whole menu, which spec §3 rules out.

    Two blind scouts drawing 30 and 29 from 59 cover ~44 distinct experiments
    against `llm_pc`'s 59 -- a 25% coverage deficit at the top budget point.
    """
    from evaluation.chamber_pipeline.orchestrator import _budget_k_for
    from evaluation.chamber_pipeline.run_experiment import M6_SPEC

    ks = [_budget_k_for("lt", f) for f in M6_SPEC.budget_fractions]
    assert ks == [6, 30, 45], ks


class TestConservationIsNotGatedOnAggregatorSpend:
    """`verify()` is graph-wide; only `tree_would_refuse` is aggregator-only.

    Gating both on aggregator spend let a cell whose SCOUTS overran, but whose
    aggregator reported no usage, record `certified=None` -- dropping out of
    H-C's denominator instead of counting as the failure it is, and biasing
    the reported compliance rate upward. Reverting the ungating previously
    left the whole suite green.
    """

    def test_a_scout_overrun_is_recorded_as_a_failure_not_as_unmeasured(
        self,
    ) -> None:
        """Behavioral, not a source grep.

        An earlier version of this test asserted the string ``agg_tokens > 0``
        was absent from ``run_cell``'s source near ``verify()``. That passes
        with the gate fully restored under any other spelling (``if
        agg_tokens:``), and fails when the neighbouring comment is merely
        reworded -- wrong in both directions. This drives ``run_cell`` instead.
        """
        import numpy as np

        from evaluation.chamber_pipeline.agents import _node_names
        from evaluation.chamber_pipeline.coordination import build_fan_in_graph

        def overrun_agent(adapter: object, **_kw: object) -> pd.DataFrame:
            graph = build_fan_in_graph(k=6, c95=2205, a95=8557, multiple=4)
            graph.monitor_for("scout_a").usage.add_tokens(10**7)
            adapter.delegation_graph = graph  # type: ignore[attr-defined]
            nodes = _node_names(adapter)
            return pd.DataFrame(
                np.zeros((len(nodes), len(nodes)), dtype=int), index=nodes, columns=nodes
            )

        record = run_cell(
            spec=AgentSpec(name="overrun", run=overrun_agent, chambers=("lt",), kind="non_llm"),
            chamber="lt",
            configuration="standard",
            budget_k=6,
            seed=0,
        )
        assert record.status == "ok"
        # The exact configuration the gate hid: aggregator silent, scout over.
        assert record.aggregator_tokens == 0
        assert record.conservation_certified is False, (
            "a scout overrun with a silent aggregator must count as a "
            "conservation failure, not drop out of H-C's denominator"
        )
        # tree_would_refuse stays aggregator-gated -- that gate is correct.
        assert record.tree_would_refuse is None

    def test_a_graph_whose_scouts_overran_does_not_verify(self) -> None:
        """The condition the gate used to hide."""
        from agent_contracts.core.delegation import ConservationViolationError
        from evaluation.chamber_pipeline.coordination import build_fan_in_graph

        graph = build_fan_in_graph(k=6, c95=2205, a95=8557, multiple=4)
        graph.monitor_for("scout_a").usage.add_tokens(10**7)
        assert graph.monitor_for("aggregator").usage.tokens == 0
        with pytest.raises(ConservationViolationError):
            graph.verify()


class TestModelOverride:
    """`--model` makes the model a first-class, recorded sweep parameter.

    DeepSeek's 0423 snapshot began reasoning ~4x harder per call on
    2026-08-13 under unchanged weights, which took `llm_pc` k=30 from 4.1 to
    27.4 min/cell. Comparing snapshots is therefore an experiment we have to
    run, not a config edit -- and the model that produced each row has to be
    recorded next to it.
    """

    def test_model_override_reaches_an_llm_agent(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import _build_agent_kwargs

        kwargs = _build_agent_kwargs(
            get_spec("llm_pc"), 6, 0, 0.05, None, model="openrouter/x/y-0731"
        )
        assert kwargs["model"] == "openrouter/x/y-0731"

    def test_no_override_leaves_the_agent_default_untouched(self) -> None:
        """Absent the flag, the agent signature default must still apply."""
        from evaluation.chamber_pipeline.orchestrator import _build_agent_kwargs

        assert "model" not in _build_agent_kwargs(get_spec("llm_pc"), 6, 0, 0.05, None)

    def test_a_non_llm_agent_never_receives_model(self) -> None:
        """`random_agent` has no `model` parameter; passing one is a TypeError."""
        from evaluation.chamber_pipeline.orchestrator import _build_agent_kwargs

        kwargs = _build_agent_kwargs(
            get_spec("random"), 6, 0, 0.05, None, model="openrouter/x/y-0731"
        )
        assert "model" not in kwargs

    def test_the_override_wins_over_static_kwargs(self) -> None:
        """A spec-level default must not silently outrank an explicit flag."""
        from types import MappingProxyType

        from evaluation.chamber_pipeline.agents import llm_pc_agent
        from evaluation.chamber_pipeline.orchestrator import AgentSpec, _build_agent_kwargs

        spec = AgentSpec(
            name="pinned",
            run=llm_pc_agent,
            chambers=("lt",),
            kind="llm",
            accepts_llm=True,
            static_kwargs=MappingProxyType({"model": "openrouter/pinned/old"}),
        )
        kwargs = _build_agent_kwargs(spec, 6, 0, 0.05, None, model="openrouter/x/y-0731")
        assert kwargs["model"] == "openrouter/x/y-0731"

    def test_cli_exposes_model_and_run_cell_records_it(self) -> None:
        from evaluation.chamber_pipeline.run_experiment import build_arg_parser

        args = build_arg_parser().parse_args(
            ["--variants", "llm_pc", "--out", "x.parquet", "--model", "openrouter/x/y-0731"]
        )
        assert args.model == "openrouter/x/y-0731"


class TestParallelSweep:
    """M4c: cells run at ~1.3% CPU, so serial execution wastes the machine.

    Process-level, not threads. `_PcDegeneracyHandler` attaches to the global
    `evaluation.chamber_pipeline.inference` logger for the duration of a cell,
    so N concurrent cells in one process would give every handler every other
    cell's records and silently corrupt `n_pc_degeneracy`. Processes make that
    impossible rather than relying on a correct thread-affinity filter.
    """

    def _spec(self) -> SweepSpec:
        return SweepSpec(
            chambers=("lt",),
            budget_fractions=(0.10,),
            agent_names=("random",),
            seeds=(0, 1, 2, 3),
            configuration="standard",
        )

    def test_parallel_matches_serial_exactly(self) -> None:
        """Same cells, same seeds -> same scores. Concurrency must not
        perturb a result; the cell seeds are the only randomness source."""
        serial = run_sweep(self._spec())
        parallel = run_sweep(self._spec(), max_workers=2)
        assert [r.status for r in parallel] == [r.status for r in serial]
        assert [(r.agent_name, r.budget_k, r.seed) for r in parallel] == [
            (r.agent_name, r.budget_k, r.seed) for r in serial
        ]
        assert [r.shd for r in parallel] == [r.shd for r in serial]
        assert [r.f1 for r in parallel] == [r.f1 for r in serial]

    def test_records_come_back_in_cell_order_not_completion_order(self) -> None:
        """Reproducibility: Parquet row order must not depend on which worker
        finished first.

        Tests `order_by_cell_index` directly with DELIBERATELY shuffled input.
        An earlier version ran two live pools and compared them, which proved
        nothing: `random` cells are fast and uniform, so completion order
        equals submission order and the test passed with the reordering
        deleted entirely.
        """
        from evaluation.chamber_pipeline.orchestrator import order_by_cell_index

        serial = run_sweep(self._spec())
        # Simulate workers finishing in reverse, then middle-out.
        scrambled = [(3, serial[3]), (1, serial[1]), (0, serial[0]), (2, serial[2])]
        assert [r.seed for r in order_by_cell_index(scrambled)] == [r.seed for r in serial]

    def test_max_workers_actually_dispatches_to_the_parallel_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards against the parallel branch silently falling back to serial:
        a sweep that quietly runs serially still returns correct records, just
        10x slower, and no other test would notice."""
        from evaluation.chamber_pipeline import orchestrator as _orch

        called: list[dict[str, object]] = []

        def _fake(sweep, cells, on_cell, model, temperature, max_workers, llm):  # type: ignore[no-untyped-def]
            called.append({"max_workers": max_workers, "temperature": temperature})
            return []

        monkeypatch.setattr(_orch, "_run_sweep_parallel", _fake)
        run_sweep(self._spec(), max_workers=4, temperature=0.0)
        # Both are asserted, and `temperature` is the more fragile: 0.0 is
        # falsy, so any `if temperature:` guard on the way down would drop a
        # pinned zero and silently restore provider-default sampling while
        # every column still looked right.
        assert called == [{"max_workers": 4, "temperature": 0.0}]

    def test_on_cell_fires_once_per_cell(self) -> None:
        """The CLI writes the checkpoint sidecar from this callback. It must
        run in the PARENT, once per cell, so sidecar appends stay serial and
        the checkpoint needs no locking of its own."""
        seen: list[tuple[str, int]] = []
        run_sweep(
            self._spec(),
            max_workers=2,
            on_cell=lambda rec, idx, total: seen.append((rec.agent_name, rec.seed)),
        )
        assert len(seen) == 4
        assert len(set(seen)) == 4

    def test_a_non_picklable_llm_is_refused_up_front(self) -> None:
        """Fail loudly at the call, not with an opaque pickling error 400
        cells into a paid sweep."""
        with pytest.raises(ValueError, match="max_workers"):
            run_sweep(self._spec(), max_workers=2, llm=lambda **kw: {})

    def test_default_is_the_untouched_serial_path(self) -> None:
        import inspect

        sig = inspect.signature(run_sweep)
        assert sig.parameters["max_workers"].default is None


class TestCalibrationIsBudgetDependent:
    """`a95` is a function of k, and pretending otherwise breaks conservation.

    The old docstring claimed per-call cost is "driven by the role's prompt,
    not by `k`". Measured untruncated: aggregator spend medians 8,557 at k=30
    and 16,980 at k=45. With one fixed constant, 6 of 9 graph cells at k=45
    failed `verify()` -- a budgeting artifact that would have been reported as
    0% H-C compliance for the fan-in rungs.
    """

    def test_calibration_takes_the_budget(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import _ladder_calibration, get_spec

        _, _, a95_30, _ = _ladder_calibration(get_spec("fan_in_spec"), 30)
        _, _, a95_45, _ = _ladder_calibration(get_spec("fan_in_spec"), 45)
        assert a95_45 > a95_30, "a95 must grow with k; reconcile prompts grow with k"

    def test_an_unmeasured_budget_raises_rather_than_extrapolating(self) -> None:
        """Silently guessing is how a fixed constant became an artifact.

        A wrong `a95` does not fail loudly -- it produces plausible
        conservation numbers that are really statements about provisioning.
        Refuse the cell instead.
        """
        from evaluation.chamber_pipeline.orchestrator import _ladder_calibration, get_spec

        with pytest.raises(ValueError, match="not calibrated"):
            _ladder_calibration(get_spec("fan_in_spec"), 17)

    def test_the_measured_budgets_cover_the_m6_grid(self) -> None:
        """Every budget M6 runs must have a measurement behind it."""
        from evaluation.chamber_pipeline.orchestrator import (
            _budget_k_for,
            _ladder_calibration,
            get_spec,
        )
        from evaluation.chamber_pipeline.run_experiment import M6_SPEC

        for fraction in M6_SPEC.budget_fractions:
            k = _budget_k_for("lt", fraction)
            _ladder_calibration(get_spec("fan_in_spec"), k)  # must not raise

    def test_capacity_at_k45_clears_the_measured_spend(self) -> None:
        """The failure this fixes: 9 cells spent 9,783-25,168; capacity was
        12,836."""
        from evaluation.chamber_pipeline.coordination import build_fan_in_graph
        from evaluation.chamber_pipeline.orchestrator import (
            _PROVISION_MULTIPLE,
            _ladder_calibration,
            get_spec,
        )
        from evaluation.chamber_pipeline.tree_accounting import dag_capacity

        c95_a, _, a95, _ = _ladder_calibration(get_spec("fan_in_spec"), 45)
        graph = build_fan_in_graph(k=45, c95=c95_a, a95=a95, multiple=_PROVISION_MULTIPLE)
        assert dag_capacity(graph, "aggregator") >= 25168  # worst cell measured


pytest.importorskip("causallearn")

# A model that IS in `PROVIDER_ORDER_BY_MODEL`. Placeholder ids like "m" used
# to work because an unlisted model silently inherited the flash pin; that
# fallback is now a SweepConfigurationError, since with `allow_fallbacks:
# False` the pin is a hard constraint rather than a preference.
_PINNED_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"


class TestCollinearDropCounting:
    """The collinear-drop count must reach the RunRecord.

    An uncounted degradation path is the failure mode this project has hit
    three times: truncation that varied with k, a provider that returned
    empty content, and a `wt_walks_v1` curve that pointed downhill. Each
    looked like a finding because nothing recorded the harness giving way.
    """

    def test_run_cell_records_collinear_drops(self) -> None:
        from evaluation.chamber_pipeline.inference import run_pc

        # TWO duplicates, not one. With a single duplicate the assertion
        # cannot tell "columns dropped" from "warnings emitted" -- both are
        # 1 -- and a handler that counted warnings would pass. Verified by
        # mutation: `self.count += 1` fails this test and passed the
        # one-duplicate version.
        rng = np.random.default_rng(0)
        base = rng.normal(size=400)
        data = pd.DataFrame(
            {
                "a": base,
                "b": 2.0 * base + rng.normal(scale=0.5, size=400),
                "dup1": base + rng.normal(scale=1e-7, size=400),
                "dup2": base + rng.normal(scale=1e-7, size=400),
            }
        )
        inference_logger = logging.getLogger("evaluation.chamber_pipeline.inference")
        handler = _PcCollinearHandler()
        inference_logger.addHandler(handler)
        prev = inference_logger.level
        inference_logger.setLevel(logging.WARNING)
        try:
            run_pc(data, ["a", "b", "dup1", "dup2"], alpha=0.05, seed=0)
        finally:
            inference_logger.removeHandler(handler)
            inference_logger.setLevel(prev)

        assert handler.count == 2, (
            "two columns ('dup1','dup2') are numerical duplicates emitted in "
            "ONE warning; the handler must count COLUMNS dropped, not warnings"
        )


class TestChamberKeyedCalibration:
    """Calibration constants are per-chamber, and unmeasured ones are inert.

    Every `_ROLE_C95` / `_A95_RECONCILE_BY_K` / `_C95_NEGOTIATE` figure was
    measured on LT, whose menu is 59 experiments. WT's is 28, so its prompts
    -- and therefore its token costs -- are different quantities. A dict keyed
    by `k` alone silently applies LT's numbers to WT.
    """

    def test_lt_calibration_still_resolves(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            AGENT_REGISTRY,
            _ladder_calibration,
        )

        spec = next(s for s in AGENT_REGISTRY if s.name == "fan_in_homog")
        c95_a, c95_b, a95, _ = _ladder_calibration(spec, 30, chamber="lt")
        assert (c95_a, c95_b, a95) == (2205, 2205, 11427)

    def test_lt_budget_does_not_leak_into_another_chamber(self) -> None:
        """k=30 is calibrated on LT and NOT on WT; asking for WT must raise.

        Before chamber keying this returned LT's 11,427 for a WT cell without
        complaint -- the same silent-plausible-number failure that let a single
        `_A95_RECONCILE` survive until the k=45 gate.
        """
        from evaluation.chamber_pipeline.orchestrator import (
            AGENT_REGISTRY,
            _ladder_calibration,
        )

        spec = next(s for s in AGENT_REGISTRY if s.name == "fan_in_homog")
        with pytest.raises(ValueError, match="not calibrated"):
            _ladder_calibration(spec, 30, chamber="wt")

    def test_provisional_entries_are_declared_not_silent(self) -> None:
        """A provisional entry must be flagged so conservation can be voided.

        Provisional numbers exist only so a calibration gate can RUN on a new
        chamber; they are not measurements. `is_provisional_calibration` is
        what `run_cell` consults to force `conservation_certified` to None, so
        a gate cannot contribute plausible H-C figures that are really
        statements about provisioning.
        """
        from evaluation.chamber_pipeline.orchestrator import (
            _PROVISIONAL_CALIBRATION,
            is_provisional_calibration,
        )

        for chamber, k in _PROVISIONAL_CALIBRATION:
            assert is_provisional_calibration(chamber, k) is True
        assert is_provisional_calibration("lt", 30) is False


class TestRequestTimeoutIsCalibrated:
    """The per-request timeout must clear measured call latency with margin.

    `DEFAULT_REQUEST_TIMEOUT_SECONDS` was 30.0 with a comment calling it
    "generous for normal completions (~1-15s)". That was true in May 2026 and
    was invalidated twice over: by raising the token caps to 32768, and by the
    provider-side reasoning increase in August. Measured on the WT gate, the
    MEDIAN cell had a mean per-call time of 30.3s -- the timeout sat at the
    middle of the distribution, and 21 of 42 cells were above it.

    Why that is a moderator and not just noise: `num_retries=3` rescues most
    over-runs, so failures surface only where latency is highest -- the
    heaviest arms at the largest budget. The error rate then correlates with
    both the topology IV and the budget axis, which is survivorship bias
    pointed straight at the arms under comparison.

    Pinning the ratio rather than the number so the relationship stays
    checkable when either figure moves.
    """

    def test_timeout_clears_measured_p99_with_margin(self) -> None:
        assert _CountingLLM.DEFAULT_REQUEST_TIMEOUT_SECONDS >= (
            3 * _CountingLLM.MEASURED_CALL_P99_SECONDS
        ), (
            "the request timeout must sit well clear of observed call latency; "
            "at parity it burns retries on healthy calls and fails the tail"
        )

    def test_timeout_stays_below_the_cell_timeout_safety_net(self) -> None:
        """It must still be a per-request bound, not a second cell timeout.

        The constant exists to surface a STUCK call -- the original symptom was
        12+ minutes at 0% CPU inside `_ssl__SSLSocket_read`. If it grows past
        the cell timeout it stops catching that.
        """
        assert _CountingLLM.DEFAULT_REQUEST_TIMEOUT_SECONDS < 1800


# ---------------------------------------------------------------------------
# PC provenance -- the three parameters that silently determine the graph
# ---------------------------------------------------------------------------


def test_pc_provenance_survives_a_runtime_constant_reassignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamp must follow `run_pc`, not the module constant.

    Python binds default arguments at def time, so reassigning
    `DEFAULT_MAX_ROWS` after import moves the constant while `run_pc` keeps
    using the old value. A stamp read from the constant would then report a
    configuration that never ran -- which is strictly worse than no stamp,
    because it looks like provenance. Editing the constant in the source
    file cannot catch this: that changes both together, so the assertion
    would hold by construction.
    """
    from evaluation.chamber_pipeline import inference

    monkeypatch.setattr(inference, "DEFAULT_MAX_ROWS", 9999)
    defaults = inference.pc_call_defaults()
    assert defaults["max_rows"] != 9999
    assert defaults["max_rows"] == DEFAULT_MAX_ROWS


def test_runtime_fingerprint_is_read_from_numpy_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BLAS name must come from the installed numpy at call time.

    A hardcoded string would be provenance-shaped and wrong on exactly the
    machine where it matters. macOS/Accelerate and Linux/OpenBLAS produce
    different graphs from byte-identical inputs, so this field is what stops
    two sweeps from being pooled across that boundary.
    """
    import numpy as np

    from evaluation.chamber_pipeline.inference import runtime_fingerprint

    monkeypatch.setattr(
        np,
        "show_config",
        lambda *a, **k: {"Build Dependencies": {"blas": {"name": "sentinel-blas"}}},
    )
    assert runtime_fingerprint()["blas"] == "sentinel-blas"


def test_runtime_fingerprint_degrades_to_unknown_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numpy build without the dicts API must not abort a 20-hour sweep."""
    import numpy as np

    from evaluation.chamber_pipeline.inference import runtime_fingerprint

    def _boom(*a: object, **k: object) -> dict[str, object]:
        raise TypeError("show_config('dicts') unsupported")

    monkeypatch.setattr(np, "show_config", _boom)
    assert runtime_fingerprint()["blas"] == "unknown"


@requires_causalchamber
class TestRunCellRecordsPcProvenance:
    """Every cell must state the PC configuration that produced it.

    `runs/m6-controls.parquet` disagrees with a later `random` run on the
    same seed, the same committed code and the same dependency versions
    (41 predicted edges vs 22 -- a structurally different graph, not a
    threshold flip). It stayed undiagnosable because none of alpha, the row
    cap or the collinearity threshold travelled with the row.
    """

    def test_ok_record_carries_the_alpha_actually_used(self) -> None:
        record = run_cell(
            spec=get_spec("random"),
            chamber="lt",
            configuration="standard",
            budget_k=2,
            seed=0,
            pc_alpha=0.01,
        )
        assert record.status == "ok"
        # The value passed, NOT run_cell's 0.05 default -- a stamp that
        # hardcoded the default would be worse than no stamp at all.
        assert record.pc_alpha == 0.01
        assert record.pc_max_rows == DEFAULT_MAX_ROWS
        assert record.pc_collinearity_threshold == DEFAULT_COLLINEARITY_THRESHOLD

    def test_skipped_record_carries_it_too(self) -> None:
        """A re-run needs the skipped cell's configuration as much as an ok one."""
        record = run_cell(
            spec=get_spec("greedy_ig_lite"),  # LT-only, so WT skips
            chamber="wt",
            configuration="standard",
            budget_k=2,
            seed=0,
            pc_alpha=0.01,
        )
        assert record.status == "skipped"
        assert record.pc_alpha == 0.01


class TestZeroVarianceCounting:
    """The zero-variance count must reach the RunRecord, and stay its own path.

    Three handlers now scrape one logger. That is exactly the shape in which a
    counter starts double-counting: `_PcCollinearHandler` matches "dropped",
    and so does the zero-variance warning.
    """

    @staticmethod
    def _run(data, names, **kwargs):
        """Run PC with all three handlers attached; return their counts."""
        from evaluation.chamber_pipeline.inference import run_pc

        inference_logger = logging.getLogger("evaluation.chamber_pipeline.inference")
        handlers = {
            "zerovar": _PcZeroVarianceHandler(),
            "collinear": _PcCollinearHandler(),
            "degeneracy": _PcDegeneracyHandler(),
        }
        for h in handlers.values():
            inference_logger.addHandler(h)
        prev = inference_logger.level
        inference_logger.setLevel(logging.WARNING)
        try:
            run_pc(data, names, alpha=0.05, seed=0, **kwargs)
        finally:
            for h in handlers.values():
                inference_logger.removeHandler(h)
            inference_logger.setLevel(prev)
        return {k: h.count for k, h in handlers.items()}

    def test_counts_columns_not_warnings(self) -> None:
        # TWO constant columns, emitted in ONE warning -- so a handler that
        # incremented by 1 per warning would read 1 and fail here. Same
        # mutation guard as the collinear test above.
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "a": rng.normal(size=300),
                "b": rng.normal(size=300),
                "flat1": np.zeros(300),
                "flat2": np.full(300, 3.0),
            }
        )
        counts = self._run(data, ["a", "b", "flat1", "flat2"])
        assert counts["zerovar"] == 2, "must count COLUMNS dropped, not warnings emitted"

    def test_pc_warning_markers_are_unambiguous(self) -> None:
        """Each degradation path must increment its own handler and no other.

        Asserted by driving all three paths rather than by inspecting the
        message strings, so a reworded warning that collides with another
        handler's marker fails here instead of silently inflating a count.
        """
        rng = np.random.default_rng(1)

        # Path 1: constant columns only.
        base = rng.normal(size=300)
        zero_var = pd.DataFrame(
            {"a": base, "b": 2.0 * base + rng.normal(scale=0.5, size=300), "flat": np.zeros(300)}
        )
        counts = self._run(zero_var, ["a", "b", "flat"])
        assert counts == {"zerovar": 1, "collinear": 0, "degeneracy": 0}

        # Path 2: a numerically duplicate column, no constant one.
        collinear = pd.DataFrame(
            {
                "a": base,
                "b": 2.0 * base + rng.normal(scale=0.5, size=300),
                "dup": base + rng.normal(scale=1e-7, size=300),
            }
        )
        counts = self._run(collinear, ["a", "b", "dup"])
        assert counts == {"zerovar": 0, "collinear": 1, "degeneracy": 0}

        # Path 3: exact collinearity with the filter OFF -> singular fallback.
        singular = pd.DataFrame({"x": base, "y": 2.0 * base, "z": rng.normal(size=300)})
        counts = self._run(singular, ["x", "y", "z"], collinearity_threshold=None)
        assert counts == {"zerovar": 0, "collinear": 0, "degeneracy": 1}


def test_rotation_inflates_attempts_but_not_logical_calls() -> None:
    """`fallback_rate`'s denominator must not grow when providers misbehave.

    `calls` records one entry per provider ATTEMPT so rotation shows up in
    cost. Using that as the denominator for a per-decision rate means a cell
    reports a LOWER degradation rate the worse the serving stack behaves --
    a harness statistic that moves with conditions, which is the class of
    defect the validity report exists to catch.
    """
    responses = [
        {"choices": [{"message": {"content": ""}, "finish_reason": "error"}]},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    ]
    wrapper = _CountingLLM(target=lambda **_: responses.pop(0))
    wrapper(model=_PINNED_MODEL, messages=[])

    n, _ti, _to, _c, _fb, attempts = _read_llm_metrics(wrapper)
    assert attempts == 2, "both provider attempts are billed and must be counted"
    assert n == 1, "one logical call, whatever it took to satisfy it"


class TestSweepConfigurationErrors:
    """Configuration faults must abort, not become an error RATE.

    `run_cell` turns in-cell exceptions into `status="error"` records so one
    bad cell cannot kill a 20-hour sweep. Applied to a configuration fault
    that is exactly wrong: every cell raises identically, `done_cell_keys`
    excludes errored cells so they can be retried, and each resume re-attempts
    all of them forever while the message naming the fix sits truncated in
    `error_message`.
    """

    def test_uncalibrated_budget_raises_instead_of_recording_an_error(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import _A95_RECONCILE_BY_K

        # A WT budget that is deliberately not in the calibration table.
        assert ("wt", 3) not in _A95_RECONCILE_BY_K
        with pytest.raises(SweepConfigurationError, match="not calibrated"):
            run_cell(get_spec("team"), "wt", "standard", budget_k=3, seed=0, llm=lambda **_: None)

    def test_menu_size_drift_is_checked_for_every_arm_not_just_uncontracted(
        self, monkeypatch
    ) -> None:
        """`_budget_k_for` converts fractions to k through `MENU_SIZES` for
        EVERY arm, so a stale table mis-budgets contracted arms too. The guard
        used to sit inside the `ignores_budget` branch."""
        from evaluation.chamber_pipeline import orchestrator as orch

        monkeypatch.setitem(orch.MENU_SIZES, "lt", 58)  # live menu is 59
        with pytest.raises(SweepConfigurationError, match="disagrees with"):
            run_cell(get_spec("random"), "lt", "standard", budget_k=3, seed=0)


class TestParallelWorkerFailure:
    """A dead worker costs one cell, not the sweep.

    `run_cell` converts in-cell exceptions to records, so a raise out of
    `fut.result()` means the WORKER died -- OOM-killed, or a result that would
    not pickle. Uncaught it escapes the `with ProcessPoolExecutor(...)` block,
    whose `__exit__` calls `shutdown(wait=True)`: the same wait-on-exit shape
    this project already root-caused for ThreadPoolExecutor, discarding every
    in-flight cell with no sidecar line.
    """

    @staticmethod
    def _fake_pool_raising(exc: BaseException):
        class _Fut:
            def __init__(self, payload):
                self.payload = payload

            def result(self):
                raise exc

        class _Pool:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, _fn, payload):
                return _Fut(payload)

        return _Pool

    def test_a_surviving_pool_turns_one_cell_fault_into_one_error_record(self, monkeypatch) -> None:
        """A fault the pool SURVIVES costs one cell, not the sweep.

        A result that fails to pickle is the canonical case: the worker is
        fine, this cell is not. Distinguished from `BrokenProcessPool`, which
        kills every remaining future and is handled by aborting instead --
        treating that as a per-cell fault turned a sweep that died at cell 250
        into an instant completion with 200 fabricated error rows and exit 0.
        """
        import concurrent.futures as cf

        from evaluation.chamber_pipeline.orchestrator import SweepSpec, run_sweep

        monkeypatch.setattr(
            cf, "ProcessPoolExecutor", self._fake_pool_raising(TypeError("cannot pickle"))
        )
        monkeypatch.setattr(cf, "as_completed", lambda futs: list(futs))

        sweep = SweepSpec(
            chambers=("lt",), agent_names=("random",), budget_fractions=(0.1,), seeds=range(2)
        )
        records = run_sweep(sweep, max_workers=2)

        assert len(records) == 2, "every cell is accounted for, none silently dropped"
        assert all(r.status == "error" for r in records)
        assert all(r.error_type == "TypeError" for r in records)
        assert all("worker process died" in (r.error_message or "") for r in records)
        assert all(r.blas_backend for r in records)

    def test_a_configuration_error_still_aborts_the_parallel_sweep(self, monkeypatch) -> None:
        import concurrent.futures as cf

        from evaluation.chamber_pipeline.orchestrator import SweepSpec, run_sweep

        monkeypatch.setattr(
            cf, "ProcessPoolExecutor", self._fake_pool_raising(SweepConfigurationError("bad k"))
        )
        monkeypatch.setattr(cf, "as_completed", lambda futs: list(futs))
        sweep = SweepSpec(
            chambers=("lt",), agent_names=("random",), budget_fractions=(0.1,), seeds=range(2)
        )
        with pytest.raises(SweepConfigurationError):
            run_sweep(sweep, max_workers=2)


@pytest.mark.parametrize("arm", ["random", "llm_pc"])
def test_counters_are_none_when_pc_never_ran_for_a_non_llm_only_arm(arm: str) -> None:
    """The gate must be "did `run_pc` execute", not "is this llm_only".

    Seven agents return `_empty_adjacency` early -- zero budget, empty menu, or
    no frames after every selection failed -- and `random` at k=0 is the
    cheapest of them. Under the old name-based gate such a cell recorded
    0/0/0 with an all-zeros adjacency, which reads in the validity report
    exactly like a clean PC run on a graph it recovered nothing from.
    """
    record = run_cell(get_spec(arm), "lt", "standard", budget_k=0, seed=0)
    assert record.status == "ok"
    assert record.f1 == 0.0
    assert record.n_pc_degeneracies is None
    assert record.n_collinear_dropped is None
    assert record.n_zero_variance_dropped is None


def test_an_unpinned_model_raises_rather_than_recording_an_error_row() -> None:
    """`_provider_order_for` fires inside `_CountingLLM.__call__`, so it lands
    in the AGENT-INVOCATION try, not the adapter-construction one. Guarding
    only the first left `--model <unlisted>` producing N identical error rows
    that every resume re-attempts -- the loop the class exists to prevent.
    """
    with pytest.raises(SweepConfigurationError, match="unknown-model"):
        run_cell(
            get_spec("llm_pc"),
            "lt",
            "standard",
            budget_k=2,
            seed=0,
            model="openrouter/foo/unknown-model",
        )


def test_a_broken_pool_aborts_instead_of_fabricating_the_rest_of_the_sweep(
    monkeypatch,
) -> None:
    """Every remaining future raises the same BrokenProcessPool, so treating it
    as a per-cell fault turns a sweep that died at cell 250 into an instant
    completion with 200 fabricated error rows and exit 0."""
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool

    from evaluation.chamber_pipeline.orchestrator import SweepSpec, run_sweep

    monkeypatch.setattr(
        cf,
        "ProcessPoolExecutor",
        TestParallelWorkerFailure._fake_pool_raising(BrokenProcessPool("worker died")),
    )
    monkeypatch.setattr(cf, "as_completed", lambda futs: list(futs))
    sweep = SweepSpec(
        chambers=("lt",), agent_names=("random",), budget_fractions=(0.1,), seeds=range(2)
    )
    with pytest.raises(RuntimeError, match="worker pool died"):
        run_sweep(sweep, max_workers=2)


def test_the_purchase_roster_is_recorded_for_every_arm() -> None:
    """WHICH experiments were bought, not just how many.

    Recorded at the adapter rather than per agent: seven agents can each
    forget to report their picks, `query_intervention` cannot. Without the
    roster, an arm that matches the loop on distinct-experiment coverage and
    still loses on accuracy cannot be explained after the fact -- two arms can
    buy 30 distinct experiments and touch very different numbers of graph
    variables, since one variable has up to three menu entries.
    """
    record = run_cell(get_spec("random"), "lt", "standard", budget_k=5, seed=0)
    assert record.status == "ok"
    names = (record.chosen_experiments or "").split(",")
    assert len(names) == 5, "one entry per unit of budget spent, in spending order"
    assert len(set(names)) == 5
    # The roster is the ground for the variable-coverage question: distinct
    # menu entries is NOT distinct perturbed variables.
    variables = {n.removeprefix("uniform_").rsplit("_", 1)[0] for n in names}
    assert len(variables) <= len(names)


def test_a_failed_query_does_not_enter_the_roster() -> None:
    """Charge-on-success: a bad name costs no budget and must not appear as a
    purchase, or the roster and the budget disagree."""
    from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

    adapter = create_contracted_chamber_agent(
        chamber="lt", configuration="standard", intervention_budget=3
    )
    real = adapter.available_experiments()[0]
    adapter.query_intervention(real)
    with pytest.raises(KeyError):
        adapter.query_intervention("no_such_experiment")
    assert adapter.purchased == [real]


@pytest.mark.parametrize("arm", ["random", "llm_pc", "one_shot"])
def test_every_arm_reports_distinct_experiment_count(arm: str) -> None:
    """The reference arm must not be missing from its own comparison.

    Only the multi-agent agents set `coordination_stats`, so `llm_pc` reported
    None and silently dropped out of any analysis grouped on this column --
    including the coverage curve that the fan-in redundancy decomposition is
    measured against.
    """
    llm = None
    if arm != "random":

        def llm(**kw):  # type: ignore[misc]
            menu = [
                line
                for line in kw["messages"][-1]["content"]
                .split("Menu:\n")[-1]
                .split("\n\n")[0]
                .split("\n")
                if line
            ]
            resp = type("R", (), {})()
            resp.choices = [
                type(
                    "C",
                    (),
                    {
                        "message": type(
                            "M",
                            (),
                            {"content": menu[0] if arm != "one_shot" else "\n".join(menu[:4])},
                        )()
                    },
                )()
            ]
            resp.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
            return resp

    record = run_cell(get_spec(arm), "lt", "standard", budget_k=4, seed=0, llm=llm)
    assert record.status == "ok", record.error_message
    assert record.n_experiments_distinct is not None
    assert record.n_experiments_distinct == len(set((record.chosen_experiments or "").split(",")))


class TestTeamVarsplitIsAControlNotARung:
    """`team_varsplit` must be budget-identical to `team` and outside the ladder.

    The arm exists to isolate ONE change -- what the pools partition on -- so
    any difference in provisioning or in call count would confound exactly the
    contrast it is built for.
    """

    def test_it_is_not_listed_as_a_ladder_rung(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import LADDER_ORDER

        assert "team_varsplit" not in LADDER_ORDER
        assert "team" in LADDER_ORDER

    def test_its_calibration_is_identical_to_team(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            _ladder_calibration,
            get_spec,
        )

        for k in (6, 30, 45):
            assert _ladder_calibration(get_spec("team_varsplit"), k) == (
                _ladder_calibration(get_spec("team"), k)
            )

    def test_it_declares_the_same_negotiation_cost_as_team(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import get_spec

        team, var = get_spec("team"), get_spec("team_varsplit")
        assert var.negotiation_rounds == team.negotiation_rounds
        assert var.scout_roles == team.scout_roles
        assert var.extra_kwargs == team.extra_kwargs


class TestTemperatureIsRecordedAndRouted:
    """Temperature must reach the arms that declare it, and only those.

    The failure this guards is silent in both directions: an unrouted pin looks
    like a pinned run in the logs while sampling stays at the provider default,
    and a pin forced onto the scout roles reintroduces the degeneracy
    `_SCOUT_TEMPERATURE` exists to prevent.
    """

    def test_llm_pc_receives_a_pinned_temperature(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        kwargs = _build_agent_kwargs(get_spec("llm_pc"), 30, 0, 0.05, None, temperature=0.0)
        assert kwargs["temperature"] == 0.0

    def test_an_unset_temperature_sends_no_field_at_all(self) -> None:
        """`None` must mean 'omit', not 'send None' — the recorded corpus was
        produced with the field absent."""
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        kwargs = _build_agent_kwargs(get_spec("llm_pc"), 30, 0, 0.05, None)
        assert "temperature" not in kwargs

    def test_arms_without_the_parameter_are_not_handed_one(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import (
            _build_agent_kwargs,
            get_spec,
        )

        for name in ("random", "team", "fan_in_homog"):
            kwargs = _build_agent_kwargs(get_spec(name), 30, 0, 0.05, None, temperature=0.0)
            assert "temperature" not in kwargs, name

    def test_the_record_carries_the_field(self) -> None:
        from evaluation.chamber_pipeline.results import RunRecord

        assert "temperature" in RunRecord.__dataclass_fields__
        assert RunRecord.__dataclass_fields__["temperature"].default is None
