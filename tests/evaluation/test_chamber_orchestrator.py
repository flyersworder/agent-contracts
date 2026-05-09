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
    _PcDegeneracyHandler,
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
