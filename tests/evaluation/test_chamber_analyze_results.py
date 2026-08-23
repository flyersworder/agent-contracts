"""Tests for the chamber-pillar analysis layer.

Covers `evaluation.chamber_pipeline.analyze_results` — load_records,
aggregate_pareto, plotting, M4 acceptance check, and CLI entry.

Tests run against synthetic RunRecord data so this file does not
need `causalchamber` or any LLM. Validates the figure-generation
pipeline end-to-end before M4b commits real OpenRouter spend.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless tests

import pandas as pd
import pytest

from evaluation.chamber_pipeline.analyze_results import (
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    aggregate_pareto,
    build_arg_parser,
    check_m4_acceptance,
    format_acceptance_summary,
    load_records,
    main,
    make_pareto_figure,
    plot_pareto,
)
from evaluation.chamber_pipeline.results import RunRecord, write_records_parquet

# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


def _record(
    chamber: str = "lt",
    agent_name: str = "random",
    budget_k: int = 6,
    budget_fraction: float = 0.10,
    seed: int = 0,
    status: str = "ok",
    shd: float | None = 30.0,
    f1: float | None = 0.5,
    n_pc_degeneracies: int | None = 0,
    **overrides,
) -> RunRecord:
    """Build a RunRecord with sensible defaults for synthetic test data."""
    base = {
        "chamber": chamber,
        "configuration": "standard",
        "agent_name": agent_name,
        "budget_k": budget_k,
        "budget_fraction": budget_fraction,
        "seed": seed,
        "status": status,
        "started_at": "2026-05-09T00:00:00",
        "finished_at": "2026-05-09T00:00:01",
        "shd": shd,
        "f1": f1,
        "n_edges_predicted": 20,
        "n_edges_truth": 57,
        "wall_time_seconds": 1.0,
        "n_llm_calls": None,
        "n_pc_degeneracies": n_pc_degeneracies,
    }
    base.update(overrides)
    return RunRecord(**base)


def _synthetic_pilot_records(n_seeds: int = 30) -> list[RunRecord]:
    """Generate a synthetic 450-cell M4 pilot result.

    Hand-crafted so the resulting Pareto curves are monotonic and
    Random sits below LLM variants at every budget — i.e., the
    synthetic data passes the M4 acceptance check. Used by tests
    that verify the analyzer's "pass" path.
    """
    # Mean SHD per (variant, budget_fraction). Random is highest (worst);
    # LLM variants beat it; planner_reasoner is best at higher budgets.
    means: dict[tuple[str, float], float] = {
        ("random", 0.10): 80.0,
        ("random", 0.50): 65.0,
        ("random", 1.00): 50.0,
        ("greedy_ig_lite", 0.10): 75.0,
        ("greedy_ig_lite", 0.50): 55.0,
        ("greedy_ig_lite", 1.00): 40.0,
        ("llm_only", 0.10): 70.0,
        ("llm_only", 0.50): 50.0,
        ("llm_only", 1.00): 35.0,
        ("llm_pc", 0.10): 65.0,
        ("llm_pc", 0.50): 45.0,
        ("llm_pc", 1.00): 30.0,
        ("planner_reasoner", 0.10): 65.0,
        ("planner_reasoner", 0.50): 42.0,
        ("planner_reasoner", 1.00): 28.0,
    }
    bf_to_k = {0.10: 6, 0.50: 30, 1.00: 59}
    records: list[RunRecord] = []
    import random as _rng

    rng = _rng.Random(42)
    for (variant, bf), shd_mean in means.items():
        for seed in range(n_seeds):
            # Add small Gaussian-ish noise so std is non-zero.
            noise = (rng.random() - 0.5) * 4.0
            records.append(
                _record(
                    agent_name=variant,
                    budget_fraction=bf,
                    budget_k=bf_to_k[bf],
                    seed=seed,
                    shd=shd_mean + noise,
                    f1=max(0.0, min(1.0, 0.7 - shd_mean / 200.0)),
                )
            )
    return records


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


class TestLoadRecords:
    """File-format detection + schema validation."""

    def test_loads_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.parquet")
            write_records_parquet([_record(), _record(seed=1)], path)
            df = load_records(path)
            assert len(df) == 2

    def test_loads_csv(self) -> None:
        from evaluation.chamber_pipeline.results import write_records_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            write_records_csv([_record(), _record(seed=1)], path)
            df = load_records(path)
            assert len(df) == 2

    def test_unknown_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.txt")
            Path(path).write_text("not a real file")
            with pytest.raises(ValueError, match="extension"):
                load_records(path)

    def test_missing_columns_raises(self) -> None:
        """File missing required columns surfaces a clear error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.parquet")
            pd.DataFrame({"foo": [1, 2]}).to_parquet(path, index=False)
            with pytest.raises(ValueError, match="missing required columns"):
                load_records(path)


# ---------------------------------------------------------------------------
# aggregate_pareto
# ---------------------------------------------------------------------------


class TestAggregatePareto:
    """Per-cell aggregation into per-(chamber, agent, budget) Pareto points."""

    def test_drops_non_ok_cells(self) -> None:
        records = [
            _record(seed=0, shd=30.0),
            _record(seed=1, shd=40.0),
            _record(seed=2, status="skipped", shd=None),
            _record(seed=3, status="error", shd=None),
        ]
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        # Only the 2 ok cells are aggregated.
        row = agg.iloc[0]
        assert row["n_seeds"] == 2
        assert row["shd_mean"] == 35.0

    def test_groups_by_chamber_agent_budget(self) -> None:
        records = [_record(chamber="lt", agent_name="random", seed=s) for s in range(3)] + [
            _record(chamber="lt", agent_name="llm_pc", seed=s) for s in range(3)
        ]
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        assert len(agg) == 2
        assert set(agg["agent_name"]) == {"random", "llm_pc"}

    def test_sem_is_zero_for_single_seed(self) -> None:
        """std/sqrt(1) of a single observation is NaN → coerced to 0.0."""
        df = pd.DataFrame.from_records([_record(seed=0, shd=42.0).to_dict()])
        agg = aggregate_pareto(df)
        assert agg.iloc[0]["shd_sem"] == 0.0

    def test_empty_input_yields_empty_output(self) -> None:
        df = pd.DataFrame.from_records([_record(status="error", shd=None).to_dict()])
        agg = aggregate_pareto(df)
        assert agg.empty


# ---------------------------------------------------------------------------
# Plotting (smoke + content checks)
# ---------------------------------------------------------------------------


class TestPlotPareto:
    """Plot-rendering smoke tests."""

    def test_renders_for_lt(self) -> None:
        records = _synthetic_pilot_records(n_seeds=5)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)

        ax = plot_pareto(agg, chamber="lt")
        # Every variant PRESENT IN THE DATA shows up in the legend (more
        # robust than counting Line2D objects, which errorbar inflates by
        # adding cap lines). Compared against the data rather than against
        # all of VARIANT_LABELS: the ladder arms are registered and styled
        # but absent from this fixture, and a legend entry for a variant with
        # no cells would be the bug, not the fix.
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        expected = {VARIANT_LABELS[v] for v in agg["agent_name"].unique()}
        assert legend_labels == expected
        # Title mentions chamber.
        assert "LT" in ax.get_title()

    def test_renders_f1_metric(self) -> None:
        records = _synthetic_pilot_records(n_seeds=3)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        ax = plot_pareto(agg, chamber="lt", metric="f1")
        assert "F1" in ax.get_ylabel() or "F1" in ax.get_title()

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            plot_pareto(pd.DataFrame(), chamber="lt", metric="bogus")

    def test_empty_chamber_renders_message(self) -> None:
        """Plotting against a missing chamber doesn't crash; shows a placeholder."""
        records = _synthetic_pilot_records(n_seeds=3)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        # 'wt' isn't in the synthetic data.
        ax = plot_pareto(agg, chamber="wt")
        # No data lines were drawn.
        assert len(ax.get_lines()) == 0


class TestMakeParetoFigure:
    """End-to-end figure construction."""

    def test_single_chamber_returns_single_panel(self) -> None:
        records = _synthetic_pilot_records(n_seeds=3)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        fig = make_pareto_figure(agg)
        # One panel.
        assert len(fig.axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_combined_two_chambers_returns_two_panels(self) -> None:
        # Synth records for both chambers.
        records_lt = _synthetic_pilot_records(n_seeds=3)
        records_wt = [
            _record(
                chamber="wt",
                agent_name=v,
                budget_fraction=bf,
                budget_k=int(28 * bf),
                seed=s,
                shd=50.0 - bf * 20,
            )
            for v in ("random", "llm_only", "llm_pc", "planner_reasoner")
            for bf in (0.10, 0.50, 1.00)
            for s in range(3)
        ]
        df = pd.DataFrame.from_records([r.to_dict() for r in records_lt + records_wt])
        agg = aggregate_pareto(df)
        fig = make_pareto_figure(agg, combined=True)
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# M4 acceptance check
# ---------------------------------------------------------------------------


class TestM4AcceptanceCheck:
    """Plan §9 M4 acceptance criterion verification."""

    def test_synthetic_pass_data_passes(self) -> None:
        """Hand-crafted monotonic-and-LLM-beats-Random data → overall_pass=True."""
        records = _synthetic_pilot_records(n_seeds=30)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        result = check_m4_acceptance(agg, chamber="lt")
        assert result["overall_pass"] is True
        # All five variants are monotonic.
        assert all(result["monotonic"].values())

    def test_random_better_than_llm_fails_dominance_check(self) -> None:
        """If Random somehow has the lowest SHD, dominance fails."""
        # Construct data where llm_pc is WORSE than random at all budgets.
        records: list[RunRecord] = []
        for variant, shds in [
            ("random", [30.0, 25.0, 20.0]),
            ("llm_pc", [40.0, 35.0, 30.0]),
        ]:
            for bf, shd_val in zip([0.10, 0.50, 1.00], shds, strict=True):
                for seed in range(5):
                    records.append(
                        _record(
                            agent_name=variant,
                            budget_fraction=bf,
                            budget_k=int(59 * bf),
                            seed=seed,
                            shd=shd_val,
                        )
                    )
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        result = check_m4_acceptance(agg, chamber="lt")
        # llm_pc never beats random → highest-budget dominance fails → overall fail.
        assert result["overall_pass"] is False
        assert result["random_dominated"][1.00] == []  # nobody beats random at k/M=1.00

    def test_non_monotonic_violation_flagged(self) -> None:
        """A variant whose SHD spikes upward at a later budget is flagged."""
        records: list[RunRecord] = []
        # llm_pc goes 30 → 50 → 25 (non-monotonic: 30 → 50)
        for bf, shd_val in zip([0.10, 0.50, 1.00], [30.0, 50.0, 25.0], strict=True):
            for seed in range(20):
                records.append(
                    _record(
                        agent_name="llm_pc",
                        budget_fraction=bf,
                        budget_k=int(59 * bf),
                        seed=seed,
                        shd=shd_val,
                    )
                )
        # Add random for the dominance check.
        for bf, shd_val in zip([0.10, 0.50, 1.00], [80.0, 70.0, 60.0], strict=True):
            for seed in range(20):
                records.append(
                    _record(
                        agent_name="random",
                        budget_fraction=bf,
                        budget_k=int(59 * bf),
                        seed=seed,
                        shd=shd_val,
                    )
                )
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        result = check_m4_acceptance(agg, chamber="lt")
        assert result["monotonic"]["llm_pc"] is False
        assert "llm_pc" in result["monotonic_violations"]

    def test_format_summary_contains_key_phrases(self) -> None:
        records = _synthetic_pilot_records(n_seeds=10)
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
        agg = aggregate_pareto(df)
        result = check_m4_acceptance(agg, chamber="lt")
        summary = format_acceptance_summary(result)
        assert "M4 acceptance criteria" in summary
        assert "Pareto curve monotonic" in summary
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    """The CLI entry point — argparse + figure writes + acceptance exit codes."""

    def test_arg_parser_accepts_basic_args(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--input", "/tmp/x.parquet"])
        assert args.input == "/tmp/x.parquet"
        assert args.out_dir is None
        assert args.combined is False

    def test_cli_writes_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "data.parquet")
            out_dir = os.path.join(tmp, "figs")
            records = _synthetic_pilot_records(n_seeds=5)
            write_records_parquet(records, in_path)
            rc = main(["--input", in_path, "--out-dir", out_dir])
            assert rc == 0
            # Two figures: shd + f1.
            assert os.path.exists(os.path.join(out_dir, "pareto_shd.png"))
            assert os.path.exists(os.path.join(out_dir, "pareto_f1.png"))

    def test_cli_acceptance_check_returns_zero_on_pass(self, capsys) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "data.parquet")
            records = _synthetic_pilot_records(n_seeds=30)
            write_records_parquet(records, in_path)
            rc = main(["--input", in_path, "--check-m4-acceptance"])
            assert rc == 0
            out = capsys.readouterr().out
            assert "PASS" in out

    def test_cli_returns_nonzero_when_no_ok_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "data.parquet")
            # Only error cells.
            records = [_record(status="error", shd=None) for _ in range(5)]
            write_records_parquet(records, in_path)
            rc = main(["--input", in_path])
            assert rc == 1


# ---------------------------------------------------------------------------
# Constants smoke
# ---------------------------------------------------------------------------


def test_variant_color_label_keys_match_order() -> None:
    """The three variant-keyed dicts share the same key set."""
    assert set(VARIANT_COLORS.keys()) == set(VARIANT_LABELS.keys())
    assert set(VARIANT_ORDER) == set(VARIANT_COLORS.keys())


def test_every_registered_variant_has_full_plot_styling():
    """A registered arm missing from these tables is silently dropped.

    Every figure and summary table iterates `VARIANT_ORDER`, so an arm absent
    from it renders as nothing at all -- no error, no warning, just a missing
    curve in the paper's headline figure.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        VARIANT_COLORS,
        VARIANT_LABELS,
        VARIANT_LINESTYLES,
        VARIANT_MARKERS,
        VARIANT_ORDER,
    )
    from evaluation.chamber_pipeline.orchestrator import AGENT_REGISTRY

    registered = {spec.name for spec in AGENT_REGISTRY}
    assert registered <= set(VARIANT_ORDER), registered - set(VARIANT_ORDER)
    for table in (
        VARIANT_COLORS,
        VARIANT_LABELS,
        VARIANT_MARKERS,
        VARIANT_LINESTYLES,
    ):
        assert registered <= set(table), registered - set(table)
