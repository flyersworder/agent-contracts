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
import numpy as np

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


# ---------------------------------------------------------------------------
# M6 coordination ladder (Task 9)
# ---------------------------------------------------------------------------

LADDER_RUNGS = ["llm_pc", "fan_in_homog", "fan_in_spec", "planner_reasoner", "team"]


def _synthetic_ladder(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for agent in LADDER_RUNGS:
        for k in (6, 30, 45):
            for seed in range(n):
                rows.append(
                    {
                        "chamber": "lt",
                        "agent_name": agent,
                        "budget_k": k,
                        "budget_fraction": k / 59.0,
                        "seed": seed,
                        "status": "ok",
                        "f1": float(rng.normal(0.4, 0.04)),
                        "shd": 55.0,
                        "tokens_in": 1000,
                        "tokens_out": 500,
                        "wall_time_seconds": 300.0,
                        "n_llm_calls": k,
                        "n_selection_fallbacks": 0,
                        "overlap_frac": 0.3,
                    }
                )
    return pd.DataFrame(rows)


def test_ladder_frame_has_one_row_per_rung_and_budget() -> None:
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    out = ladder_frame(_synthetic_ladder())
    assert len(out) == len(LADDER_RUNGS) * 3
    assert {"f1_mean", "failure_rate", "overlap_frac_mean"} <= set(out.columns)


def test_ladder_frame_drops_non_ladder_arms() -> None:
    """`runs/m4-pilot.parquet` carries `random`, `greedy_ig`, and `llm_only`.

    The M6 table reuses that file for its `llm_pc` and `planner_reasoner`
    rows, so without a filter three non-rungs land in the ladder table and
    the rung-ordered x-axis silently gains three positions.
    """
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    df = _synthetic_ladder(n=3)
    intruder = df[df.agent_name == "llm_pc"].copy()
    intruder["agent_name"] = "random"
    out = ladder_frame(pd.concat([df, intruder], ignore_index=True))
    assert set(out.agent_name) == set(LADDER_RUNGS)


def test_ladder_frame_rows_are_in_rung_order() -> None:
    """Ladder order is loop, ensemble, parallel-roles, chain, team.

    `VARIANT_ORDER` is plan §5.1 order, which puts `planner_reasoner` BEFORE
    the fan-in arms -- reusing it would plot the chain rung as if it were
    less coordinated than the ensembles.
    """
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    out = ladder_frame(_synthetic_ladder(n=3))
    first_seen = list(dict.fromkeys(out.agent_name))
    assert first_seen == LADDER_RUNGS


def test_mde_matches_the_closed_form() -> None:
    from evaluation.chamber_pipeline.analyze_results import minimum_detectable_effect

    df = _synthetic_ladder()
    got = minimum_detectable_effect(df, "llm_pc", 30)
    sd = df[(df.agent_name == "llm_pc") & (df.budget_k == 30)].f1.std()
    assert abs(got - 2.8 * sd * np.sqrt(2 / 30)) < 1e-9


def test_mde_uses_ddof_1_not_the_numpy_default() -> None:
    """pandas `Series.std()` is ddof=1; `np.std` is ddof=0.

    The two differ by sqrt(30/29) ~ 1.7 % at n=30 -- small enough to look
    like noise, large enough to move an equivalence bound.
    """
    from evaluation.chamber_pipeline.analyze_results import minimum_detectable_effect

    df = _synthetic_ladder()
    sub = df[(df.agent_name == "llm_pc") & (df.budget_k == 30)].f1
    got = minimum_detectable_effect(df, "llm_pc", 30)
    ddof0 = 2.8 * float(np.std(sub, ddof=0)) * np.sqrt(2 / 30)
    assert abs(got - ddof0) > 1e-4


def test_failure_rate_counts_non_ok_cells() -> None:
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    df = _synthetic_ladder()
    df.loc[df.index[:3], "status"] = "error"
    out = ladder_frame(df)
    assert out.failure_rate.max() > 0


def test_failure_rate_is_computed_before_ok_filtering() -> None:
    """Filtering to ok-cells first makes every failure rate exactly 0.

    That is the whole point of the column for H-C: `planner_reasoner` timed
    out on 8 of 30 cells at k=59 in M4b, and a table that reports 0% there
    hides the arm's defining weakness.
    """
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    df = _synthetic_ladder()
    mask = (df.agent_name == "team") & (df.budget_k == 45)
    idx = df[mask].index[:6]
    df.loc[idx, "status"] = "error"
    out = ladder_frame(df)
    row = out[(out.agent_name == "team") & (out.budget_k == 45)].iloc[0]
    assert row.failure_rate == pytest.approx(6 / 30)
    assert row.n_ok == 24


def test_ladder_frame_refuses_to_merge_two_chambers() -> None:
    """LT and WT have different menu sizes, so k=30 is a different budget.

    Silently averaging them produces a row that describes neither.
    """
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    df = _synthetic_ladder(n=3)
    other = df.copy()
    other["chamber"] = "wt"
    with pytest.raises(ValueError, match="chamber"):
        ladder_frame(pd.concat([df, other], ignore_index=True))


def test_plot_ladder_writes_three_panels() -> None:
    from evaluation.chamber_pipeline.analyze_results import plot_ladder

    with tempfile.TemporaryDirectory() as tmp:
        paths = plot_ladder(_synthetic_ladder(n=5), Path(tmp))
        assert len(paths) == 3
        for p in paths:
            assert p.exists() and p.stat().st_size > 0


def test_ladder_summary_never_prints_a_delta_without_its_mde() -> None:
    """Spec §6: the paper reports an equivalence bound, not a null.

    A rung-vs-rung accuracy difference printed bare reads as a finding. The
    ladder's whole risk is that the rungs land within noise of each other,
    so every delta must carry the smallest effect this design could resolve.
    """
    from evaluation.chamber_pipeline.analyze_results import format_ladder_summary

    text = format_ladder_summary(_synthetic_ladder())
    delta_lines = [ln for ln in text.splitlines() if "delta" in ln.lower()]
    assert delta_lines, "summary printed no comparison at all"
    for line in delta_lines:
        assert "MDE" in line or "mde" in line, line


def test_ladder_summary_marks_a_within_noise_difference() -> None:
    """All rungs drawn from one distribution: nothing is resolvable."""
    from evaluation.chamber_pipeline.analyze_results import format_ladder_summary

    text = format_ladder_summary(_synthetic_ladder())
    assert "below MDE" in text


def test_ladder_summary_reports_a_real_separation_as_resolved() -> None:
    """M4b's actual gap: F1 0.75 vs 0.40 is ~9x the MDE and must not read
    as 'below MDE'."""
    from evaluation.chamber_pipeline.analyze_results import format_ladder_summary

    df = _synthetic_ladder()
    mask = (df.agent_name == "team") & (df.budget_k == 45)
    df.loc[mask, "f1"] = df.loc[mask, "f1"] + 0.35
    text = format_ladder_summary(df)
    team45 = [ln for ln in text.splitlines() if "Team" in ln and " 45 " in ln]
    assert team45, text
    assert "below MDE" not in team45[0], team45[0]


def test_ladder_frame_tolerates_a_pre_task8_frame() -> None:
    """`runs/m4-pilot.parquet` has no `overlap_frac` column.

    M6 reuses those rows for two of its five rungs, so a hard reference to a
    Task-8 column makes the ladder table unbuildable on exactly the data it
    was designed to join. Absent optional columns must read as NaN, not
    raise.
    """
    from evaluation.chamber_pipeline.analyze_results import ladder_frame

    df = _synthetic_ladder(n=3).drop(columns=["overlap_frac"])
    out = ladder_frame(df)
    assert len(out) == len(LADDER_RUNGS) * 3
    assert out["overlap_frac_mean"].isna().all()


def test_ladder_cli_runs_as_a_module() -> None:
    """Importing the module is not the same as running it.

    Every other test imports `analyze_results`, which executes the whole file
    top to bottom -- so a function defined AFTER the `__main__` guard is
    reachable from tests and a `NameError` under `python -m`. Only a
    subprocess catches that ordering.
    """
    import subprocess
    import sys as _sys

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cells.parquet"
        _synthetic_ladder(n=3).to_parquet(path)
        proc = subprocess.run(
            [
                _sys.executable,
                "-m",
                "evaluation.chamber_pipeline.analyze_results",
                "--input",
                str(path),
                "--ladder",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "MDE" in proc.stdout


# ---------------------------------------------------------------------------
# Harness validity report
# ---------------------------------------------------------------------------


def _validity_frame(
    fallbacks_by_k=None, errors_by_k=None, n=6, budgets=(6, 30, 45)
) -> pd.DataFrame:
    fallbacks_by_k = fallbacks_by_k or {}
    errors_by_k = errors_by_k or {}
    rows = []
    for agent in ("llm_pc", "team"):
        for k in budgets:
            for seed in range(n):
                is_err = seed < errors_by_k.get((agent, k), 0)
                rows.append(
                    {
                        "chamber": "lt",
                        "agent_name": agent,
                        "budget_k": k,
                        "budget_fraction": k / 59.0,
                        "seed": seed,
                        "status": "error" if is_err else "ok",
                        "f1": 0.4,
                        "shd": 55.0,
                        "tokens_in": 1000,
                        "tokens_out": 500,
                        "wall_time_seconds": 300.0,
                        "n_llm_calls": k,
                        "n_selection_fallbacks": fallbacks_by_k.get((agent, k), 0),
                        "n_pc_degeneracies": 0,
                        # Present-and-zero, not absent. A frame missing this
                        # column is UNMEASURED, not clean -- which is the
                        # distinction `missing_columns` exists to make.
                        "n_collinear_dropped": 0,
                        "n_zero_variance_dropped": 0,
                        # Present, so the frame can be CONFIRMED
                        # single-configuration. Absent, it would be
                        # UNVERIFIABLE -- which is a real warning, not noise.
                        "blas_backend": "scipy-openblas",
                        "platform_tag": "Linux-x86_64",
                        "conservation_certified": True,
                    }
                )
    return pd.DataFrame(rows)


def test_validity_report_is_clean_when_nothing_degrades() -> None:
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    report = harness_validity_report(_validity_frame())
    assert len(report) == 6  # 2 arms x 3 budgets
    assert validity_warnings(report) == []


def test_a_flat_nonzero_fallback_rate_is_flagged() -> None:
    """Any degradation is a warning, even if it does not vary with k.

    The RATE must be held flat, not the count. An earlier version fixed the
    count at 3 while `n_llm_calls = k`, so the rate was 0.50/0.10/0.067 and
    tripped the moderator branch instead -- the flat-rate branch it claimed to
    test was never reached, and deleting that branch left it green.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    budgets = (10, 30, 50)  # fb = 0.2*k is exact on these
    df = _validity_frame(
        fallbacks_by_k={("llm_pc", k): int(0.2 * k) for k in budgets}, budgets=budgets
    )
    report = harness_validity_report(df)
    llm_pc = report[report.agent_name == "llm_pc"]
    assert llm_pc.fallback_rate.max() - llm_pc.fallback_rate.min() < 1e-9  # truly flat
    warnings = validity_warnings(report)
    assert any("DEGRADED" in w and "fallback" in w for w in warnings), warnings
    assert not any("MODERATOR" in w and "fallback" in w for w in warnings), warnings


def test_a_k_varying_fallback_rate_is_flagged_as_a_moderator() -> None:
    """The signature that cost us today: 0/36 at k=6, 43% at k=30.

    A rate that changes with the budget is not merely degradation -- it is a
    moderator correlated with the independent variable, so it biases the
    measured effect rather than only adding noise. It must be called out
    distinctly from a flat rate.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame(fallbacks_by_k={("llm_pc", 6): 0, ("llm_pc", 30): 13, ("llm_pc", 45): 20})
    warnings = validity_warnings(harness_validity_report(df))
    assert any("varies with budget" in w for w in warnings), warnings
    assert any("llm_pc" in w for w in warnings), warnings


def test_a_k_varying_error_rate_is_flagged() -> None:
    """M4b's real case: 8 of 8 errors at planner_reasoner k=59 and none
    elsewhere, which deletes the slowest cells from one arm's mean."""
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame(errors_by_k={("team", 45): 2})
    warnings = validity_warnings(harness_validity_report(df))
    assert any("error rate" in w and "varies with budget" in w for w in warnings), warnings


def test_an_unrecorded_degradation_path_is_reported_as_unmeasured_not_clean() -> None:
    """A missing column must never read as a zero rate.

    `runs/m4-pilot.parquet` predates `n_selection_fallbacks`, so a report over
    it shows fallback_rate 0.00 -- identical to a harness that never fell back.
    In fact ~43% of its k=30 selections were random. Silence about an
    unmeasured path is exactly the failure this report exists to prevent, so
    absence has to be louder than zero.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame().drop(columns=["n_selection_fallbacks"])
    warnings = validity_warnings(harness_validity_report(df))
    assert any("UNMEASURED" in w and "n_selection_fallbacks" in w for w in warnings), warnings


def test_ladder_output_puts_harness_validity_before_accuracy() -> None:
    """Order is the point, not decoration.

    A degradation rate that varies with budget biases the accuracy numbers, so
    reading accuracy first means reading numbers you may have to discard. The
    whole M4b pilot was interpreted before anyone checked whether the harness
    was working. Print the warnings above the table so they cannot be skipped.
    """
    import io
    from contextlib import redirect_stdout

    from evaluation.chamber_pipeline.analyze_results import main

    df = _synthetic_ladder(n=3)
    df.loc[df.index[:5], "n_selection_fallbacks"] = 4  # a real degradation
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cells.parquet"
        df.to_parquet(path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--input", str(path), "--ladder"])
    out = buf.getvalue()

    assert "HARNESS VALIDITY" in out, out[-800:]
    assert "DEGRADED" in out
    # The validity block must come FIRST.
    assert out.index("HARNESS VALIDITY") < out.index("Rung"), (
        "accuracy table printed above the validity warnings"
    )


def test_conservation_variance_is_not_an_accuracy_moderator() -> None:
    """A k-varying conservation rate is a FINDING, not a confound.

    `conservation_certified` comes from a post-hoc `verify()`. Token budgets
    are non-binding at execution -- node monitors record tokens for
    certification arithmetic and must not halt; only interventions are
    live-gated -- so a conservation failure cannot change what the agent did
    or what PC inferred. Warning that it "biases the measured effect" is
    wrong, and it would fire forever: we predicted k-varying conservation from
    the 48.8x cost spread at k=6.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame(n=10)
    mask = (df.agent_name == "llm_pc") & (df.budget_k == 45)
    df["conservation_certified"] = True
    df.loc[mask, "conservation_certified"] = False
    warnings = validity_warnings(harness_validity_report(df))
    moderators = [w for w in warnings if "MODERATOR" in w]
    assert not any("conservation" in w for w in moderators), moderators
    # Still reported, just not as a bias.
    assert any("conservation" in w and "FINDING" in w for w in warnings), warnings


def test_pc_degeneracy_variance_is_an_accuracy_moderator() -> None:
    """Degenerate PC returns an all-zeros adjacency, so it zeroes F1
    directly. A rate that varies with k biases the comparison."""
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame(n=10)
    df["n_pc_degeneracies"] = 0
    df.loc[(df.agent_name == "llm_pc") & (df.budget_k == 6), "n_pc_degeneracies"] = 1
    warnings = validity_warnings(harness_validity_report(df))
    assert any("MODERATOR" in w and "degeneracy" in w for w in warnings), warnings


def test_collinear_drop_rate_is_a_fraction_not_a_count() -> None:
    """`collinear_drop_rate` must live in [0, 1].

    Every consumer -- `_VARIATION_EPS` (0.02) and the DEGRADED threshold --
    assumes a rate. WT drops three redundant barometers in essentially every
    cell, so a columns-per-cell encoding sits near 3.1 and its ordinary
    sampling wobble (3.27 vs 2.90) reads as a 0.37 "variation with budget",
    false-alarming as a MODERATOR on a path that is in fact perfectly flat.

    The magnitude is not lost -- it moves to `collinear_cols_mean`.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    df = _validity_frame()
    # Three columns dropped in every cell: constant, and far above 1.0 if
    # anyone re-encodes this as a count.
    df["n_collinear_dropped"] = 3

    report = harness_validity_report(df)
    assert (report.collinear_drop_rate <= 1.0).all()
    assert (report.collinear_drop_rate == 1.0).all()
    assert (report.collinear_cols_mean == 3.0).all()

    moderators = [w for w in validity_warnings(report) if w.startswith("MODERATOR")]
    assert not moderators, f"flat path must not be a moderator; got {moderators}"


def test_suspend_between_the_two_clocks_is_surfaced() -> None:
    """A machine that sleeps mid-sweep must not look like slow cells.

    `wall_time_seconds` comes from `time.perf_counter()`, which on macOS does
    not advance while the system is asleep; `started_at`/`finished_at` are
    wall-clock and do. On 2026-08-25 the WT gate ran 1.01h of active worker
    time inside a 6.66h span -- 5.6h of idle sleep on battery -- and the gap
    was initially misread as a 5x error in the cost model.

    Surfacing it costs nothing (both columns already exist) and turns an
    invisible multi-day stretch into a reported number.
    """
    from evaluation.chamber_pipeline.analyze_results import harness_validity_report

    df = _validity_frame()
    # One cell that "took" 100s of compute but spanned 700s of wall clock.
    df.loc[df.index[0], "started_at"] = "2026-08-25T10:00:00"
    df.loc[df.index[0], "finished_at"] = "2026-08-25T10:11:40"
    df.loc[df.index[0], "wall_time_seconds"] = 100.0

    report = harness_validity_report(df)
    assert "suspend_seconds" in report.columns
    row = report[
        (report.agent_name == df.iloc[0]["agent_name"])
        & (report.budget_k == df.iloc[0]["budget_k"])
    ].iloc[0]
    assert row.suspend_seconds >= 500, (
        "a 700s span around 100s of compute is ~600s of suspend; reporting 0 "
        "would hide exactly the condition that stretched a 10h sweep to days"
    )


def test_absent_zero_variance_column_is_reported_but_not_as_contamination() -> None:
    """Its absence is a gap in decomposition, not evidence of a biased arm.

    Distinguished from the `UNMEASURED` notices because a zero-variance drop
    cannot bias an arm-vs-arm comparison: every arm at one budget faces the
    same padding. Conflating the two would teach a reader to discount the
    warning that does mean contamination.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    frame = _validity_frame().drop(columns=["n_zero_variance_dropped"])
    warns = validity_warnings(harness_validity_report(frame))
    assert len(warns) == 1, warns
    assert warns[0].startswith("NOT RECORDED: n_zero_variance_dropped")
    assert "UNMEASURED" not in warns[0]


def test_validity_warnings_on_an_empty_selection_is_empty_not_a_crash() -> None:
    """`--ladder` on a WT file while `--check-chamber` still defaults to 'lt'
    filters to zero rows. `harness_validity_report` then builds
    `pd.DataFrame([])`, which has NO COLUMNS, and the groupby raised
    KeyError('agent_name') -- a bare traceback where an empty table belongs.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    empty = _validity_frame().iloc[0:0]
    report = harness_validity_report(empty)
    assert report.empty
    assert validity_warnings(report) == []


class TestProvenanceHomogeneity:
    """Rows from two configurations must not become one mean.

    The rule "never pool rows whose `blas_backend` differs" lived in prose in
    three documents and in no code, while every RunRecord carried the field
    and no analyzer read it. Register entry 10 measures the cost of breaking
    it at |dF1| = 0.055 -- larger than most effects this pillar reports.
    """

    @staticmethod
    def _frame(n: int = 4, **overrides) -> pd.DataFrame:
        base = {
            "chamber": "lt",
            "agent_name": "llm_pc",
            "budget_k": 30,
            "budget_fraction": 0.5,
            "seed": 0,
            "status": "ok",
            "shd": 50,
            "f1": 0.4,
            "wall_time_seconds": 10.0,
            "started_at": "2026-08-29T10:00:00",
            "finished_at": "2026-08-29T10:00:10",
            "blas_backend": "scipy-openblas",
            "platform_tag": "Linux-x86_64",
        }
        rows = []
        for i in range(n):
            row = dict(base, seed=i)
            row.update({k: v[i] for k, v in overrides.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_a_homogeneous_frame_passes(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import provenance_problems

        assert provenance_problems(self._frame()) == []

    def test_two_backends_are_refused_by_the_aggregators(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import (
            MixedProvenanceError,
            aggregate_pareto,
            ladder_frame,
        )

        mixed = self._frame(
            blas_backend=["scipy-openblas", "scipy-openblas", "accelerate", "accelerate"]
        )
        for fn in (aggregate_pareto, ladder_frame):
            with pytest.raises(MixedProvenanceError, match="blas_backend"):
                fn(mixed)

    def test_a_partly_stamped_frame_is_refused(self) -> None:
        """The mix anyone is actually likely to make: a legacy Parquet
        concatenated with a current one. Judging on `dropna().unique()` alone
        calls this homogeneous -- one distinct backend plus a block of nulls --
        so the guard would miss `m4-pilot` + `m6-lt-loop-curve`, the exact
        pooling the register warns about."""
        from evaluation.chamber_pipeline.analyze_results import (
            MixedProvenanceError,
            provenance_problems,
            require_homogeneous_provenance,
        )

        partial = self._frame(
            blas_backend=["scipy-openblas", "scipy-openblas", None, None],
            platform_tag=["Linux-x86_64", "Linux-x86_64", None, None],
        )
        problems = provenance_problems(partial)
        assert any("UNSTAMPED on 2" in p for p in problems)
        with pytest.raises(MixedProvenanceError):
            require_homogeneous_provenance(partial)

    def test_a_fully_unstamped_frame_is_allowed_but_flagged_unverifiable(self) -> None:
        """Absent is weaker than mixed: a pre-2026-08-26 sweep is analysable
        on its own, it just cannot be CONFIRMED single-configuration."""
        from evaluation.chamber_pipeline.analyze_results import (
            harness_validity_report,
            provenance_problems,
            validity_warnings,
        )

        legacy = self._frame().drop(columns=["blas_backend", "platform_tag"])
        assert provenance_problems(legacy) == []
        warns = validity_warnings(harness_validity_report(legacy))
        assert any(w.startswith("UNVERIFIABLE: blas_backend") for w in warns)

    def test_the_escape_hatch_exists_for_measuring_the_difference(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import require_homogeneous_provenance

        mixed = self._frame(
            blas_backend=["scipy-openblas", "scipy-openblas", "accelerate", "accelerate"]
        )
        require_homogeneous_provenance(mixed, allow_mixed=True)  # must not raise


def test_a_partly_recorded_counter_is_flagged_not_averaged_as_zero() -> None:
    """`fillna(0)` makes unrecorded rows read as clean ones.

    Reachable by normal workflow: the JSONL sidecar resumes across a counter's
    introduction, so older lines carry None and newer ones an int. Measured on
    this frame the report shows mean 19.0 for a true 38 and rate 0.50 for a
    true 1.00, so the warning is the only thing standing between a reader and
    a halved degradation rate.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    frame = _validity_frame()
    frame["n_pc_degeneracies"] = 0
    frame["n_zero_variance_dropped"] = [38 if i % 2 == 0 else None for i in range(len(frame))]
    warns = validity_warnings(harness_validity_report(frame))
    assert any(w.startswith("PARTIALLY RECORDED: n_zero_variance_dropped") for w in warns)


def test_a_counter_null_because_pc_never_ran_is_not_flagged_as_partial() -> None:
    """A null counter is legitimate when inference did not happen -- and since
    `n_pc_degeneracies` is now itself None exactly then, the two cases are
    distinguishable without guessing from the arm name."""
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    frame = _validity_frame()
    frame["n_pc_degeneracies"] = [0 if i % 2 == 0 else None for i in range(len(frame))]
    frame["n_zero_variance_dropped"] = [3 if i % 2 == 0 else None for i in range(len(frame))]
    warns = validity_warnings(harness_validity_report(frame))
    assert not any(w.startswith("PARTIALLY RECORDED") for w in warns)


def test_allow_mixed_provenance_reaches_the_aggregators() -> None:
    """A flag honoured only at load time is a flag that does nothing.

    `main` calls `aggregate_pareto` on every invocation and both aggregators
    hold the same guard, so the advertised escape hatch raised anyway.
    """
    from evaluation.chamber_pipeline.analyze_results import (
        MixedProvenanceError,
        aggregate_pareto,
        ladder_frame,
    )

    frame = _validity_frame()
    frame["blas_backend"] = ["scipy-openblas"] * (len(frame) // 2) + ["accelerate"] * (
        len(frame) - len(frame) // 2
    )
    for fn in (aggregate_pareto, ladder_frame):
        with pytest.raises(MixedProvenanceError):
            fn(frame)
        fn(frame, allow_mixed_provenance=True)  # must not raise


def test_an_all_null_provenance_column_is_unverifiable_not_confirmed() -> None:
    """A Parquet consolidated from a fully-legacy sidecar has the column and
    no values. Skipping it as homogeneous reports "single configuration"
    where nothing was checked."""
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        provenance_problems,
        validity_warnings,
    )

    frame = _validity_frame()
    frame["blas_backend"] = None
    frame["platform_tag"] = None
    assert provenance_problems(frame) == []
    warns = validity_warnings(harness_validity_report(frame))
    assert any(w.startswith("UNVERIFIABLE: blas_backend") for w in warns)
    assert any(w.startswith("UNVERIFIABLE: platform_tag") for w in warns)


def test_a_non_llm_arm_does_not_trip_the_partial_counter_check() -> None:
    """`random` and `greedy_ig` run PC and make no LLM calls, so
    `n_selection_fallbacks` is legitimately null while `n_pc_degeneracies` is
    not. Testing an LLM counter against `pc_ran` flagged every ladder frame
    carrying a random baseline."""
    from evaluation.chamber_pipeline.analyze_results import (
        harness_validity_report,
        validity_warnings,
    )

    frame = _validity_frame()
    frame["n_pc_degeneracies"] = 0
    frame["n_llm_calls"] = [30 if i % 2 == 0 else None for i in range(len(frame))]
    frame["n_selection_fallbacks"] = [0 if i % 2 == 0 else None for i in range(len(frame))]
    warns = validity_warnings(harness_validity_report(frame))
    assert not any(w.startswith("PARTIALLY RECORDED") for w in warns)


class TestCostFrontier:
    """Accuracy against coordination cost, in calls rather than dollars."""

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        base = {
            "chamber": "lt",
            "budget_fraction": 0.5,
            "status": "ok",
            "shd": 50,
            "wall_time_seconds": 1.0,
            "started_at": "x",
            "finished_at": "y",
            "blas_backend": "scipy-openblas",
            "platform_tag": "Linux-x86_64",
        }
        return pd.DataFrame(
            [
                dict(base, agent_name=a, budget_k=30, seed=i, f1=f, n_llm_calls=c, cost_usd=u)
                for a, f, c, u in rows
                for i in range(3)
            ]
        )

    def test_an_arm_beaten_on_both_axes_is_dominated(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import cost_frontier

        out = cost_frontier(
            self._frame([("cheap_good", 0.5, 3, 0.01), ("dear_bad", 0.4, 30, 0.10)])
        )
        by = dict(zip(out.agent_name, out.dominated, strict=True))
        assert by["dear_bad"] is True
        assert by["cheap_good"] is False

    def test_the_most_accurate_arm_is_optimal_however_dear(self) -> None:
        """Pareto, not value-for-money: nothing beats it on BOTH axes."""
        from evaluation.chamber_pipeline.analyze_results import cost_frontier

        out = cost_frontier(
            self._frame([("cheap_bad", 0.3, 3, 0.01), ("dear_best", 0.6, 50, 0.20)])
        )
        by = dict(zip(out.agent_name, out.dominated, strict=True))
        assert by["dear_best"] is False
        assert by["cheap_bad"] is False, "cheapest is also on the frontier"

    def test_a_tie_on_both_axes_dominates_neither(self) -> None:
        """`<=` and `>=` alone would call two identical arms dominated."""
        from evaluation.chamber_pipeline.analyze_results import cost_frontier

        out = cost_frontier(self._frame([("a", 0.4, 10, 0.05), ("b", 0.4, 10, 0.05)]))
        assert not out.dominated.any()

    def test_the_table_stays_aligned_for_a_long_label(self) -> None:
        from evaluation.chamber_pipeline.analyze_results import format_cost_frontier

        text = format_cost_frontier(
            self._frame([("fan_in_spec", 0.4, 31, 0.1), ("llm_pc", 0.42, 30, 0.1)])
        )
        rows = [ln for ln in text.splitlines() if "0.4" in ln]
        assert len({ln.index("0.4") for ln in rows}) == 1, "F1 column must line up"
