"""Analysis layer for chamber-pillar sweep results.

Consumes Parquet/CSV output from `run_experiment.py` and produces
plan §5.3's headline Pareto figure (SHD vs intervention budget,
one line per variant per chamber). Also provides a quick check
against plan §9 M4's acceptance criterion ("preliminary Pareto
curve monotonic; Random sits below LLM variants").

Usage:

    # Generate Pareto plots from a finished sweep
    python -m evaluation.chamber_pipeline.analyze_results \\
        --input runs/m4-pilot.parquet --out-dir runs/m4-figures/

    # Check M4 acceptance criteria + print summary
    python -m evaluation.chamber_pipeline.analyze_results \\
        --input runs/m4-pilot.parquet --check-m4-acceptance

Design choices:

- **Aggregation is data-driven**, not config-driven. The analysis
  layer reads whatever (chamber, agent_name, budget_fraction) cells
  are present in the input Parquet and aggregates across seeds.
  Adding a new variant or budget level needs no analyzer change —
  the figure picks it up automatically.
- **Per-chamber figures, not combined**. LT and WT have different
  variant counts (5 vs 4 per plan §5.1) and different intervention
  scales (M=59 vs M=28). One figure per chamber keeps comparisons
  clean. A `--combined` mode produces a side-by-side grid for
  publication; default is per-chamber.
- **Acceptance check is permissive on monotonicity**. The plan
  §9 M4 criterion says "preliminary Pareto curve monotonic," but
  N=30 seeds with the LLM stochasticity of DeepSeek Flash will
  produce some non-monotonicities at budget transitions. We
  measure "weakly monotonic" (each step is no more than 1.5sigma
  above the previous step's mean) and flag deviations rather
  than fail outright.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Plan §5.3 figure: one color per variant, consistent across panels.
# Order matches plan §5.1 row numbering for the legend.
VARIANT_COLORS: dict[str, str] = {
    "random": "#888888",  # neutral gray — Pareto floor
    "greedy_ig_lite": "#1f77b4",  # blue — non-LLM principled
    "llm_only": "#ff7f0e",  # orange — pure LLM
    "llm_pc": "#2ca02c",  # green — main hybrid
    "planner_reasoner": "#d62728",  # red — multi-agent ⭐
    # M6 coordination ladder
    "fan_in_homog": "#9467bd",  # purple — ensemble
    "fan_in_spec": "#8c564b",  # brown — parallel roles
    "team": "#e377c2",  # pink — negotiation
    "fan_in_agg": "#17becf",  # cyan — rung-1 ablation, not a rung
}

VARIANT_LABELS: dict[str, str] = {
    "random": "Random",
    "greedy_ig_lite": "GreedyIG-lite",
    "llm_only": "LLM-only",
    "llm_pc": "LLM+PC",
    "planner_reasoner": "Planner+Reasoner",
    "fan_in_homog": "Ensemble (fan-in)",
    "fan_in_spec": "Parallel roles (fan-in)",
    "team": "Team (negotiation)",
    "fan_in_agg": "Ensemble (aggregator honored)",
}

# Marker + linestyle per variant so curves stay distinguishable when the
# figure is printed in black and white (colors alone collapse in grayscale).
VARIANT_MARKERS: dict[str, str] = {
    "random": "o",
    "greedy_ig_lite": "s",
    "llm_only": "^",
    "llm_pc": "D",
    "planner_reasoner": "v",
    "fan_in_homog": "P",
    "fan_in_spec": "X",
    "team": "*",
    "fan_in_agg": "P",
}

VARIANT_LINESTYLES: dict[str, str | tuple[int, tuple[int, ...]]] = {
    "random": ":",
    "greedy_ig_lite": "--",
    "llm_only": "-",
    "llm_pc": "-.",
    "planner_reasoner": (0, (3, 1, 1, 1, 1, 1)),
    "fan_in_homog": (0, (5, 1)),
    "fan_in_spec": (0, (1, 1)),
    "team": (0, (3, 5, 1, 5)),
    "fan_in_agg": (0, (3, 1, 1, 1)),
}

# Variant rendering order in legend (matches plan §5.3 description top-to-bottom).
VARIANT_ORDER: tuple[str, ...] = (
    "random",
    "greedy_ig_lite",
    "llm_only",
    "llm_pc",
    "planner_reasoner",
    # M6 ladder, in rung order. Every figure and summary table iterates this
    # tuple, so an arm missing here is silently dropped from the output --
    # no error, no warning, just an absent curve.
    "fan_in_homog",
    "fan_in_spec",
    # Ablation of rung 1, ordered beside it. Deliberately NOT in
    # `LADDER_ORDER`: it is not a rung, and the ladder table must not list it.
    "fan_in_agg",
    "team",
)


# ---------------------------------------------------------------------------
# IO + aggregation
# ---------------------------------------------------------------------------


def load_records(path: str | Path) -> pd.DataFrame:
    """Load a Parquet or CSV produced by `run_experiment.py`.

    Auto-detects format from the file extension. Validates the schema
    by checking for the columns the rest of this module depends on —
    surfaces a clear error if the file is from an old M4a build (e.g.,
    pre-M4a.1 sweeps lack `tokens_in`/`cost_usd` columns; we tolerate
    this for backward-compat).

    Returns:
        DataFrame with one row per cell. Schema matches `RunRecord.to_dict`.
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unrecognized extension {path.suffix!r}; expected .parquet or .csv")
    required_cols = {
        "chamber",
        "agent_name",
        "budget_k",
        "budget_fraction",
        "seed",
        "status",
        "shd",
        "f1",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file missing required columns: {sorted(missing)}. "
            f"Was this Parquet produced by `run_experiment.py`?"
        )
    return df


def aggregate_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cell-level records into per-(chamber, agent, budget) Pareto points.

    Drops non-"ok" cells (skipped/error) before aggregating — those
    don't contribute to the Pareto curve. Computes mean and standard
    error of the mean (SEM = std / sqrt(n)) for SHD and F1, plus the
    sample count.

    Args:
        df: Cell-level DataFrame from `load_records`.

    Returns:
        DataFrame with one row per (chamber, agent_name, budget_fraction)
        combination. Columns: chamber, agent_name, budget_k, budget_fraction,
        n_seeds, shd_mean, shd_sem, f1_mean, f1_sem.
    """
    ok_only = df[df["status"] == "ok"].copy()
    if ok_only.empty:
        return pd.DataFrame(
            columns=[
                "chamber",
                "agent_name",
                "budget_k",
                "budget_fraction",
                "n_seeds",
                "shd_mean",
                "shd_sem",
                "f1_mean",
                "f1_sem",
            ]
        )

    grouped = ok_only.groupby(
        ["chamber", "agent_name", "budget_k", "budget_fraction"], as_index=False
    )
    agg = grouped.agg(
        n_seeds=("seed", "count"),
        shd_mean=("shd", "mean"),
        shd_std=("shd", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    # Standard error of the mean for the error bars. For n=1 the std
    # is NaN; surface that as 0.0 (single observation has no spread).
    agg["shd_sem"] = (agg["shd_std"] / np.sqrt(agg["n_seeds"])).fillna(0.0)
    agg["f1_sem"] = (agg["f1_std"] / np.sqrt(agg["n_seeds"])).fillna(0.0)
    return agg.drop(columns=["shd_std", "f1_std"])


# M6 coordination ladder, in rung order: loop, ensemble, parallel roles,
# chain, team. NOT `VARIANT_ORDER`, which is plan §5.1 numbering and places
# `planner_reasoner` before the fan-in arms -- plotting the chain rung as if
# it were less coordinated than the ensembles, which is the one axis the
# ladder exists to order.
LADDER_ORDER: tuple[str, ...] = (
    "llm_pc",
    "fan_in_homog",
    "fan_in_spec",
    "planner_reasoner",
    "team",
)

# 1.96 (two-sided alpha=0.05) + 0.84 (80% power). The paper reports an
# equivalence bound rather than a null, so every accuracy comparison is
# printed next to the smallest difference this design could have detected.
_MDE_Z = 2.8


def ladder_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ladder rung, budget) for the M6 coordination table.

    `failure_rate` is computed BEFORE dropping non-"ok" cells. Filtering
    first makes every rate exactly 0.0, which would erase the M4b finding
    that `planner_reasoner` timed out on 8 of 30 cells at its top budget --
    the arm's defining weakness and half of hypothesis H-A.

    Non-ladder arms are dropped. The M6 analysis reuses
    `runs/m4-pilot.parquet` for its `llm_pc` and `planner_reasoner` rows,
    and that file also carries `random`, `greedy_ig`, and `llm_only`.

    Args:
        df: Cell-level DataFrame from `load_records`, one chamber only.

    Returns:
        DataFrame ordered by `LADDER_ORDER` then budget, with columns:
        agent_name, budget_k, n_cells, n_ok, failure_rate, f1_mean, f1_sd,
        shd_mean, tokens_mean, wall_time_mean, overlap_frac_mean.

    Raises:
        ValueError: If `df` mixes chambers. LT and WT have different menu
            sizes, so the same `budget_k` is a different budget in each and
            averaging them describes neither.
    """
    if "chamber" in df.columns and df["chamber"].nunique(dropna=True) > 1:
        found = sorted(df["chamber"].dropna().unique())
        raise ValueError(f"ladder_frame needs a single chamber; got {found}. Filter first.")

    rungs = df[df["agent_name"].isin(LADDER_ORDER)]
    if rungs.empty:
        return pd.DataFrame(
            columns=[
                "agent_name",
                "budget_k",
                "n_cells",
                "n_ok",
                "failure_rate",
                "f1_mean",
                "f1_sd",
                "shd_mean",
                "tokens_mean",
                "wall_time_mean",
                "overlap_frac_mean",
            ]
        )

    # Denominator: every attempted cell. Numerator comes from ok-cells only,
    # so the two are deliberately computed on different frames.
    attempted = rungs.groupby(["agent_name", "budget_k"], as_index=False).agg(
        n_cells=("seed", "count"),
        n_ok=("status", lambda s: int((s == "ok").sum())),
    )

    ok_only = rungs[rungs["status"] == "ok"]
    tokens = ok_only["tokens_in"].fillna(0) + ok_only["tokens_out"].fillna(0)
    ok_only = ok_only.assign(_tokens_total=tokens)
    # `runs/m4-pilot.parquet` predates the Task-8 columns and supplies two of
    # the five rungs, so a hard reference here makes the ladder table
    # unbuildable on exactly the data it exists to join. Absent optional
    # columns read as NaN.
    for optional in ("overlap_frac", "wall_time_seconds"):
        if optional not in ok_only.columns:
            ok_only = ok_only.assign(**{optional: float("nan")})
    scored = ok_only.groupby(["agent_name", "budget_k"], as_index=False).agg(
        f1_mean=("f1", "mean"),
        f1_sd=("f1", "std"),
        shd_mean=("shd", "mean"),
        tokens_mean=("_tokens_total", "mean"),
        wall_time_mean=("wall_time_seconds", "mean"),
        overlap_frac_mean=("overlap_frac", "mean"),
    )

    # Left join on `attempted`: an all-error (rung, budget) has no ok-cells
    # and so no `scored` row, but must still appear with failure_rate 1.0
    # rather than vanishing from the table.
    out = attempted.merge(scored, on=["agent_name", "budget_k"], how="left")
    out["failure_rate"] = 1.0 - (out["n_ok"] / out["n_cells"])

    rung_rank = {name: i for i, name in enumerate(LADDER_ORDER)}
    out = out.sort_values(
        by=["agent_name", "budget_k"],
        key=lambda col: col.map(rung_rank) if col.name == "agent_name" else col,
    ).reset_index(drop=True)
    return out[
        [
            "agent_name",
            "budget_k",
            "n_cells",
            "n_ok",
            "failure_rate",
            "f1_mean",
            "f1_sd",
            "shd_mean",
            "tokens_mean",
            "wall_time_mean",
            "overlap_frac_mean",
        ]
    ]


def minimum_detectable_effect(df: pd.DataFrame, agent: str, budget_k: int) -> float:
    """Smallest F1 difference this design could detect at 80% power.

    `2.8 * sd * sqrt(2/n)` for a two-sample comparison at n seeds per arm.

    The SD is the within-arm per-cell SD over ok-cells with **ddof=1**
    (pandas' `Series.std()` default). A NumPy implementation defaulting to
    ddof=0 differs by sqrt(n/(n-1)) -- 1.7 % at n=30, small enough to read
    as noise and large enough to move the equivalence bound.

    Args:
        df: Cell-level DataFrame.
        agent: Arm name.
        budget_k: Budget to slice on.

    Returns:
        The MDE in F1 units, or `nan` if fewer than two ok-cells exist.
    """
    sub = df[(df["agent_name"] == agent) & (df["budget_k"] == budget_k)]
    sub = sub[sub["status"] == "ok"]
    n = len(sub)
    if n < 2:
        return float("nan")
    sd = float(sub["f1"].std())  # ddof=1
    return _MDE_Z * sd * float(np.sqrt(2 / n))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pareto(
    agg: pd.DataFrame,
    chamber: str,
    metric: str = "shd",
    ax: Axes | None = None,
) -> Axes:
    """Plot the Pareto curve for one chamber, one metric (SHD or F1).

    Args:
        agg: Aggregated DataFrame from `aggregate_pareto`.
        chamber: Which chamber to plot.
        metric: "shd" (lower is better) or "f1" (higher is better).
        ax: Optional matplotlib Axes to draw on. If None, a new
            Figure+Axes is created with default sizing.

    Returns:
        The Axes object (so callers can apply additional styling).
    """
    if metric not in ("shd", "f1"):
        raise ValueError(f"metric must be 'shd' or 'f1'; got {metric!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    chamber_df = agg[agg["chamber"] == chamber]
    if chamber_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No 'ok' cells for chamber={chamber!r}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot one line per variant, in the canonical order (so the legend
    # always lists Random first → Planner+Reasoner last regardless of
    # which variants are present).
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    for variant in VARIANT_ORDER:
        v_df = chamber_df[chamber_df["agent_name"] == variant].sort_values("budget_fraction")
        if v_df.empty:
            continue
        ax.errorbar(
            v_df["budget_fraction"],
            v_df[mean_col],
            yerr=v_df[sem_col],
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "#000000"),
            marker=VARIANT_MARKERS.get(variant, "o"),
            linestyle=VARIANT_LINESTYLES.get(variant, "-"),
            capsize=4,
            linewidth=1.6,
            markersize=6,
        )

    ax.set_xlabel("Intervention budget fraction (k / M)")
    ax.set_ylabel({"shd": "Mean SHD (lower is better)", "f1": "Mean F1 (higher is better)"}[metric])
    ax.set_title(f"Chamber {chamber.upper()}: causal-discovery {metric.upper()} vs budget")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return ax


def make_pareto_figure(
    agg: pd.DataFrame,
    metric: str = "shd",
    combined: bool = False,
) -> Figure:
    """Produce the publication-quality Pareto figure.

    Args:
        agg: Aggregated DataFrame from `aggregate_pareto`.
        metric: "shd" or "f1".
        combined: If True, side-by-side LT + WT panels in one figure
            (matches plan §5.3 description). If False, returns a
            single-panel figure for whichever chamber is present.
            For a single-chamber input (e.g., M4 pilot with LT only),
            combined=True still produces a single panel.

    Returns:
        matplotlib Figure ready for `.savefig(...)` or `plt.show()`.
    """
    chambers_present = sorted(agg["chamber"].unique())

    if not combined or len(chambers_present) <= 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        chamber = chambers_present[0] if chambers_present else "lt"
        plot_pareto(agg, chamber, metric=metric, ax=ax)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, len(chambers_present), figsize=(7 * len(chambers_present), 5))
    if len(chambers_present) == 1:
        axes = [axes]
    for ax, chamber in zip(axes, chambers_present, strict=True):
        plot_pareto(agg, chamber, metric=metric, ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# M4 acceptance check
# ---------------------------------------------------------------------------


def check_m4_acceptance(agg: pd.DataFrame, chamber: str = "lt") -> dict[str, Any]:
    """Verify the plan §9 M4 acceptance criterion against aggregated data.

    Two sub-criteria from the plan:
      1. "Preliminary Pareto curve monotonic" — for each variant,
         mean SHD should weakly decrease as budget increases.
         "Weakly" = each step is no more than 1.5sigma above the previous
         step's mean (DeepSeek Flash stochasticity at N=30 will produce
         occasional non-monotonicities; we tolerate 1.5sigma to avoid
         flagging noise as failures).
      2. "Random sits below LLM variants" — at each budget, mean SHD
         for `random` should be ≥ the mean for at least one LLM-bearing
         variant (i.e., the LLM is doing some work).

    Args:
        agg: Aggregated DataFrame from `aggregate_pareto`.
        chamber: Which chamber to evaluate. Default "lt" (the M4 pilot).

    Returns:
        Dict with keys:
          monotonic: dict[variant_name, bool]
          monotonic_violations: dict[variant_name, list[(budget_low, budget_high, sigmas)]]
          random_dominated: dict[budget_fraction, list[variant_name]]
              — variants beating Random at each budget
          overall_pass: bool — True iff ALL variants are monotonic AND
              at least one LLM variant beats Random at the highest budget
    """
    chamber_df = agg[agg["chamber"] == chamber].copy()

    monotonic: dict[str, bool] = {}
    violations: dict[str, list[tuple[float, float, float]]] = {}

    for variant in VARIANT_ORDER:
        v_df = chamber_df[chamber_df["agent_name"] == variant].sort_values("budget_fraction")
        if v_df.empty or len(v_df) < 2:
            continue
        is_mono = True
        v_violations: list[tuple[float, float, float]] = []
        prev_row = None
        for _, row in v_df.iterrows():
            if prev_row is not None and row["shd_mean"] > prev_row["shd_mean"]:
                # Non-monotonic step. Quantify in sigma of the larger step's SEM.
                sem = max(prev_row["shd_sem"], row["shd_sem"], 1e-9)
                sigmas = (row["shd_mean"] - prev_row["shd_mean"]) / sem
                if sigmas > 1.5:
                    is_mono = False
                    v_violations.append(
                        (prev_row["budget_fraction"], row["budget_fraction"], float(sigmas))
                    )
            prev_row = row
        monotonic[variant] = is_mono
        if v_violations:
            violations[variant] = v_violations

    # Random-dominance check at each budget.
    random_dominated: dict[float, list[str]] = {}
    budget_fractions = sorted(chamber_df["budget_fraction"].unique())
    for bf in budget_fractions:
        bf_df = chamber_df[chamber_df["budget_fraction"] == bf]
        random_row = bf_df[bf_df["agent_name"] == "random"]
        if random_row.empty:
            continue
        random_shd = float(random_row["shd_mean"].iloc[0])
        beating: list[str] = []
        for variant in ("greedy_ig_lite", "llm_only", "llm_pc", "planner_reasoner"):
            v_row = bf_df[bf_df["agent_name"] == variant]
            if v_row.empty:
                continue
            if float(v_row["shd_mean"].iloc[0]) < random_shd:
                beating.append(variant)
        random_dominated[float(bf)] = beating

    # Overall pass: all variants present are monotonic AND at least one LLM
    # variant beats Random at the highest budget tested.
    all_mono = all(monotonic.values()) if monotonic else False
    highest_bf = budget_fractions[-1] if budget_fractions else 0.0
    llm_beats_random_at_max = bool(
        random_dominated.get(highest_bf)
        and any(
            v in random_dominated[highest_bf] for v in ("llm_only", "llm_pc", "planner_reasoner")
        )
    )
    overall_pass = all_mono and llm_beats_random_at_max

    return {
        "monotonic": monotonic,
        "monotonic_violations": violations,
        "random_dominated": random_dominated,
        "overall_pass": overall_pass,
    }


def format_acceptance_summary(result: dict[str, Any]) -> str:
    """Pretty-print an acceptance-check result for stdout."""
    lines: list[str] = []
    lines.append("M4 acceptance criteria (plan §9):")
    lines.append("")
    lines.append("  1. Pareto curve monotonic per variant (allowing 1.5sigma noise):")
    for variant in VARIANT_ORDER:
        mono = result["monotonic"].get(variant)
        if mono is None:
            lines.append(f"     {variant:20s} (not present)")
        else:
            mark = "✓" if mono else "✗"
            lines.append(f"     {mark} {variant}")
            for low, high, sigmas in result["monotonic_violations"].get(variant, []):
                lines.append(
                    f"         violation: SHD increased {sigmas:.1f}sigma between k/M={low:.2f} and {high:.2f}"
                )
    lines.append("")
    lines.append("  2. Random dominated by LLM variants at each budget:")
    for bf, beaters in sorted(result["random_dominated"].items()):
        if not beaters:
            lines.append(f"     k/M={bf:.2f}: ✗ NO variant beats Random")
        else:
            lines.append(f"     k/M={bf:.2f}: ✓ Random beaten by: {', '.join(beaters)}")
    lines.append("")
    lines.append(
        f"Overall: {'✓ PASS' if result['overall_pass'] else '✗ FAIL'} "
        f"(monotonic AND LLM-beats-Random-at-max-budget)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_results",
        description=(
            "Analyze chamber-pillar sweep results. Produces plan §5.3 Pareto "
            "figures (SHD and F1) and optionally checks plan §9 M4 acceptance."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Parquet/CSV file produced by run_experiment.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save figures. Created if needed. If omitted, only the summary is printed.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Side-by-side LT + WT panels (default: one figure per chamber).",
    )
    parser.add_argument(
        "--check-m4-acceptance",
        action="store_true",
        help="Print plan §9 M4 acceptance check (per-variant monotonic + LLM-beats-Random).",
    )
    parser.add_argument(
        "--ladder",
        action="store_true",
        help="Print the M6 coordination-ladder table (with MDE) and, with --out-dir, its panels.",
    )
    parser.add_argument(
        "--check-chamber",
        type=str,
        default="lt",
        help="Chamber to evaluate for the acceptance check. Default: lt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    df = load_records(args.input)
    print(f"Loaded {len(df)} records from {args.input}")
    n_ok = (df["status"] == "ok").sum()
    n_skipped = (df["status"] == "skipped").sum()
    n_error = (df["status"] == "error").sum()
    print(f"  ok: {n_ok}, skipped: {n_skipped}, error: {n_error}")

    agg = aggregate_pareto(df)
    if agg.empty:
        print("No 'ok' cells to analyze; bailing.")
        return 1
    print(f"Aggregated to {len(agg)} (chamber, agent, budget) Pareto points.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for metric in ("shd", "f1"):
            fig = make_pareto_figure(agg, metric=metric, combined=args.combined)
            fname = f"pareto_{metric}_combined.png" if args.combined else f"pareto_{metric}.png"
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_path}")

    if args.ladder:
        # One chamber only -- `ladder_frame` refuses a mixed frame, since the
        # same budget_k is a different budget under a different menu size.
        ladder_df = df[df["chamber"] == args.check_chamber] if "chamber" in df.columns else df
        # Validity BEFORE accuracy, deliberately. A degradation rate that
        # varies with budget biases the accuracy numbers, so reading the table
        # first means reading numbers that may have to be discarded. The M4b
        # pilot was interpreted in full before anyone checked whether the
        # harness was working; ~43% of its k=30 selections were random.
        print()
        print("=== HARNESS VALIDITY ===")
        report = harness_validity_report(ladder_df)
        warnings = validity_warnings(report)
        if warnings:
            for warning in warnings:
                print(f"  {warning}")
            print(
                "\n  Read these before the table below. A MODERATOR warning means "
                "the\n  measured effect is biased, not merely noisy."
            )
        else:
            print("  No degradation detected on any recorded path.")
        print()
        print(format_ladder_summary(ladder_df))
        if args.out_dir:
            for written in plot_ladder(ladder_df, Path(args.out_dir)):
                print(f"Wrote {written}")

    if args.check_m4_acceptance:
        result = check_m4_acceptance(agg, chamber=args.check_chamber)
        print()
        print(format_acceptance_summary(result))
        return 0 if result["overall_pass"] else 2

    return 0


def _suspend_seconds(ok: pd.DataFrame) -> float:
    """Total wall-clock time these cells spanned but did not spend computing.

    Returns 0.0 when the timestamp columns are absent or unparseable rather
    than raising: this is a diagnostic, and a frame that predates the columns
    should still produce a validity report.
    """
    needed = {"started_at", "finished_at", "wall_time_seconds"}
    if not needed.issubset(ok.columns) or not len(ok):
        return 0.0
    try:
        start = pd.to_datetime(ok["started_at"], errors="coerce")
        end = pd.to_datetime(ok["finished_at"], errors="coerce")
    except (TypeError, ValueError):
        return 0.0
    span = (end - start).dt.total_seconds()
    gap = span - ok["wall_time_seconds"]
    # Negative gaps are clock jitter on sub-second cells, not negative sleep.
    return float(gap[gap > 0].sum())


def harness_validity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(arm, budget) rates for every scaffold degradation path.

    Accuracy columns say what the agents scored; these say whether the harness
    was working while they scored it. Keeping them separate matters because a
    degraded cell still reports `status="ok"` and a plausible F1 -- that is how
    a 43% random-selection rate at k=30 survived a full pilot unnoticed.

    Args:
        df: Cell-level DataFrame from `load_records`.

    Returns:
        One row per (agent_name, budget_k): n_cells, error_rate,
        fallback_rate (fallbacks per LLM call), pc_degeneracy_rate,
        conservation_fail_rate, wall_mean, wall_p95.
    """
    out = []
    for (agent, budget), grp in df.groupby(["agent_name", "budget_k"], sort=True):
        ok = grp[grp["status"] == "ok"]
        calls = ok["n_llm_calls"].sum() if "n_llm_calls" in ok.columns else 0
        fb = ok["n_selection_fallbacks"].sum() if "n_selection_fallbacks" in ok.columns else 0
        pc = ok["n_pc_degeneracies"].sum() if "n_pc_degeneracies" in ok.columns else 0
        if "n_collinear_dropped" in ok.columns and len(ok):
            dropped = ok["n_collinear_dropped"].fillna(0)
            # FRACTION OF CELLS that dropped anything, not columns-per-cell.
            # `_VARIATION_EPS` and the DEGRADED threshold both assume a rate
            # in [0, 1]; a per-cell count (~3.1 on WT, where three redundant
            # barometers are dropped every time) makes a 0.37 wobble look
            # like a 0.02 moderator and false-alarms on every budget.
            coll_rate = float((dropped > 0).mean())
            coll_mean = float(dropped.mean())
        else:
            coll_rate = 0.0
            coll_mean = 0.0
        if "conservation_certified" in ok.columns:
            judged = ok[ok["conservation_certified"].notna()]
            cons_fail = (
                float((~judged["conservation_certified"].astype(bool)).mean())
                if len(judged)
                else float("nan")
            )
        else:
            cons_fail = float("nan")
        out.append(
            {
                "agent_name": agent,
                "budget_k": int(budget),
                "n_cells": len(grp),
                "n_ok": len(ok),
                # Denominator is every ATTEMPTED cell: an errored cell is the
                # degradation, so filtering to ok-cells first would report 0.
                "error_rate": 1.0 - (len(ok) / len(grp)) if len(grp) else float("nan"),
                # No denominator but non-zero fallbacks is NOT a clean rate.
                # Surface it as 1.0 so `validity_warnings` flags it, rather
                # than dividing by a missing `n_llm_calls` into a silent 0.0.
                "fallback_rate": (float(fb) / float(calls)) if calls else (1.0 if fb else 0.0),
                "pc_degeneracy_rate": (float(pc) / len(ok)) if len(ok) else 0.0,
                # Reported separately from pc_degeneracy_rate because a
                # collinear drop is a LOCAL loss (the dropped node makes no
                # claim, the rest of the graph is still inferred) rather than
                # a total one. Two numbers, deliberately: the rate drives the
                # warnings, the mean says how much of the graph went silent.
                "collinear_drop_rate": coll_rate,
                "collinear_cols_mean": coll_mean,
                "conservation_fail_rate": cons_fail,
                # Wall-CLOCK span minus active compute. `wall_time_seconds`
                # is `time.perf_counter()`, which on macOS does not advance
                # while the system is asleep, so this difference is suspend
                # time. Reported because it is otherwise invisible: the
                # 2026-08-25 WT gate ran 1.01h of work inside a 6.66h span.
                # It is NOT a contaminating path -- sleeping between LLM calls
                # cannot change what the agent chose -- it is an operations
                # number. Run sweeps under `caffeinate -is`.
                "suspend_seconds": _suspend_seconds(ok),
                "wall_mean": ok["wall_time_seconds"].mean() if len(ok) else float("nan"),
                "wall_p95": ok["wall_time_seconds"].quantile(0.95) if len(ok) else float("nan"),
            }
        )
    report = pd.DataFrame(out)
    # Record which source columns were ABSENT. Without this a frame that never
    # recorded a path is indistinguishable from one where the path never fired,
    # and `validity_warnings` would call an unmeasured harness clean.
    report.attrs["missing_columns"] = [
        c
        for c in (
            "n_selection_fallbacks",
            "n_pc_degeneracies",
            "n_collinear_dropped",
            "conservation_certified",
        )
        if c not in df.columns
    ]
    return report


# A rate this far apart across budgets counts as varying rather than jitter.
_VARIATION_EPS = 0.02


def validity_warnings(report: pd.DataFrame) -> list[str]:
    """Turn a validity report into explicit warnings, worst kind first.

    Two severities, and the distinction is the whole point:

      * A **flat** non-zero rate degrades every cell about equally. It adds
        noise and weakens power.
      * A rate that **varies with budget** is a moderator correlated with the
        independent variable. It biases the measured effect. This is what made
        `llm_pc` beat `random` at k=6 (+0.034) and not at k=30 (+0.018): the
        selection-truncation rate went 0% -> 43% across those budgets, so the
        treatment was being removed in proportion to the x-axis.

    Args:
        report: Output of `harness_validity_report`.

    Returns:
        Human-readable warnings; empty if every path is clean.
    """
    warnings: list[str] = []
    # Paths that CONTAMINATE the outcome, versus paths that ARE the outcome.
    # Only the first class can bias an accuracy comparison:
    #   * a selection fallback replaces the LLM's choice with `rng.choice`
    #   * a degenerate PC returns an all-zeros adjacency, zeroing F1
    #   * an errored cell drops out of the mean (survivorship)
    # `conservation_certified` and `tree_would_refuse` come from a post-hoc
    # `verify()`. Token budgets are non-binding at execution -- node monitors
    # record tokens for certification arithmetic and must not halt; only
    # interventions are live-gated -- so a conservation failure cannot change
    # what the agent did or what PC inferred. Calling its k-variance a bias is
    # wrong, and it would fire permanently: k=6's 48.8x aggregator cost spread
    # makes provisioning there unpredictable BY MEASUREMENT, which is one of
    # the paper's findings, not a defect.
    # `collinear_drop_rate` is contaminating too, and less obviously so than
    # the others: dropping a numerically duplicate column forfeits every edge
    # incident to it, and how many columns are duplicate depends on WHICH
    # experiments were bought -- so the rate can move with the budget axis.
    # On WT this is load-bearing: four barometers read the same quantity in
    # the `standard` configuration, and 13 of 42 true edges point into them.
    contaminating = {
        "fallback_rate",
        "error_rate",
        "pc_degeneracy_rate",
        "collinear_drop_rate",
    }
    for column in report.attrs.get("missing_columns", []):
        warnings.append(
            f"UNMEASURED: {column} is absent from these records, so its rate "
            "reads 0.00 here. That is not evidence the path is clean -- "
            "`runs/m4-pilot.parquet` shows 0.00 for it while ~43% of its k=30 "
            "selections were in fact random."
        )
    rates = (
        ("fallback_rate", "selection fallback rate"),
        ("error_rate", "cell error rate"),
        ("conservation_fail_rate", "conservation failure rate"),
        ("pc_degeneracy_rate", "PC degeneracy rate"),
        ("collinear_drop_rate", "collinear column-drop rate"),
    )

    # Varying-with-budget first: these bias, they do not merely blur.
    for column, label in rates:
        for agent, grp in report.groupby("agent_name", sort=True):
            vals = grp[column].dropna()
            if len(vals) < 2:
                continue
            if float(vals.max() - vals.min()) > _VARIATION_EPS:
                by_k = ", ".join(
                    f"k={int(r.budget_k)}:{getattr(r, column):.2f}"
                    for r in grp.sort_values("budget_k").itertuples()
                )
                if column in contaminating:
                    warnings.append(
                        f"MODERATOR: {agent} {label} varies with budget ({by_k}); "
                        "this biases the measured effect, it does not merely add noise"
                    )
                else:
                    warnings.append(
                        f"FINDING: {agent} {label} varies with budget ({by_k}); "
                        "post-hoc certification, so it does NOT bias accuracy -- "
                        "report it as a result about provisioning"
                    )

    for column, label in rates:
        flagged = report[report[column].fillna(0.0) > 0.0]
        for row in flagged.itertuples():
            warnings.append(
                f"DEGRADED: {row.agent_name} k={int(row.budget_k)} {label} = "
                f"{getattr(row, column):.2f}"
            )
    return warnings


def format_ladder_summary(df: pd.DataFrame, reference: str = "llm_pc") -> str:
    """Render the M6 ladder table, every delta paired with its MDE.

    Spec §6 reports an equivalence bound rather than a null. A bare
    rung-vs-rung accuracy difference reads as a finding, and the ladder's
    central risk is that the rungs land within noise of one another -- so a
    delta smaller than the minimum detectable effect is printed as
    "below MDE" rather than as a number the reader might interpret.

    Args:
        df: Cell-level DataFrame, one chamber only.
        reference: Rung every other rung is compared against. Defaults to
            the loop rung, which is hypothesis H-A's baseline.

    Returns:
        A printable multi-line table.
    """
    frame = ladder_frame(df)
    if frame.empty:
        return "No ladder cells to summarize."

    header = (
        f"{'Rung':<26}{'k':>4}{'n_ok':>6}{'fail':>7}"
        f"{'F1':>8}{'MDE':>8}{'delta vs ' + reference:>22}"
    )
    lines = [header, "-" * len(header)]

    for _, row in frame.iterrows():
        rung = str(row["agent_name"])
        budget = int(row["budget_k"])
        mde = minimum_detectable_effect(df, rung, budget)
        ref_rows = frame[(frame["agent_name"] == reference) & (frame["budget_k"] == budget)]

        if rung == reference or ref_rows.empty:
            verdict = "--"
        else:
            delta = float(row["f1_mean"]) - float(ref_rows.iloc[0]["f1_mean"])
            ref_mde = minimum_detectable_effect(df, reference, budget)
            # Compare against the wider of the two arms' bounds: the pair is
            # only resolvable if it clears whichever arm is noisier.
            bound = (
                max(m for m in (mde, ref_mde) if not np.isnan(m))
                if not (np.isnan(mde) and np.isnan(ref_mde))
                else float("nan")
            )
            if np.isnan(bound) or abs(delta) < bound:
                verdict = f"{delta:+.3f} (below MDE)"
            else:
                verdict = f"{delta:+.3f} (resolved)"

        f1 = float(row["f1_mean"])
        lines.append(
            f"{VARIANT_LABELS.get(rung, rung):<26}{budget:>4}{int(row['n_ok']):>6}"
            f"{row['failure_rate']:>7.2f}{f1:>8.3f}{mde:>8.3f}{verdict:>22}"
        )

    lines.append("")
    lines.append(
        "MDE = 2.8 * sd * sqrt(2/n), the smallest F1 difference detectable at "
        "80% power (alpha=0.05, two-sided). A delta below it is not evidence "
        "of equality -- only that this design cannot resolve it."
    )
    return "\n".join(lines)


def plot_ladder(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Three ladder panels: accuracy, cost, and failure rate.

    Rungs on the x-axis in `LADDER_ORDER`, one line per budget. Rung index
    rather than a coordination score, because the ladder is ordinal -- the
    spacing between rungs carries no meaning and a numeric axis would imply
    it does.

    Args:
        df: Cell-level DataFrame, one chamber only.
        out_dir: Directory to write the PNGs into. Created if absent.

    Returns:
        The three written paths, in panel order.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    frame = ladder_frame(df)

    panels = (
        ("ladder_f1.png", "f1_mean", "F1 (higher is better)"),
        ("ladder_tokens.png", "tokens_mean", "Tokens per cell"),
        ("ladder_failures.png", "failure_rate", "Failure rate"),
    )
    present = [r for r in LADDER_ORDER if r in set(frame["agent_name"])]
    xs = range(len(present))
    written: list[Path] = []

    for filename, column, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for budget in sorted(frame["budget_k"].unique()):
            at_budget = frame[frame["budget_k"] == budget].set_index("agent_name")
            ys = [at_budget[column].get(rung, float("nan")) for rung in present]
            ax.plot(list(xs), ys, marker="o", label=f"k={budget}")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([VARIANT_LABELS.get(r, r) for r in present], rotation=20, ha="right")
        ax.set_xlabel("Coordination rung")
        ax.set_ylabel(ylabel)
        ax.legend(title="Budget")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        target = out_path / filename
        fig.savefig(target, dpi=150)
        plt.close(fig)
        written.append(target)

    return written


__all__ = [
    "LADDER_ORDER",
    "VARIANT_COLORS",
    "VARIANT_LABELS",
    "VARIANT_ORDER",
    "aggregate_pareto",
    "build_arg_parser",
    "check_m4_acceptance",
    "format_acceptance_summary",
    "format_ladder_summary",
    "harness_validity_report",
    "ladder_frame",
    "load_records",
    "main",
    "make_pareto_figure",
    "minimum_detectable_effect",
    "plot_ladder",
    "plot_pareto",
    "validity_warnings",
]


if __name__ == "__main__":
    sys.exit(main())
