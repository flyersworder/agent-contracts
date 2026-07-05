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
}

VARIANT_LABELS: dict[str, str] = {
    "random": "Random",
    "greedy_ig_lite": "GreedyIG-lite",
    "llm_only": "LLM-only",
    "llm_pc": "LLM+PC",
    "planner_reasoner": "Planner+Reasoner",
}

# Marker + linestyle per variant so curves stay distinguishable when the
# figure is printed in black and white (colors alone collapse in grayscale).
VARIANT_MARKERS: dict[str, str] = {
    "random": "o",
    "greedy_ig_lite": "s",
    "llm_only": "^",
    "llm_pc": "D",
    "planner_reasoner": "v",
}

VARIANT_LINESTYLES: dict[str, str | tuple[int, tuple[int, ...]]] = {
    "random": ":",
    "greedy_ig_lite": "--",
    "llm_only": "-",
    "llm_pc": "-.",
    "planner_reasoner": (0, (3, 1, 1, 1, 1, 1)),
}

# Variant rendering order in legend (matches plan §5.3 description top-to-bottom).
VARIANT_ORDER: tuple[str, ...] = (
    "random",
    "greedy_ig_lite",
    "llm_only",
    "llm_pc",
    "planner_reasoner",
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

    if args.check_m4_acceptance:
        result = check_m4_acceptance(agg, chamber=args.check_chamber)
        print()
        print(format_acceptance_summary(result))
        return 0 if result["overall_pass"] else 2

    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "VARIANT_COLORS",
    "VARIANT_LABELS",
    "VARIANT_ORDER",
    "aggregate_pareto",
    "build_arg_parser",
    "check_m4_acceptance",
    "format_acceptance_summary",
    "load_records",
    "main",
    "make_pareto_figure",
    "plot_pareto",
]
