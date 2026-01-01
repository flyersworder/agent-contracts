"""Analysis script for Good Enough experiment.

Generates bootstrap statistics and publication-quality figures for COINE 2026.

Usage:
    uv run python -m evaluation.good_enough.analyze_results \
        --input results/good_enough/experiment_YYYYMMDD_HHMMSS.json

Output:
    - Figures saved to evaluation/good_enough/figures/
    - Analysis JSON saved alongside input file
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Type aliases
TrialDict = dict[str, Any]

# Style configuration for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "unconstrained": "#E74C3C",  # Red - wasteful
    "contracted": "#2ECC71",  # Green - efficient
}
CONDITION_ORDER = ["unconstrained", "contracted"]
CONDITION_LABELS = {"unconstrained": "UNCONSTRAINED", "contracted": "CONTRACTED"}


def load_results(path: str) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def extract_trials_by_condition(data: dict[str, Any]) -> dict[str, list[TrialDict]]:
    """Extract trials grouped by condition."""
    trials_by_cond: dict[str, list[TrialDict]] = {
        "unconstrained": [],
        "contracted": [],
    }
    for trial in data["trials"]:
        trials_by_cond["unconstrained"].append(trial["unconstrained"])
        trials_by_cond["contracted"].append(trial["contracted"])
    return trials_by_cond


def bootstrap_ci(
    values: list[float] | list[int],
    stat_func: Any = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Data values
        stat_func: Statistic function (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95)
        seed: Random seed

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    values_arr = np.array(values)
    n = len(values_arr)
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(values_arr, size=n, replace=True)
        boot_stats.append(stat_func(boot_sample))
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(np.mean(boot_stats)), float(lower), float(upper)


def bootstrap_difference(
    values1: list[float],
    values2: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, bool]:
    """Compute bootstrap CI for difference between two groups.

    Returns:
        Tuple of (mean_diff, lower_ci, upper_ci, is_significant)
    """
    rng = np.random.default_rng(seed)
    arr1, arr2 = np.array(values1), np.array(values2)
    diffs = []
    for _ in range(n_bootstrap):
        boot1 = rng.choice(arr1, size=len(arr1), replace=True)
        boot2 = rng.choice(arr2, size=len(arr2), replace=True)
        diffs.append(boot2.mean() - boot1.mean())
    mean_diff = float(np.mean(diffs))
    lower = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    upper = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    significant = lower > 0 or upper < 0  # CI excludes zero
    return mean_diff, lower, upper, significant


def bootstrap_ratio(
    values1: list[float],
    values2: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for ratio (values2 / values1).

    Useful for computing reduction percentages.
    """
    rng = np.random.default_rng(seed)
    arr1, arr2 = np.array(values1), np.array(values2)
    ratios = []
    for _ in range(n_bootstrap):
        boot1 = rng.choice(arr1, size=len(arr1), replace=True)
        boot2 = rng.choice(arr2, size=len(arr2), replace=True)
        if boot1.mean() > 0:
            ratios.append(boot2.mean() / boot1.mean())
    mean_ratio = float(np.mean(ratios))
    lower = float(np.percentile(ratios, (1 - ci) / 2 * 100))
    upper = float(np.percentile(ratios, (1 + ci) / 2 * 100))
    return mean_ratio, lower, upper


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean2 - mean1) / pooled_std)


def compute_sample_size_adequacy(n: int, effect_size: float) -> dict[str, Any]:
    """Assess whether sample size is adequate for detecting effect.

    Uses rule of thumb based on Cohen's guidelines:
    - Large effect (d >= 0.8): n >= 20 per group
    - Medium effect (d >= 0.5): n >= 50 per group
    - Small effect (d >= 0.2): n >= 200 per group

    Args:
        n: Sample size
        effect_size: Observed Cohen's d

    Returns:
        Dictionary with adequacy assessment
    """
    abs_d = abs(effect_size)

    if abs_d >= 0.8:
        required_n = 20
        effect_category = "large"
    elif abs_d >= 0.5:
        required_n = 50
        effect_category = "medium"
    elif abs_d >= 0.2:
        required_n = 200
        effect_category = "small"
    else:
        required_n = 500
        effect_category = "negligible"

    return {
        "n": n,
        "effect_size": effect_size,
        "effect_category": effect_category,
        "required_n": required_n,
        "adequate": n >= required_n,
        "power_note": f"n={n} is {'adequate' if n >= required_n else 'insufficient'} "
        f"for {effect_category} effect (d={abs_d:.2f})",
    }


def compute_statistics(trials_by_cond: dict[str, list[TrialDict]]) -> dict[str, Any]:
    """Compute comprehensive statistics for both conditions."""
    stats: dict[str, Any] = {
        "conditions": {},
        "comparisons": {},
        "sample_adequacy": {},
    }

    # Extract metrics for each condition
    for cond in CONDITION_ORDER:
        trials = trials_by_cond[cond]
        n = len(trials)

        # Iterations
        iterations = [t["iterations"] for t in trials]
        iter_mean, iter_low, iter_high = bootstrap_ci(iterations)

        # Tokens
        tokens = [t["total_tokens"] for t in trials]
        tok_mean, tok_low, tok_high = bootstrap_ci(tokens)

        # Quality
        qualities = [t["final_quality"] for t in trials]
        qual_mean, qual_low, qual_high = bootstrap_ci(qualities)

        # Early stop rate
        early_stops = [1 if t["stopped_early"] else 0 for t in trials]
        early_mean, early_low, early_high = bootstrap_ci(early_stops)

        stats["conditions"][cond] = {
            "n": n,
            "iterations": {
                "mean": iter_mean,
                "ci_low": iter_low,
                "ci_high": iter_high,
                "std": float(np.std(iterations, ddof=1)) if len(iterations) > 1 else 0,
            },
            "tokens": {
                "mean": tok_mean,
                "ci_low": tok_low,
                "ci_high": tok_high,
                "std": float(np.std(tokens, ddof=1)) if len(tokens) > 1 else 0,
            },
            "quality": {
                "mean": qual_mean,
                "ci_low": qual_low,
                "ci_high": qual_high,
                "std": float(np.std(qualities, ddof=1)) if len(qualities) > 1 else 0,
            },
            "early_stop_rate": {
                "mean": early_mean,
                "ci_low": early_low,
                "ci_high": early_high,
            },
        }

    # Compute comparisons (CONTRACTED vs UNCONSTRAINED)
    uc_trials = trials_by_cond["unconstrained"]
    ct_trials = trials_by_cond["contracted"]

    uc_iter = [t["iterations"] for t in uc_trials]
    ct_iter = [t["iterations"] for t in ct_trials]
    uc_tok = [t["total_tokens"] for t in uc_trials]
    ct_tok = [t["total_tokens"] for t in ct_trials]
    uc_qual = [t["final_quality"] for t in uc_trials]
    ct_qual = [t["final_quality"] for t in ct_trials]

    # Iteration reduction
    iter_diff, iter_low, iter_high, iter_sig = bootstrap_difference(uc_iter, ct_iter)
    iter_ratio, _ratio_low, _ratio_high = bootstrap_ratio(uc_iter, ct_iter)
    iter_reduction_pct = (1 - iter_ratio) * 100
    iter_d = cohens_d(uc_iter, ct_iter)

    stats["comparisons"]["iterations"] = {
        "difference": iter_diff,
        "ci_low": iter_low,
        "ci_high": iter_high,
        "significant": iter_sig,
        "reduction_ratio": iter_ratio,
        "reduction_pct": iter_reduction_pct,
        "cohens_d": iter_d,
    }

    # Token reduction
    tok_diff, tok_low, tok_high, tok_sig = bootstrap_difference(uc_tok, ct_tok)
    tok_ratio, _tok_ratio_low, _tok_ratio_high = bootstrap_ratio(uc_tok, ct_tok)
    tok_reduction_pct = (1 - tok_ratio) * 100
    tok_d = cohens_d(uc_tok, ct_tok)

    stats["comparisons"]["tokens"] = {
        "difference": tok_diff,
        "ci_low": tok_low,
        "ci_high": tok_high,
        "significant": tok_sig,
        "reduction_ratio": tok_ratio,
        "reduction_pct": tok_reduction_pct,
        "cohens_d": tok_d,
    }

    # Quality difference (expect ~0)
    qual_diff, qual_low, qual_high, qual_sig = bootstrap_difference(uc_qual, ct_qual)
    qual_d = cohens_d(uc_qual, ct_qual)

    stats["comparisons"]["quality"] = {
        "difference": qual_diff,
        "ci_low": qual_low,
        "ci_high": qual_high,
        "significant": qual_sig,
        "cohens_d": qual_d,
        "equivalent": not qual_sig,  # Quality is equivalent if difference is not significant
    }

    # Sample size adequacy
    n = len(uc_trials)
    stats["sample_adequacy"] = {
        "iterations": compute_sample_size_adequacy(n, iter_d),
        "tokens": compute_sample_size_adequacy(n, tok_d),
        "quality": compute_sample_size_adequacy(n, qual_d),
    }

    return stats


def plot_iterations_comparison(
    stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot iteration comparison between conditions."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    conditions = CONDITION_ORDER
    means = [stats["conditions"][c]["iterations"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["iterations"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["iterations"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    x = np.arange(len(conditions))
    colors = [COLORS[c] for c in conditions]

    bars = ax.bar(x, means, yerr=errors, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Average Iterations", fontsize=12)
    ax.set_title("Iterations to Complete Email Drafting\n(95% Bootstrap CI)", fontsize=14)

    # Add reduction annotation
    reduction = stats["comparisons"]["iterations"]["reduction_pct"]
    ax.annotate(
        f"{reduction:.0f}% fewer\niterations",
        xy=(1, means[1] + 0.3),
        ha="center",
        fontsize=11,
        color=COLORS["contracted"],
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_iterations.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig_iterations.pdf", bbox_inches="tight")
    plt.close()


def plot_tokens_comparison(
    stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot token usage comparison."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    conditions = CONDITION_ORDER
    means = [stats["conditions"][c]["tokens"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["tokens"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["tokens"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    x = np.arange(len(conditions))
    colors = [COLORS[c] for c in conditions]

    bars = ax.bar(x, means, yerr=errors, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Average Tokens", fontsize=12)
    ax.set_title("Token Usage for Email Drafting\n(95% Bootstrap CI)", fontsize=14)

    # Add reduction annotation
    reduction = stats["comparisons"]["tokens"]["reduction_pct"]
    ax.annotate(
        f"{reduction:.0f}% fewer\ntokens",
        xy=(1, means[1] + means[0] * 0.05),
        ha="center",
        fontsize=11,
        color=COLORS["contracted"],
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_tokens.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig_tokens.pdf", bbox_inches="tight")
    plt.close()


def plot_quality_comparison(
    stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot quality comparison (should be similar)."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    conditions = CONDITION_ORDER
    means = [stats["conditions"][c]["quality"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["quality"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["quality"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    x = np.arange(len(conditions))
    colors = [COLORS[c] for c in conditions]

    bars = ax.bar(x, means, yerr=errors, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Final Quality Score", fontsize=12)
    ax.set_title("Email Quality at Stopping Point\n(95% Bootstrap CI)", fontsize=14)
    ax.set_ylim(0, 1.1)

    # Add equivalence annotation
    diff = stats["comparisons"]["quality"]["difference"]
    if stats["comparisons"]["quality"]["equivalent"]:
        ax.annotate(
            f"Δ = {diff:+.2f}\n(not significant)",
            xy=(0.5, 0.5),
            ha="center",
            fontsize=11,
            color="gray",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_quality.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig_quality.pdf", bbox_inches="tight")
    plt.close()


def plot_combined_summary(
    stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Create combined summary figure for paper."""
    _fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    conditions = CONDITION_ORDER

    # Panel 1: Iterations
    ax = axes[0]
    means = [stats["conditions"][c]["iterations"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["iterations"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["iterations"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    x = np.arange(len(conditions))
    colors = [COLORS[c] for c in conditions]

    bars = ax.bar(x, means, yerr=errors, capsize=6, color=colors, edgecolor="black", linewidth=1.2)
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylabel("Iterations", fontsize=11)
    ax.set_title(
        f"(a) Iterations\n({stats['comparisons']['iterations']['reduction_pct']:.0f}% reduction)",
        fontsize=12,
    )

    # Panel 2: Tokens
    ax = axes[1]
    means = [stats["conditions"][c]["tokens"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["tokens"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["tokens"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    bars = ax.bar(x, means, yerr=errors, capsize=6, color=colors, edgecolor="black", linewidth=1.2)
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylabel("Tokens", fontsize=11)
    ax.set_title(
        f"(b) Token Usage\n({stats['comparisons']['tokens']['reduction_pct']:.0f}% reduction)",
        fontsize=12,
    )

    # Panel 3: Quality
    ax = axes[2]
    means = [stats["conditions"][c]["quality"]["mean"] for c in conditions]
    ci_lows = [stats["conditions"][c]["quality"]["ci_low"] for c in conditions]
    ci_highs = [stats["conditions"][c]["quality"]["ci_high"] for c in conditions]
    errors = [
        [m - low for m, low in zip(means, ci_lows, strict=False)],
        [h - m for m, h in zip(means, ci_highs, strict=False)],
    ]

    bars = ax.bar(x, means, yerr=errors, capsize=6, color=colors, edgecolor="black", linewidth=1.2)
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylabel("Quality Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    qual_note = "equivalent" if stats["comparisons"]["quality"]["equivalent"] else "different"
    ax.set_title(f"(c) Final Quality\n({qual_note})", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_combined.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig_combined.pdf", bbox_inches="tight")
    plt.close()


def plot_early_stop_rate(
    stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot early stop rate comparison."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    conditions = CONDITION_ORDER
    means = [stats["conditions"][c]["early_stop_rate"]["mean"] * 100 for c in conditions]

    x = np.arange(len(conditions))
    colors = [COLORS[c] for c in conditions]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, mean in zip(bars, means, strict=False):
        ax.annotate(
            f"{mean:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Early Stop Rate (%)", fontsize=12)
    ax.set_title("'Good Enough' Recognition Rate\n(Stopped before max iterations)", fontsize=14)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_early_stop.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig_early_stop.pdf", bbox_inches="tight")
    plt.close()


def print_summary(stats: dict[str, Any], config: dict[str, Any]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nSample size: n = {config['n_scenarios']} scenarios")
    print(f"Quality threshold (Q_min): {config['quality_threshold']}")
    print(f"Max iterations: {config['max_iterations']}")

    print("\n" + "-" * 70)
    print("MAIN RESULTS (95% Bootstrap CI)")
    print("-" * 70)

    # Iterations
    uc_iter = stats["conditions"]["unconstrained"]["iterations"]
    ct_iter = stats["conditions"]["contracted"]["iterations"]
    iter_comp = stats["comparisons"]["iterations"]
    print("\nIterations:")
    print(
        f"  UNCONSTRAINED: {uc_iter['mean']:.1f} [{uc_iter['ci_low']:.1f}, {uc_iter['ci_high']:.1f}]"
    )
    print(
        f"  CONTRACTED:    {ct_iter['mean']:.1f} [{ct_iter['ci_low']:.1f}, {ct_iter['ci_high']:.1f}]"
    )
    print(
        f"  Reduction:     {iter_comp['reduction_pct']:.1f}% (Cohen's d = {iter_comp['cohens_d']:.2f})"
    )
    print(f"  Significant:   {'YES' if iter_comp['significant'] else 'NO'}")

    # Tokens
    uc_tok = stats["conditions"]["unconstrained"]["tokens"]
    ct_tok = stats["conditions"]["contracted"]["tokens"]
    tok_comp = stats["comparisons"]["tokens"]
    print("\nTokens:")
    print(
        f"  UNCONSTRAINED: {uc_tok['mean']:,.0f} [{uc_tok['ci_low']:,.0f}, {uc_tok['ci_high']:,.0f}]"
    )
    print(
        f"  CONTRACTED:    {ct_tok['mean']:,.0f} [{ct_tok['ci_low']:,.0f}, {ct_tok['ci_high']:,.0f}]"
    )
    print(
        f"  Reduction:     {tok_comp['reduction_pct']:.1f}% (Cohen's d = {tok_comp['cohens_d']:.2f})"
    )
    print(f"  Significant:   {'YES' if tok_comp['significant'] else 'NO'}")

    # Quality
    uc_qual = stats["conditions"]["unconstrained"]["quality"]
    ct_qual = stats["conditions"]["contracted"]["quality"]
    qual_comp = stats["comparisons"]["quality"]
    print("\nQuality:")
    print(
        f"  UNCONSTRAINED: {uc_qual['mean']:.3f} [{uc_qual['ci_low']:.3f}, {uc_qual['ci_high']:.3f}]"
    )
    print(
        f"  CONTRACTED:    {ct_qual['mean']:.3f} [{ct_qual['ci_low']:.3f}, {ct_qual['ci_high']:.3f}]"
    )
    print(
        f"  Difference:    {qual_comp['difference']:+.3f} (Cohen's d = {qual_comp['cohens_d']:.2f})"
    )
    print(f"  Equivalent:    {'YES' if qual_comp['equivalent'] else 'NO'}")

    # Early stop rate
    uc_early = stats["conditions"]["unconstrained"]["early_stop_rate"]
    ct_early = stats["conditions"]["contracted"]["early_stop_rate"]
    print("\nEarly Stop Rate:")
    print(f"  UNCONSTRAINED: {uc_early['mean'] * 100:.0f}%")
    print(f"  CONTRACTED:    {ct_early['mean'] * 100:.0f}%")

    # Sample adequacy
    print("\n" + "-" * 70)
    print("SAMPLE SIZE ADEQUACY")
    print("-" * 70)
    for metric, adequacy in stats["sample_adequacy"].items():
        status = "✓" if adequacy["adequate"] else "⚠"
        print(f"  {status} {metric}: {adequacy['power_note']}")

    print("\n" + "=" * 70)


def analyze(input_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """Run full analysis on experiment results.

    Args:
        input_path: Path to experiment results JSON
        output_dir: Directory for figures (default: evaluation/good_enough/figures)

    Returns:
        Analysis statistics dictionary
    """
    # Load data
    data = load_results(input_path)
    config = data["config"]

    # Set up output directory
    fig_dir = Path("evaluation/good_enough/figures") if output_dir is None else Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Extract trials
    trials_by_cond = extract_trials_by_condition(data)

    # Compute statistics
    stats = compute_statistics(trials_by_cond)

    # Print summary
    print_summary(stats, config)

    # Generate figures
    print("\nGenerating figures...")
    plot_iterations_comparison(stats, fig_dir)
    plot_tokens_comparison(stats, fig_dir)
    plot_quality_comparison(stats, fig_dir)
    plot_combined_summary(stats, fig_dir)
    plot_early_stop_rate(stats, fig_dir)
    print(f"Figures saved to: {fig_dir}/")

    # Save analysis JSON
    input_file = Path(input_path)
    analysis_file = (
        input_file.parent / f"analysis_{input_file.stem.replace('experiment_', '')}.json"
    )
    with open(analysis_file, "w") as f:
        json.dump({"config": config, "statistics": stats}, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Good Enough experiment results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to experiment results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for figures (default: evaluation/good_enough/figures)",
    )

    args = parser.parse_args()
    analyze(args.input, args.output_dir)


if __name__ == "__main__":
    main()
