"""Analysis script for logic reasoning strategy modes experiment.

Generates bootstrap statistics and publication-quality figures for COINE 2026.

Usage:
    uv run python -m evaluation.strategy_modes.analyze_logic_results \
        --input results/strategy_modes/logic_openr1_20251229_184159.json

Output:
    - Figures saved to evaluation/strategy_modes/figures/
    - Analysis JSON saved alongside input file
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Type aliases for cleaner signatures
TrialDict = dict[str, Any]
TrialsByMode = dict[str, list[TrialDict]]

# Style configuration for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "urgent": "#E74C3C",  # Red
    "economical": "#3498DB",  # Blue
    "balanced": "#2ECC71",  # Green
}
MODE_ORDER = ["urgent", "economical", "balanced"]
MODE_LABELS = {"urgent": "URGENT", "economical": "ECONOMICAL", "balanced": "BALANCED"}


def load_results(path: str) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def extract_trials_by_mode(data: dict[str, Any]) -> TrialsByMode:
    """Extract trials grouped by mode."""
    trials_by_mode: TrialsByMode = {mode: [] for mode in MODE_ORDER}
    for trial in data["trials"]:
        mode = trial["mode"]
        if mode in trials_by_mode:
            trials_by_mode[mode].append(trial)
    return trials_by_mode


def bootstrap_ci(
    values: list[float] | list[int],
    stat_func: Any = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    np.random.seed(seed)
    values_arr = np.array(values)
    n = len(values_arr)
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(values_arr, size=n, replace=True)
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
    """Compute bootstrap CI for difference between two groups."""
    np.random.seed(seed)
    arr1, arr2 = np.array(values1), np.array(values2)
    diffs = []
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(arr1, size=len(arr1), replace=True)
        boot2 = np.random.choice(arr2, size=len(arr2), replace=True)
        diffs.append(boot2.mean() - boot1.mean())
    mean_diff = float(np.mean(diffs))
    lower = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    upper = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    significant = lower > 0 or upper < 0  # CI excludes zero
    return mean_diff, lower, upper, significant


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean2 - mean1) / pooled_std)


def compute_statistics(trials_by_mode: TrialsByMode) -> dict[str, Any]:
    """Compute comprehensive statistics for all modes."""
    stats: dict[str, Any] = {"modes": {}, "comparisons": {}}

    for mode in MODE_ORDER:
        trials = trials_by_mode[mode]
        n_total = len(trials)

        # Overall success rate (correct / total) - PRIMARY METRIC
        # This is what users ultimately care about: "Did you get the right answer?"
        all_correct = [1 if t["correct"] else 0 for t in trials]
        overall_mean, overall_low, overall_high = bootstrap_ci(all_correct)
        n_correct = sum(all_correct)

        # Completion rate (completed / total) - DIAGNOSTIC METRIC
        # Measures: "Did the API return a response without timeout?"
        successes = [1 if t["success"] else 0 for t in trials]
        comp_mean, comp_low, comp_high = bootstrap_ci(successes)

        # Timeout rate
        timeouts = [1 if t["timed_out"] else 0 for t in trials]
        to_mean, to_low, to_high = bootstrap_ci(timeouts)

        # Among completed trials
        successful = [t for t in trials if t["success"]]
        n_completed = len(successful)

        if successful:
            # Accuracy among completed (correct / completed) - DIAGNOSTIC METRIC
            correct = [1 if t["correct"] else 0 for t in successful]
            acc_mean, acc_low, acc_high = bootstrap_ci(correct)

            # Tokens
            tokens = [t["tokens_used"] for t in successful]
            tok_mean, tok_low, tok_high = bootstrap_ci(tokens)
            tok_std = float(np.std(tokens, ddof=1)) if len(tokens) > 1 else 0

            # Reasoning tokens
            reasoning = [t["reasoning_tokens"] for t in successful]
            reas_mean, reas_low, reas_high = bootstrap_ci(reasoning)

            # Execution time
            times = [t["execution_time"] for t in successful]
            time_mean, time_low, time_high = bootstrap_ci(times)
        else:
            acc_mean, acc_low, acc_high = 0, 0, 0
            tok_mean, tok_low, tok_high, tok_std = 0, 0, 0, 0
            reas_mean, reas_low, reas_high = 0, 0, 0
            time_mean, time_low, time_high = 0, 0, 0
            tokens, reasoning, times, correct = [], [], [], []

        stats["modes"][mode] = {
            "n_total": n_total,
            "n_completed": n_completed,
            "n_correct": n_correct,
            # PRIMARY METRIC: Overall success rate (correct / total)
            "overall_success_rate": {
                "mean": overall_mean,
                "ci_low": overall_low,
                "ci_high": overall_high,
            },
            # DIAGNOSTIC METRICS: Separate completion and accuracy
            "completion_rate": {"mean": comp_mean, "ci_low": comp_low, "ci_high": comp_high},
            "timeout_rate": {"mean": to_mean, "ci_low": to_low, "ci_high": to_high},
            "accuracy_given_completion": {"mean": acc_mean, "ci_low": acc_low, "ci_high": acc_high},
            "tokens": {
                "mean": tok_mean,
                "ci_low": tok_low,
                "ci_high": tok_high,
                "std": tok_std,
            },
            "reasoning_tokens": {
                "mean": reas_mean,
                "ci_low": reas_low,
                "ci_high": reas_high,
            },
            "execution_time": {
                "mean": time_mean,
                "ci_low": time_low,
                "ci_high": time_high,
            },
            # Raw values for effect size computation
            "_raw": {
                "all_correct": all_correct,  # For overall success rate comparison
                "completions": successes,
                "correct_given_completion": correct,
                "tokens": tokens,
                "reasoning": reasoning,
                "times": times,
            },
        }

    # Pairwise comparisons
    comparisons = [
        ("urgent", "balanced"),
        ("economical", "balanced"),
        ("urgent", "economical"),
    ]

    for mode1, mode2 in comparisons:
        key = f"{mode1}_vs_{mode2}"
        raw1 = stats["modes"][mode1]["_raw"]
        raw2 = stats["modes"][mode2]["_raw"]

        # Overall success rate difference (PRIMARY COMPARISON)
        diff_mean, diff_low, diff_high, sig = bootstrap_difference(
            raw1["all_correct"], raw2["all_correct"]
        )

        stats["comparisons"][key] = {
            "overall_success_diff": {
                "mean": diff_mean,
                "ci_low": diff_low,
                "ci_high": diff_high,
                "significant": sig,
            }
        }

        # Completion rate difference (DIAGNOSTIC)
        comp_diff_mean, comp_diff_low, comp_diff_high, comp_sig = bootstrap_difference(
            raw1["completions"], raw2["completions"]
        )
        stats["comparisons"][key]["completion_rate_diff"] = {
            "mean": comp_diff_mean,
            "ci_low": comp_diff_low,
            "ci_high": comp_diff_high,
            "significant": comp_sig,
        }

        # Effect sizes (among completed trials)
        if raw1["tokens"] and raw2["tokens"]:
            stats["comparisons"][key]["cohens_d_tokens"] = cohens_d(raw1["tokens"], raw2["tokens"])
        if raw1["reasoning"] and raw2["reasoning"]:
            stats["comparisons"][key]["cohens_d_reasoning"] = cohens_d(
                raw1["reasoning"], raw2["reasoning"]
            )
        if raw1["times"] and raw2["times"]:
            stats["comparisons"][key]["cohens_d_time"] = cohens_d(raw1["times"], raw2["times"])

    # Clean up raw data from output
    for mode in MODE_ORDER:
        del stats["modes"][mode]["_raw"]

    return stats


def plot_success_rate(trials_by_mode: TrialsByMode, output_dir: Path) -> None:
    """Plot success rate comparison with CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    means, errors_low, errors_high = [], [], []
    for mode in MODE_ORDER:
        trials = trials_by_mode[mode]
        successes = [1 if t["success"] else 0 for t in trials]
        mean, low, high = bootstrap_ci(successes)
        means.append(mean * 100)
        errors_low.append((mean - low) * 100)
        errors_high.append((high - mean) * 100)

    x = np.arange(len(MODE_ORDER))
    colors = [COLORS[m] for m in MODE_ORDER]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x,
        means,
        yerr=[errors_low, errors_high],
        fmt="none",
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    ax.set_ylabel("Success Rate (%)", fontsize=14)
    ax.set_xlabel("Contract Mode", fontsize=14)
    ax.set_title("Success Rate by Contract Mode\n(95% Bootstrap CI)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], fontsize=12)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, mean in zip(bars, means, strict=True):
        ax.annotate(
            f"{mean:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_logic_success_rate.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_logic_success_rate.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(trials_by_mode: TrialsByMode, output_dir: Path) -> None:
    """Plot accuracy comparison (among successful trials)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    means, errors_low, errors_high = [], [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            correct = [1 if t["correct"] else 0 for t in successful]
            mean, low, high = bootstrap_ci(correct)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean * 100)
        errors_low.append((mean - low) * 100)
        errors_high.append((high - mean) * 100)

    x = np.arange(len(MODE_ORDER))
    colors = [COLORS[m] for m in MODE_ORDER]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x,
        means,
        yerr=[errors_low, errors_high],
        fmt="none",
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Contract Mode", fontsize=14)
    ax.set_title(
        "Accuracy by Contract Mode (Among Successful Trials)\n(95% Bootstrap CI)",
        fontsize=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], fontsize=12)
    ax.set_ylim(0, 105)

    for bar, mean in zip(bars, means, strict=True):
        ax.annotate(
            f"{mean:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_logic_accuracy.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_logic_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tokens(trials_by_mode: TrialsByMode, output_dir: Path) -> None:
    """Plot token usage comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    means, errors_low, errors_high = [], [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            tokens = [t["tokens_used"] for t in successful]
            mean, low, high = bootstrap_ci(tokens)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean)
        errors_low.append(mean - low)
        errors_high.append(high - mean)

    x = np.arange(len(MODE_ORDER))
    colors = [COLORS[m] for m in MODE_ORDER]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x,
        means,
        yerr=[errors_low, errors_high],
        fmt="none",
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    ax.set_ylabel("Average Tokens", fontsize=14)
    ax.set_xlabel("Contract Mode", fontsize=14)
    ax.set_title("Token Usage by Contract Mode\n(95% Bootstrap CI)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], fontsize=12)

    for bar, mean in zip(bars, means, strict=True):
        ax.annotate(
            f"{mean:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_logic_tokens.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_logic_tokens.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reasoning_tokens(trials_by_mode: TrialsByMode, output_dir: Path) -> None:
    """Plot reasoning token usage comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    means, errors_low, errors_high = [], [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            reasoning = [t["reasoning_tokens"] for t in successful]
            mean, low, high = bootstrap_ci(reasoning)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean)
        errors_low.append(mean - low)
        errors_high.append(high - mean)

    x = np.arange(len(MODE_ORDER))
    colors = [COLORS[m] for m in MODE_ORDER]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x,
        means,
        yerr=[errors_low, errors_high],
        fmt="none",
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
    )

    ax.set_ylabel("Reasoning Tokens", fontsize=14)
    ax.set_xlabel("Contract Mode", fontsize=14)
    ax.set_title("Reasoning Token Usage by Contract Mode\n(95% Bootstrap CI)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], fontsize=12)

    for bar, mean in zip(bars, means, strict=True):
        ax.annotate(
            f"{mean:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_logic_reasoning_tokens.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_logic_reasoning_tokens.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_summary(trials_by_mode: TrialsByMode, output_dir: Path) -> None:
    """Create a 2x2 combined figure for paper."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Success Rate (top-left)
    ax = axes[0, 0]
    means, errors = [], []
    for mode in MODE_ORDER:
        trials = trials_by_mode[mode]
        successes = [1 if t["success"] else 0 for t in trials]
        mean, low, high = bootstrap_ci(successes)
        means.append(mean * 100)
        errors.append([(mean - low) * 100, (high - mean) * 100])
    x = np.arange(len(MODE_ORDER))
    colors = [COLORS[m] for m in MODE_ORDER]
    ax.bar(x, means, color=colors, edgecolor="black")
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt="none", color="black", capsize=5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(a) Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax.set_ylim(0, 105)

    # 2. Accuracy (top-right)
    ax = axes[0, 1]
    means, errors = [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            correct = [1 if t["correct"] else 0 for t in successful]
            mean, low, high = bootstrap_ci(correct)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean * 100)
        errors.append([(mean - low) * 100, (high - mean) * 100])
    ax.bar(x, means, color=colors, edgecolor="black")
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt="none", color="black", capsize=5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy (Among Successful)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax.set_ylim(0, 105)

    # 3. Token Usage (bottom-left)
    ax = axes[1, 0]
    means, errors = [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            tokens = [t["tokens_used"] for t in successful]
            mean, low, high = bootstrap_ci(tokens)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean)
        errors.append([mean - low, high - mean])
    ax.bar(x, means, color=colors, edgecolor="black")
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt="none", color="black", capsize=5)
    ax.set_ylabel("Average Tokens")
    ax.set_title("(c) Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])

    # 4. Reasoning Tokens (bottom-right)
    ax = axes[1, 1]
    means, errors = [], []
    for mode in MODE_ORDER:
        successful = [t for t in trials_by_mode[mode] if t["success"]]
        if successful:
            reasoning = [t["reasoning_tokens"] for t in successful]
            mean, low, high = bootstrap_ci(reasoning)
        else:
            mean, low, high = 0, 0, 0
        means.append(mean)
        errors.append([mean - low, high - mean])
    ax.bar(x, means, color=colors, edgecolor="black")
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt="none", color="black", capsize=5)
    ax.set_ylabel("Reasoning Tokens")
    ax.set_title("(d) Reasoning Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])

    plt.suptitle(
        "Logic Reasoning: Contract Mode Comparison\nOpenR1 Logic Puzzles (n=50, medium difficulty)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig_logic_combined.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fig_logic_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_results(input_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """Run full analysis on experiment results."""
    data = load_results(input_path)
    trials_by_mode = extract_trials_by_mode(data)

    # Compute statistics
    stats = compute_statistics(trials_by_mode)

    # Add experiment metadata
    stats["experiment"] = data["experiment"]
    stats["n_problems"] = data["experiment"].get("n_problems_actual", len(data["trials"]) // 3)

    # Generate figures
    fig_dir = Path(output_dir) if output_dir else Path("evaluation/strategy_modes/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_success_rate(trials_by_mode, fig_dir)
    plot_accuracy(trials_by_mode, fig_dir)
    plot_tokens(trials_by_mode, fig_dir)
    plot_reasoning_tokens(trials_by_mode, fig_dir)
    plot_combined_summary(trials_by_mode, fig_dir)
    print(f"Figures saved to {fig_dir}")

    # Save analysis JSON
    input_file = Path(input_path)
    analysis_path = input_file.parent / f"analysis_{input_file.stem}.json"
    with open(analysis_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Analysis saved to {analysis_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nExperiment: {stats['experiment']['type']}")
    print(f"Dataset: {stats['experiment']['dataset']}")
    print(f"Difficulty: {stats['experiment'].get('difficulty', 'all')}")
    print(f"N problems: {stats['n_problems']}")

    print("\n" + "-" * 70)
    print("MODE STATISTICS (95% Bootstrap CI)")
    print("-" * 70)

    for mode in MODE_ORDER:
        m = stats["modes"][mode]
        print(
            f"\n{MODE_LABELS[mode]} (n={m['n_total']}, correct={m['n_correct']}, completed={m['n_completed']}):"
        )

        # Primary metric
        osr = m["overall_success_rate"]
        print(
            f"  Overall Success Rate: {osr['mean']:.1%} [{osr['ci_low']:.1%}, {osr['ci_high']:.1%}]  ← PRIMARY"
        )

        # Diagnostic metrics
        cr = m["completion_rate"]
        print(f"  Completion Rate:      {cr['mean']:.1%} [{cr['ci_low']:.1%}, {cr['ci_high']:.1%}]")
        acc = m["accuracy_given_completion"]
        print(
            f"  Accuracy|Completed:   {acc['mean']:.1%} [{acc['ci_low']:.1%}, {acc['ci_high']:.1%}]"
        )

        # Resource metrics
        tok = m["tokens"]
        print(
            f"  Avg Tokens:           {tok['mean']:,.0f} [{tok['ci_low']:,.0f}, {tok['ci_high']:,.0f}]"
        )
        reas = m["reasoning_tokens"]
        print(
            f"  Reasoning Tokens:     {reas['mean']:,.0f} [{reas['ci_low']:,.0f}, {reas['ci_high']:,.0f}]"
        )

    print("\n" + "-" * 70)
    print("KEY COMPARISONS")
    print("-" * 70)

    for comp_key, comp in stats["comparisons"].items():
        print(f"\n{comp_key.replace('_', ' ').upper()}:")

        # Primary comparison
        osr_diff = comp["overall_success_diff"]
        sig = "✅ SIGNIFICANT" if osr_diff["significant"] else "⚠️ Not significant"
        print(
            f"  Overall Success Diff: {osr_diff['mean']:+.1%} "
            f"[{osr_diff['ci_low']:+.1%}, {osr_diff['ci_high']:+.1%}] {sig}"
        )

        # Diagnostic comparison
        cr_diff = comp["completion_rate_diff"]
        cr_sig = "✅" if cr_diff["significant"] else "⚠️"
        print(
            f"  Completion Rate Diff: {cr_diff['mean']:+.1%} "
            f"[{cr_diff['ci_low']:+.1%}, {cr_diff['ci_high']:+.1%}] {cr_sig}"
        )

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze logic experiment results")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/strategy_modes/figures",
        help="Directory for output figures",
    )

    args = parser.parse_args()
    analyze_results(args.input, args.output_dir)


if __name__ == "__main__":
    main()
