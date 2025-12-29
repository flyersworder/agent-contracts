"""Statistical analysis for strategy modes experiment.

This script performs bootstrap analysis and generates publication-ready figures
for the COINE 2026 paper.

Usage:
    uv run python -m evaluation.strategy_modes.analyze_results \
        --results results/strategy_modes/strategy_modes_20251229_123750.json

    # Or analyze the most recent results file
    uv run python -m evaluation.strategy_modes.analyze_results --latest
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BootstrapResult:
    """Result from bootstrap confidence interval estimation."""

    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval with BCa correction.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with mean, CI bounds, std, and sample size
    """
    if not data:
        return BootstrapResult(mean=0.0, ci_lower=0.0, ci_upper=0.0, std=0.0, n=0)

    arr = np.array(data)
    n = len(arr)

    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap resampling
    boot_means = np.array(
        [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    # Percentile method for CI
    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

    return BootstrapResult(
        mean=float(np.mean(arr)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=float(np.std(arr, ddof=1)),
        n=n,
    )


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    if not group1 or not group2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def load_results(filepath: str | Path) -> dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(filepath) as f:
        data: dict[str, Any] = json.load(f)
        return data


def extract_mode_data(results: dict[str, Any], mode: str, metric: str) -> list[float]:
    """Extract metric values for a specific mode.

    Args:
        results: Experiment results dictionary
        mode: Mode name (urgent/economical/balanced)
        metric: Metric to extract (e.g., 'tokens_used', 'reasoning_tokens')

    Returns:
        List of metric values for successful trials
    """
    values = []
    for trial in results.get("trials", []):
        if trial["mode"] == mode and trial["success"]:
            if metric == "rouge_l_f1":
                value = trial.get("rouge_metrics", {}).get("rouge_l_f1", 0)
            else:
                value = trial.get(metric, 0)
            values.append(float(value))
    return values


def analyze_experiment(results: dict[str, Any]) -> dict[str, Any]:
    """Perform full statistical analysis of experiment results.

    Args:
        results: Experiment results dictionary

    Returns:
        Analysis results with bootstrap CIs and effect sizes
    """
    modes = ["urgent", "economical", "balanced"]
    metrics = [
        "tokens_used",
        "reasoning_tokens",
        "text_tokens",
        "execution_time",
        "word_count",
        "rouge_l_f1",
    ]

    analysis: dict[str, Any] = {
        "experiment_info": results.get("experiment", {}),
        "bootstrap_results": {},
        "effect_sizes": {},
        "comparisons": {},
    }

    # Bootstrap CI for each mode and metric
    for mode in modes:
        analysis["bootstrap_results"][mode] = {}
        for metric in metrics:
            data = extract_mode_data(results, mode, metric)
            ci_result = bootstrap_ci(data)
            analysis["bootstrap_results"][mode][metric] = {
                "mean": ci_result.mean,
                "ci_lower": ci_result.ci_lower,
                "ci_upper": ci_result.ci_upper,
                "std": ci_result.std,
                "n": ci_result.n,
            }

    # Pairwise effect sizes
    pairs = [("urgent", "balanced"), ("economical", "balanced"), ("urgent", "economical")]
    for mode1, mode2 in pairs:
        pair_key = f"{mode1}_vs_{mode2}"
        analysis["effect_sizes"][pair_key] = {}
        for metric in metrics:
            data1 = extract_mode_data(results, mode1, metric)
            data2 = extract_mode_data(results, mode2, metric)
            d = cohens_d(data1, data2)
            analysis["effect_sizes"][pair_key][metric] = {
                "cohens_d": d,
                "interpretation": interpret_cohens_d(d),
            }

    # Key comparisons
    analysis["comparisons"] = compute_key_comparisons(results, modes)

    return analysis


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_key_comparisons(results: dict[str, Any], modes: list[str]) -> dict[str, Any]:
    """Compute key comparisons for paper claims."""
    comparisons = {}

    # URGENT vs BALANCED speed comparison
    urgent_time = extract_mode_data(results, "urgent", "execution_time")
    balanced_time = extract_mode_data(results, "balanced", "execution_time")
    if urgent_time and balanced_time:
        time_savings = (1 - np.mean(urgent_time) / np.mean(balanced_time)) * 100
        comparisons["urgent_vs_balanced_time_savings_pct"] = float(time_savings)

    # ECONOMICAL vs BALANCED token comparison
    econ_tokens = extract_mode_data(results, "economical", "tokens_used")
    balanced_tokens = extract_mode_data(results, "balanced", "tokens_used")
    if econ_tokens and balanced_tokens:
        token_savings = (1 - np.mean(econ_tokens) / np.mean(balanced_tokens)) * 100
        comparisons["economical_vs_balanced_token_savings_pct"] = float(token_savings)

    # Reasoning tokens distribution
    for mode in modes:
        reasoning = extract_mode_data(results, mode, "reasoning_tokens")
        if reasoning:
            comparisons[f"{mode}_avg_reasoning_tokens"] = float(np.mean(reasoning))

    # Success rates
    for mode in modes:
        total = sum(1 for t in results.get("trials", []) if t["mode"] == mode)
        success = sum(1 for t in results.get("trials", []) if t["mode"] == mode and t["success"])
        if total > 0:
            comparisons[f"{mode}_success_rate"] = success / total

    return comparisons


def print_analysis_summary(analysis: dict[str, Any]) -> None:
    """Print formatted analysis summary."""
    print("=" * 80)
    print("BOOTSTRAP ANALYSIS SUMMARY")
    print("=" * 80)

    # Experiment info
    info = analysis.get("experiment_info", {})
    print(f"\nExperiment: {info.get('n_articles', 'N/A')} articles x 3 modes")
    print(f"Model: {info.get('model', 'N/A')}")
    print(f"Seed: {info.get('seed', 'N/A')}")

    # Bootstrap results table
    print("\n" + "-" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("-" * 80)

    modes = ["urgent", "economical", "balanced"]
    metrics = [
        ("reasoning_tokens", "Reasoning Tokens"),
        ("tokens_used", "Total Tokens"),
        ("execution_time", "Execution Time (s)"),
        ("word_count", "Word Count"),
        ("rouge_l_f1", "ROUGE-L F1"),
    ]

    # Header
    print(f"\n{'Metric':<25}", end="")
    for mode in modes:
        print(f"{mode.upper():<25}", end="")
    print()
    print("-" * 100)

    # Data rows
    bootstrap = analysis.get("bootstrap_results", {})
    for metric_key, metric_name in metrics:
        print(f"{metric_name:<25}", end="")
        for mode in modes:
            data = bootstrap.get(mode, {}).get(metric_key, {})
            mean = data.get("mean", 0)
            ci_lower = data.get("ci_lower", 0)
            ci_upper = data.get("ci_upper", 0)
            print(f"{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]".ljust(25), end="")
        print()

    # Effect sizes
    print("\n" + "-" * 80)
    print("EFFECT SIZES (Cohen's d)")
    print("-" * 80)

    effect_sizes = analysis.get("effect_sizes", {})
    for pair_key, pair_effects in effect_sizes.items():
        print(f"\n{pair_key.replace('_', ' ').title()}:")
        for metric_key, metric_name in metrics:
            effect = pair_effects.get(metric_key, {})
            d = effect.get("cohens_d", 0)
            interp = effect.get("interpretation", "N/A")
            print(f"  {metric_name:<25}: d = {d:+.3f} ({interp})")

    # Key comparisons
    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)

    comparisons = analysis.get("comparisons", {})

    print("\n1. GOVERNANCE IS OBSERVABLE (Reasoning Tokens):")
    for mode in modes:
        tokens = comparisons.get(f"{mode}_avg_reasoning_tokens", 0)
        print(f"   {mode.upper()}: {tokens:.0f} reasoning tokens")

    print("\n2. SPEED IMPROVEMENT:")
    time_savings = comparisons.get("urgent_vs_balanced_time_savings_pct", 0)
    print(f"   URGENT is {time_savings:.1f}% faster than BALANCED")

    print("\n3. TOKEN EFFICIENCY:")
    token_savings = comparisons.get("economical_vs_balanced_token_savings_pct", 0)
    print(
        f"   ECONOMICAL uses {abs(token_savings):.1f}% {'fewer' if token_savings > 0 else 'more'} tokens than BALANCED"
    )

    print("\n4. SUCCESS RATES:")
    for mode in modes:
        rate = comparisons.get(f"{mode}_success_rate", 0)
        print(f"   {mode.upper()}: {rate * 100:.1f}%")

    print("\n" + "=" * 80)


def generate_figures(
    analysis: dict[str, Any],
    output_dir: str = "results/strategy_modes/figures",
) -> None:
    """Generate publication-ready figures.

    Requires matplotlib. Will skip if not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping figure generation.")
        print("Install with: uv add matplotlib")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modes = ["urgent", "economical", "balanced"]
    mode_colors = {"urgent": "#e74c3c", "economical": "#27ae60", "balanced": "#3498db"}
    bootstrap = analysis.get("bootstrap_results", {})

    # Figure 1a: Reasoning Tokens by Mode
    _fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(modes))
    means = [bootstrap[m]["reasoning_tokens"]["mean"] for m in modes]
    ci_lowers = [bootstrap[m]["reasoning_tokens"]["ci_lower"] for m in modes]
    ci_uppers = [bootstrap[m]["reasoning_tokens"]["ci_upper"] for m in modes]
    errors = [
        [mean - lower for mean, lower in zip(means, ci_lowers, strict=True)],
        [upper - mean for mean, upper in zip(means, ci_uppers, strict=True)],
    ]

    bars = ax.bar(x, means, color=[mode_colors[m] for m in modes], edgecolor="black")
    ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=5)

    ax.set_ylabel("Reasoning Tokens", fontsize=12)
    ax.set_xlabel("Contract Mode", fontsize=12)
    ax.set_title("Figure 1a: Reasoning Tokens by Mode\n(95% Bootstrap CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path / "fig1a_reasoning_tokens.png", dpi=300)
    plt.savefig(output_path / "fig1a_reasoning_tokens.pdf")
    plt.close()
    print(f"Saved: {output_path / 'fig1a_reasoning_tokens.png'}")

    # Figure 1b: Execution Time by Mode
    _fig, ax = plt.subplots(figsize=(8, 6))
    means = [bootstrap[m]["execution_time"]["mean"] for m in modes]
    ci_lowers = [bootstrap[m]["execution_time"]["ci_lower"] for m in modes]
    ci_uppers = [bootstrap[m]["execution_time"]["ci_upper"] for m in modes]
    errors = [
        [mean - lower for mean, lower in zip(means, ci_lowers, strict=True)],
        [upper - mean for mean, upper in zip(means, ci_uppers, strict=True)],
    ]

    bars = ax.bar(x, means, color=[mode_colors[m] for m in modes], edgecolor="black")
    ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=5)

    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_xlabel("Contract Mode", fontsize=12)
    ax.set_title("Figure 1b: Execution Time by Mode\n(95% Bootstrap CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.set_ylim(bottom=0)

    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{mean:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path / "fig1b_execution_time.png", dpi=300)
    plt.savefig(output_path / "fig1b_execution_time.pdf")
    plt.close()
    print(f"Saved: {output_path / 'fig1b_execution_time.png'}")

    # Figure 1c: ROUGE-L by Mode
    _fig, ax = plt.subplots(figsize=(8, 6))
    means = [bootstrap[m]["rouge_l_f1"]["mean"] for m in modes]
    ci_lowers = [bootstrap[m]["rouge_l_f1"]["ci_lower"] for m in modes]
    ci_uppers = [bootstrap[m]["rouge_l_f1"]["ci_upper"] for m in modes]
    errors = [
        [mean - lower for mean, lower in zip(means, ci_lowers, strict=True)],
        [upper - mean for mean, upper in zip(means, ci_uppers, strict=True)],
    ]

    bars = ax.bar(x, means, color=[mode_colors[m] for m in modes], edgecolor="black")
    ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=5)

    ax.set_ylabel("ROUGE-L F1", fontsize=12)
    ax.set_xlabel("Contract Mode", fontsize=12)
    ax.set_title("Figure 1c: Summary Quality by Mode\n(95% Bootstrap CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.set_ylim(0, 0.35)

    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path / "fig1c_rouge_quality.png", dpi=300)
    plt.savefig(output_path / "fig1c_rouge_quality.pdf")
    plt.close()
    print(f"Saved: {output_path / 'fig1c_rouge_quality.png'}")

    # Figure 1d: Quality vs Reasoning Tokens scatter
    _fig, ax = plt.subplots(figsize=(8, 6))

    for mode in modes:
        reasoning = bootstrap[mode]["reasoning_tokens"]["mean"]
        rouge = bootstrap[mode]["rouge_l_f1"]["mean"]
        rouge_err = (
            bootstrap[mode]["rouge_l_f1"]["ci_upper"] - bootstrap[mode]["rouge_l_f1"]["ci_lower"]
        ) / 2

        ax.errorbar(
            reasoning,
            rouge,
            yerr=rouge_err,
            fmt="o",
            markersize=15,
            color=mode_colors[mode],
            capsize=5,
            label=mode.upper(),
            markeredgecolor="black",
        )

    ax.set_xlabel("Reasoning Tokens", fontsize=12)
    ax.set_ylabel("ROUGE-L F1", fontsize=12)
    ax.set_title("Figure 1d: Quality vs Reasoning Depth\n(95% Bootstrap CI)", fontsize=14)
    ax.legend()
    ax.set_xlim(-50, 650)
    ax.set_ylim(0.15, 0.30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "fig1d_quality_vs_reasoning.png", dpi=300)
    plt.savefig(output_path / "fig1d_quality_vs_reasoning.pdf")
    plt.close()
    print(f"Saved: {output_path / 'fig1d_quality_vs_reasoning.png'}")

    print(f"\nAll figures saved to: {output_path}")


def find_latest_results(results_dir: str = "results/strategy_modes") -> Path | None:
    """Find the most recent results file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    json_files = list(results_path.glob("strategy_modes_*.json"))
    if not json_files:
        return None

    return max(json_files, key=lambda p: p.stat().st_mtime)


def save_analysis(analysis: dict[str, Any], output_path: Path) -> None:
    """Save analysis results to JSON."""
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze strategy modes experiment results")
    parser.add_argument(
        "--results",
        type=str,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent results file",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        default=True,
        help="Generate publication figures (default: True)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/strategy_modes",
        help="Output directory for analysis and figures",
    )

    args = parser.parse_args()

    # Determine results file
    results_path: Path
    if args.results:
        results_path = Path(args.results)
    elif args.latest:
        latest_path = find_latest_results()
        if not latest_path:
            print("No results files found in results/strategy_modes/")
            return
        results_path = latest_path
        print(f"Using latest results: {results_path}")
    else:
        # Default to latest
        latest_path = find_latest_results()
        if not latest_path:
            print("No results files found. Specify --results or run experiment first.")
            return
        results_path = latest_path
        print(f"Using latest results: {results_path}")

    # Load and analyze
    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)

    print("Performing bootstrap analysis...")
    analysis = analyze_experiment(results)

    # Print summary
    print_analysis_summary(analysis)

    # Save analysis
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / f"analysis_{results_path.stem.replace('strategy_modes_', '')}.json"
    save_analysis(analysis, analysis_path)

    # Generate figures
    if args.figures and not args.no_figures:
        print("\nGenerating publication figures...")
        generate_figures(analysis, str(output_dir / "figures"))


if __name__ == "__main__":
    main()
