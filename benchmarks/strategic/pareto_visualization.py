"""Pareto frontier visualization for strategic optimization results.

This script visualizes the quality-cost-time tradeoffs across contract modes
using matplotlib to show the Pareto-optimal frontier.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: uv pip install matplotlib")
    exit(1)

from agent_contracts.core import ContractMode
from benchmarks.strategic.strategic_optimization_test import (
    BenchmarkResult,
    StrategicOptimizationBenchmark,
)


def visualize_2d_tradeoff(
    results: dict[ContractMode, BenchmarkResult],
    x_metric: str,
    y_metric: str,
    title: str,
) -> None:
    """Visualize a 2D tradeoff between two metrics.

    Args:
        results: Benchmark results from all modes
        x_metric: Metric for x-axis ('tokens_used', 'time_seconds', 'quality')
        y_metric: Metric for y-axis
        title: Plot title
    """
    # Extract data
    modes = [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]
    colors = {"urgent": "#FF6B6B", "balanced": "#4ECDC4", "economical": "#95E1D3"}
    labels = {"urgent": "URGENT", "balanced": "BALANCED", "economical": "ECONOMICAL"}

    x_values = []
    y_values = []
    mode_labels = []
    mode_colors = []

    for mode in modes:
        result = results[mode]
        x = getattr(result, x_metric)
        y = getattr(result, y_metric)

        x_values.append(x)
        y_values.append(y)
        mode_labels.append(labels[mode.value])
        mode_colors.append(colors[mode.value])

    # Create plot
    plt.figure(figsize=(10, 7))

    # Plot points
    for i, _mode in enumerate(modes):
        plt.scatter(
            x_values[i],
            y_values[i],
            c=mode_colors[i],
            s=300,
            alpha=0.6,
            edgecolors="black",
            linewidth=2,
            label=mode_labels[i],
            zorder=3,
        )

    # Connect Pareto frontier with line
    plt.plot(x_values, y_values, "k--", alpha=0.3, zorder=1)

    # Add labels for each point
    for i, label in enumerate(mode_labels):
        plt.annotate(
            label,
            (x_values[i], y_values[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Formatting
    plt.xlabel(x_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.ylabel(y_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="best", fontsize=10)

    # Add Pareto frontier label
    plt.text(
        0.02,
        0.98,
        "Pareto Frontier",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()


def visualize_all_metrics(results: dict[ContractMode, BenchmarkResult]) -> None:
    """Create a comprehensive visualization of all metrics.

    Args:
        results: Benchmark results from all modes
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Strategic Optimization: Pareto Frontier Analysis",
        fontsize=16,
        fontweight="bold",
    )

    modes = [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]
    colors = {"urgent": "#FF6B6B", "balanced": "#4ECDC4", "economical": "#95E1D3"}
    labels = {"urgent": "URGENT", "balanced": "BALANCED", "economical": "ECONOMICAL"}

    # 1. Cost vs Time tradeoff (top-left)
    ax = axes[0, 0]
    for mode in modes:
        res = results[mode]
        ax.scatter(
            res.cost_usd,
            res.time_seconds,
            c=colors[mode.value],
            s=300,
            alpha=0.6,
            edgecolors="black",
            linewidth=2,
            label=labels[mode.value],
        )
        ax.annotate(
            labels[mode.value],
            (res.cost_usd, res.time_seconds),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Cost (USD)", fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontweight="bold")
    ax.set_title("Cost vs Time Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Quality vs Cost tradeoff (top-right)
    ax = axes[0, 1]
    for mode in modes:
        res = results[mode]
        ax.scatter(
            res.cost_usd,
            res.quality,
            c=colors[mode.value],
            s=300,
            alpha=0.6,
            edgecolors="black",
            linewidth=2,
            label=labels[mode.value],
        )
        ax.annotate(
            labels[mode.value],
            (res.cost_usd, res.quality),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Cost (USD)", fontweight="bold")
    ax.set_ylabel("Quality", fontweight="bold")
    ax.set_title("Quality vs Cost Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Quality vs Time tradeoff (bottom-left)
    ax = axes[1, 0]
    for mode in modes:
        res = results[mode]
        ax.scatter(
            res.time_seconds,
            res.quality,
            c=colors[mode.value],
            s=300,
            alpha=0.6,
            edgecolors="black",
            linewidth=2,
            label=labels[mode.value],
        )
        ax.annotate(
            labels[mode.value],
            (res.time_seconds, res.quality),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Quality", fontweight="bold")
    ax.set_title("Quality vs Time Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Bar chart comparison (bottom-right)
    ax = axes[1, 1]
    x = range(len(modes))
    width = 0.25

    # Normalize metrics for comparison
    max_tokens = max(results[m].tokens_used for m in modes)
    max_time = max(results[m].time_seconds for m in modes)

    tokens_norm = [results[m].tokens_used / max_tokens for m in modes]
    time_norm = [results[m].time_seconds / max_time for m in modes]
    quality = [results[m].quality for m in modes]

    ax.bar([i - width for i in x], tokens_norm, width, label="Tokens (normalized)", alpha=0.7)
    ax.bar(x, time_norm, width, label="Time (normalized)", alpha=0.7)
    ax.bar([i + width for i in x], quality, width, label="Quality", alpha=0.7)

    ax.set_xlabel("Mode", fontweight="bold")
    ax.set_ylabel("Normalized Value", fontweight="bold")
    ax.set_title("Metric Comparison Across Modes")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m.value] for m in modes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()


def main() -> None:
    """Run benchmark and visualize results."""
    print("Running strategic optimization benchmark...")
    benchmark = StrategicOptimizationBenchmark(verbose=False)
    results = benchmark.run_multi_task_scenario()

    print("\nGenerating visualizations...")

    # Create comprehensive visualization
    visualize_all_metrics(results)
    plt.savefig("pareto_frontier_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: pareto_frontier_analysis.png")

    # Show plot
    plt.show()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
