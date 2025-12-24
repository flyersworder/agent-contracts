"""Experiment runner for strategy modes evaluation.

This script runs the URGENT vs ECONOMICAL vs BALANCED comparison experiment
using CNN/DailyMail summarization tasks.

Usage:
    # Run with default settings (10 articles, all modes)
    python -m evaluation.strategy_modes.run_experiment

    # Run with specific settings
    python -m evaluation.strategy_modes.run_experiment \\
        --n-articles 50 \\
        --model gemini/gemini-2.5-flash \\
        --seed 42

    # Run single mode only
    python -m evaluation.strategy_modes.run_experiment --mode economical
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .orchestrator import StrategyModesRunner, TrialResult, compute_mode_statistics
from .tasks import get_task_statistics, load_tasks


def run_experiment(
    n_articles: int = 10,
    modes: list[str] | None = None,
    model: str = "gpt-4o-mini",
    token_budget: int = 4000,
    random_seed: int = 42,
    output_dir: str = "results/strategy_modes",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the strategy modes experiment.

    Args:
        n_articles: Number of articles to evaluate
        modes: List of modes to test (default: all three)
        model: LLM model to use
        token_budget: Token budget per task
        random_seed: Random seed for reproducibility
        output_dir: Directory for results
        verbose: If True, print progress

    Returns:
        Dictionary with experiment results
    """
    if modes is None:
        modes = ["urgent", "economical", "balanced"]

    print("=" * 80)
    print("STRATEGY MODES EXPERIMENT")
    print("Comparing URGENT vs ECONOMICAL vs BALANCED")
    print("=" * 80)
    print()

    # Initialize results
    results: dict[str, Any] = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "n_articles": n_articles,
            "modes": modes,
            "model": model,
            "token_budget": token_budget,
            "seed": random_seed,
        },
        "trials": [],
        "summary": {},
    }

    # Load tasks
    print("Loading tasks from CNN/DailyMail...")
    tasks = load_tasks(limit=n_articles, random_seed=random_seed)
    stats = get_task_statistics(tasks)
    print(f"Loaded {stats['total']} articles")
    print(f"  Article length: {stats['article_length']['avg']:.0f} chars (avg)")
    print(f"  Reference length: {stats['reference_length']['avg']:.0f} chars (avg)")
    print()

    # Initialize runner
    runner = StrategyModesRunner(
        model=model,
        token_budget=token_budget,
    )

    # Run trials
    total_trials = len(tasks) * len(modes)
    trial_num = 0

    all_results: dict[str, list[TrialResult]] = {mode: [] for mode in modes}

    for task in tasks:
        print(f"\n[Article] {task.task_id}")
        print(f"  Length: {task.article_length} chars")

        for mode in modes:
            trial_num += 1
            print(f"\n  [{trial_num}/{total_trials}] {mode.upper()} mode...")

            try:
                trial_result = runner.run_task(task, mode, verbose=verbose)
                all_results[mode].append(trial_result)
                results["trials"].append(trial_result.to_dict())

                status = "✓" if trial_result.success else "✗"
                print(
                    f"    {status} Tokens: {trial_result.tokens_used:,} | "
                    f"Words: {trial_result.word_count} | "
                    f"ROUGE-L: {trial_result.rouge_metrics.rouge_l_f1:.3f}"
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                # Record failed trial
                failed_trial = TrialResult(
                    task_id=task.task_id,
                    mode=mode,
                    success=False,
                    error=str(e),
                )
                all_results[mode].append(failed_trial)
                results["trials"].append(failed_trial.to_dict())

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("COMPUTING SUMMARY STATISTICS")
    print("=" * 80)

    for mode in modes:
        mode_stats = compute_mode_statistics(all_results[mode])
        results["summary"][mode] = mode_stats

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, output_dir)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    summary = results.get("summary", {})
    modes = list(summary.keys())

    if not modes:
        print("No results to summarize")
        return

    # Header
    print(f"\n{'Metric':<25}", end="")
    for mode in modes:
        print(f"{mode.upper():<18}", end="")
    print()
    print("-" * (25 + 18 * len(modes)))

    # Metrics
    metrics: list[tuple[str, str, Callable[[float], str]]] = [
        ("Success Rate", "success_rate", lambda x: f"{x * 100:.1f}%"),
        ("Avg Tokens", "avg_tokens", lambda x: f"{x:,.0f}"),
        ("Std Tokens", "std_tokens", lambda x: f"{x:,.0f}"),
        ("Avg Word Count", "avg_word_count", lambda x: f"{x:.0f}"),
        ("Avg ROUGE-L F1", "avg_rouge_l_f1", lambda x: f"{x:.3f}"),
        ("Std ROUGE-L F1", "std_rouge_l_f1", lambda x: f"{x:.3f}"),
        ("Avg Exec Time (s)", "avg_execution_time", lambda x: f"{x:.2f}"),
    ]

    for name, key, fmt in metrics:
        print(f"{name:<25}", end="")
        for mode in modes:
            mode_stats = summary.get(mode, {})
            value = mode_stats.get(key, 0)
            print(f"{fmt(value):<18}", end="")
        print()

    # Comparison
    print("\n" + "-" * 40)
    print("KEY COMPARISONS:")
    print("-" * 40)

    if "economical" in summary and "balanced" in summary:
        econ_tokens = summary["economical"].get("avg_tokens", 0)
        bal_tokens = summary["balanced"].get("avg_tokens", 0)
        if bal_tokens > 0:
            token_savings = (1 - econ_tokens / bal_tokens) * 100
            print(f"  ECONOMICAL vs BALANCED token savings: {token_savings:.1f}%")

    if "urgent" in summary and "balanced" in summary:
        urg_time = summary["urgent"].get("avg_execution_time", 0)
        bal_time = summary["balanced"].get("avg_execution_time", 0)
        if bal_time > 0:
            time_savings = (1 - urg_time / bal_time) * 100
            print(f"  URGENT vs BALANCED time savings: {time_savings:.1f}%")

    # Quality comparison
    print("\n  Quality maintained across modes:")
    for mode in modes:
        rouge = summary.get(mode, {}).get("avg_rouge_l_f1", 0)
        print(f"    {mode.upper()}: ROUGE-L = {rouge:.3f}")


def save_results(results: dict[str, Any], output_dir: str) -> Path:
    """Save results to JSON file.

    Args:
        results: Experiment results
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"strategy_modes_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run strategy modes experiment for Agent Contracts"
    )
    parser.add_argument(
        "--n-articles",
        type=int,
        default=10,
        help="Number of articles to evaluate (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["urgent", "economical", "balanced"],
        help="Run only specified mode (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash",
        help="LLM model to use (default: gemini/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=4000,
        help="Token budget per task (default: 4000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/strategy_modes",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Determine modes to run
    modes = [args.mode] if args.mode else None

    # Run experiment
    run_experiment(
        n_articles=args.n_articles,
        modes=modes,
        model=args.model,
        token_budget=args.token_budget,
        random_seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
