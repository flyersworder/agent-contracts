"""Experiment runner for logic reasoning strategy modes evaluation.

This script runs the URGENT vs ECONOMICAL vs BALANCED comparison experiment
using OpenR1 Logic Puzzles (Feb 2025, guaranteed uncontaminated).

Usage:
    # Run with default settings (20 problems, all modes)
    uv run python -m evaluation.strategy_modes.run_logic_experiment

    # Run with more problems
    uv run python -m evaluation.strategy_modes.run_logic_experiment --n-problems 50

    # Run single mode only
    uv run python -m evaluation.strategy_modes.run_logic_experiment --mode balanced

Dataset: sunyiyou/openr1_logic_and_puzzles_1k_nm
- 1000 logic/word problems from February 2025
- ~672 with simple numeric answers (for deterministic evaluation)
- Tractable difficulty: ~5-15 seconds per problem
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .logic_orchestrator import LogicModesRunner, LogicTrialResult, compute_logic_statistics
from .logic_tasks import get_logic_task_statistics, load_logic_tasks


def run_experiment(
    n_problems: int = 20,
    modes: list[str] | None = None,
    model: str = "gemini/gemini-2.5-flash",
    token_budget: int = 8000,
    random_seed: int = 42,
    output_dir: str = "results/strategy_modes",
    verbose: bool = True,
    enable_timeout: bool = True,
) -> dict[str, Any]:
    """Run the logic reasoning strategy modes experiment.

    Args:
        n_problems: Number of logic problems to evaluate
        modes: List of modes to test (default: all three)
        model: LLM model to use
        token_budget: Token budget per task
        random_seed: Random seed for reproducibility
        output_dir: Directory for results
        verbose: If True, print progress
        enable_timeout: If True, enforce mode-specific API timeouts

    Returns:
        Dictionary with experiment results
    """
    if modes is None:
        modes = ["urgent", "economical", "balanced"]

    print("=" * 80)
    print("LOGIC REASONING STRATEGY MODES EXPERIMENT")
    print("Comparing URGENT vs ECONOMICAL vs BALANCED on OpenR1 Logic Puzzles")
    print("=" * 80)
    print()

    # Initialize results
    results: dict[str, Any] = {
        "experiment": {
            "type": "logic_reasoning",
            "dataset": "openr1_logic_puzzles",
            "timestamp": datetime.now().isoformat(),
            "n_problems": n_problems,
            "modes": modes,
            "model": model,
            "token_budget": token_budget,
            "seed": random_seed,
            "enable_timeout": enable_timeout,
            "timeout_config": {
                "urgent": 15.0,
                "economical": 30.0,
                "balanced": 45.0,
            }
            if enable_timeout
            else None,
            "reasoning_effort_config": {
                "urgent": "none",
                "balanced": "medium",
                "economical": "low",
            },
        },
        "trials": [],
        "summary": {},
    }

    # Load tasks
    print(f"Loading {n_problems} logic problems...")
    tasks = load_logic_tasks(
        limit=n_problems,
        numeric_only=True,
        random_seed=random_seed,
    )
    stats = get_logic_task_statistics(tasks)
    print(f"Loaded {stats['total']} problems")
    print(f"  Sources: {stats.get('sources', {})}")
    print()

    # Update results with actual count
    results["experiment"]["n_problems_actual"] = len(tasks)

    # Initialize runner
    runner = LogicModesRunner(
        model=model,
        token_budget=token_budget,
        enable_timeout=enable_timeout,
    )

    # Run trials
    total_trials = len(tasks) * len(modes)
    trial_num = 0

    all_results: dict[str, list[LogicTrialResult]] = {mode: [] for mode in modes}

    for task in tasks:
        source_str = task.source if task.source else "Unknown"
        print(f"\n[Problem] {task.task_id} (Source: {source_str})")
        print(f"  Q: {task.question[:100]}...")

        for mode in modes:
            trial_num += 1
            print(f"\n  [{trial_num}/{total_trials}] {mode.upper()} mode...")

            try:
                trial_result = runner.run_task(task, mode, verbose=verbose)
                all_results[mode].append(trial_result)
                results["trials"].append(trial_result.to_dict())

                status = "CORRECT" if trial_result.correct else "WRONG"
                print(
                    f"    {status} | Predicted: {trial_result.predicted_answer} | "
                    f"Expected: {task.answer} | Tokens: {trial_result.tokens_used:,} | "
                    f"Time: {trial_result.execution_time:.1f}s"
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                failed_trial = LogicTrialResult(
                    task_id=task.task_id,
                    mode=mode,
                    success=False,
                    error=str(e),
                    expected_answer=task.answer,
                    source=task.source,
                )
                all_results[mode].append(failed_trial)
                results["trials"].append(failed_trial.to_dict())

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("COMPUTING SUMMARY STATISTICS")
    print("=" * 80)

    for mode in modes:
        mode_stats = compute_logic_statistics(all_results[mode])
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
    metrics: list[tuple[str, str, Callable[[Any], str]]] = [
        ("Success Rate", "success_rate", lambda x: f"{x * 100:.1f}%"),
        ("ACCURACY", "accuracy", lambda x: f"{x * 100:.1f}%"),
        ("Timeout Rate", "timeout_rate", lambda x: f"{x * 100:.1f}%"),
        ("Timeout Limit (s)", "timeout_seconds", lambda x: f"{x:.0f}" if x else "None"),
        ("Reasoning Effort", "reasoning_effort", lambda x: str(x) if x else "N/A"),
        ("Avg Tokens", "avg_tokens", lambda x: f"{x:,.0f}"),
        ("Avg Reasoning Tokens", "avg_reasoning_tokens", lambda x: f"{x:,.0f}"),
        ("Avg Exec Time (s)", "avg_execution_time", lambda x: f"{x:.2f}"),
    ]

    for name, key, fmt in metrics:
        print(f"{name:<25}", end="")
        for mode in modes:
            mode_stats = summary.get(mode, {})
            value = mode_stats.get(key, 0)
            print(f"{fmt(value):<18}", end="")
        print()

    # Key comparison
    print("\n" + "-" * 40)
    print("KEY FINDINGS:")
    print("-" * 40)

    if "urgent" in summary and "balanced" in summary:
        urg_acc = summary["urgent"].get("accuracy", 0)
        bal_acc = summary["balanced"].get("accuracy", 0)
        diff = (bal_acc - urg_acc) * 100
        print(f"  BALANCED vs URGENT accuracy difference: {diff:+.1f}%")

        urg_tokens = summary["urgent"].get("avg_tokens", 0)
        bal_tokens = summary["balanced"].get("avg_tokens", 0)
        token_diff = ((bal_tokens - urg_tokens) / urg_tokens * 100) if urg_tokens > 0 else 0
        print(
            f"  Avg tokens: URGENT={urg_tokens:.0f}, BALANCED={bal_tokens:.0f} ({token_diff:+.0f}%)"
        )

        urg_time = summary["urgent"].get("avg_execution_time", 0)
        bal_time = summary["balanced"].get("avg_execution_time", 0)
        time_diff = ((bal_time - urg_time) / urg_time * 100) if urg_time > 0 else 0
        print(f"  Avg time: URGENT={urg_time:.1f}s, BALANCED={bal_time:.1f}s ({time_diff:+.0f}%)")

    if "economical" in summary and "balanced" in summary:
        eco_acc = summary["economical"].get("accuracy", 0)
        bal_acc = summary["balanced"].get("accuracy", 0)
        print(f"  BALANCED vs ECONOMICAL accuracy difference: {(bal_acc - eco_acc) * 100:+.1f}%")


def save_results(results: dict[str, Any], output_dir: str) -> Path:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"logic_openr1_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run logic reasoning strategy modes experiment")
    parser.add_argument(
        "--n-problems",
        type=int,
        default=20,
        help="Number of problems to evaluate (default: 20)",
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
        default=8000,
        help="Token budget per task (default: 8000)",
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
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable mode-specific timeout enforcement",
    )

    args = parser.parse_args()

    modes = [args.mode] if args.mode else None

    run_experiment(
        n_problems=args.n_problems,
        modes=modes,
        model=args.model,
        token_budget=args.token_budget,
        random_seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        enable_timeout=not args.no_timeout,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
