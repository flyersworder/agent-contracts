"""Experiment runner for math reasoning strategy modes evaluation.

This script runs the URGENT vs ECONOMICAL vs BALANCED comparison experiment
using MathArena 2025 competition math problems (guaranteed uncontaminated).

Usage:
    # Run with default settings (SMT 2025, 20 problems, all modes)
    uv run python -m evaluation.strategy_modes.run_math_experiment

    # Run with specific dataset
    uv run python -m evaluation.strategy_modes.run_math_experiment \
        --dataset aime_2025 \
        --n-problems 15

    # Run with all integer-answer problems from SMT 2025
    uv run python -m evaluation.strategy_modes.run_math_experiment \
        --dataset smt_2025 \
        --n-problems 28

    # Run single mode only
    uv run python -m evaluation.strategy_modes.run_math_experiment --mode balanced

Available datasets:
    - smt_2025: Stanford Math Tournament 2025 (28 integer-answer problems)
    - aime_2025: AIME 2025 I & II combined (30 problems)
    - aime_2025_I: AIME 2025 I only (15 problems)
    - aime_2025_II: AIME 2025 II only (15 problems)
    - cmimc_2025: Carnegie Mellon Math Competition 2025
    - brumo_2025: BRUMO 2025
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .math_orchestrator import MathModesRunner, MathTrialResult, compute_mode_statistics
from .math_tasks import get_task_statistics, list_available_datasets, load_math_tasks


def run_experiment(
    dataset: str = "smt_2025",
    n_problems: int | None = None,
    modes: list[str] | None = None,
    model: str = "gemini/gemini-2.5-flash",
    token_budget: int = 20000,
    random_seed: int = 42,
    output_dir: str = "results/strategy_modes",
    verbose: bool = True,
    enable_timeout: bool = True,
    integer_only: bool = True,
) -> dict[str, Any]:
    """Run the math reasoning strategy modes experiment.

    Args:
        dataset: MathArena dataset to use (e.g., "smt_2025", "aime_2025")
        n_problems: Number of math problems to evaluate (None = all available)
        modes: List of modes to test (default: all three)
        model: LLM model to use
        token_budget: Token budget per task
        random_seed: Random seed for reproducibility
        output_dir: Directory for results
        verbose: If True, print progress
        enable_timeout: If True, enforce mode-specific API timeouts
        integer_only: If True, only use problems with integer answers

    Returns:
        Dictionary with experiment results
    """
    if modes is None:
        modes = ["urgent", "economical", "balanced"]

    print("=" * 80)
    print("MATH REASONING STRATEGY MODES EXPERIMENT")
    print("Comparing URGENT vs ECONOMICAL vs BALANCED on MathArena 2025")
    print("=" * 80)
    print()

    # Initialize results
    results: dict[str, Any] = {
        "experiment": {
            "type": "math_reasoning",
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "n_problems": n_problems,
            "modes": modes,
            "model": model,
            "token_budget": token_budget,
            "seed": random_seed,
            "integer_only": integer_only,
            "enable_timeout": enable_timeout,
            "timeout_config": {
                "urgent": 45.0,
                "economical": 60.0,
                "balanced": 120.0,
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
    print(f"Loading problems from {dataset}...")
    tasks = load_math_tasks(
        dataset=dataset,
        limit=n_problems,
        integer_only=integer_only,
        random_seed=random_seed,
    )
    stats = get_task_statistics(tasks)
    print(f"Loaded {stats['total']} problems")
    print(f"  Problem types: {stats.get('problem_types', {})}")
    print(f"  Integer answers: {stats.get('integer_answers', 0)}")
    print()

    # Update results with actual count
    results["experiment"]["n_problems_actual"] = len(tasks)

    # Initialize runner
    runner = MathModesRunner(
        model=model,
        token_budget=token_budget,
        enable_timeout=enable_timeout,
    )

    # Run trials
    total_trials = len(tasks) * len(modes)
    trial_num = 0

    all_results: dict[str, list[MathTrialResult]] = {mode: [] for mode in modes}

    for task in tasks:
        type_str = ", ".join(task.problem_type) if task.problem_type else "Unknown"
        print(f"\n[Problem] {task.task_id} ({type_str})")
        print(f"  Q: {task.question[:80]}...")

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
                    f"Expected: {task.answer} | Tokens: {trial_result.tokens_used:,}"
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                failed_trial = MathTrialResult(
                    task_id=task.task_id,
                    mode=mode,
                    success=False,
                    error=str(e),
                    expected_answer=task.answer,
                    problem_type=task.problem_type,
                    dataset=task.dataset,
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
    save_results(results, output_dir, dataset)

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

    # Accuracy by problem type
    print("\n" + "-" * 40)
    print("ACCURACY BY PROBLEM TYPE:")
    print("-" * 40)

    # Collect all problem types across modes
    all_types: set[str] = set()
    for mode in modes:
        by_type = summary.get(mode, {}).get("accuracy_by_type", {})
        all_types.update(by_type.keys())

    for ptype in sorted(all_types):
        print(f"\n  {ptype}:", end="")
        for mode in modes:
            by_type = summary.get(mode, {}).get("accuracy_by_type", {})
            acc = by_type.get(ptype, 0)
            print(f"  {mode.upper()}: {acc * 100:.0f}%", end="")
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
        print(f"  Avg tokens: URGENT={urg_tokens:.0f}, BALANCED={bal_tokens:.0f}")

        urg_time = summary["urgent"].get("avg_execution_time", 0)
        bal_time = summary["balanced"].get("avg_execution_time", 0)
        print(f"  Avg time: URGENT={urg_time:.1f}s, BALANCED={bal_time:.1f}s")


def save_results(results: dict[str, Any], output_dir: str, dataset: str) -> Path:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"math_{dataset}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run math reasoning strategy modes experiment")
    parser.add_argument(
        "--dataset",
        default="smt_2025",
        choices=list(list_available_datasets().keys()),
        help="MathArena dataset to use (default: smt_2025)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        help="Number of problems to evaluate (default: all available)",
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
    parser.add_argument(
        "--include-non-integer",
        action="store_true",
        help="Include problems with non-integer answers (fractions, expressions)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available MathArena datasets:")
        for name, path in list_available_datasets().items():
            print(f"  {name}: {path}")
        return

    modes = [args.mode] if args.mode else None

    run_experiment(
        dataset=args.dataset,
        n_problems=args.n_problems,
        modes=modes,
        model=args.model,
        token_budget=args.token_budget,
        random_seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        enable_timeout=not args.no_timeout,
        integer_only=not args.include_non_integer,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
