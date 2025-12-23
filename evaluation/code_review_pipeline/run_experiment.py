"""Experiment runner for code review pipeline evaluation.

This script runs the CONTRACTED vs UNCONTRACTED comparison experiment
using LiveCodeBench problems.

Usage:
    # Run with default settings (10 problems, both conditions)
    python -m evaluation.code_review_pipeline.run_experiment

    # Run with specific settings
    python -m evaluation.code_review_pipeline.run_experiment \\
        --n-problems 50 \\
        --difficulty medium \\
        --after-date 2025-02-01 \\
        --seed 42

    # Run only contracted condition
    python -m evaluation.code_review_pipeline.run_experiment --contracted-only
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .agents import PipelineConfig
from .orchestrator import ContractedPipeline, PipelineResult, UncontractedPipeline
from .tasks import CodeTask, TaskDifficulty, get_task_statistics, load_tasks


@dataclass
class ExperimentConfig:
    """Configuration for the experiment.

    Attributes:
        n_problems: Number of problems to test
        difficulty: Filter by difficulty (None = all)
        after_date: Only problems after this date
        random_seed: Random seed for reproducibility
        conditions: Which conditions to run
        output_dir: Where to save results
    """

    n_problems: int = 10
    difficulty: str | None = None
    after_date: str = "2025-02-01"
    random_seed: int = 42
    conditions: list[str] | None = None  # ["CONTRACTED", "UNCONTRACTED"]
    output_dir: str = "evaluation/results/code_review"

    def __post_init__(self) -> None:
        if self.conditions is None:
            self.conditions = ["CONTRACTED", "UNCONTRACTED"]


@dataclass
class ExperimentResults:
    """Results from the experiment.

    Attributes:
        config: Experiment configuration
        started_at: When experiment started
        completed_at: When experiment completed
        trials: List of trial results
        summary: Summary statistics
    """

    config: dict[str, Any]
    started_at: str
    completed_at: str | None = None
    trials: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None


def run_single_trial(
    task: CodeTask,
    condition: str,
    config: PipelineConfig,
) -> PipelineResult:
    """Run a single trial for a task and condition.

    Args:
        task: The coding task
        condition: "CONTRACTED" or "UNCONTRACTED"
        config: Pipeline configuration

    Returns:
        PipelineResult with execution details
    """
    pipeline: ContractedPipeline | UncontractedPipeline
    if condition == "CONTRACTED":
        pipeline = ContractedPipeline(config)
    else:
        pipeline = UncontractedPipeline(config)

    return pipeline.run(task)


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run the full experiment.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResults with all trial data
    """
    print("=" * 80)
    print("CODE REVIEW PIPELINE EXPERIMENT")
    print("Comparing CONTRACTED vs UNCONTRACTED execution")
    print("=" * 80)
    print()

    # Initialize results
    results = ExperimentResults(
        config=asdict(config),
        started_at=datetime.now().isoformat(),
        trials=[],
    )

    # Load tasks
    print("Loading tasks from LiveCodeBench...")
    difficulty = TaskDifficulty(config.difficulty) if config.difficulty else None
    tasks = load_tasks(
        difficulty=difficulty,
        after_date=config.after_date,
        limit=config.n_problems,
        random_seed=config.random_seed,
    )

    stats = get_task_statistics(tasks)
    print(f"Loaded {stats['total']} tasks")
    print(f"  Difficulty distribution: {stats['by_difficulty']}")
    print(f"  Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print()

    # Calculate total trials
    total_trials = len(tasks) * len(config.conditions or [])
    print(
        f"Running {total_trials} trials ({len(tasks)} tasks x {len(config.conditions or [])} conditions)"
    )
    print()

    # Run trials
    trial_num = 0
    for task in tasks:
        # Get difficulty-appropriate config
        pipeline_config = PipelineConfig.for_difficulty(task.difficulty.value)

        for condition in config.conditions or []:
            trial_num += 1
            print(f"[{trial_num}/{total_trials}] {task.title[:40]}... ({condition})")

            start_time = time.time()
            try:
                result = run_single_trial(task, condition, pipeline_config)

                # Convert to dict for storage
                trial_dict = result.to_dict()
                trial_dict["trial_num"] = trial_num
                trial_dict["condition"] = condition
                trial_dict["duration_seconds"] = time.time() - start_time

                status = "✓" if result.success else "✗"
                print(
                    f"  {status} | Iterations: {result.num_iterations} | "
                    f"Tokens: {result.total_tokens:,} | "
                    f"LLM Calls: {result.total_llm_calls}"
                )

                if result.runaway_prevented:
                    print("  ⚠️  Runaway prevented (hit iteration limit)")

            except Exception as e:
                print(f"  ERROR: {e}")
                trial_dict = {
                    "trial_num": trial_num,
                    "task_id": task.task_id,
                    "task_title": task.title,
                    "difficulty": task.difficulty.value,
                    "condition": condition,
                    "success": False,
                    "error": str(e),
                    "duration_seconds": time.time() - start_time,
                }

            if results.trials is not None:
                results.trials.append(trial_dict)

            # Save intermediate results every 10 trials
            if trial_num % 10 == 0:
                save_results(results, intermediate=True, config=config)

    results.completed_at = datetime.now().isoformat()
    results.summary = compute_summary(results.trials or [])

    return results


def compute_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from trials.

    Args:
        trials: List of trial result dictionaries

    Returns:
        Summary statistics dictionary
    """
    summary: dict[str, Any] = {
        "total_trials": len(trials),
        "by_condition": {},
        "by_difficulty": {},
    }

    # Group by condition
    for condition in ["CONTRACTED", "UNCONTRACTED"]:
        cond_trials = [t for t in trials if t.get("condition") == condition]
        if not cond_trials:
            continue

        successes = sum(1 for t in cond_trials if t.get("success", False))
        runaways = sum(1 for t in cond_trials if t.get("runaway_prevented", False))
        total_tokens = sum(t.get("total_tokens", 0) for t in cond_trials)
        total_llm_calls = sum(t.get("total_llm_calls", 0) for t in cond_trials)
        iterations = [t.get("num_iterations", 0) for t in cond_trials]

        summary["by_condition"][condition] = {
            "n_trials": len(cond_trials),
            "success_rate": successes / len(cond_trials) if cond_trials else 0,
            "successes": successes,
            "runaway_prevented": runaways,
            "avg_tokens": total_tokens / len(cond_trials) if cond_trials else 0,
            "avg_llm_calls": total_llm_calls / len(cond_trials) if cond_trials else 0,
            "avg_iterations": sum(iterations) / len(iterations) if iterations else 0,
            "max_iterations": max(iterations) if iterations else 0,
            "token_variance": compute_variance([t.get("total_tokens", 0) for t in cond_trials]),
        }

    # Group by difficulty
    for difficulty in ["easy", "medium", "hard"]:
        diff_trials = [t for t in trials if t.get("difficulty") == difficulty]
        if not diff_trials:
            continue

        successes = sum(1 for t in diff_trials if t.get("success", False))

        summary["by_difficulty"][difficulty] = {
            "n_trials": len(diff_trials),
            "success_rate": successes / len(diff_trials) if diff_trials else 0,
        }

    return summary


def compute_variance(values: list[float | int]) -> float:
    """Compute variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def save_results(
    results: ExperimentResults,
    intermediate: bool = False,
    config: ExperimentConfig | None = None,
) -> Path:
    """Save results to JSON file.

    Args:
        results: Experiment results
        intermediate: Whether this is an intermediate save
        config: Experiment config for output directory

    Returns:
        Path to saved file
    """
    output_dir = Path(config.output_dir if config else "evaluation/results/code_review")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_intermediate" if intermediate else ""
    output_file = output_dir / f"experiment_{timestamp}{suffix}.json"

    # Convert to serializable dict
    data = {
        "config": results.config,
        "started_at": results.started_at,
        "completed_at": results.completed_at,
        "trials": results.trials,
        "summary": results.summary,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def print_summary(results: ExperimentResults) -> None:
    """Print summary of experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    if not results.summary:
        print("No summary available")
        return

    print(f"\nTotal trials: {results.summary['total_trials']}")

    # By condition
    print("\n" + "-" * 40)
    print("BY CONDITION:")
    print("-" * 40)

    print(f"\n{'Metric':<25} {'CONTRACTED':<20} {'UNCONTRACTED':<20}")
    print("-" * 65)

    contracted = results.summary["by_condition"].get("CONTRACTED", {})
    uncontracted = results.summary["by_condition"].get("UNCONTRACTED", {})

    metrics: list[tuple[str, str, Callable[[float | int], str]]] = [
        ("Success Rate", "success_rate", lambda x: f"{x * 100:.1f}%"),
        ("Avg Iterations", "avg_iterations", lambda x: f"{x:.1f}"),
        ("Avg Tokens", "avg_tokens", lambda x: f"{x:,.0f}"),
        ("Avg LLM Calls", "avg_llm_calls", lambda x: f"{x:.1f}"),
        ("Runaway Prevented", "runaway_prevented", lambda x: f"{x}"),
        ("Token Variance", "token_variance", lambda x: f"{x:,.0f}"),
    ]

    for name, key, fmt in metrics:
        c_val = fmt(contracted.get(key, 0))
        u_val = fmt(uncontracted.get(key, 0))
        print(f"{name:<25} {c_val:<20} {u_val:<20}")

    # By difficulty
    if results.summary["by_difficulty"]:
        print("\n" + "-" * 40)
        print("BY DIFFICULTY:")
        print("-" * 40)

        for diff, stats in results.summary["by_difficulty"].items():
            print(f"\n{diff.upper()}:")
            print(f"  Trials: {stats['n_trials']}")
            print(f"  Success Rate: {stats['success_rate'] * 100:.1f}%")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run code review pipeline experiment")
    parser.add_argument(
        "--n-problems",
        type=int,
        default=10,
        help="Number of problems to test (default: 10)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty (default: all)",
    )
    parser.add_argument(
        "--after-date",
        default="2025-02-01",
        help="Only problems after this date (default: 2025-02-01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--contracted-only",
        action="store_true",
        help="Run only contracted condition",
    )
    parser.add_argument(
        "--uncontracted-only",
        action="store_true",
        help="Run only uncontracted condition",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/code_review",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine conditions
    conditions = ["CONTRACTED", "UNCONTRACTED"]
    if args.contracted_only:
        conditions = ["CONTRACTED"]
    elif args.uncontracted_only:
        conditions = ["UNCONTRACTED"]

    config = ExperimentConfig(
        n_problems=args.n_problems,
        difficulty=args.difficulty,
        after_date=args.after_date,
        random_seed=args.seed,
        conditions=conditions,
        output_dir=args.output_dir,
    )

    results = run_experiment(config)
    print_summary(results)
    save_results(results, config=config)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
