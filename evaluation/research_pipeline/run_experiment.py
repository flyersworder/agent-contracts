#!/usr/bin/env python3
"""COINE 2026 Evaluation Experiment Runner.

This script runs the multi-agent research report generation experiment
to validate the Agent Contracts framework.

Experimental Design:
- n=25 research topics x 2 conditions = 50 trials
- Conditions: UNCONTRACTED (baseline) vs CONTRACTED (treatment)
- Within-subjects design: same topics in both conditions

Claims to Validate:
1. Contracts prevent runaway execution (the $47K problem)
2. Conservation laws enable safe delegation (Σbᵢ ≤ B)
3. Lifecycle provides clear accountability
4. Multi-agent workflows benefit most from contracts

Usage:
    # Full experiment (n=25)
    python run_experiment.py

    # Quick test (n=3)
    python run_experiment.py --quick

    # Single topic test
    python run_experiment.py --topic tech_01

    # Contracted only
    python run_experiment.py --mode contracted
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load environment
load_dotenv()


def run_experiment(
    n_topics: int = 25,
    mode: str = "both",
    topic_id: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the evaluation experiment.

    Args:
        n_topics: Number of topics to use (max 25)
        mode: "both", "contracted", or "uncontracted"
        topic_id: Specific topic ID to run (overrides n_topics)
        seed: Random seed for reproducibility
        verbose: Print progress messages
        output_dir: Directory for results (default: evaluation/results)

    Returns:
        Dictionary with experiment results
    """
    from evaluation.research_pipeline.orchestrator import (
        ContractedPipeline,
        SuccessCriteria,
        UncontractedPipeline,
    )
    from evaluation.research_pipeline.topics import ALL_TOPICS, get_topic

    # Set random seed
    random.seed(seed)

    # Select topics
    if topic_id:
        topic = get_topic(topic_id)
        if not topic:
            print(f"❌ Topic not found: {topic_id}")
            sys.exit(1)
        topics = [topic]
    else:
        topics = ALL_TOPICS[:n_topics]
        random.shuffle(topics)

    if verbose:
        print(f"\n{'=' * 70}")
        print("  COINE 2026 Evaluation Experiment")
        print(f"{'=' * 70}")
        print(f"\nTopics: {len(topics)}")
        print(f"Mode: {mode}")
        print(f"Seed: {seed}")
        print(f"{'=' * 70}\n")

    # Initialize pipelines
    uncontracted = (
        UncontractedPipeline(verbose=verbose) if mode in ("both", "uncontracted") else None
    )
    contracted = ContractedPipeline(verbose=verbose) if mode in ("both", "contracted") else None

    # Success criteria
    criteria = SuccessCriteria()

    # Results storage
    results: dict[str, Any] = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "n_topics": len(topics),
            "mode": mode,
            "seed": seed,
        },
        "topics": [t.id for t in topics],
        "trials": [],
        "summary": {},
    }

    # Run trials
    for i, topic in enumerate(topics):
        if verbose:
            print(f"\n[{i + 1}/{len(topics)}] Topic: {topic.title}")
            print(f"  Category: {topic.category} | Difficulty: {topic.difficulty}")

        trial_result: dict[str, Any] = {
            "topic_id": topic.id,
            "topic_title": topic.title,
            "category": topic.category,
            "difficulty": topic.difficulty,
        }

        # Run UNCONTRACTED condition
        if uncontracted:
            if verbose:
                print("\n  === UNCONTRACTED ===")

            try:
                unc_result = uncontracted.run(topic)
                score, success = criteria.evaluate(unc_result)

                trial_result["uncontracted"] = {
                    "success": unc_result.success,
                    "total_tokens": unc_result.total_tokens,
                    "tokens_by_agent": unc_result.tokens_by_agent,
                    "word_count": unc_result.word_count,
                    "citation_count": unc_result.citation_count,
                    "execution_time": unc_result.execution_time_seconds,
                    "quality_score": score,
                    "meets_criteria": success,
                    "error": unc_result.error,
                }

                if verbose:
                    print(f"    Tokens: {unc_result.total_tokens:,}")
                    print(f"    Words: {unc_result.word_count:,}")
                    print(f"    Citations: {unc_result.citation_count}")
                    print(f"    Quality: {score:.2f} ({'✅' if success else '❌'})")

            except Exception as e:
                trial_result["uncontracted"] = {"error": str(e), "success": False}
                if verbose:
                    print(f"    ❌ Error: {e}")

        # Run CONTRACTED condition
        if contracted:
            if verbose:
                print("\n  === CONTRACTED ===")

            try:
                con_result = contracted.run(topic)
                score, success = criteria.evaluate(con_result)

                trial_result["contracted"] = {
                    "success": con_result.success,
                    "total_tokens": con_result.total_tokens,
                    "tokens_by_agent": con_result.tokens_by_agent,
                    "word_count": con_result.word_count,
                    "citation_count": con_result.citation_count,
                    "execution_time": con_result.execution_time_seconds,
                    "budget_compliant": con_result.budget_compliant,
                    "conservation_violations": con_result.conservation_violations,
                    "quality_score": score,
                    "meets_criteria": success,
                    "error": con_result.error,
                }

                if verbose:
                    print(f"    Tokens: {con_result.total_tokens:,}")
                    print(f"    Budget: {'✅' if con_result.budget_compliant else '❌'}")
                    print(f"    Words: {con_result.word_count:,}")
                    print(f"    Citations: {con_result.citation_count}")
                    print(f"    Quality: {score:.2f} ({'✅' if success else '❌'})")

            except Exception as e:
                trial_result["contracted"] = {"error": str(e), "success": False}
                if verbose:
                    print(f"    ❌ Error: {e}")

        results["trials"].append(trial_result)

    # Calculate summary statistics
    results["summary"] = calculate_summary(results["trials"], mode)

    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Results saved to: {output_file}")
        print(f"{'=' * 70}")
        print_summary(results["summary"])

    return results


def calculate_summary(trials: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Calculate summary statistics from trials.

    Args:
        trials: List of trial results
        mode: Experiment mode

    Returns:
        Dictionary with summary statistics
    """
    summary: dict[str, Any] = {"n_trials": len(trials)}

    for condition in ["uncontracted", "contracted"]:
        if mode not in ("both", condition):
            continue

        cond_results = [t.get(condition, {}) for t in trials if condition in t]

        if not cond_results:
            continue

        # Success rate
        successes = [r for r in cond_results if r.get("success", False)]
        summary[f"{condition}_success_rate"] = len(successes) / len(cond_results)

        # Token statistics
        tokens = [r.get("total_tokens", 0) for r in successes]
        if tokens:
            summary[f"{condition}_avg_tokens"] = sum(tokens) / len(tokens)
            summary[f"{condition}_min_tokens"] = min(tokens)
            summary[f"{condition}_max_tokens"] = max(tokens)

        # Quality scores
        scores = [r.get("quality_score", 0) for r in successes]
        if scores:
            summary[f"{condition}_avg_quality"] = sum(scores) / len(scores)

        # Criteria met rate
        meets = [r for r in successes if r.get("meets_criteria", False)]
        summary[f"{condition}_criteria_met_rate"] = (
            len(meets) / len(cond_results) if cond_results else 0
        )

        # Contracted-specific metrics
        if condition == "contracted":
            budget_compliant = [r for r in cond_results if r.get("budget_compliant", False)]
            summary["contracted_budget_compliance"] = len(budget_compliant) / len(cond_results)

            violations = sum(r.get("conservation_violations", 0) for r in cond_results)
            summary["contracted_conservation_violations"] = violations

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print formatted summary."""
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTrials: {summary.get('n_trials', 0)}")

    if "uncontracted_success_rate" in summary:
        print("\n  UNCONTRACTED:")
        print(f"    Success Rate: {summary['uncontracted_success_rate']:.1%}")
        print(f"    Avg Tokens: {summary.get('uncontracted_avg_tokens', 0):,.0f}")
        print(f"    Avg Quality: {summary.get('uncontracted_avg_quality', 0):.2f}")
        print(f"    Criteria Met: {summary.get('uncontracted_criteria_met_rate', 0):.1%}")

    if "contracted_success_rate" in summary:
        print("\n  CONTRACTED:")
        print(f"    Success Rate: {summary['contracted_success_rate']:.1%}")
        print(f"    Avg Tokens: {summary.get('contracted_avg_tokens', 0):,.0f}")
        print(f"    Avg Quality: {summary.get('contracted_avg_quality', 0):.2f}")
        print(f"    Criteria Met: {summary.get('contracted_criteria_met_rate', 0):.1%}")
        print(f"    Budget Compliance: {summary.get('contracted_budget_compliance', 0):.1%}")
        print(
            f"    Conservation Violations: {summary.get('contracted_conservation_violations', 0)}"
        )

    print(f"\n{'=' * 70}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COINE 2026 Evaluation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 3 topics",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Run single topic by ID (e.g., tech_01)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "contracted", "uncontracted"],
        default="both",
        help="Experiment mode (default: both)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=25,
        help="Number of topics (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    n_topics = 3 if args.quick else args.n

    run_experiment(
        n_topics=n_topics,
        mode=args.mode,
        topic_id=args.topic,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
