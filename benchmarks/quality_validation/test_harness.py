"""Test harness for quality evaluator validation study."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.research_agent.evaluator import QualityEvaluator  # noqa: E402


def load_samples(samples_dir: Path) -> list[dict[str, Any]]:
    """Load all sample answers."""
    samples = []
    for sample_file in sorted(samples_dir.glob("sample_*.json")):
        with open(sample_file) as f:
            sample = json.load(f)
            sample["sample_id"] = sample_file.stem  # e.g., "sample_1_fair"
            samples.append(sample)
    return samples


def run_test_retest(
    samples: list[dict[str, Any]],
    n_runs: int,
    evaluator: QualityEvaluator,
    output_file: Path,
) -> dict[str, Any]:
    """Run test-retest reliability study.

    Args:
        samples: List of sample answers to evaluate
        n_runs: Number of evaluation runs per sample
        evaluator: Quality evaluator instance
        output_file: Path to save results

    Returns:
        Dictionary with all results and metadata
    """
    results: dict[str, Any] = {
        "study_phase": "test_retest",
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(samples),
        "n_runs_per_sample": n_runs,
        "total_evaluations": len(samples) * n_runs,
        "evaluator_config": {
            "model": evaluator.judge_model,
            "use_multiple_judges": evaluator.use_multiple_judges,
            "use_hybrid_scoring": evaluator.use_hybrid_scoring,
        },
        "samples": [],
    }

    total_cost = 0.0
    total_evaluations = 0

    print(f"\n{'=' * 80}")
    print("QUALITY VALIDATION STUDY: TEST-RETEST RELIABILITY")
    print(f"{'=' * 80}")
    print(f"Samples: {len(samples)}")
    print(f"Runs per sample: {n_runs}")
    print(f"Total evaluations: {len(samples) * n_runs}")
    print(f"Evaluator: {evaluator.judge_model}")
    print(f"{'=' * 80}\n")

    for i, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        question = sample["question"]
        answer = sample["answer"]
        original_score = sample["quality_score"]

        print(f"Sample {i}/{len(samples)}: {sample_id}")
        print(f"  Original score: {original_score:.1f}")
        print(f"  Question: {question[:60]}...")
        print(f"  Running {n_runs} evaluations...", end=" ", flush=True)

        runs_data: list[dict[str, Any]] = []
        start_time = time.time()

        for run_id in range(1, n_runs + 1):
            # Evaluate the answer
            quality_score = evaluator.evaluate(question, answer)

            # Estimate cost (rough approximation)
            # Input: ~650 tokens (question + answer + rubric)
            # Output: ~180 tokens
            # Reasoning (medium): ~1,200 tokens
            # If using multiple judges (3x): multiply by 3
            eval_cost = 0.001  # Base cost per evaluation
            if evaluator.use_multiple_judges:
                eval_cost *= 3  # 3 judge runs

            total_cost += eval_cost
            total_evaluations += 1

            runs_data.append(
                {
                    "run_id": run_id,
                    "accuracy": quality_score.accuracy,
                    "completeness": quality_score.completeness,
                    "coherence": quality_score.coherence,
                    "total": quality_score.total,
                    "explanation": quality_score.explanation[:200] + "..."  # Truncate
                    if len(quality_score.explanation) > 200
                    else quality_score.explanation,
                    "estimated_cost": eval_cost,
                }
            )

            # Add small delay to avoid rate limits
            time.sleep(0.5)

        elapsed = time.time() - start_time

        # Calculate statistics
        total_scores = [r["total"] for r in runs_data]
        accuracy_scores = [r["accuracy"] for r in runs_data]
        completeness_scores = [r["completeness"] for r in runs_data]
        coherence_scores = [r["coherence"] for r in runs_data]

        mean_total = sum(total_scores) / len(total_scores)
        std_total = (sum((x - mean_total) ** 2 for x in total_scores) / len(total_scores)) ** 0.5
        cv_total = (std_total / mean_total * 100) if mean_total > 0 else 0
        min_total = min(total_scores)
        max_total = max(total_scores)
        range_total = max_total - min_total

        sample_result = {
            "sample_id": sample_id,
            "question_id": sample["question_id"],
            "question": question,
            "answer_word_count": sample["word_count"],
            "original_score": original_score,
            "runs": runs_data,
            "statistics": {
                "n_runs": n_runs,
                "total": {
                    "mean": mean_total,
                    "std_dev": std_total,
                    "cv_percent": cv_total,
                    "min": min_total,
                    "max": max_total,
                    "range": range_total,
                },
                "accuracy": {
                    "mean": sum(accuracy_scores) / len(accuracy_scores),
                    "std_dev": (
                        sum(
                            (x - sum(accuracy_scores) / len(accuracy_scores)) ** 2
                            for x in accuracy_scores
                        )
                        / len(accuracy_scores)
                    )
                    ** 0.5,
                },
                "completeness": {
                    "mean": sum(completeness_scores) / len(completeness_scores),
                    "std_dev": (
                        sum(
                            (x - sum(completeness_scores) / len(completeness_scores)) ** 2
                            for x in completeness_scores
                        )
                        / len(completeness_scores)
                    )
                    ** 0.5,
                },
                "coherence": {
                    "mean": sum(coherence_scores) / len(coherence_scores),
                    "std_dev": (
                        sum(
                            (x - sum(coherence_scores) / len(coherence_scores)) ** 2
                            for x in coherence_scores
                        )
                        / len(coherence_scores)
                    )
                    ** 0.5,
                },
                "elapsed_time": elapsed,
            },
        }

        results["samples"].append(sample_result)

        print(f"✓ Done in {elapsed:.1f}s")
        print(
            f"    Mean: {mean_total:.1f}, SD: {std_total:.2f}, CV: {cv_total:.1f}%, "
            f"Range: {range_total:.1f} ({min_total:.1f}-{max_total:.1f})"
        )

    # Calculate overall statistics
    all_cvs = [s["statistics"]["total"]["cv_percent"] for s in results["samples"]]
    all_ranges = [s["statistics"]["total"]["range"] for s in results["samples"]]

    results["overall_statistics"] = {
        "average_cv_percent": sum(all_cvs) / len(all_cvs),
        "average_range": sum(all_ranges) / len(all_ranges),
        "total_cost_estimate": total_cost,
        "total_evaluations": total_evaluations,
    }

    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")
    print(f"{'=' * 80}")
    print(f"Average CV: {results['overall_statistics']['average_cv_percent']:.1f}%")
    print(f"Average Range: {results['overall_statistics']['average_range']:.1f} points")
    print(f"Total Cost: ${total_cost:.3f}")
    print(f"Total Evaluations: {total_evaluations}")
    print(f"{'=' * 80}\n")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_file}")

    return results


def main() -> None:
    """Run validation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Run quality evaluator validation study")
    parser.add_argument(
        "--phase",
        choices=["pilot", "core", "extended"],
        default="pilot",
        help="Validation phase (pilot=3 runs, core=5 runs, extended=10 runs)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        help="Override number of runs (overrides phase default)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("benchmarks/quality_validation/samples"),
        help="Directory containing sample answers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/quality_validation/results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine number of runs based on phase or override
    n_runs = args.n_runs or {"pilot": 3, "core": 5, "extended": 10}[args.phase]

    # Load samples
    samples = load_samples(args.samples_dir)
    if not samples:
        print(f"❌ No samples found in {args.samples_dir}")
        return

    print(f"Loaded {len(samples)} samples from {args.samples_dir}")

    # Initialize evaluator (same configuration as research benchmark)
    evaluator = QualityEvaluator(
        judge_model="gemini/gemini-2.5-flash-preview-09-2025",
        use_multiple_judges=True,  # 3 judges → median
        use_hybrid_scoring=True,  # 60% LLM + 40% rules
    )

    # Run study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"phase1_{args.phase}_results_{timestamp}.json"

    results = run_test_retest(
        samples=samples,
        n_runs=n_runs,
        evaluator=evaluator,
        output_file=output_file,
    )

    # Print decision guidance
    avg_cv = results["overall_statistics"]["average_cv_percent"]
    print("\n" + "=" * 80)
    print("DECISION GUIDANCE")
    print("=" * 80)

    if avg_cv < 5:
        print("✅ PASS: Evaluator is highly reliable (CV < 5%)")
        print("   → Recommendation: Stop here, current system is excellent")
    elif avg_cv < 8:
        print("✅ ACCEPTABLE: Evaluator has good reliability (CV < 8%)")
        print("   → Recommendation: Add confidence intervals, document as limitation")
        if args.phase == "pilot":
            print("   → Consider running core validation (N=5) for confirmation")
    elif avg_cv < 10:
        print("⚠️  MARGINAL: Evaluator has acceptable reliability (CV < 10%)")
        print("   → Recommendation: Matches SOTA, but room for improvement")
        if args.phase in ["pilot", "core"]:
            print("   → Consider running extended validation (N=10) for deeper analysis")
    else:
        print("❌ CONCERNING: Evaluator shows high variance (CV >= 10%)")
        print("   → Recommendation: Proceed to Phase 2 (inter-rater reliability)")
        print("   → Investigate: Model variance vs implementation issues")

    print("=" * 80)


if __name__ == "__main__":
    main()
