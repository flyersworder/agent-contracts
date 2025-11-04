"""Test reasoning_effort impact on bimodal behavior."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_reasoning_effort(
    sample_file: Path,
    reasoning_effort: str,
    n_runs: int = 10,
) -> dict[str, Any]:
    """Test evaluator with different reasoning effort levels.

    Args:
        sample_file: Path to sample JSON file
        reasoning_effort: "low", "medium", or "high"
        n_runs: Number of test runs

    Returns:
        Results dictionary
    """
    # Load sample
    with open(sample_file) as f:
        sample = json.load(f)

    question = sample["question"]
    answer = sample["answer"]
    original_score = sample["quality_score"]

    print(f"\n{'=' * 80}")
    print(f"REASONING EFFORT TEST: {reasoning_effort.upper()}")
    print(f"{'=' * 80}")
    print(f"Sample: {sample_file.name}")
    print(f"Original score: {original_score:.1f}")
    print(f"Question: {question[:60]}...")
    print(f"Runs: {n_runs}")
    print(f"{'=' * 80}\n")

    # Create evaluator with specified reasoning effort
    # Note: We need to modify QualityEvaluator to accept reasoning_effort
    # For now, we'll create a custom evaluation function

    results = {
        "sample_file": sample_file.name,
        "reasoning_effort": reasoning_effort,
        "n_runs": n_runs,
        "original_score": original_score,
        "runs": [],
    }

    start_time = time.time()

    for run_id in range(1, n_runs + 1):
        print(f"Run {run_id}/{n_runs}...", end=" ", flush=True)

        # Manually run evaluation with custom reasoning_effort
        from litellm import completion

        prompt = f"""You are an expert evaluator assessing the quality of research answers. Evaluate the following answer on three dimensions:

1. **Accuracy (0-10)**: Are the facts, explanations, and technical details correct? Are there any significant errors or misconceptions?

2. **Completeness (0-10)**: Does the answer address all aspects of the question? Are key points covered? Is important context included?

3. **Coherence (0-10)**: Is the answer well-structured, logically organized, and easy to understand? Does it flow naturally?

Research Question:
{question}

Answer to Evaluate:
{answer}

Provide your evaluation in the following format:

Accuracy: [score 0-10]
Completeness: [score 0-10]
Coherence: [score 0-10]

Explanation:
[2-3 sentences explaining the scores, highlighting strengths and weaknesses]"""

        # Run 3 evaluations and take median (same as QualityEvaluator)
        scores_list = []
        for _ in range(3):
            response = completion(
                model="gemini/gemini-2.5-flash-preview-09-2025",
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=reasoning_effort,
                temperature=0,
            )

            content = response["choices"][0]["message"]["content"]

            # Parse scores
            accuracy, completeness, coherence, _explanation = parse_evaluation(content)
            scores_list.append(
                {
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "coherence": coherence,
                }
            )

        # Take median
        import statistics

        accuracy = statistics.median([s["accuracy"] for s in scores_list])
        completeness = statistics.median([s["completeness"] for s in scores_list])
        coherence = statistics.median([s["coherence"] for s in scores_list])

        # Apply hybrid scoring (60% LLM + 40% rules) - simplified
        # For this test, we'll use pure LLM scores to isolate reasoning_effort effect
        total = (accuracy + completeness + coherence) / 30 * 100

        results["runs"].append(
            {
                "run_id": run_id,
                "accuracy": accuracy,
                "completeness": completeness,
                "coherence": coherence,
                "total": total,
            }
        )

        print(f"→ {total:.1f}")
        time.sleep(0.5)  # Rate limit

    elapsed = time.time() - start_time

    # Calculate statistics
    totals = [r["total"] for r in results["runs"]]
    mean_total = sum(totals) / len(totals)
    std_total = (sum((x - mean_total) ** 2 for x in totals) / len(totals)) ** 0.5
    cv_total = (std_total / mean_total * 100) if mean_total > 0 else 0

    from collections import Counter

    unique_scores = Counter(totals)

    results["statistics"] = {
        "mean": mean_total,
        "std_dev": std_total,
        "cv_percent": cv_total,
        "min": min(totals),
        "max": max(totals),
        "range": max(totals) - min(totals),
        "unique_scores": len(unique_scores),
        "elapsed_time": elapsed,
    }

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean: {mean_total:.1f}")
    print(f"SD: {std_total:.2f}")
    print(f"CV: {cv_total:.1f}%")
    print(f"Range: {max(totals) - min(totals):.1f} ({min(totals):.1f}-{max(totals):.1f})")
    print(f"Unique scores: {len(unique_scores)}")
    print("\nScore distribution:")
    for score, count in sorted(unique_scores.items()):
        pct = count / len(totals) * 100
        bar = "█" * int(pct / 5)
        print(f"  {score:.1f}: {count:2d} runs ({pct:4.1f}%) {bar}")
    print(f"{'=' * 80}\n")

    return results


def parse_evaluation(content: str) -> tuple[float, float, float, str]:
    """Parse evaluation from LLM response."""
    lines = content.strip().split("\n")

    accuracy = 0.0
    completeness = 0.0
    coherence = 0.0
    explanation = ""
    explanation_started = False

    for line in lines:
        line = line.strip()

        if line.lower().startswith("accuracy:"):
            try:
                accuracy = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                accuracy = 7.0

        elif line.lower().startswith("completeness:"):
            try:
                completeness = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                completeness = 7.0

        elif line.lower().startswith("coherence:"):
            try:
                coherence = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                coherence = 7.0

        elif line.lower().startswith("explanation:"):
            explanation_started = True
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                explanation = parts[1].strip()

        elif explanation_started and line:
            explanation += " " + line

    # Clamp scores
    accuracy = max(0.0, min(10.0, accuracy))
    completeness = max(0.0, min(10.0, completeness))
    coherence = max(0.0, min(10.0, coherence))

    return accuracy, completeness, coherence, explanation.strip()


def main() -> None:
    """Run reasoning effort test."""
    sample_file = Path("benchmarks/quality_validation/extended_samples/sample_2.json")

    print("\nTesting Hypothesis 2: Reasoning Token Stochasticity")
    print("Comparing reasoning_effort='medium' (original) vs 'low'")
    print()

    # Test with low reasoning effort
    results_low = test_reasoning_effort(
        sample_file=sample_file,
        reasoning_effort="low",
        n_runs=10,
    )

    # Save results
    output_dir = Path("benchmarks/quality_validation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reasoning_effort_low_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results_low, f, indent=2)

    print(f"✓ Results saved to {output_file}")

    # Compare with original (reasoning_effort="medium", CV=10.3%)
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH ORIGINAL (reasoning_effort='medium')")
    print(f"{'=' * 80}")
    print("Original (medium): CV = 10.3%, Bimodal (78.0 vs 96.0)")
    print(f"New (low):         CV = {results_low['statistics']['cv_percent']:.1f}%")
    print()

    cv_low = results_low["statistics"]["cv_percent"]
    if cv_low < 5:
        print("✅ HYPOTHESIS CONFIRMED: Low reasoning effort reduces variance significantly!")
        print("   → Reasoning tokens are the primary cause of bimodal behavior")
    elif cv_low < 8:
        print("⚠️  PARTIAL CONFIRMATION: Low reasoning effort reduces variance moderately")
        print("   → Reasoning tokens contribute but not the only factor")
    else:
        print("❌ HYPOTHESIS REJECTED: Low reasoning effort does not reduce variance")
        print("   → Bimodal behavior is not primarily from reasoning tokens")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
