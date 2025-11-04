"""Select diverse answer samples for quality validation study."""

import json
from pathlib import Path
from typing import Any, cast


def load_benchmark_results(result_file: Path) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(result_file) as f:
        return cast("dict[str, Any]", json.load(f))


def select_diverse_samples(result_files: list[Path], output_dir: Path) -> None:
    """Select 5 diverse samples spanning quality levels.

    Selection criteria:
    - 1 excellent (90+)
    - 2 good (75-85)
    - 1 fair (60-70)
    - 1 poor (<60)

    Also consider:
    - Length distribution (short, medium, long)
    - Question difficulty (easy, medium, hard)
    """
    # Load all results
    all_samples = []

    for result_file in result_files:
        data = load_benchmark_results(result_file)

        # Extract answers with metadata (handle list format)
        results_list = data.get("results", [])
        if isinstance(results_list, list):
            for result in results_list:
                quality_score = result.get("quality_score", {}).get("total", 0)
                answer_text = result.get("final_answer", "")
                word_count = len(answer_text.split())

                all_samples.append(
                    {
                        "question_id": result.get("question_id", ""),
                        "question": result.get("question_text", ""),
                        "agent_type": result.get("agent_type", ""),
                        "answer": answer_text,
                        "quality_score": quality_score,
                        "word_count": word_count,
                        "accuracy": result.get("quality_score", {}).get("accuracy", 0),
                        "completeness": result.get("quality_score", {}).get("completeness", 0),
                        "coherence": result.get("quality_score", {}).get("coherence", 0),
                        "source_file": result_file.name,
                    }
                )

    # Sort by quality score
    all_samples.sort(key=lambda x: x["quality_score"])

    # Select diverse samples based on actual distribution
    # We have: 70-80 range (14 samples) and 90+ range (24 samples)
    selected = []

    # 1. Lowest (74-75) - lower bound
    lowest_samples = [s for s in all_samples if 74 <= s["quality_score"] < 76]
    if lowest_samples:
        selected.append(lowest_samples[0])  # Take first (lowest)

    # 2. Medium-low (77-78) - shorter answer
    medium_low = [s for s in all_samples if 77 <= s["quality_score"] < 79]
    if medium_low:
        medium_low.sort(key=lambda x: x["word_count"])
        selected.append(medium_low[len(medium_low) // 3])  # Shorter

    # 3. Medium-high (78-79) - longer answer
    medium_high = [s for s in all_samples if 78 <= s["quality_score"] < 79]
    if medium_high:
        medium_high.sort(key=lambda x: x["word_count"])
        selected.append(medium_high[2 * len(medium_high) // 3])  # Longer

    # 4. High (92-94) - lower excellent
    high_samples = [s for s in all_samples if 92 <= s["quality_score"] < 94]
    if high_samples:
        selected.append(high_samples[len(high_samples) // 2])

    # 5. Highest (95+) - top excellent
    highest_samples = [s for s in all_samples if s["quality_score"] >= 95]
    if highest_samples:
        selected.append(highest_samples[len(highest_samples) // 2])

    # Save selected samples
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(selected, 1):
        quality_label = (
            "excellent"
            if sample["quality_score"] >= 90
            else "good"
            if sample["quality_score"] >= 75
            else "fair"
            if sample["quality_score"] >= 60
            else "poor"
        )

        output_file = output_dir / f"sample_{i}_{quality_label}.json"

        with open(output_file, "w") as f:
            json.dump(sample, f, indent=2)

        print(
            f"Sample {i}: {quality_label} (score: {sample['quality_score']:.1f}, "
            f"words: {sample['word_count']}, question: {sample['question_id']})"
        )

    print(f"\n✓ Selected {len(selected)} diverse samples")
    print(f"✓ Saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    # Find all benchmark result files
    results_dir = Path("benchmarks/research_agent/results")
    result_files = sorted(results_dir.glob("benchmark_results_*.json"))

    if not result_files:
        print("❌ No benchmark results found")
        return

    print(f"Found {len(result_files)} benchmark result files")

    # Select diverse samples
    output_dir = Path("benchmarks/quality_validation/samples")
    select_diverse_samples(result_files, output_dir)


if __name__ == "__main__":
    main()
