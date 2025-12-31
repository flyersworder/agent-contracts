#!/usr/bin/env python3
"""Soft Cutoff Experiment: Demonstrating Time-Quality Tradeoff.

This experiment shows how soft cutoff enables graceful degradation:
- Shorter time budget → Lower quality (truncated) but still usable output
- Longer time budget → Higher quality (complete) output

The key insight: For summarization, partial output has value.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Any

from agent_contracts import Contract, ContractMode, ResourceConstraints, TemporalConstraints
from agent_contracts.core.contract import ExecutionConfig
from agent_contracts.core.executor import ContractExecutor
from evaluation.strategy_modes.metrics import compute_rouge
from evaluation.strategy_modes.tasks import load_tasks


def run_trial(article: str, reference: str, timeout: float, soft_cutoff: bool) -> dict[str, Any]:
    """Run a single summarization trial."""
    contract = Contract(
        id=f"soft-cutoff-{timeout}s",
        name="Soft Cutoff Summarization",
        description=f"Summarization with {timeout}s timeout",
        mode=ContractMode.URGENT,
        resources=ResourceConstraints(
            tokens=8000,  # Generous token budget
            cost_usd=0.10,
        ),
        temporal=TemporalConstraints(
            max_duration=timedelta(minutes=5),
        ),
        execution=ExecutionConfig(
            model="gemini/gemini-2.5-flash",
            temperature=0.3,
            timeout_seconds=timeout,
        ),
    )

    prompt = f"""Summarize the following news article in 2-3 paragraphs.
Be comprehensive and include all key points.

Article:
{article}

Summary:"""

    executor = ContractExecutor(
        contract=contract,
        strict_mode=False,
        soft_cutoff=soft_cutoff,
    )

    result = executor.run(query=prompt)

    # Compute ROUGE if we have output
    rouge_l = 0.0
    if result.output:
        rouge_metrics = compute_rouge(result.output, reference)
        rouge_l = rouge_metrics.rouge_l_f1

    return {
        "success": result.success,
        "truncated": result.truncated,
        "output": result.output or "",
        "output_length": len(result.output or ""),
        "word_count": len((result.output or "").split()),
        "rouge_l": rouge_l,
        "tokens_used": result.tokens_used,
        "timeout_seconds": timeout,
        "contract_state": result.contract_state.value,
        "error": result.error or "",
    }


def run_experiment(n_articles: int = 20, seed: int = 42) -> dict[float, dict[str, Any]]:
    """Run the soft cutoff experiment.

    Tests multiple timeout levels to show the time-quality gradient:
    - 1.5s: Very aggressive, likely to truncate
    - 3.0s: Moderate, some truncation expected
    - 5.0s: Comfortable, mostly complete
    - 10.0s: Generous, should complete fully
    """
    print("=" * 70)
    print("SOFT CUTOFF EXPERIMENT: Time-Quality Tradeoff")
    print("=" * 70)

    # Load tasks
    print(f"\nLoading {n_articles} CNN/DailyMail articles (seed={seed})...")
    tasks = load_tasks(limit=n_articles, random_seed=seed)
    print(f"Loaded {len(tasks)} articles")

    # Timeout levels to test
    timeout_levels = [1.5, 3.0, 5.0, 10.0]

    results: dict[float, list[dict[str, Any]]] = {timeout: [] for timeout in timeout_levels}

    for i, task in enumerate(tasks):
        print(f"\n[Article {i + 1}/{len(tasks)}] {task.task_id[:30]}...")

        for timeout in timeout_levels:
            print(f"  Timeout {timeout}s: ", end="", flush=True)

            trial = run_trial(
                article=task.article,
                reference=task.reference_summary,
                timeout=timeout,
                soft_cutoff=True,
            )
            trial["task_id"] = task.task_id
            results[timeout].append(trial)

            status = (
                "✓ Complete"
                if not trial["truncated"]
                else f"⚡ Truncated ({trial['word_count']} words)"
            )
            if not trial["success"] and not trial["truncated"]:
                status = f"✗ Failed: {trial['error'][:30]}"
            print(f"{status}, ROUGE-L: {trial['rouge_l']:.3f}")

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for timeout in timeout_levels:
        trials = results[timeout]
        n_total = len(trials)
        n_truncated = sum(1 for t in trials if t["truncated"])
        n_complete = sum(1 for t in trials if t["success"] and not t["truncated"])
        n_failed = sum(1 for t in trials if not t["success"] and not t["truncated"])

        # Quality metrics for complete vs truncated
        complete_trials = [t for t in trials if t["success"] and not t["truncated"]]
        truncated_trials = [t for t in trials if t["truncated"]]

        avg_rouge_all = sum(t["rouge_l"] for t in trials) / n_total if n_total else 0
        avg_rouge_complete = (
            sum(t["rouge_l"] for t in complete_trials) / len(complete_trials)
            if complete_trials
            else 0
        )
        avg_rouge_truncated = (
            sum(t["rouge_l"] for t in truncated_trials) / len(truncated_trials)
            if truncated_trials
            else 0
        )

        avg_words_complete = (
            sum(t["word_count"] for t in complete_trials) / len(complete_trials)
            if complete_trials
            else 0
        )
        avg_words_truncated = (
            sum(t["word_count"] for t in truncated_trials) / len(truncated_trials)
            if truncated_trials
            else 0
        )

        summary[timeout] = {
            "n_total": n_total,
            "n_complete": n_complete,
            "n_truncated": n_truncated,
            "n_failed": n_failed,
            "truncation_rate": n_truncated / n_total if n_total else 0,
            "avg_rouge_all": avg_rouge_all,
            "avg_rouge_complete": avg_rouge_complete,
            "avg_rouge_truncated": avg_rouge_truncated,
            "avg_words_complete": avg_words_complete,
            "avg_words_truncated": avg_words_truncated,
        }

        print(f"\nTimeout: {timeout}s")
        print(f"  Complete: {n_complete}/{n_total} ({n_complete * 100 / n_total:.0f}%)")
        print(f"  Truncated: {n_truncated}/{n_total} ({n_truncated * 100 / n_total:.0f}%)")
        print(f"  Failed: {n_failed}/{n_total}")
        print("  ---")
        print(f"  ROUGE-L (all): {avg_rouge_all:.3f}")
        if complete_trials:
            print(
                f"  ROUGE-L (complete): {avg_rouge_complete:.3f} ({avg_words_complete:.0f} words avg)"
            )
        if truncated_trials:
            print(
                f"  ROUGE-L (truncated): {avg_rouge_truncated:.3f} ({avg_words_truncated:.0f} words avg)"
            )

    # Show the tradeoff
    print("\n" + "=" * 70)
    print("TIME-QUALITY TRADEOFF")
    print("=" * 70)
    print("\n  Timeout  | Truncation | ROUGE-L | Interpretation")
    print("  " + "-" * 55)
    for timeout in timeout_levels:
        s = summary[timeout]
        interp = (
            "Too aggressive"
            if s["truncation_rate"] > 0.5
            else "Good tradeoff"
            if s["truncation_rate"] > 0.1
            else "Comfortable"
        )
        print(
            f"  {timeout:5.1f}s   |   {s['truncation_rate'] * 100:5.1f}%   |  {s['avg_rouge_all']:.3f}  | {interp}"
        )

    # Save results
    output_dir = Path("results/strategy_modes")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"soft_cutoff_experiment_{timestamp}.json"

    output_data = {
        "experiment": {
            "type": "soft_cutoff_tradeoff",
            "timestamp": datetime.now().isoformat(),
            "n_articles": n_articles,
            "seed": seed,
            "timeout_levels": timeout_levels,
            "model": "gemini/gemini-2.5-flash",
            "soft_cutoff": True,
        },
        "summary": summary,
        "trials": {str(t): results[t] for t in timeout_levels},
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run soft cutoff experiment")
    parser.add_argument("--n-articles", type=int, default=20, help="Number of articles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(n_articles=args.n_articles, seed=args.seed)
