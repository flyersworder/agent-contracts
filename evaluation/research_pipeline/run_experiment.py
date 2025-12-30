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

    # With LLM-as-judge evaluation (IndeterminacyAwareEvaluator)
    python run_experiment.py --quick --evaluate

    # Custom evaluation settings (default judge: gemini-2.5-flash-lite)
    python run_experiment.py --evaluate --judge-model gemini/gemini-2.0-flash --num-judges 5
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load environment
load_dotenv()

if TYPE_CHECKING:
    from evaluation.research_pipeline.evaluator import (
        ReportQualityScore,
        ResearchReportEvaluator,
    )
    from evaluation.research_pipeline.topics import ResearchTopic


def _evaluate_report(
    evaluator: "ResearchReportEvaluator",
    topic: "ResearchTopic",
    report: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run LLM evaluation on a report and return metrics.

    Args:
        evaluator: ResearchReportEvaluator instance
        topic: Research topic
        report: Generated report text
        verbose: Print evaluation progress

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        if verbose:
            print("    📊 Running LLM evaluation...")

        quality: ReportQualityScore = evaluator.evaluate(topic, report)

        # Extract key metrics from indeterminacy score
        ind_score = quality.indeterminacy_score

        # Calculate average indeterminacy across dimensions
        avg_indeterminacy = (
            ind_score.accuracy.indeterminacy
            + ind_score.completeness.indeterminacy
            + ind_score.coherence.indeterminacy
        ) / 3

        return {
            "llm_overall_score": quality.overall_score,
            "llm_accuracy": ind_score.accuracy.point_estimate,
            "llm_accuracy_indeterminacy": ind_score.accuracy.indeterminacy,
            "llm_completeness": ind_score.completeness.point_estimate,
            "llm_completeness_indeterminacy": ind_score.completeness.indeterminacy,
            "llm_coherence": ind_score.coherence.point_estimate,
            "llm_coherence_indeterminacy": ind_score.coherence.indeterminacy,
            "llm_total": ind_score.total,
            "llm_judge_agreement": ind_score.judge_agreement,
            "llm_avg_indeterminacy": avg_indeterminacy,
            "covers_key_aspects": quality.covers_key_aspects,
        }
    except Exception as e:
        if verbose:
            print(f"    ⚠️ LLM evaluation failed: {e}")
        return {"llm_error": str(e)}


def run_experiment(
    n_topics: int = 25,
    mode: str = "both",
    topic_id: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    output_dir: Path | None = None,
    evaluate: bool = False,
    judge_model: str = "gemini/gemini-2.5-flash-lite",
    num_judges: int = 3,
) -> dict[str, Any]:
    """Run the evaluation experiment.

    Args:
        n_topics: Number of topics to use (max 25)
        mode: "both", "contracted", or "uncontracted"
        topic_id: Specific topic ID to run (overrides n_topics)
        seed: Random seed for reproducibility
        verbose: Print progress messages
        output_dir: Directory for results (default: evaluation/results)
        evaluate: Use LLM-as-judge evaluation (IndeterminacyAwareEvaluator)
        judge_model: LLM model for quality evaluation
        num_judges: Number of independent LLM evaluations to aggregate

    Returns:
        Dictionary with experiment results
    """
    from evaluation.research_pipeline.evaluator import ResearchReportEvaluator
    from evaluation.research_pipeline.orchestrator import (
        ContractedPipeline,
        SuccessCriteria,
        UncontractedPipeline,
    )
    from evaluation.research_pipeline.topics import ALL_TOPICS, get_topic

    # Initialize LLM evaluator if requested
    evaluator: ResearchReportEvaluator | None = None
    if evaluate:
        evaluator = ResearchReportEvaluator(
            judge_model=judge_model,
            num_judges=num_judges,
        )
        if verbose:
            print(f"📊 LLM Evaluation enabled: {judge_model} x {num_judges} judges")

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
        print(f"LLM Evaluation: {'✅ Enabled' if evaluate else '❌ Disabled'}")
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
            "llm_evaluation": evaluate,
            "judge_model": judge_model if evaluate else None,
            "num_judges": num_judges if evaluate else None,
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
                    "total_thinking_tokens": unc_result.total_thinking_tokens,
                    "thinking_tokens_by_agent": unc_result.thinking_tokens_by_agent,
                    "total_llm_calls": unc_result.total_llm_calls,
                    "llm_calls_by_agent": unc_result.llm_calls_by_agent,
                    "word_count": unc_result.word_count,
                    "citation_count": unc_result.citation_count,
                    "web_searches": unc_result.web_searches,
                    "grounding_data": unc_result.grounding_data,
                    "tool_usage": unc_result.tool_usage,
                    "execution_time": unc_result.execution_time_seconds,
                    "quality_score": score,
                    "meets_criteria": success,
                    "error": unc_result.error,
                }

                if verbose:
                    print(f"    Tokens: {unc_result.total_tokens:,}")
                    if unc_result.total_thinking_tokens > 0:
                        ratio = unc_result.total_thinking_tokens / max(1, unc_result.total_tokens)
                        print(
                            f"    Thinking Tokens: {unc_result.total_thinking_tokens:,} ({ratio:.1%})"
                        )
                    print(f"    Web Searches: {unc_result.web_searches}")
                    print(f"    Words: {unc_result.word_count:,}")
                    print(f"    Citations: {unc_result.citation_count}")
                    print(f"    Quality: {score:.2f} ({'✅' if success else '❌'})")

                # Run LLM evaluation if enabled and report was generated
                if evaluator and unc_result.success and unc_result.report:
                    llm_metrics = _evaluate_report(evaluator, topic, unc_result.report, verbose)
                    trial_result["uncontracted"].update(llm_metrics)
                    if verbose and "llm_overall_score" in llm_metrics:
                        print(f"    LLM Score: {llm_metrics['llm_overall_score']:.1f}")
                        print(f"    Indeterminacy: {llm_metrics['llm_avg_indeterminacy']:.2f}")

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
                    "total_thinking_tokens": con_result.total_thinking_tokens,
                    "thinking_tokens_by_agent": con_result.thinking_tokens_by_agent,
                    "total_llm_calls": con_result.total_llm_calls,
                    "llm_calls_by_agent": con_result.llm_calls_by_agent,
                    "word_count": con_result.word_count,
                    "citation_count": con_result.citation_count,
                    "web_searches": con_result.web_searches,
                    "grounding_data": con_result.grounding_data,
                    "tool_usage": con_result.tool_usage,
                    "execution_time": con_result.execution_time_seconds,
                    "budget_compliant": con_result.budget_compliant,
                    "conservation_violations": con_result.conservation_violations,
                    "quality_score": score,
                    "meets_criteria": success,
                    "error": con_result.error,
                }

                if verbose:
                    print(f"    Tokens: {con_result.total_tokens:,}")
                    if con_result.total_thinking_tokens > 0:
                        ratio = con_result.total_thinking_tokens / max(1, con_result.total_tokens)
                        print(
                            f"    Thinking Tokens: {con_result.total_thinking_tokens:,} ({ratio:.1%})"
                        )
                    print(f"    Budget: {'✅' if con_result.budget_compliant else '❌'}")
                    print(f"    Web Searches: {con_result.web_searches}")
                    print(f"    Words: {con_result.word_count:,}")
                    print(f"    Citations: {con_result.citation_count}")
                    print(f"    Quality: {score:.2f} ({'✅' if success else '❌'})")

                # Run LLM evaluation if enabled and report was generated
                if evaluator and con_result.success and con_result.report:
                    llm_metrics = _evaluate_report(evaluator, topic, con_result.report, verbose)
                    trial_result["contracted"].update(llm_metrics)
                    if verbose and "llm_overall_score" in llm_metrics:
                        print(f"    LLM Score: {llm_metrics['llm_overall_score']:.1f}")
                        print(f"    Indeterminacy: {llm_metrics['llm_avg_indeterminacy']:.2f}")

            except Exception as e:
                trial_result["contracted"] = {"error": str(e), "success": False}
                if verbose:
                    print(f"    ❌ Error: {e}")

        results["trials"].append(trial_result)

    # Calculate summary statistics
    results["summary"] = calculate_summary(results["trials"], mode)

    # Save results to results/research_pipeline/ (consistent with strategy_modes)
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "research_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"research_pipeline_{timestamp}.json"

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

        # Thinking token statistics (Gemini 2.5+ reasoning tokens)
        thinking_tokens = [r.get("total_thinking_tokens", 0) for r in successes]
        if thinking_tokens and sum(thinking_tokens) > 0:
            summary[f"{condition}_avg_thinking_tokens"] = sum(thinking_tokens) / len(
                thinking_tokens
            )
            summary[f"{condition}_total_thinking_tokens"] = sum(thinking_tokens)
            # Calculate thinking token ratio (thinking / total)
            total_all_tokens = sum(tokens)
            if total_all_tokens > 0:
                summary[f"{condition}_thinking_ratio"] = sum(thinking_tokens) / total_all_tokens

        # Web search statistics (grounding tool tracking)
        web_searches = [r.get("web_searches", 0) for r in successes]
        if web_searches:
            summary[f"{condition}_avg_web_searches"] = sum(web_searches) / len(web_searches)
            summary[f"{condition}_total_web_searches"] = sum(web_searches)

        # LLM call statistics (iteration tracking)
        llm_calls = [r.get("total_llm_calls", 0) for r in successes]
        if llm_calls:
            summary[f"{condition}_avg_llm_calls"] = sum(llm_calls) / len(llm_calls)
            summary[f"{condition}_total_llm_calls"] = sum(llm_calls)

        # Quality scores (rule-based)
        scores = [r.get("quality_score", 0) for r in successes]
        if scores:
            summary[f"{condition}_avg_quality"] = sum(scores) / len(scores)

        # LLM quality scores (if available)
        llm_scores = [r.get("llm_overall_score", 0) for r in successes if "llm_overall_score" in r]
        if llm_scores:
            summary[f"{condition}_avg_llm_quality"] = sum(llm_scores) / len(llm_scores)

            # Individual dimension scores
            for dim in ["accuracy", "completeness", "coherence"]:
                dim_scores = [r.get(f"llm_{dim}", 0) for r in successes if f"llm_{dim}" in r]
                if dim_scores:
                    summary[f"{condition}_avg_llm_{dim}"] = sum(dim_scores) / len(dim_scores)

            # Average indeterminacy
            indeterminacy = [
                r.get("llm_avg_indeterminacy", 0) for r in successes if "llm_avg_indeterminacy" in r
            ]
            if indeterminacy:
                summary[f"{condition}_avg_indeterminacy"] = sum(indeterminacy) / len(indeterminacy)

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

    for condition in ["uncontracted", "contracted"]:
        if f"{condition}_success_rate" not in summary:
            continue

        print(f"\n  {condition.upper()}:")
        print(f"    Success Rate: {summary[f'{condition}_success_rate']:.1%}")
        print(f"    Avg Tokens: {summary.get(f'{condition}_avg_tokens', 0):,.0f}")
        # Show thinking tokens if available (Gemini 2.5+ models)
        if f"{condition}_avg_thinking_tokens" in summary:
            thinking_ratio = summary.get(f"{condition}_thinking_ratio", 0)
            print(
                f"    Avg Thinking Tokens: {summary[f'{condition}_avg_thinking_tokens']:,.0f} ({thinking_ratio:.1%} of total)"
            )
        print(f"    Avg LLM Calls: {summary.get(f'{condition}_avg_llm_calls', 0):.1f}")
        print(f"    Avg Web Searches: {summary.get(f'{condition}_avg_web_searches', 0):.1f}")
        print(f"    Avg Quality (rule-based): {summary.get(f'{condition}_avg_quality', 0):.2f}")
        print(f"    Criteria Met: {summary.get(f'{condition}_criteria_met_rate', 0):.1%}")

        # LLM quality scores (if available)
        if f"{condition}_avg_llm_quality" in summary:
            print("    --- LLM Evaluation ---")
            print(f"    Avg LLM Quality: {summary[f'{condition}_avg_llm_quality']:.1f}")
            print(f"      Accuracy: {summary.get(f'{condition}_avg_llm_accuracy', 0):.1f}")
            print(f"      Completeness: {summary.get(f'{condition}_avg_llm_completeness', 0):.1f}")
            print(f"      Coherence: {summary.get(f'{condition}_avg_llm_coherence', 0):.1f}")
            print(f"    Avg Indeterminacy: {summary.get(f'{condition}_avg_indeterminacy', 0):.2f}")

        # Contracted-specific metrics
        if condition == "contracted":
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable LLM-as-judge evaluation (IndeterminacyAwareEvaluator)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini/gemini-2.5-flash-lite",
        help="LLM model for evaluation (default: gemini/gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--num-judges",
        type=int,
        default=3,
        help="Number of LLM evaluations to aggregate (default: 3)",
    )

    args = parser.parse_args()

    n_topics = 3 if args.quick else args.n

    run_experiment(
        n_topics=n_topics,
        mode=args.mode,
        topic_id=args.topic,
        seed=args.seed,
        verbose=not args.quiet,
        evaluate=args.evaluate,
        judge_model=args.judge_model,
        num_judges=args.num_judges,
    )


if __name__ == "__main__":
    main()
