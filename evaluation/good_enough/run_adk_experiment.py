#!/usr/bin/env python3
"""Run the Good Enough experiment with Google ADK agents.

This is the TRUE AGENTIC version of the experiment where:
- Agents have tools and decide when to call them
- The agent internally decides when to submit the email
- UNCONSTRAINED: No stopping criteria, keeps improving
- CONTRACTED: Has Q_min threshold, stops when "good enough"

Usage:
    python -m evaluation.good_enough.run_adk_experiment --n-scenarios 10
    python -m evaluation.good_enough.run_adk_experiment --quick  # 3 scenarios
"""

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .adk_agents import AdkAgentResult, ContractedAdkAgent, UnconstrainedAdkAgent
from .scenarios import load_scenarios


@dataclass
class ExperimentSummary:
    """Summary statistics for the experiment."""

    n_scenarios: int = 0

    # UNCONSTRAINED stats
    unconstrained_avg_iterations: float = 0.0
    unconstrained_avg_tokens: float = 0.0
    unconstrained_avg_quality: float = 0.0
    unconstrained_early_stop_rate: float = 0.0

    # CONTRACTED stats
    contracted_avg_iterations: float = 0.0
    contracted_avg_tokens: float = 0.0
    contracted_avg_quality: float = 0.0
    contracted_early_stop_rate: float = 0.0

    # Differences
    iteration_reduction: float = 0.0
    token_reduction: float = 0.0
    quality_difference: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_scenarios": self.n_scenarios,
            "unconstrained": {
                "avg_iterations": self.unconstrained_avg_iterations,
                "avg_tokens": self.unconstrained_avg_tokens,
                "avg_quality": self.unconstrained_avg_quality,
                "early_stop_rate": self.unconstrained_early_stop_rate,
            },
            "contracted": {
                "avg_iterations": self.contracted_avg_iterations,
                "avg_tokens": self.contracted_avg_tokens,
                "avg_quality": self.contracted_avg_quality,
                "early_stop_rate": self.contracted_early_stop_rate,
            },
            "differences": {
                "iteration_reduction_pct": self.iteration_reduction,
                "token_reduction_pct": self.token_reduction,
                "quality_difference": self.quality_difference,
            },
        }


@dataclass
class ExperimentResults:
    """Full experiment results."""

    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    summary: ExperimentSummary = field(default_factory=ExperimentSummary)
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment": "good_enough_adk",
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": self.summary.to_dict(),
            "trials": self.trials,
        }


def run_experiment(
    n_scenarios: int = 10,
    max_llm_calls: int = 30,
    quality_threshold: float = 0.80,
    num_judges: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResults:
    """Run the Good Enough experiment with ADK agents.

    Args:
        n_scenarios: Number of email scenarios to test
        max_llm_calls: Maximum LLM calls per agent (safety limit)
        quality_threshold: Q_min threshold for CONTRACTED agent
        num_judges: Number of judges for quality evaluation
        seed: Random seed for scenario selection
        verbose: Print progress

    Returns:
        ExperimentResults with all trial data
    """
    print("=" * 70)
    print("GOOD ENOUGH EXPERIMENT (Google ADK - True Agentic)")
    print("=" * 70)
    print(f"\nConfig: n={n_scenarios}, max_calls={max_llm_calls}, Q_min={quality_threshold}")
    print(f"Judges: {num_judges}")

    # Load scenarios
    scenarios = load_scenarios(limit=n_scenarios, random_seed=seed)
    print(f"\nLoaded {len(scenarios)} email scenarios")

    # Initialize agents
    print("\nInitializing ADK agents...")
    unconstrained_agent = UnconstrainedAdkAgent(
        max_llm_calls=max_llm_calls,
        num_judges=num_judges,
    )
    contracted_agent = ContractedAdkAgent(
        max_llm_calls=max_llm_calls,
        quality_threshold=quality_threshold,
        num_judges=num_judges,
    )

    results = ExperimentResults(
        timestamp=datetime.now().isoformat(),
        config={
            "n_scenarios": n_scenarios,
            "max_llm_calls": max_llm_calls,
            "quality_threshold": quality_threshold,
            "num_judges": num_judges,
            "seed": seed,
            "framework": "google_adk",
            "agent_type": "true_agentic",
        },
    )

    unconstrained_results: list[AdkAgentResult] = []
    contracted_results: list[AdkAgentResult] = []

    for i, scenario in enumerate(scenarios):
        print(f"\n[Scenario {i + 1}/{len(scenarios)}] {scenario.id} ({scenario.category})")

        # Run UNCONSTRAINED
        print("\n  UNCONSTRAINED (no Q_min):")
        uc_result = unconstrained_agent.run(scenario, verbose=verbose)
        unconstrained_results.append(uc_result)
        print(
            f"  → Iterations: {uc_result.iterations}, Tokens: {uc_result.total_tokens:,}, Q: {uc_result.final_quality:.2f}"
        )
        print(f"  → Stop reason: {uc_result.stop_reason}")

        # Run CONTRACTED
        print(f"\n  CONTRACTED (Q_min={quality_threshold}):")
        ct_result = contracted_agent.run(scenario, verbose=verbose)
        contracted_results.append(ct_result)
        print(
            f"  → Iterations: {ct_result.iterations}, Tokens: {ct_result.total_tokens:,}, Q: {ct_result.final_quality:.2f}"
        )
        print(f"  → Stop reason: {ct_result.stop_reason}")
        if ct_result.stopped_early:
            print("  ✓ Agent recognized 'good enough' and stopped!")

        # Store trial
        results.trials.append(
            {
                "scenario_id": scenario.id,
                "category": scenario.category,
                "unconstrained": uc_result.to_dict(),
                "contracted": ct_result.to_dict(),
            }
        )

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = ExperimentSummary(n_scenarios=len(scenarios))

    # UNCONSTRAINED stats
    uc_iterations = [r.iterations for r in unconstrained_results]
    uc_tokens = [r.total_tokens for r in unconstrained_results]
    uc_qualities = [r.final_quality for r in unconstrained_results]

    summary.unconstrained_avg_iterations = statistics.mean(uc_iterations)
    summary.unconstrained_avg_tokens = statistics.mean(uc_tokens)
    summary.unconstrained_avg_quality = statistics.mean(uc_qualities)
    summary.unconstrained_early_stop_rate = sum(
        1 for r in unconstrained_results if r.stopped_early
    ) / len(unconstrained_results)

    # CONTRACTED stats
    ct_iterations = [r.iterations for r in contracted_results]
    ct_tokens = [r.total_tokens for r in contracted_results]
    ct_qualities = [r.final_quality for r in contracted_results]

    summary.contracted_avg_iterations = statistics.mean(ct_iterations)
    summary.contracted_avg_tokens = statistics.mean(ct_tokens)
    summary.contracted_avg_quality = statistics.mean(ct_qualities)
    summary.contracted_early_stop_rate = sum(
        1 for r in contracted_results if r.stopped_early
    ) / len(contracted_results)

    # Differences
    if summary.unconstrained_avg_iterations > 0:
        summary.iteration_reduction = (
            (summary.unconstrained_avg_iterations - summary.contracted_avg_iterations)
            / summary.unconstrained_avg_iterations
            * 100
        )
    if summary.unconstrained_avg_tokens > 0:
        summary.token_reduction = (
            (summary.unconstrained_avg_tokens - summary.contracted_avg_tokens)
            / summary.unconstrained_avg_tokens
            * 100
        )
    summary.quality_difference = summary.contracted_avg_quality - summary.unconstrained_avg_quality

    results.summary = summary

    # Print summary
    print("\n                     UNCONSTRAINED    CONTRACTED      DIFFERENCE")
    print("  " + "-" * 62)
    print(
        f"  Avg Iterations     {summary.unconstrained_avg_iterations:>10.1f}    {summary.contracted_avg_iterations:>10.1f}    {summary.iteration_reduction:>+8.1f}%"
    )
    print(
        f"  Avg Tokens         {summary.unconstrained_avg_tokens:>10,.0f}    {summary.contracted_avg_tokens:>10,.0f}    {summary.token_reduction:>+8.1f}%"
    )
    print(
        f"  Avg Quality        {summary.unconstrained_avg_quality:>10.2f}    {summary.contracted_avg_quality:>10.2f}    {summary.quality_difference:>+8.2f}"
    )
    print(
        f"  Early Stop Rate    {summary.unconstrained_early_stop_rate * 100:>9.0f}%    {summary.contracted_early_stop_rate * 100:>9.0f}%"
    )

    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    if summary.iteration_reduction > 0 and abs(summary.quality_difference) < 0.15:
        print("\n  ✓ CONTRACTED agent recognized 'good enough' and stopped early")
        print(f"    - {summary.iteration_reduction:.0f}% fewer iterations")
        print(f"    - {summary.token_reduction:.0f}% fewer tokens")
        print(f"    - Quality difference: {summary.quality_difference:+.2f} (comparable)")
        print("\n  → TRUE AGENTIC BEHAVIOR: The agent itself decided when to stop,")
        print("    based on its understanding of the quality contract.")
    else:
        print("\n  Results require further analysis.")

    # Save results
    output_dir = Path("results/good_enough")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"adk_experiment_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Good Enough experiment with ADK")
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=10,
        help="Number of email scenarios (default: 10)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=30,
        help="Maximum LLM calls per agent (default: 30)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.80,
        help="Q_min threshold for CONTRACTED (default: 0.80)",
    )
    parser.add_argument(
        "--num-judges",
        type=int,
        default=3,
        help="Number of judges for quality evaluation (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 3 scenarios",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )

    args = parser.parse_args()

    if args.quick:
        args.n_scenarios = 3
        args.max_llm_calls = 20

    run_experiment(
        n_scenarios=args.n_scenarios,
        max_llm_calls=args.max_llm_calls,
        quality_threshold=args.quality_threshold,
        num_judges=args.num_judges,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
