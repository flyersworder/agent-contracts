#!/usr/bin/env python3
"""Run the Good Enough experiment focused on CRISIS scenarios.

This experiment specifically tests crisis communications where:
- Time pressure is explicit (regulatory deadlines, customer panic, etc.)
- CONTRACTED agents have iteration limits as part of their contract
- UNCONSTRAINED agents may over-iterate despite urgency

The key hypothesis: In crisis scenarios, CONTRACTED agents will stop sooner
because their contract specifies both quality threshold AND iteration limits.

Usage:
    python -m evaluation.good_enough.run_crisis_experiment
    python -m evaluation.good_enough.run_crisis_experiment --verbose
"""

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .adk_agents import AdkAgentResult, ContractedAdkAgent, UnconstrainedAdkAgent
from .scenarios import CRISIS_SCENARIOS


@dataclass
class CrisisExperimentSummary:
    """Summary statistics for the crisis experiment."""

    n_scenarios: int = 0

    # UNCONSTRAINED stats
    unconstrained_avg_iterations: float = 0.0
    unconstrained_avg_tokens: float = 0.0
    unconstrained_avg_quality: float = 0.0
    unconstrained_early_stop_rate: float = 0.0

    # CONTRACTED stats (with iteration limits)
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
class CrisisExperimentResults:
    """Full crisis experiment results."""

    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    summary: CrisisExperimentSummary = field(default_factory=CrisisExperimentSummary)
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment": "good_enough_crisis",
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": self.summary.to_dict(),
            "trials": self.trials,
        }


def run_crisis_experiment(
    max_llm_calls: int = 30,
    quality_threshold: float = 0.80,
    num_judges: int = 3,
    verbose: bool = True,
) -> CrisisExperimentResults:
    """Run the Good Enough experiment on CRISIS scenarios.

    In crisis scenarios:
    - CONTRACTED agents have both quality threshold AND iteration limits
    - UNCONSTRAINED agents have neither

    The hypothesis is that CONTRACTED agents will stop sooner because
    their contract explicitly tells them to prioritize speed over perfection.

    Args:
        max_llm_calls: Maximum LLM calls per agent (safety limit)
        quality_threshold: Q_min threshold for CONTRACTED agent
        num_judges: Number of judges for quality evaluation
        verbose: Print progress

    Returns:
        CrisisExperimentResults with all trial data
    """
    print("=" * 70)
    print("CRISIS COMMUNICATION EXPERIMENT (Google ADK)")
    print("=" * 70)
    print("\n⚠️  Testing time-critical scenarios where SPEED matters")
    print(f"\nConfig: max_calls={max_llm_calls}, Q_min={quality_threshold}")
    print(f"Judges: {num_judges}")
    print(f"Crisis scenarios: {len(CRISIS_SCENARIOS)}")

    results = CrisisExperimentResults(
        timestamp=datetime.now().isoformat(),
        config={
            "n_scenarios": len(CRISIS_SCENARIOS),
            "max_llm_calls": max_llm_calls,
            "quality_threshold": quality_threshold,
            "num_judges": num_judges,
            "framework": "google_adk",
            "experiment_type": "crisis_communication",
        },
    )

    unconstrained_results: list[AdkAgentResult] = []
    contracted_results: list[AdkAgentResult] = []

    for i, scenario in enumerate(CRISIS_SCENARIOS):
        print(f"\n{'=' * 70}")
        print(f"[Scenario {i + 1}/{len(CRISIS_SCENARIOS)}] {scenario.id}")
        print(f"  Urgency: {scenario.urgency.upper()}")
        print(f"  Time Pressure: {scenario.time_pressure}")
        print(f"  Max Iterations (contract): {scenario.max_iterations}")
        print("=" * 70)

        # Create UNCONSTRAINED agent (no iteration limit in contract)
        unconstrained_agent = UnconstrainedAdkAgent(
            max_llm_calls=max_llm_calls,
            num_judges=num_judges,
        )

        # Create CONTRACTED agent WITH scenario-specific iteration limit
        contracted_agent = ContractedAdkAgent(
            max_llm_calls=max_llm_calls,
            quality_threshold=quality_threshold,
            num_judges=num_judges,
            scenario_max_iterations=scenario.max_iterations,  # Crisis limit!
        )

        # Run UNCONSTRAINED
        print("\n  UNCONSTRAINED (no iteration limit):")
        print("  → Agent decides when to stop based on subjective judgment")
        uc_result = unconstrained_agent.run(scenario, verbose=verbose)
        unconstrained_results.append(uc_result)
        print(
            f"  → Iterations: {uc_result.iterations}, Tokens: {uc_result.total_tokens:,}, Q: {uc_result.final_quality:.2f}"
        )
        print(f"  → Stop reason: {uc_result.stop_reason}")

        # Run CONTRACTED
        print(f"\n  CONTRACTED (Q_min={quality_threshold}, max_iter={scenario.max_iterations}):")
        print("  → Agent has explicit contract with quality AND iteration limits")
        ct_result = contracted_agent.run(scenario, verbose=verbose)
        contracted_results.append(ct_result)
        print(
            f"  → Iterations: {ct_result.iterations}, Tokens: {ct_result.total_tokens:,}, Q: {ct_result.final_quality:.2f}"
        )
        print(f"  → Stop reason: {ct_result.stop_reason}")

        # Check if contracted agent respected iteration limit
        if ct_result.iterations <= (scenario.max_iterations or 999):
            print(
                f"  ✓ Agent respected iteration limit ({ct_result.iterations} <= {scenario.max_iterations})"
            )
        else:
            print(
                f"  ⚠ Agent exceeded iteration limit ({ct_result.iterations} > {scenario.max_iterations})"
            )

        # Store trial
        results.trials.append(
            {
                "scenario_id": scenario.id,
                "category": scenario.category,
                "urgency": scenario.urgency,
                "time_pressure": scenario.time_pressure,
                "max_iterations": scenario.max_iterations,
                "unconstrained": uc_result.to_dict(),
                "contracted": ct_result.to_dict(),
            }
        )

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("CRISIS EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    summary = CrisisExperimentSummary(n_scenarios=len(CRISIS_SCENARIOS))

    # UNCONSTRAINED stats
    uc_iterations = [r.iterations for r in unconstrained_results]
    uc_tokens = [r.total_tokens for r in unconstrained_results]
    uc_qualities = [r.final_quality for r in unconstrained_results]

    summary.unconstrained_avg_iterations = statistics.mean(uc_iterations) if uc_iterations else 0
    summary.unconstrained_avg_tokens = statistics.mean(uc_tokens) if uc_tokens else 0
    summary.unconstrained_avg_quality = statistics.mean(uc_qualities) if uc_qualities else 0
    summary.unconstrained_early_stop_rate = (
        sum(1 for r in unconstrained_results if r.stopped_early) / len(unconstrained_results)
        if unconstrained_results
        else 0
    )

    # CONTRACTED stats
    ct_iterations = [r.iterations for r in contracted_results]
    ct_tokens = [r.total_tokens for r in contracted_results]
    ct_qualities = [r.final_quality for r in contracted_results]

    summary.contracted_avg_iterations = statistics.mean(ct_iterations) if ct_iterations else 0
    summary.contracted_avg_tokens = statistics.mean(ct_tokens) if ct_tokens else 0
    summary.contracted_avg_quality = statistics.mean(ct_qualities) if ct_qualities else 0
    summary.contracted_early_stop_rate = (
        sum(1 for r in contracted_results if r.stopped_early) / len(contracted_results)
        if contracted_results
        else 0
    )

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
    print("KEY FINDING: CRISIS COMMUNICATION")
    print("=" * 70)
    if summary.iteration_reduction > 0:
        print("\n  ✓ CONTRACTED agent with iteration limits stopped sooner")
        print(f"    - {summary.iteration_reduction:.0f}% fewer iterations")
        print(f"    - {summary.token_reduction:.0f}% fewer tokens")
        print(f"    - Quality difference: {summary.quality_difference:+.2f}")
        print("\n  → In crisis scenarios, the contract's iteration limit enforces")
        print("    'good enough NOW' rather than 'perfect later'")
        print("\n  → This demonstrates Agent Contracts' value for time-critical tasks:")
        print("    - Regulatory compliance (GDPR notification deadlines)")
        print("    - Customer trust (outage communication)")
        print("    - Security (vulnerability disclosure windows)")
    else:
        print("\n  Results require further analysis.")
        print("  Both agents may be behaving similarly in crisis scenarios.")

    # Save results
    output_dir = Path("results/good_enough")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"crisis_experiment_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Crisis Communication experiment")
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
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )

    args = parser.parse_args()

    run_crisis_experiment(
        max_llm_calls=args.max_llm_calls,
        quality_threshold=args.quality_threshold,
        num_judges=args.num_judges,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
