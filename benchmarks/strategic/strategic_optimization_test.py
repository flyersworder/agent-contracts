"""Strategic optimization benchmark: Test Pareto frontier of quality-cost-time tradeoffs.

This benchmark validates that the three contract modes (URGENT, BALANCED, ECONOMICAL)
produce a Pareto-optimal frontier where no mode strictly dominates another, and each
mode optimizes for its intended dimension:
- URGENT: Minimize time (fast execution)
- BALANCED: Balance quality, cost, and time
- ECONOMICAL: Minimize cost (efficient resource usage)

The benchmark executes simulated tasks under all three modes and verifies:
1. No mode is strictly dominated (Pareto optimality)
2. URGENT has lowest time
3. ECONOMICAL has lowest cost
4. Each mode achieves acceptable quality
"""

import time
from dataclasses import dataclass

from agent_contracts.core import (
    Contract,
    ContractMode,
    ResourceConstraints,
    ResourceUsage,
)
from agent_contracts.core.planning import (
    Task,
    TaskPriority,
    estimate_quality_cost_time,
    plan_resource_allocation,
    recommend_strategy,
)


@dataclass
class BenchmarkResult:
    """Result from running a task under a specific mode.

    Attributes:
        mode: The contract mode used
        quality: Achieved quality score (0.0 to 1.0)
        tokens_used: Total tokens consumed
        time_seconds: Execution time in seconds
        cost_usd: Total cost in USD
    """

    mode: ContractMode
    quality: float
    tokens_used: int
    time_seconds: float
    cost_usd: float


class StrategicOptimizationBenchmark:
    """Benchmark for testing strategic optimization across contract modes."""

    def __init__(self, verbose: bool = True):
        """Initialize the benchmark.

        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose

    def simulate_task_execution(
        self,
        contract: Contract,
        task_description: str,
        approach: str = "standard",
    ) -> BenchmarkResult:
        """Simulate executing a task under a contract mode.

        This simulates task execution using the contract's mode to determine
        quality-cost-time tradeoffs based on the planning utilities.

        Args:
            contract: The contract with mode and constraints
            task_description: Description of the task
            approach: Execution approach (thorough/standard/quick/minimal)

        Returns:
            Benchmark result with quality, cost, and time metrics
        """
        # Use planning utilities to estimate tradeoffs
        quality, tokens, time_estimate = estimate_quality_cost_time(approach, contract)

        # Simulate execution time with some variance
        start_time = time.time()
        time.sleep(time_estimate / 1000)  # Convert to actual seconds (scaled down)
        actual_time = time.time() - start_time

        # Calculate cost (using simplified pricing: $0.01 per 1000 tokens)
        cost = tokens * 0.01 / 1000

        return BenchmarkResult(
            mode=contract.mode,
            quality=quality,
            tokens_used=tokens,
            time_seconds=actual_time,
            cost_usd=cost,
        )

    def run_multi_task_scenario(self) -> dict[ContractMode, BenchmarkResult]:
        """Run a multi-task scenario under all three modes.

        This simulates a realistic scenario where an agent needs to:
        1. Research a topic
        2. Analyze findings
        3. Generate a report

        Returns:
            Dictionary mapping mode to benchmark results
        """
        # Define the tasks
        tasks = [
            Task(
                id="research",
                name="Research Topic",
                estimated_tokens=3000,
                estimated_time=120.0,
                estimated_quality=0.85,
                priority=TaskPriority.HIGH,
                required=True,
            ),
            Task(
                id="analyze",
                name="Analyze Findings",
                estimated_tokens=2000,
                estimated_time=90.0,
                estimated_quality=0.90,
                priority=TaskPriority.HIGH,
                required=True,
            ),
            Task(
                id="report",
                name="Generate Report",
                estimated_tokens=1500,
                estimated_time=60.0,
                estimated_quality=0.80,
                priority=TaskPriority.MEDIUM,
                required=False,
            ),
        ]

        results = {}

        # Run under each mode
        for mode in [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]:
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Running scenario under {mode.value.upper()} mode")
                print(f"{'=' * 60}")

            # Create contract for this mode
            contract = Contract(
                id=f"multi-task-{mode.value}",
                name=f"Multi-Task Scenario ({mode.value})",
                mode=mode,
                resources=ResourceConstraints(
                    tokens=10000,  # Generous budget for all modes
                    api_calls=20,
                    cost_usd=0.50,
                ),
            )

            # Plan resource allocation
            allocations = plan_resource_allocation(tasks, contract)

            if self.verbose:
                print("\nResource Allocation:")
                for alloc in allocations:
                    task = next(t for t in tasks if t.id == alloc.task_id)
                    print(
                        f"  {task.name}: {alloc.allocated_tokens} tokens, "
                        f"{alloc.allocated_time:.1f}s, quality {alloc.expected_quality:.2f}"
                    )

            # Simulate execution (using aggregate metrics)
            total_tokens = sum(a.allocated_tokens for a in allocations)
            total_time_estimate = sum(a.allocated_time for a in allocations)
            avg_quality = sum(a.expected_quality for a in allocations) / len(allocations)

            # Simulate actual execution
            start_time = time.time()
            time.sleep(total_time_estimate / 1000)  # Scaled down
            actual_time = time.time() - start_time

            # Calculate cost
            cost = total_tokens * 0.01 / 1000

            result = BenchmarkResult(
                mode=mode,
                quality=avg_quality,
                tokens_used=total_tokens,
                time_seconds=actual_time,
                cost_usd=cost,
            )

            results[mode] = result

            if self.verbose:
                print("\nResults:")
                print(f"  Quality: {result.quality:.2f}")
                print(f"  Tokens: {result.tokens_used:,}")
                print(f"  Time: {result.time_seconds:.3f}s")
                print(f"  Cost: ${result.cost_usd:.5f}")

                # Get strategic recommendation
                usage = ResourceUsage(tokens=result.tokens_used)
                recommendation = recommend_strategy(contract, usage)
                print("\nStrategic Assessment:")
                print(f"  Risk Level: {recommendation.risk_level}")
                print(f"  Budget Utilization: {recommendation.budget_utilization:.1%}")
                print(f"  Recommendation: {recommendation.recommended_approach}")

        return results

    def verify_pareto_frontier(self, results: dict[ContractMode, BenchmarkResult]) -> bool:
        """Verify that results form a Pareto frontier.

        A Pareto frontier means no mode is strictly dominated by another.
        Mode A dominates mode B if A is better in all dimensions (quality, cost, time).

        Args:
            results: Results from all three modes

        Returns:
            True if results form a Pareto frontier, False otherwise
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Pareto Frontier Analysis")
            print(f"{'=' * 60}")

        modes = list(results.keys())

        # Check for strict domination
        dominated = set()
        for i, mode_a in enumerate(modes):
            for j, mode_b in enumerate(modes):
                if i == j:
                    continue

                res_a = results[mode_a]
                res_b = results[mode_b]

                # Mode A dominates B if A is better in all dimensions
                # (higher quality, lower cost, lower time)
                quality_better = res_a.quality >= res_b.quality
                cost_better = res_a.cost_usd <= res_b.cost_usd
                time_better = res_a.time_seconds <= res_b.time_seconds

                # Strict domination requires at least one strict inequality
                strictly_dominates = quality_better and cost_better and time_better
                has_strict = (
                    res_a.quality > res_b.quality
                    or res_a.cost_usd < res_b.cost_usd
                    or res_a.time_seconds < res_b.time_seconds
                )

                if strictly_dominates and has_strict:
                    dominated.add(mode_b)
                    if self.verbose:
                        print(f"  ⚠️  {mode_a.value} dominates {mode_b.value}")

        if not dominated:
            if self.verbose:
                print("  ✅ No mode is strictly dominated - Pareto frontier achieved!")
            return True
        else:
            if self.verbose:
                print(f"  ❌ Dominated modes: {[m.value for m in dominated]}")
            return False

    def verify_mode_objectives(self, results: dict[ContractMode, BenchmarkResult]) -> bool:
        """Verify that each mode achieves its intended objective.

        - URGENT should have lowest time
        - ECONOMICAL should have lowest cost
        - All modes should maintain acceptable quality

        Args:
            results: Results from all three modes

        Returns:
            True if all modes meet their objectives, False otherwise
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Mode Objective Verification")
            print(f"{'=' * 60}")

        all_passed = True

        # Extract results
        urgent = results[ContractMode.URGENT]
        balanced = results[ContractMode.BALANCED]
        economical = results[ContractMode.ECONOMICAL]

        # Check URGENT has lowest time
        urgent_fastest = urgent.time_seconds <= min(balanced.time_seconds, economical.time_seconds)
        if self.verbose:
            print("\n  URGENT mode (optimize speed):")
            print(f"    Time: {urgent.time_seconds:.3f}s")
            print("    ✅ Fastest" if urgent_fastest else "    ❌ Not fastest")
        all_passed = all_passed and urgent_fastest

        # Check ECONOMICAL has lowest cost
        economical_cheapest = economical.cost_usd <= min(urgent.cost_usd, balanced.cost_usd)
        if self.verbose:
            print("\n  ECONOMICAL mode (optimize cost):")
            print(f"    Cost: ${economical.cost_usd:.5f}")
            print("    ✅ Cheapest" if economical_cheapest else "    ❌ Not cheapest")
        all_passed = all_passed and economical_cheapest

        # Check all modes maintain quality
        min_quality_threshold = 0.7
        urgent_quality_ok = urgent.quality >= min_quality_threshold
        balanced_quality_ok = balanced.quality >= min_quality_threshold
        economical_quality_ok = economical.quality >= min_quality_threshold

        if self.verbose:
            print(f"\n  Quality (threshold: {min_quality_threshold}):")
            print(f"    URGENT: {urgent.quality:.2f} " + ("✅" if urgent_quality_ok else "❌"))
            print(
                f"    BALANCED: {balanced.quality:.2f} " + ("✅" if balanced_quality_ok else "❌")
            )
            print(
                f"    ECONOMICAL: {economical.quality:.2f} "
                + ("✅" if economical_quality_ok else "❌")
            )

        all_passed = (
            all_passed and urgent_quality_ok and balanced_quality_ok and economical_quality_ok
        )

        return all_passed

    def print_summary_table(self, results: dict[ContractMode, BenchmarkResult]) -> None:
        """Print a formatted summary table of results.

        Args:
            results: Results from all three modes
        """
        print(f"\n{'=' * 60}")
        print("Strategic Optimization Results")
        print(f"{'=' * 60}")
        print(f"\n{'Mode':<15} {'Quality':<10} {'Tokens':<10} {'Time (s)':<10} {'Cost ($)'}")
        print("-" * 60)

        for mode in [ContractMode.URGENT, ContractMode.BALANCED, ContractMode.ECONOMICAL]:
            res = results[mode]
            print(
                f"{mode.value.upper():<15} {res.quality:<10.2f} {res.tokens_used:<10,} "
                f"{res.time_seconds:<10.3f} ${res.cost_usd:.5f}"
            )

        print()

    def run(self) -> bool:
        """Run the complete strategic optimization benchmark.

        Returns:
            True if all tests pass, False otherwise
        """
        print("\n" + "=" * 60)
        print("STRATEGIC OPTIMIZATION BENCHMARK (H2)")
        print("Testing Quality-Cost-Time Pareto Frontier")
        print("=" * 60)

        # Run multi-task scenario
        results = self.run_multi_task_scenario()

        # Print summary
        self.print_summary_table(results)

        # Verify Pareto frontier
        pareto_ok = self.verify_pareto_frontier(results)

        # Verify mode objectives
        objectives_ok = self.verify_mode_objectives(results)

        # Final verdict
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(f"  Pareto Frontier: {'✅ PASS' if pareto_ok else '❌ FAIL'}")
        print(f"  Mode Objectives: {'✅ PASS' if objectives_ok else '❌ FAIL'}")
        print(f"\n  Overall: {'✅ PASS' if (pareto_ok and objectives_ok) else '❌ FAIL'}")
        print()

        return pareto_ok and objectives_ok


def main() -> int:
    """Run the strategic optimization benchmark."""
    benchmark = StrategicOptimizationBenchmark(verbose=True)
    success = benchmark.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
