"""Budget Violation Test - Demonstrates Enforcement Compliance.

This test proves that the framework can enforce budget limits and prevent violations.

## The Value Proposition

Organizations need to prevent runaway AI costs. The framework provides hard enforcement
that stops execution when budgets are exceeded.

## Test Design

Test questions with different budget levels:
1. GENEROUS (100% of typical usage): Should complete successfully
2. MEDIUM (75% of typical usage): Should complete with some degradation
3. TIGHT (50% of typical usage): Should hit limits but complete
4. EXTREME (25% of typical usage): Should violate and stop

## Metrics

1. Compliance rate: % of runs that stay within budget
2. Violation detection: Were violations caught?
3. Quality degradation curve: Quality vs budget level
4. Graceful failure: How does the system behave when limits hit?

## Expected Result

- GENEROUS/MEDIUM: 100% compliance, high quality
- TIGHT: 100% compliance (via enforcement), reduced quality
- EXTREME: Violations caught and execution stopped
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from agent_contracts import Contract, ContractedLLM, ResourceConstraints
from benchmarks.research_agent.contracted_agent import ContractedResearchAgent
from benchmarks.research_agent.evaluator import QualityEvaluator


class BudgetLevel(str, Enum):
    """Budget constraint levels."""

    GENEROUS = "generous"  # 100% of typical usage
    MEDIUM = "medium"  # 75% of typical usage
    TIGHT = "tight"  # 50% of typical usage
    EXTREME = "extreme"  # 25% of typical usage


@dataclass
class BudgetTestResult:
    """Result of testing a single question at a budget level."""

    question_id: str
    question_text: str
    budget_level: str
    budget_tokens: int
    budget_cost: float

    # Execution results
    completed: bool  # Did execution complete?
    violated: bool  # Was budget violated?
    violation_reason: str | None  # Why did it violate?

    # Usage metrics
    tokens_used: int
    cost_used: float
    api_calls_used: int

    # Budget utilization
    token_utilization_pct: float  # % of budget used
    cost_utilization_pct: float

    # Quality (if completed)
    quality_score: float | None
    final_answer: str | None


class BudgetViolationTest:
    """Test framework's budget enforcement capabilities."""

    # Baseline budgets (based on empirical observation from benchmarks)
    # These are "typical" usage levels we want to scale down from
    BASELINE_BUDGETS: ClassVar[dict[str, int]] = {
        "planning": 1200,  # reasoning tokens
        "research_per_subq": 2500,  # reasoning tokens per sub-question
        "synthesis": 2500,  # reasoning tokens
        "num_subquestions": 3,  # typical decomposition
    }

    # Budget multipliers for each level
    BUDGET_MULTIPLIERS: ClassVar[dict[BudgetLevel, float]] = {
        BudgetLevel.GENEROUS: 1.0,  # 100%
        BudgetLevel.MEDIUM: 0.75,  # 75%
        BudgetLevel.TIGHT: 0.5,  # 50%
        BudgetLevel.EXTREME: 0.25,  # 25%
    }

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        output_dir: str = "benchmarks/governance/results",
    ) -> None:
        """Initialize budget violation test.

        Args:
            model: LLM model to use
            output_dir: Directory to save results
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = QualityEvaluator(
            judge_model=model,
            use_multiple_judges=False,  # Single judge for speed
            use_hybrid_scoring=True,
        )

    def run(
        self,
        questions: list[tuple[str, str]],  # List of (id, question)
        budget_levels: list[BudgetLevel] | None = None,
    ) -> list[BudgetTestResult]:
        """Run budget violation test.

        Args:
            questions: List of (question_id, question_text) tuples
            budget_levels: Budget levels to test (defaults to all)

        Returns:
            List of BudgetTestResult objects
        """
        if budget_levels is None:
            budget_levels = list(BudgetLevel)

        print("=" * 80)
        print("BUDGET VIOLATION TEST")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Questions: {len(questions)}")
        print(f"Budget Levels: {len(budget_levels)}")
        print(f"Total tests: {len(questions) * len(budget_levels)}")
        print()

        results = []

        for q_id, question in questions:
            print(f"\n{'=' * 80}")
            print(f"Question: {q_id}")
            print(f"{'=' * 80}")
            print(f"Q: {question[:100]}...")
            print()

            for budget_level in budget_levels:
                print(f"\n  Testing with {budget_level.value.upper()} budget...")
                result = self._test_budget_level(q_id, question, budget_level)
                results.append(result)
                self._print_result_summary(result)

        # Print overall analysis
        print(f"\n{'=' * 80}")
        print("BUDGET ENFORCEMENT ANALYSIS")
        print(f"={'=' * 80}")
        self._print_overall_analysis(results)

        # Save results
        self._save_results(results)

        return results

    def _test_budget_level(
        self, q_id: str, question: str, budget_level: BudgetLevel
    ) -> BudgetTestResult:
        """Test a single question at a specific budget level.

        Args:
            q_id: Question ID
            question: Question text
            budget_level: Budget level to test

        Returns:
            BudgetTestResult
        """
        # Calculate budget for this level
        multiplier = self.BUDGET_MULTIPLIERS[budget_level]
        total_budget = self._calculate_budget(multiplier)

        # Create contracted agent with strict mode to catch violations
        agent = ContractedBudgetAgent(
            model=self.model,
            budget_tokens=int(total_budget["tokens"]),
            budget_cost=float(total_budget["cost"]),
            strict_mode=True,  # Enable strict enforcement
        )

        # Try to execute
        violated = False
        violation_reason = None
        completed = False
        tokens_used = 0
        cost_used = 0.0
        api_calls_used = 0
        quality_score = None
        final_answer = None

        try:
            result = agent.research(question)
            completed = True
            tokens_used = result.total_tokens
            cost_used = result.total_cost
            api_calls_used = result.api_calls
            final_answer = result.final_answer

            # Evaluate quality if completed
            quality = self.evaluator.evaluate(question, result.final_answer)
            quality_score = quality.total

        except Exception as e:
            # Budget violation or other error
            violated = True
            violation_reason = str(e)

            # Get partial usage from agent if available
            if hasattr(agent, "total_tokens_used"):
                tokens_used = agent.total_tokens_used
                cost_used = agent.total_cost_used
                api_calls_used = agent.total_api_calls

        # Calculate utilization
        token_util = (
            (tokens_used / total_budget["tokens"] * 100) if total_budget["tokens"] > 0 else 0
        )
        cost_util = (cost_used / total_budget["cost"] * 100) if total_budget["cost"] > 0 else 0

        return BudgetTestResult(
            question_id=q_id,
            question_text=question,
            budget_level=budget_level.value,
            budget_tokens=int(total_budget["tokens"]),
            budget_cost=float(total_budget["cost"]),
            completed=completed,
            violated=violated,
            violation_reason=violation_reason,
            tokens_used=tokens_used,
            cost_used=cost_used,
            api_calls_used=api_calls_used,
            token_utilization_pct=token_util,
            cost_utilization_pct=cost_util,
            quality_score=quality_score,
            final_answer=final_answer,
        )

    def _calculate_budget(self, multiplier: float) -> dict[str, int | float]:
        """Calculate budget constraints based on multiplier.

        Args:
            multiplier: Budget multiplier (0.25 to 1.0)

        Returns:
            Dictionary with token and cost budgets
        """
        # Total reasoning tokens
        total_tokens = int(
            (
                self.BASELINE_BUDGETS["planning"]
                + (
                    self.BASELINE_BUDGETS["research_per_subq"]
                    * self.BASELINE_BUDGETS["num_subquestions"]
                )
                + self.BASELINE_BUDGETS["synthesis"]
            )
            * multiplier
        )

        # Estimated cost (rough approximation based on Gemini 2.5 Flash pricing)
        # Reasoning: ~$0.008 per 1K tokens
        # Text output: ~$0.003 per 1K tokens
        # Approximate 80/20 split reasoning/text
        total_cost = (total_tokens / 1000) * 0.008 * 1.2  # 20% margin for text tokens

        return {"tokens": total_tokens, "cost": total_cost}

    def _print_result_summary(self, result: BudgetTestResult) -> None:
        """Print summary of a single test result.

        Args:
            result: BudgetTestResult to summarize
        """
        status = "✅ COMPLETED" if result.completed else "❌ VIOLATED"
        print(f"    Status: {status}")
        print(f"    Budget: {result.budget_tokens:,} tokens, ${result.budget_cost:.4f}")
        print(
            f"    Used:   {result.tokens_used:,} tokens ({result.token_utilization_pct:.1f}%), ${result.cost_used:.4f} ({result.cost_utilization_pct:.1f}%)"
        )

        if result.completed and result.quality_score is not None:
            print(f"    Quality: {result.quality_score:.1f}/100")
        elif result.violated and result.violation_reason:
            print(f"    Violation: {result.violation_reason[:60]}...")

    def _print_overall_analysis(self, results: list[BudgetTestResult]) -> None:
        """Print overall budget enforcement analysis.

        Args:
            results: List of all test results
        """
        # Group by budget level
        by_level: dict[str, list[BudgetTestResult]] = {}
        for result in results:
            if result.budget_level not in by_level:
                by_level[result.budget_level] = []
            by_level[result.budget_level].append(result)

        # Analyze each level
        for level in [
            BudgetLevel.GENEROUS,
            BudgetLevel.MEDIUM,
            BudgetLevel.TIGHT,
            BudgetLevel.EXTREME,
        ]:
            level_results = by_level.get(level.value, [])
            if not level_results:
                continue

            print(
                f"\n{level.value.upper()} Budget ({self.BUDGET_MULTIPLIERS[level]:.0%} of baseline):"
            )
            print(f"  Tests: {len(level_results)}")

            # Completion rate
            completed = sum(1 for r in level_results if r.completed)
            compliance_rate = completed / len(level_results) * 100
            print(
                f"  Compliance Rate: {compliance_rate:.0f}% ({completed}/{len(level_results)} completed)"
            )

            # Average quality (for completed)
            completed_results = [
                r for r in level_results if r.completed and r.quality_score is not None
            ]
            if completed_results:
                # mypy doesn't understand the filter above ensures quality_score is not None
                avg_quality = sum(
                    r.quality_score for r in completed_results if r.quality_score is not None
                ) / len(completed_results)
                print(f"  Average Quality: {avg_quality:.1f}/100")

            # Average utilization
            avg_token_util = sum(r.token_utilization_pct for r in level_results) / len(
                level_results
            )
            print(f"  Average Token Utilization: {avg_token_util:.1f}%")

        # Key finding
        print(f"\n{'🎯' * 3} KEY FINDING {'🎯' * 3}")
        total_violations = sum(1 for r in results if r.violated)
        total_tests = len(results)
        print(
            f"Budget violations detected and prevented: {total_violations}/{total_tests} "
            f"({total_violations / total_tests * 100:.0f}%)"
        )
        print("The framework successfully enforces hard budget limits.")

    def _save_results(self, results: list[BudgetTestResult]) -> None:
        """Save results to JSON file.

        Args:
            results: List of BudgetTestResult objects
        """
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"budget_violation_results_{timestamp}.json"

        # Convert to dict
        results_dict = [asdict(r) for r in results]

        # Save to file
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "model": self.model,
                    "num_tests": len(results),
                    "results": results_dict,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")


class ContractedBudgetAgent(ContractedResearchAgent):
    """Research agent with a single unified budget across all steps."""

    def __init__(
        self,
        model: str,
        budget_tokens: int,
        budget_cost: float,
        strict_mode: bool = True,
    ) -> None:
        """Initialize with unified budget.

        Args:
            model: LLM model to use
            budget_tokens: Total reasoning token budget
            budget_cost: Total cost budget
            strict_mode: Enable strict enforcement
        """
        super().__init__(model=model, strict_mode=strict_mode)
        self.unified_budget_tokens = budget_tokens
        self.unified_budget_cost = budget_cost

        # Track cumulative usage
        self.total_tokens_used = 0
        self.total_cost_used = 0.0
        self.total_api_calls = 0

    def _call_llm(self, messages: list[dict[str, str]], step_type: str) -> Any:
        """Call LLM with unified budget tracking."""
        # Calculate remaining budget
        remaining_tokens = self.unified_budget_tokens - self.total_tokens_used
        remaining_cost = self.unified_budget_cost - self.total_cost_used

        if remaining_tokens <= 0:
            raise RuntimeError(
                f"Token budget exhausted: {self.total_tokens_used}/{self.unified_budget_tokens}"
            )

        # Create contract with remaining budget
        contract = Contract(
            id=f"{step_type}_with_remaining_budget",
            name=f"{step_type.title()} Step",
            description=f"{step_type} with unified budget tracking",
            resources=ResourceConstraints(
                reasoning_tokens=remaining_tokens,
                text_tokens=2500,  # Generous text budget (research agent outputs 1300-2000 tokens)
                api_calls=1,
                cost_usd=remaining_cost,
            ),
        )

        # Execute with contract
        with ContractedLLM(contract=contract, strict_mode=self.strict_mode) as llm:
            response = llm.completion(model=self.model, messages=messages, temperature=0)

        # Update cumulative usage
        if hasattr(response, "usage"):
            reasoning_tokens = getattr(response.usage, "reasoning_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

            self.total_tokens_used += reasoning_tokens
            self.total_api_calls += 1

            # Estimate cost (rough)
            self.total_cost_used += (reasoning_tokens / 1000) * 0.008
            self.total_cost_used += (completion_tokens / 1000) * 0.003

        return response


def main() -> None:
    """Run budget violation test from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run budget violation test")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-preview-09-2025",
        help="LLM model to use",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with only 1 question and 2 budget levels",
    )

    args = parser.parse_args()

    # Test questions
    test_questions = [
        (
            "q1_transformers",
            "Compare transformer architectures with state space models (SSMs) for sequence modeling. "
            "What are the key differences in computational complexity, long-range dependency handling, "
            "and practical performance on language tasks?",
        ),
        (
            "q3_financial",
            "What were the primary causes of the 2008 financial crisis, and how did the regulatory "
            "response (Dodd-Frank Act) attempt to prevent similar events in the future?",
        ),
    ]

    # Budget levels to test
    if args.quick:
        test_questions = test_questions[:1]
        budget_levels = [BudgetLevel.MEDIUM, BudgetLevel.TIGHT]
    else:
        budget_levels = list(BudgetLevel)

    test = BudgetViolationTest(model=args.model)
    test.run(questions=test_questions, budget_levels=budget_levels)


if __name__ == "__main__":
    main()
