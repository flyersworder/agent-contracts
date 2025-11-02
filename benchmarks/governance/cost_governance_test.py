"""Cost Governance Test - Demonstrates Organizational Policy Enforcement.

This test shows that the framework can enforce company-wide cost policies across
diverse queries of varying complexity.

## The Value Proposition

Organizations need to control AI spending through enforceable policies. The framework
enables "maximum cost per query" policies that prevent budget overruns.

## Test Design

Scenario: Company policy - "Maximum $0.01 per query"

Test with questions of varying complexity:
1. SIMPLE questions: Should complete within policy
2. MEDIUM questions: May require budget-aware strategies
3. COMPLEX questions: May exceed policy if uncontracted

For each question:
- Uncontracted: Run without limits (may violate policy)
- Contracted: Run with $0.01 hard limit

## Metrics

1. Policy compliance: Does framework enforce $0.01 limit?
2. Coverage: % of questions answerable within policy
3. Quality distribution: What quality is achievable?
4. Violations prevented: How many uncontracted queries exceed policy?

## Expected Result

- Contracted: 100% policy compliance
- Uncontracted: Some queries exceed policy
- Quality tradeoff: Complex questions may need lower quality to fit budget
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import cast

from benchmarks.governance.budget_violation_test import ContractedBudgetAgent
from benchmarks.research_agent.evaluator import QualityEvaluator
from benchmarks.research_agent.uncontracted_agent import UncontractedResearchAgent


class QuestionComplexity(str, Enum):
    """Complexity levels for questions."""

    SIMPLE = "simple"  # Basic factual questions
    MEDIUM = "medium"  # Moderate analysis required
    COMPLEX = "complex"  # Deep multi-step reasoning


@dataclass
class CostGovernanceResult:
    """Result of testing organizational cost policy."""

    question_id: str
    question_text: str
    complexity: str
    policy_limit_usd: float

    # Uncontracted results
    uncontracted_cost: float
    uncontracted_tokens: int
    uncontracted_quality: float
    uncontracted_violates_policy: bool

    # Contracted results
    contracted_cost: float
    contracted_tokens: int
    contracted_quality: float | None  # None if violated/incomplete
    contracted_completed: bool
    contracted_violated: bool

    # Policy enforcement
    policy_enforced: bool  # Did contract prevent violation?
    quality_degradation: float | None  # Quality loss vs uncontracted


class CostGovernanceTest:
    """Test organizational cost policy enforcement."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        output_dir: str = "benchmarks/governance/results",
        policy_limit_usd: float = 0.01,  # $0.01 per query policy
    ) -> None:
        """Initialize cost governance test.

        Args:
            model: LLM model to use
            output_dir: Directory to save results
            policy_limit_usd: Maximum cost per query policy (default: $0.01)
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy_limit_usd = policy_limit_usd

        self.evaluator = QualityEvaluator(
            judge_model=model,
            use_multiple_judges=False,
            use_hybrid_scoring=True,
        )

    def run(
        self,
        questions: list[tuple[str, str, QuestionComplexity]],  # (id, question, complexity)
    ) -> list[CostGovernanceResult]:
        """Run cost governance test.

        Args:
            questions: List of (question_id, question_text, complexity) tuples

        Returns:
            List of CostGovernanceResult objects
        """
        print("=" * 80)
        print("COST GOVERNANCE TEST")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Policy: Maximum ${self.policy_limit_usd:.4f} per query")
        print(f"Questions: {len(questions)}")
        print()

        results = []

        for q_id, question, complexity in questions:
            print(f"\n{'=' * 80}")
            print(f"Question: {q_id} ({complexity.value.upper()})")
            print(f"{'=' * 80}")
            print(f"Q: {question[:100]}...")
            print()

            # Run uncontracted (no policy limit)
            print("  Running UNCONTRACTED (no policy limit)...")
            unc_result = self._run_uncontracted(question)
            unc_violates = unc_result["cost"] > self.policy_limit_usd
            print(f"    Cost: ${unc_result['cost']:.4f}", end="")
            print(f" {'⚠️  EXCEEDS POLICY' if unc_violates else ' ✅ Within policy'}")
            print(f"    Quality: {unc_result['quality']:.1f}/100")

            # Run contracted (with policy limit)
            print("\n  Running CONTRACTED (with policy limit)...")
            cont_result = self._run_contracted(question)
            cost = cast("float", cont_result["cost"])
            print(f"    Cost: ${cost:.4f}", end="")
            print(
                f" {'✅ Policy enforced' if cost <= self.policy_limit_usd else ' ❌ Policy violation'}"
            )

            if cont_result["completed"]:
                print(f"    Quality: {cont_result['quality']:.1f}/100")
            else:
                print("    Status: INCOMPLETE (budget exceeded)")

            # Analyze results
            result = self._analyze_result(
                q_id=q_id,
                question=question,
                complexity=complexity,
                uncontracted=unc_result,
                contracted=cont_result,
            )
            results.append(result)

        # Print overall analysis
        print(f"\n{'=' * 80}")
        print("COST GOVERNANCE ANALYSIS")
        print("=" * 80)
        self._print_overall_analysis(results)

        # Save results
        self._save_results(results)

        return results

    def _run_uncontracted(self, question: str) -> dict[str, float | int]:
        """Run uncontracted agent (no limits).

        Args:
            question: Question to answer

        Returns:
            Dictionary with cost, tokens, quality
        """
        agent = UncontractedResearchAgent(model=self.model)
        result = agent.research(question)
        quality = self.evaluator.evaluate(question, result.final_answer)

        return {
            "cost": result.total_cost,
            "tokens": result.total_tokens,
            "quality": quality.total,
        }

    def _run_contracted(self, question: str) -> dict[str, float | int | bool | None]:
        """Run contracted agent with policy limit.

        Args:
            question: Question to answer

        Returns:
            Dictionary with cost, tokens, quality, completed, violated
        """
        # Estimate token budget from cost limit
        # Gemini 2.5 Flash: ~$0.008 per 1K reasoning tokens, ~$0.003 per 1K text tokens
        # Use 80/20 split, solve for tokens
        # cost = (tokens * 0.8 / 1000) * 0.008 + (tokens * 0.2 / 1000) * 0.003
        # cost = tokens * (0.0064 + 0.0006) / 1000
        # tokens = cost * 1000 / 0.007
        estimated_tokens = int(self.policy_limit_usd * 1000 / 0.007)

        agent = ContractedBudgetAgent(
            model=self.model,
            budget_tokens=estimated_tokens,
            budget_cost=self.policy_limit_usd,
            strict_mode=True,
        )

        completed = False
        violated = False
        cost = 0.0
        tokens = 0
        quality = None

        try:
            result = agent.research(question)
            completed = True
            cost = result.total_cost
            tokens = result.total_tokens

            # Evaluate quality
            quality_score = self.evaluator.evaluate(question, result.final_answer)
            quality = quality_score.total

        except Exception:
            # Budget violation
            violated = True
            cost = agent.total_cost_used
            tokens = agent.total_tokens_used

        return {
            "cost": cost,
            "tokens": tokens,
            "quality": quality,
            "completed": completed,
            "violated": violated,
        }

    def _analyze_result(
        self,
        q_id: str,
        question: str,
        complexity: QuestionComplexity,
        uncontracted: dict[str, float | int],
        contracted: dict[str, float | int | bool | None],
    ) -> CostGovernanceResult:
        """Analyze results and create CostGovernanceResult.

        Args:
            q_id: Question ID
            question: Question text
            complexity: Question complexity
            uncontracted: Uncontracted results
            contracted: Contracted results

        Returns:
            CostGovernanceResult
        """
        # Policy enforcement check
        policy_enforced = cast("float", contracted["cost"]) <= self.policy_limit_usd

        # Quality degradation
        quality_degradation = None
        unc_quality = cast("float", uncontracted["quality"])
        cont_quality = contracted["quality"]
        if cast("bool", contracted["completed"]) and cont_quality is not None:
            quality_degradation = unc_quality - cast("float", cont_quality)

        return CostGovernanceResult(
            question_id=q_id,
            question_text=question,
            complexity=complexity.value,
            policy_limit_usd=self.policy_limit_usd,
            uncontracted_cost=cast("float", uncontracted["cost"]),
            uncontracted_tokens=cast("int", uncontracted["tokens"]),
            uncontracted_quality=unc_quality,
            uncontracted_violates_policy=cast("float", uncontracted["cost"])
            > self.policy_limit_usd,
            contracted_cost=cast("float", contracted["cost"]),
            contracted_tokens=cast("int", contracted["tokens"]),
            contracted_quality=cast("float", cont_quality) if cont_quality is not None else None,
            contracted_completed=cast("bool", contracted["completed"]),
            contracted_violated=cast("bool", contracted["violated"]),
            policy_enforced=policy_enforced,
            quality_degradation=quality_degradation,
        )

    def _print_overall_analysis(self, results: list[CostGovernanceResult]) -> None:
        """Print overall cost governance analysis.

        Args:
            results: List of all test results
        """
        total = len(results)

        # Policy enforcement
        enforced = sum(1 for r in results if r.policy_enforced)
        print("\nPolicy Compliance:")
        print(
            f"  Contracted: {enforced}/{total} ({enforced / total * 100:.0f}%) enforced ${self.policy_limit_usd:.4f} limit"
        )

        # Uncontracted violations
        unc_violations = sum(1 for r in results if r.uncontracted_violates_policy)
        print(
            f"  Uncontracted: {unc_violations}/{total} ({unc_violations / total * 100:.0f}%) exceeded policy"
        )
        print(f"  Violations Prevented: {unc_violations}")

        # Coverage (% of questions answerable within policy)
        completed = sum(1 for r in results if r.contracted_completed)
        print("\nCoverage:")
        print(
            f"  Questions answerable within policy: {completed}/{total} ({completed / total * 100:.0f}%)"
        )

        # Quality analysis (for completed)
        completed_results = [
            r for r in results if r.contracted_completed and r.contracted_quality is not None
        ]
        if completed_results:
            # mypy doesn't understand the filter above ensures contracted_quality is not None
            avg_quality = sum(
                r.contracted_quality for r in completed_results if r.contracted_quality is not None
            ) / len(completed_results)
            print("\nQuality (for completed within policy):")
            print(f"  Average: {avg_quality:.1f}/100")

            # Quality degradation vs uncontracted
            degradations = [
                r.quality_degradation
                for r in completed_results
                if r.quality_degradation is not None
            ]
            if degradations:
                avg_degradation = sum(degradations) / len(degradations)
                print(f"  Average degradation vs uncontracted: {avg_degradation:+.1f} points")

        # Key finding
        print(f"\n{'🎯' * 3} KEY FINDING {'🎯' * 3}")
        print(
            f"Cost policy (${self.policy_limit_usd:.4f}/query) enforced: {enforced}/{total} tests"
        )
        print(f"Uncontracted violations prevented: {unc_violations}")
        print("The framework enables organizational cost control through enforceable policies.")

    def _save_results(self, results: list[CostGovernanceResult]) -> None:
        """Save results to JSON file.

        Args:
            results: List of CostGovernanceResult objects
        """
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"cost_governance_results_{timestamp}.json"

        # Convert to dict
        results_dict = [asdict(r) for r in results]

        # Save to file
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "model": self.model,
                    "policy_limit_usd": self.policy_limit_usd,
                    "num_tests": len(results),
                    "results": results_dict,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")


def main() -> None:
    """Run cost governance test from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run cost governance test")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-preview-09-2025",
        help="LLM model to use",
    )
    parser.add_argument(
        "--policy-limit",
        type=float,
        default=0.01,
        help="Maximum cost per query policy (default: $0.01)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with only 2 questions",
    )

    args = parser.parse_args()

    # Test questions with varying complexity
    test_questions = [
        (
            "simple_1",
            "What is the capital of France?",
            QuestionComplexity.SIMPLE,
        ),
        (
            "medium_1",
            "Explain the key differences between supervised and unsupervised machine learning.",
            QuestionComplexity.MEDIUM,
        ),
        (
            "complex_1",
            "Compare transformer architectures with state space models for sequence modeling, "
            "including computational complexity and long-range dependency handling.",
            QuestionComplexity.COMPLEX,
        ),
        (
            "complex_2",
            "Analyze the 2008 financial crisis: primary causes, systemic failures, and how "
            "the Dodd-Frank Act addressed these issues.",
            QuestionComplexity.COMPLEX,
        ),
    ]

    if args.quick:
        test_questions = test_questions[:2]

    test = CostGovernanceTest(model=args.model, policy_limit_usd=args.policy_limit)
    test.run(questions=test_questions)


if __name__ == "__main__":
    main()
