"""Variance Reduction Test - Demonstrates Predictability Value.

This test quantifies the framework's core governance value: making AI costs predictable.

## The Value Proposition

Organizations need predictable AI costs for budgeting and planning. The framework
provides this through contract enforcement, even when it doesn't reduce average costs.

## Test Design

Run the same question multiple times (N=20) with temperature=0 to measure variance:
- Uncontracted: No budget constraints, high variance expected
- Contracted: Fixed budgets, lower variance expected

## Metrics

1. Token variance: stddev(tokens_used)
2. Cost variance: stddev(cost)
3. Quality variance: stddev(quality)
4. Variance reduction: % improvement in predictability

## Expected Result

Based on preliminary analysis: ~50% reduction in token/cost variance
"""

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.research_agent.contracted_agent import ContractedResearchAgent
from benchmarks.research_agent.evaluator import QualityEvaluator
from benchmarks.research_agent.uncontracted_agent import UncontractedResearchAgent

if TYPE_CHECKING:
    from benchmarks.research_agent.agent import ResearchAgent


@dataclass
class VarianceMetrics:
    """Metrics for variance analysis."""

    mean: float
    stddev: float
    min: float
    max: float
    coefficient_of_variation: float  # CV = stddev/mean * 100%

    @classmethod
    def from_values(cls, values: list[float]) -> "VarianceMetrics":
        """Calculate variance metrics from a list of values."""
        mean = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(
            mean=mean,
            stddev=stddev,
            min=min(values),
            max=max(values),
            coefficient_of_variation=(stddev / mean * 100) if mean > 0 else 0.0,
        )


@dataclass
class QuestionVarianceResult:
    """Variance analysis for a single question."""

    question_id: str
    question_text: str
    num_runs: int

    # Uncontracted metrics
    uncontracted_tokens: VarianceMetrics
    uncontracted_cost: VarianceMetrics
    uncontracted_quality: VarianceMetrics

    # Contracted metrics
    contracted_tokens: VarianceMetrics
    contracted_cost: VarianceMetrics
    contracted_quality: VarianceMetrics

    # Variance reduction (% improvement)
    token_variance_reduction: float
    cost_variance_reduction: float
    quality_variance_reduction: float


class VarianceReductionTest:
    """Test framework's ability to reduce cost/token variance."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        output_dir: str = "benchmarks/governance/results",
    ) -> None:
        """Initialize variance reduction test.

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
            use_hybrid_scoring=True,  # Hybrid for consistency
        )

    def run(
        self,
        questions: list[tuple[str, str]],  # List of (id, question)
        num_runs: int = 20,
    ) -> list[QuestionVarianceResult]:
        """Run variance reduction test.

        Args:
            questions: List of (question_id, question_text) tuples
            num_runs: Number of times to run each question (default 20)

        Returns:
            List of QuestionVarianceResult objects
        """
        print("=" * 80)
        print("VARIANCE REDUCTION TEST")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Questions: {len(questions)}")
        print(f"Runs per question: {num_runs}")
        print(f"Total evaluations: {len(questions) * num_runs * 2}")
        print()

        results = []

        for q_id, question in questions:
            print(f"\n{'=' * 80}")
            print(f"Question: {q_id}")
            print(f"{'=' * 80}")
            print(f"Q: {question}")
            print()

            # Run uncontracted N times
            print(f"Running UNCONTRACTED agent {num_runs} times...")
            uncontracted_data = self._run_multiple(
                question, agent_type="uncontracted", num_runs=num_runs
            )

            # Run contracted N times
            print(f"\nRunning CONTRACTED agent {num_runs} times...")
            contracted_data = self._run_multiple(
                question, agent_type="contracted", num_runs=num_runs
            )

            # Analyze variance
            result = self._analyze_variance(q_id, question, uncontracted_data, contracted_data)
            results.append(result)

            # Print summary
            self._print_question_summary(result)

        # Print overall summary
        print(f"\n{'=' * 80}")
        print("OVERALL VARIANCE REDUCTION")
        print(f"{'=' * 80}")
        self._print_overall_summary(results)

        # Save results
        self._save_results(results, num_runs)

        return results

    def _run_multiple(
        self, question: str, agent_type: str, num_runs: int
    ) -> dict[str, list[float]]:
        """Run agent multiple times and collect metrics.

        Args:
            question: Question to answer
            agent_type: "contracted" or "uncontracted"
            num_runs: Number of runs

        Returns:
            Dictionary with lists of tokens, costs, and quality scores
        """
        tokens_list: list[float] = []
        costs_list: list[float] = []
        quality_list: list[float] = []

        for i in range(num_runs):
            agent: ResearchAgent
            if agent_type == "contracted":
                agent = ContractedResearchAgent(model=self.model, strict_mode=False)
            else:
                agent = UncontractedResearchAgent(model=self.model)

            result = agent.research(question)
            quality = self.evaluator.evaluate(question, result.final_answer)

            tokens_list.append(result.total_tokens)
            costs_list.append(result.total_cost)
            quality_list.append(quality.total)

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{num_runs} runs complete")

        return {"tokens": tokens_list, "costs": costs_list, "quality": quality_list}

    def _analyze_variance(
        self,
        q_id: str,
        question: str,
        uncontracted_data: dict[str, list[float]],
        contracted_data: dict[str, list[float]],
    ) -> QuestionVarianceResult:
        """Analyze variance between contracted and uncontracted runs.

        Args:
            q_id: Question ID
            question: Question text
            uncontracted_data: Metrics from uncontracted runs
            contracted_data: Metrics from contracted runs

        Returns:
            QuestionVarianceResult with variance analysis
        """
        # Calculate variance metrics
        unc_tokens = VarianceMetrics.from_values(uncontracted_data["tokens"])
        unc_costs = VarianceMetrics.from_values(uncontracted_data["costs"])
        unc_quality = VarianceMetrics.from_values(uncontracted_data["quality"])

        cont_tokens = VarianceMetrics.from_values(contracted_data["tokens"])
        cont_costs = VarianceMetrics.from_values(contracted_data["costs"])
        cont_quality = VarianceMetrics.from_values(contracted_data["quality"])

        # Calculate variance reduction (% improvement)
        # Positive = contracted has lower variance (good)
        token_reduction = (
            (unc_tokens.stddev - cont_tokens.stddev) / unc_tokens.stddev * 100
            if unc_tokens.stddev > 0
            else 0
        )
        cost_reduction = (
            (unc_costs.stddev - cont_costs.stddev) / unc_costs.stddev * 100
            if unc_costs.stddev > 0
            else 0
        )
        quality_reduction = (
            (unc_quality.stddev - cont_quality.stddev) / unc_quality.stddev * 100
            if unc_quality.stddev > 0
            else 0
        )

        return QuestionVarianceResult(
            question_id=q_id,
            question_text=question,
            num_runs=len(uncontracted_data["tokens"]),
            uncontracted_tokens=unc_tokens,
            uncontracted_cost=unc_costs,
            uncontracted_quality=unc_quality,
            contracted_tokens=cont_tokens,
            contracted_cost=cont_costs,
            contracted_quality=cont_quality,
            token_variance_reduction=token_reduction,
            cost_variance_reduction=cost_reduction,
            quality_variance_reduction=quality_reduction,
        )

    def _print_question_summary(self, result: QuestionVarianceResult) -> None:
        """Print summary for a single question.

        Args:
            result: QuestionVarianceResult to summarize
        """
        print(f"\n{'─' * 80}")
        print("VARIANCE ANALYSIS")
        print(f"{'─' * 80}")

        # Tokens
        print(f"\nTokens ({result.num_runs} runs):")
        print(
            f"  Uncontracted: μ={result.uncontracted_tokens.mean:.0f}, stddev={result.uncontracted_tokens.stddev:.0f}, CV={result.uncontracted_tokens.coefficient_of_variation:.1f}%"
        )
        print(
            f"  Contracted:   μ={result.contracted_tokens.mean:.0f}, stddev={result.contracted_tokens.stddev:.0f}, CV={result.contracted_tokens.coefficient_of_variation:.1f}%"
        )
        print(f"  Variance Reduction: {result.token_variance_reduction:+.1f}%", end="")
        print(" ✅" if result.token_variance_reduction > 0 else " ⚠️")

        # Cost
        print(f"\nCost ({result.num_runs} runs):")
        print(
            f"  Uncontracted: μ=${result.uncontracted_cost.mean:.4f}, stddev=${result.uncontracted_cost.stddev:.4f}, CV={result.uncontracted_cost.coefficient_of_variation:.1f}%"
        )
        print(
            f"  Contracted:   μ=${result.contracted_cost.mean:.4f}, stddev=${result.contracted_cost.stddev:.4f}, CV={result.contracted_cost.coefficient_of_variation:.1f}%"
        )
        print(f"  Variance Reduction: {result.cost_variance_reduction:+.1f}%", end="")
        print(" ✅" if result.cost_variance_reduction > 0 else " ⚠️")

        # Quality
        print(f"\nQuality ({result.num_runs} runs):")
        print(
            f"  Uncontracted: μ={result.uncontracted_quality.mean:.1f}, stddev={result.uncontracted_quality.stddev:.1f}, CV={result.uncontracted_quality.coefficient_of_variation:.1f}%"
        )
        print(
            f"  Contracted:   μ={result.contracted_quality.mean:.1f}, stddev={result.contracted_quality.stddev:.1f}, CV={result.contracted_quality.coefficient_of_variation:.1f}%"
        )
        print(f"  Variance Reduction: {result.quality_variance_reduction:+.1f}%", end="")
        print(" ✅" if result.quality_variance_reduction > 0 else " ⚠️")

    def _print_overall_summary(self, results: list[QuestionVarianceResult]) -> None:
        """Print overall summary across all questions.

        Args:
            results: List of QuestionVarianceResult objects
        """
        # Average variance reductions
        avg_token_reduction = statistics.mean([r.token_variance_reduction for r in results])
        avg_cost_reduction = statistics.mean([r.cost_variance_reduction for r in results])
        avg_quality_reduction = statistics.mean([r.quality_variance_reduction for r in results])

        print(f"\nAverage Variance Reduction (across {len(results)} questions):")
        print(f"  Tokens: {avg_token_reduction:+.1f}%", end="")
        print(" ✅ PREDICTABILITY WIN" if avg_token_reduction > 25 else "")
        print(f"  Cost:   {avg_cost_reduction:+.1f}%", end="")
        print(" ✅ PREDICTABILITY WIN" if avg_cost_reduction > 25 else "")
        print(f"  Quality: {avg_quality_reduction:+.1f}%")

        # Overall CV comparison
        all_unc_cv_tokens = statistics.mean(
            [r.uncontracted_tokens.coefficient_of_variation for r in results]
        )
        all_cont_cv_tokens = statistics.mean(
            [r.contracted_tokens.coefficient_of_variation for r in results]
        )

        print("\nAverage Coefficient of Variation (Token Usage):")
        print(f"  Uncontracted: {all_unc_cv_tokens:.1f}%")
        print(f"  Contracted:   {all_cont_cv_tokens:.1f}%")
        print(f"  Improvement:  {all_unc_cv_tokens - all_cont_cv_tokens:+.1f} percentage points")

        # Key insight
        if avg_token_reduction > 25:
            print(f"\n{'🎯' * 3} KEY FINDING {'🎯' * 3}")
            print(
                f"Contracts reduce token/cost variance by ~{avg_token_reduction:.0f}%, making AI costs "
            )
            print("more predictable and easier to budget for organizations.")

    def _save_results(self, results: list[QuestionVarianceResult], num_runs: int) -> None:
        """Save results to JSON file.

        Args:
            results: List of QuestionVarianceResult objects
            num_runs: Number of runs per question
        """
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"variance_reduction_results_{timestamp}.json"

        # Convert to dict
        results_dict = [asdict(r) for r in results]

        # Calculate overall stats
        overall_stats = {
            "avg_token_variance_reduction": statistics.mean(
                [r.token_variance_reduction for r in results]
            ),
            "avg_cost_variance_reduction": statistics.mean(
                [r.cost_variance_reduction for r in results]
            ),
            "avg_quality_variance_reduction": statistics.mean(
                [r.quality_variance_reduction for r in results]
            ),
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "model": self.model,
                    "num_runs": num_runs,
                    "num_questions": len(results),
                    "overall_stats": overall_stats,
                    "results": results_dict,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_file}")


def main() -> None:
    """Run variance reduction test from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run variance reduction test")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-preview-09-2025",
        help="LLM model to use",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of runs per question (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 5 runs instead of 20",
    )

    args = parser.parse_args()

    # Test questions (subset from research benchmark)
    test_questions = [
        (
            "q1_transformers_vs_ssm",
            "Compare transformer architectures with state space models (SSMs) for sequence modeling. "
            "What are the key differences in computational complexity, long-range dependency handling, "
            "and practical performance on language tasks?",
        ),
        (
            "q3_financial_crisis",
            "What were the primary causes of the 2008 financial crisis, and how did the regulatory "
            "response (Dodd-Frank Act) attempt to prevent similar events in the future?",
        ),
        (
            "q5_microservices",
            "When should a software system use microservices architecture versus a monolithic "
            "architecture? What are the key tradeoffs in terms of complexity, scalability, and "
            "operational overhead?",
        ),
    ]

    num_runs = 5 if args.quick else args.runs

    test = VarianceReductionTest(model=args.model)
    test.run(questions=test_questions, num_runs=num_runs)


if __name__ == "__main__":
    main()
