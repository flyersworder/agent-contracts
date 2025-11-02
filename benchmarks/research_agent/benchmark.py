"""Main benchmark runner for research agent evaluation."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from benchmarks.research_agent.contracted_agent import ContractedResearchAgent
from benchmarks.research_agent.evaluator import QualityEvaluator, QualityScore
from benchmarks.research_agent.questions import RESEARCH_QUESTIONS, ResearchQuestion
from benchmarks.research_agent.uncontracted_agent import UncontractedResearchAgent


@dataclass
class BenchmarkResult:
    """Result of benchmarking an agent on a question.

    Attributes:
        question_id: Question identifier
        question_text: The research question
        agent_type: Type of agent (contracted, uncontracted)
        final_answer: Final synthesized answer
        quality_score: Quality assessment
        total_tokens: Total tokens consumed
        total_reasoning_tokens: Total reasoning tokens used
        total_text_tokens: Total text output tokens used
        total_cost: Total cost in USD
        api_calls: Number of API calls made
        cost_efficiency: Quality score per dollar (quality/cost)
        token_efficiency: Quality score per 1000 tokens (quality/tokens*1000)
    """

    question_id: str
    question_text: str
    agent_type: str
    final_answer: str
    quality_score: QualityScore
    total_tokens: int
    total_reasoning_tokens: int
    total_text_tokens: int
    total_cost: float
    api_calls: int
    cost_efficiency: float
    token_efficiency: float


class ResearchBenchmark:
    """Benchmark for evaluating research agents.

    Compares contracted vs uncontracted agents on:
    - Cost efficiency
    - Token efficiency
    - Quality preservation
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        output_dir: str = "benchmarks/research_agent/results",
    ) -> None:
        """Initialize benchmark.

        Args:
            model: LLM model to use for both agents
            output_dir: Directory to save results
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = QualityEvaluator(judge_model=model)
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        questions: list[ResearchQuestion] | None = None,
        max_questions: int | None = None,
    ) -> None:
        """Run benchmark on research questions.

        Args:
            questions: Questions to evaluate (defaults to all)
            max_questions: Maximum number of questions to run (for testing)
        """
        if questions is None:
            questions = RESEARCH_QUESTIONS

        if max_questions is not None:
            questions = questions[:max_questions]

        print(f"Running benchmark on {len(questions)} questions...")
        print(f"Model: {self.model}")
        print(f"Output: {self.output_dir}")
        print()

        for i, question in enumerate(questions, 1):
            print(f"\n{'=' * 80}")
            print(f"Question {i}/{len(questions)}: {question.id}")
            print(f"Domain: {question.domain}")
            print(f"Difficulty: {question.difficulty}/5")
            print(f"{'=' * 80}\n")
            print(f"Q: {question.question}\n")

            # Run uncontracted agent
            print("Running UNCONTRACTED agent (baseline)...")
            uncontracted_result = self._run_uncontracted(question)
            self.results.append(uncontracted_result)
            self._print_result_summary(uncontracted_result)

            # Run contracted agent
            print("\nRunning CONTRACTED agent (optimized)...")
            contracted_result = self._run_contracted(question)
            self.results.append(contracted_result)
            self._print_result_summary(contracted_result)

            # Print comparison
            self._print_comparison(uncontracted_result, contracted_result)

        # Generate final report
        print(f"\n{'=' * 80}")
        print("BENCHMARK COMPLETE")
        print(f"{'=' * 80}\n")
        self._generate_report()

    def _run_uncontracted(self, question: ResearchQuestion) -> BenchmarkResult:
        """Run uncontracted agent on a question.

        Args:
            question: Research question

        Returns:
            BenchmarkResult
        """
        agent = UncontractedResearchAgent(model=self.model)
        result = agent.research(question.question)

        # Evaluate quality
        quality = self.evaluator.evaluate(question.question, result.final_answer)

        # Calculate efficiency metrics
        cost_efficiency = quality.total / result.total_cost if result.total_cost > 0 else 0
        token_efficiency = (
            quality.total / result.total_tokens * 1000 if result.total_tokens > 0 else 0
        )

        return BenchmarkResult(
            question_id=question.id,
            question_text=question.question,
            agent_type="uncontracted",
            final_answer=result.final_answer,
            quality_score=quality,
            total_tokens=result.total_tokens,
            total_reasoning_tokens=result.total_reasoning_tokens,
            total_text_tokens=result.total_text_tokens,
            total_cost=result.total_cost,
            api_calls=result.api_calls,
            cost_efficiency=cost_efficiency,
            token_efficiency=token_efficiency,
        )

    def _run_contracted(self, question: ResearchQuestion) -> BenchmarkResult:
        """Run contracted agent on a question.

        Args:
            question: Research question

        Returns:
            BenchmarkResult
        """
        agent = ContractedResearchAgent(model=self.model, strict_mode=False)
        result = agent.research(question.question)

        # Evaluate quality
        quality = self.evaluator.evaluate(question.question, result.final_answer)

        # Calculate efficiency metrics
        cost_efficiency = quality.total / result.total_cost if result.total_cost > 0 else 0
        token_efficiency = (
            quality.total / result.total_tokens * 1000 if result.total_tokens > 0 else 0
        )

        return BenchmarkResult(
            question_id=question.id,
            question_text=question.question,
            agent_type="contracted",
            final_answer=result.final_answer,
            quality_score=quality,
            total_tokens=result.total_tokens,
            total_reasoning_tokens=result.total_reasoning_tokens,
            total_text_tokens=result.total_text_tokens,
            total_cost=result.total_cost,
            api_calls=result.api_calls,
            cost_efficiency=cost_efficiency,
            token_efficiency=token_efficiency,
        )

    def _print_result_summary(self, result: BenchmarkResult) -> None:
        """Print summary of a single result.

        Args:
            result: BenchmarkResult to summarize
        """
        print(f"  Type: {result.agent_type.upper()}")
        print(f"  Quality: {result.quality_score.total:.1f}/100")
        print(f"    - Accuracy: {result.quality_score.accuracy:.1f}/10")
        print(f"    - Completeness: {result.quality_score.completeness:.1f}/10")
        print(f"    - Coherence: {result.quality_score.coherence:.1f}/10")
        print("  Resources:")
        print(f"    - Total tokens: {result.total_tokens:,}")
        print(f"    - Reasoning tokens: {result.total_reasoning_tokens:,}")
        print(f"    - Text tokens: {result.total_text_tokens:,}")
        print(f"    - API calls: {result.api_calls}")
        print(f"    - Cost: ${result.total_cost:.4f}")
        print("  Efficiency:")
        print(f"    - Cost efficiency: {result.cost_efficiency:.1f} pts/$")
        print(f"    - Token efficiency: {result.token_efficiency:.2f} pts/1k tokens")

    def _print_comparison(self, uncontracted: BenchmarkResult, contracted: BenchmarkResult) -> None:
        """Print comparison between contracted and uncontracted.

        Args:
            uncontracted: Uncontracted agent result
            contracted: Contracted agent result
        """
        print(f"\n{'─' * 80}")
        print("COMPARISON:")
        print(f"{'─' * 80}")

        # Quality difference
        quality_diff = contracted.quality_score.total - uncontracted.quality_score.total
        quality_pct = quality_diff / uncontracted.quality_score.total * 100

        print(
            f"Quality: {contracted.quality_score.total:.1f} vs {uncontracted.quality_score.total:.1f} ",
            end="",
        )
        if quality_diff > 0:
            print(f"(+{quality_diff:.1f}, +{quality_pct:.1f}%)")
        else:
            print(f"({quality_diff:.1f}, {quality_pct:.1f}%)")

        # Token savings
        token_savings = uncontracted.total_tokens - contracted.total_tokens
        token_pct = token_savings / uncontracted.total_tokens * 100
        print(f"Tokens: {contracted.total_tokens:,} vs {uncontracted.total_tokens:,} ", end="")
        print(f"({token_savings:,} saved, {token_pct:.1f}% reduction)")

        # Reasoning token savings
        reasoning_savings = uncontracted.total_reasoning_tokens - contracted.total_reasoning_tokens
        if uncontracted.total_reasoning_tokens > 0:
            reasoning_pct = reasoning_savings / uncontracted.total_reasoning_tokens * 100
            print(
                f"Reasoning tokens: {contracted.total_reasoning_tokens:,} vs {uncontracted.total_reasoning_tokens:,} ",
                end="",
            )
            print(f"({reasoning_savings:,} saved, {reasoning_pct:.1f}% reduction)")

        # Cost savings
        cost_savings = uncontracted.total_cost - contracted.total_cost
        cost_pct = (
            cost_savings / uncontracted.total_cost * 100 if uncontracted.total_cost > 0 else 0
        )
        print(f"Cost: ${contracted.total_cost:.4f} vs ${uncontracted.total_cost:.4f} ", end="")
        print(f"(${cost_savings:.4f} saved, {cost_pct:.1f}% reduction)")

        # Cost efficiency improvement
        efficiency_gain = contracted.cost_efficiency - uncontracted.cost_efficiency
        efficiency_pct = (
            efficiency_gain / uncontracted.cost_efficiency * 100
            if uncontracted.cost_efficiency > 0
            else 0
        )
        print(
            f"Cost efficiency: {contracted.cost_efficiency:.1f} vs {uncontracted.cost_efficiency:.1f} pts/$ ",
            end="",
        )
        print(f"(+{efficiency_gain:.1f}, +{efficiency_pct:.1f}% better)")

    def _generate_report(self) -> None:
        """Generate final benchmark report."""
        # Calculate aggregate statistics
        uncontracted_results = [r for r in self.results if r.agent_type == "uncontracted"]
        contracted_results = [r for r in self.results if r.agent_type == "contracted"]

        # Average metrics
        avg_uncontracted_quality = sum(r.quality_score.total for r in uncontracted_results) / len(
            uncontracted_results
        )
        avg_contracted_quality = sum(r.quality_score.total for r in contracted_results) / len(
            contracted_results
        )

        avg_uncontracted_tokens = sum(r.total_tokens for r in uncontracted_results) / len(
            uncontracted_results
        )
        avg_contracted_tokens = sum(r.total_tokens for r in contracted_results) / len(
            contracted_results
        )

        avg_uncontracted_cost = sum(r.total_cost for r in uncontracted_results) / len(
            uncontracted_results
        )
        avg_contracted_cost = sum(r.total_cost for r in contracted_results) / len(
            contracted_results
        )

        avg_uncontracted_efficiency = sum(r.cost_efficiency for r in uncontracted_results) / len(
            uncontracted_results
        )
        avg_contracted_efficiency = sum(r.cost_efficiency for r in contracted_results) / len(
            contracted_results
        )

        # Print summary
        print("AGGREGATE RESULTS:")
        print(f"  Questions evaluated: {len(uncontracted_results)}")
        print()
        print("Average Quality:")
        print(f"  Uncontracted: {avg_uncontracted_quality:.1f}/100")
        print(f"  Contracted: {avg_contracted_quality:.1f}/100")
        quality_diff = avg_contracted_quality - avg_uncontracted_quality
        print(f"  Difference: {quality_diff:+.1f} points")
        print()
        print("Average Resource Usage:")
        print("  Tokens:")
        print(f"    Uncontracted: {avg_uncontracted_tokens:,.0f}")
        print(f"    Contracted: {avg_contracted_tokens:,.0f}")
        token_savings_pct = (
            (avg_uncontracted_tokens - avg_contracted_tokens) / avg_uncontracted_tokens * 100
        )
        print(f"    Savings: {token_savings_pct:.1f}%")
        print("  Cost:")
        print(f"    Uncontracted: ${avg_uncontracted_cost:.4f}")
        print(f"    Contracted: ${avg_contracted_cost:.4f}")
        cost_savings_pct = (
            (avg_uncontracted_cost - avg_contracted_cost) / avg_uncontracted_cost * 100
        )
        print(f"    Savings: {cost_savings_pct:.1f}%")
        print()
        print("Average Cost Efficiency (quality per $):")
        print(f"  Uncontracted: {avg_uncontracted_efficiency:.1f} pts/$")
        print(f"  Contracted: {avg_contracted_efficiency:.1f} pts/$")
        efficiency_improvement = (
            (avg_contracted_efficiency - avg_uncontracted_efficiency)
            / avg_uncontracted_efficiency
            * 100
        )
        print(f"  Improvement: {efficiency_improvement:+.1f}%")

        # Save detailed results to JSON
        self._save_results()

    def _save_results(self) -> None:
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        # Convert results to dict
        results_dict = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert quality_score to dict
            result_dict["quality_score"] = asdict(result.quality_score)
            results_dict.append(result_dict)

        # Save to file
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "model": self.model,
                    "results": results_dict,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed results saved to: {output_file}")


def main() -> None:
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run research agent benchmark")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-preview-09-2025",
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to run (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/research_agent/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    benchmark = ResearchBenchmark(model=args.model, output_dir=args.output_dir)
    benchmark.run(max_questions=args.max_questions)


if __name__ == "__main__":
    main()
