"""Quality evaluator for COINE 2026 research pipeline experiment.

This module provides quality evaluation for generated research reports,
using the IndeterminacyAwareEvaluator from the research_agent module.

The evaluator assesses:
- Accuracy: Factual correctness and technical accuracy
- Completeness: Coverage of the topic and key aspects
- Coherence: Organization and clarity of the report
"""

from dataclasses import dataclass

# Import the IndeterminacyAwareEvaluator from parent evaluation module
from ..indeterminacy_evaluator import (
    IndeterminacyAwareEvaluator,
    IndeterminacyAwareScore,
    MultiLabelScore,
    ResponseSet,
    decision_consistency,
    mse_srs_srs,
    prevalence_bias,
)
from .topics import ResearchTopic


@dataclass
class ReportQualityScore:
    """Quality score for a research report.

    Combines IndeterminacyAwareEvaluator scores with report-specific metrics.

    Attributes:
        indeterminacy_score: Full score from IndeterminacyAwareEvaluator
        word_count: Number of words in the report
        citation_count: Number of citations/references
        has_introduction: Whether report has an introduction section
        has_conclusion: Whether report has a conclusion section
        covers_key_aspects: Proportion of key aspects covered (0-1)
        overall_score: Combined overall score (0-100)
    """

    indeterminacy_score: IndeterminacyAwareScore
    word_count: int
    citation_count: int
    has_introduction: bool
    has_conclusion: bool
    covers_key_aspects: float
    overall_score: float


class ResearchReportEvaluator:
    """Evaluator for research reports using IndeterminacyAwareEvaluator.

    This evaluator combines LLM-as-judge evaluation with rule-based metrics
    specific to research reports (word count, citations, structure).

    Attributes:
        llm_evaluator: IndeterminacyAwareEvaluator instance
        min_words: Minimum word count for full score (default 2000)
        min_citations: Minimum citation count for full score (default 5)
    """

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash-lite",
        num_judges: int = 3,
        min_words: int = 2000,
        min_citations: int = 5,
    ) -> None:
        """Initialize the research report evaluator.

        Args:
            judge_model: LLM model for quality evaluation (default: gemini-2.5-flash-lite)
            num_judges: Number of independent evaluations to aggregate
            min_words: Target word count for full score
            min_citations: Target citation count for full score
        """
        self.llm_evaluator = IndeterminacyAwareEvaluator(
            judge_model=judge_model,
            num_judges=num_judges,
            use_hybrid_scoring=True,
        )
        self.min_words = min_words
        self.min_citations = min_citations

    def evaluate(self, topic: ResearchTopic, report: str) -> ReportQualityScore:
        """Evaluate a research report.

        Args:
            topic: The research topic
            report: The generated report text

        Returns:
            ReportQualityScore with detailed evaluation
        """
        # Build question from topic for LLM evaluation
        question = f"""Research Topic: {topic.title}

Description: {topic.description}

Key Aspects to Cover:
{chr(10).join(f"- {aspect}" for aspect in topic.key_aspects)}"""

        # Get indeterminacy-aware score
        indeterminacy_score = self.llm_evaluator.evaluate(question, report)

        # Calculate report-specific metrics
        word_count = len(report.split())
        citation_count = self._count_citations(report)
        has_introduction = self._has_section(report, ["introduction", "overview", "background"])
        has_conclusion = self._has_section(report, ["conclusion", "summary", "final thoughts"])
        covers_key_aspects = self._calculate_aspect_coverage(report, topic.key_aspects)

        # Calculate overall score (weighted combination)
        # - 50% LLM quality score (accuracy, completeness, coherence)
        # - 20% word count score
        # - 15% citation score
        # - 10% structure score (intro + conclusion)
        # - 5% key aspect coverage

        llm_score = indeterminacy_score.total  # 0-100
        word_score = min(100, (word_count / self.min_words) * 100)
        citation_score = min(100, (citation_count / self.min_citations) * 100)
        structure_score = 50 * has_introduction + 50 * has_conclusion
        aspect_score = covers_key_aspects * 100

        overall_score = (
            0.50 * llm_score
            + 0.20 * word_score
            + 0.15 * citation_score
            + 0.10 * structure_score
            + 0.05 * aspect_score
        )

        return ReportQualityScore(
            indeterminacy_score=indeterminacy_score,
            word_count=word_count,
            citation_count=citation_count,
            has_introduction=has_introduction,
            has_conclusion=has_conclusion,
            covers_key_aspects=covers_key_aspects,
            overall_score=overall_score,
        )

    def _count_citations(self, text: str) -> int:
        """Count citations in text (URLs or [n] references)."""
        import re

        # Count URLs
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)

        # Count [n] style references
        ref_pattern = r"\[\d+\]"
        refs = re.findall(ref_pattern, text)

        return len(set(urls)) + len(set(refs))

    def _has_section(self, text: str, keywords: list[str]) -> bool:
        """Check if text has a section with any of the given keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    def _calculate_aspect_coverage(self, text: str, key_aspects: list[str]) -> float:
        """Calculate what proportion of key aspects are covered."""
        if not key_aspects:
            return 1.0

        text_lower = text.lower()
        covered = 0

        for aspect in key_aspects:
            # Extract key terms from aspect
            terms = aspect.lower().split()
            # Check if at least half of the terms appear
            matches = sum(1 for term in terms if len(term) > 3 and term in text_lower)
            if matches >= len(terms) / 2:
                covered += 1

        return covered / len(key_aspects)


# Re-export for convenience
__all__ = [
    "IndeterminacyAwareEvaluator",
    "IndeterminacyAwareScore",
    "MultiLabelScore",
    "ReportQualityScore",
    "ResearchReportEvaluator",
    "ResponseSet",
    "decision_consistency",
    "mse_srs_srs",
    "prevalence_bias",
]
