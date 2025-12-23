"""Indeterminacy-aware LLM-as-judge evaluator.

This module implements the rating indeterminacy framework from:
"Validating LLM-as-a-Judge Systems under Rating Indeterminacy" (NeurIPS 2025)
by Guerdan, Barocas, Holstein, Wallach, Wu, and Chouldechova.

Key concepts:
- Response Set Elicitation: Ask judges to select ALL reasonable options, not just one
- Multi-label Vectors (ω): Probability that each rating option is reasonable
- Indeterminacy Signal: Judge disagreement indicates genuine ambiguity, not noise
- MSE(srs/srs): Recommended metric for comparing judge vs human ratings

Reference: https://github.com/lguerdan/indeterminacy
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from litellm import completion

# Rating ranges for each quality dimension
ACCURACY_RANGES = {
    "A": (0, 3, "Major factual errors, significant misconceptions"),
    "B": (4, 6, "Some inaccuracies, minor errors present"),
    "C": (7, 8, "Mostly accurate, few minor issues"),
    "D": (9, 10, "Highly accurate, technically correct"),
}

COMPLETENESS_RANGES = {
    "A": (0, 3, "Missing most key points, very incomplete"),
    "B": (4, 6, "Covers some aspects, misses important details"),
    "C": (7, 8, "Covers most aspects, minor gaps"),
    "D": (9, 10, "Comprehensive, addresses all aspects"),
}

COHERENCE_RANGES = {
    "A": (0, 3, "Disorganized, hard to follow"),
    "B": (4, 6, "Some structure, but unclear flow"),
    "C": (7, 8, "Well-organized, mostly clear"),
    "D": (9, 10, "Excellent structure, very clear"),
}


@dataclass
class ResponseSet:
    """A response set representing all options deemed reasonable by a judge.

    In the indeterminacy framework, a response set S ⊆ O contains all options
    that a rater considers "reasonable" for an item. This captures genuine
    ambiguity rather than forcing an arbitrary single choice.

    Attributes:
        selected_options: Set of option letters selected (e.g., {"C", "D"})
        dimension: Which quality dimension this applies to
        confidence: Optional confidence in the selection (0-1)
    """

    selected_options: set[str]
    dimension: str
    confidence: float = 1.0

    @property
    def is_determinate(self) -> bool:
        """Check if only one option was selected (no ambiguity)."""
        return len(self.selected_options) == 1

    @property
    def indeterminacy_level(self) -> float:
        """Measure of ambiguity: 1.0 = single choice, >1 = multiple reasonable options."""
        return float(len(self.selected_options))

    def to_multilabel_vector(self, options: list[str]) -> np.ndarray:
        """Convert to binary multi-label vector.

        Args:
            options: List of all possible options (e.g., ["A", "B", "C", "D"])

        Returns:
            Binary vector where 1 indicates option was selected as reasonable
        """
        return np.array([1.0 if opt in self.selected_options else 0.0 for opt in options])


@dataclass
class MultiLabelScore:
    """Multi-label probability vector for a quality dimension.

    This is the ω (omega) vector from the paper: P(option_k ∈ response_set)
    for each option k. Aggregated across multiple judges.

    Attributes:
        dimension: Quality dimension (accuracy, completeness, coherence)
        omega: Probability vector [P(A reasonable), P(B), P(C), P(D)]
        point_estimate: Traditional single-point score (for backward compatibility)
        indeterminacy: Average number of options selected per judge
        raw_response_sets: Individual response sets from each judge
    """

    dimension: str
    omega: np.ndarray  # Multi-label probability vector
    point_estimate: float  # Traditional score (0-10)
    indeterminacy: float  # Average response set size
    raw_response_sets: list[ResponseSet] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        """Check if there's significant rating indeterminacy."""
        return self.indeterminacy > 1.2  # More than 1 option on average

    def to_hard_multilabel(self, tau: float = 0.5) -> np.ndarray:
        """Convert to hard multi-label using threshold τ.

        Args:
            tau: Threshold for including an option (default 0.5 = majority)

        Returns:
            Binary vector where 1 indicates P(option) > tau
        """
        return (self.omega >= tau).astype(float)


@dataclass
class IndeterminacyAwareScore:
    """Quality assessment with full indeterminacy information.

    This extends the original QualityScore to include:
    - Response set information for each dimension
    - Multi-label probability vectors (ω)
    - Indeterminacy metrics
    - Judge disagreement signals

    Attributes:
        accuracy: Multi-label score for accuracy dimension
        completeness: Multi-label score for completeness dimension
        coherence: Multi-label score for coherence dimension
        total: Traditional total score (0-100) for backward compatibility
        explanation: Aggregated explanation from judges
        judge_agreement: Measure of agreement across judges (0-1)
    """

    accuracy: MultiLabelScore
    completeness: MultiLabelScore
    coherence: MultiLabelScore
    total: float
    explanation: str
    judge_agreement: float = 1.0

    @property
    def point_estimates(self) -> dict[str, float]:
        """Get traditional point estimates for backward compatibility."""
        return {
            "accuracy": self.accuracy.point_estimate,
            "completeness": self.completeness.point_estimate,
            "coherence": self.coherence.point_estimate,
            "total": self.total,
        }

    @property
    def indeterminacy_summary(self) -> dict[str, float]:
        """Summary of indeterminacy across dimensions."""
        return {
            "accuracy": self.accuracy.indeterminacy,
            "completeness": self.completeness.indeterminacy,
            "coherence": self.coherence.indeterminacy,
            "overall": (
                self.accuracy.indeterminacy
                + self.completeness.indeterminacy
                + self.coherence.indeterminacy
            )
            / 3,
        }

    @property
    def omega_vectors(self) -> dict[str, np.ndarray]:
        """Get all multi-label probability vectors."""
        return {
            "accuracy": self.accuracy.omega,
            "completeness": self.completeness.omega,
            "coherence": self.coherence.omega,
        }

    def is_dimension_ambiguous(self, dimension: str) -> bool:
        """Check if a specific dimension has rating indeterminacy."""
        score: MultiLabelScore = getattr(self, dimension)
        return bool(score.is_ambiguous)


class IndeterminacyAwareEvaluator:
    """Evaluates research quality using indeterminacy-aware LLM-as-judge.

    This evaluator implements the response set elicitation framework from
    the NeurIPS 2025 paper "Validating LLM-as-a-Judge Systems under Rating
    Indeterminacy".

    Key differences from traditional evaluation:
    1. Asks judges to select ALL reasonable rating ranges, not just one
    2. Tracks multi-label probability vectors (ω) for each dimension
    3. Interprets judge disagreement as a signal of genuine ambiguity
    4. Uses MSE on response sets instead of categorical agreement

    Attributes:
        judge_model: LLM model to use for evaluation
        num_judges: Number of evaluation passes (default 3)
        use_hybrid_scoring: Whether to combine with rule-based metrics
        tau: Classification threshold for downstream decisions (default 0.5)
    """

    OPTIONS: ClassVar[list[str]] = ["A", "B", "C", "D"]

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.0-flash",
        num_judges: int = 3,
        use_hybrid_scoring: bool = True,
        tau: float = 0.5,
    ) -> None:
        """Initialize indeterminacy-aware evaluator.

        Args:
            judge_model: LLM model to use for evaluation
            num_judges: Number of independent evaluations to aggregate
            use_hybrid_scoring: Whether to combine with rule-based metrics
            tau: Threshold for hard multi-label classification
        """
        self.judge_model = judge_model
        self.num_judges = num_judges
        self.use_hybrid_scoring = use_hybrid_scoring
        self.tau = tau

    def evaluate(self, question: str, answer: str) -> IndeterminacyAwareScore:
        """Evaluate answer quality with indeterminacy awareness.

        Args:
            question: Original research question
            answer: Answer to evaluate

        Returns:
            IndeterminacyAwareScore with full response set information
        """
        # Collect response sets from multiple judges
        all_response_sets: dict[str, list[ResponseSet]] = {
            "accuracy": [],
            "completeness": [],
            "coherence": [],
        }
        all_explanations: list[str] = []

        for _ in range(self.num_judges):
            judge_result = self._evaluate_single_judge(question, answer)
            for dim in ["accuracy", "completeness", "coherence"]:
                all_response_sets[dim].append(judge_result[dim])
            all_explanations.append(judge_result["explanation"])

        # Aggregate response sets into multi-label scores
        accuracy_score = self._aggregate_response_sets(all_response_sets["accuracy"], "accuracy")
        completeness_score = self._aggregate_response_sets(
            all_response_sets["completeness"], "completeness"
        )
        coherence_score = self._aggregate_response_sets(all_response_sets["coherence"], "coherence")

        # Apply hybrid scoring if enabled
        if self.use_hybrid_scoring:
            rule_scores = self._calculate_rule_based_scores(question, answer)
            accuracy_score = self._apply_hybrid_adjustment(accuracy_score, rule_scores["accuracy"])
            completeness_score = self._apply_hybrid_adjustment(
                completeness_score, rule_scores["completeness"]
            )
            coherence_score = self._apply_hybrid_adjustment(
                coherence_score, rule_scores["coherence"]
            )

        # Calculate total (for backward compatibility)
        total = (
            (
                accuracy_score.point_estimate
                + completeness_score.point_estimate
                + coherence_score.point_estimate
            )
            / 30
            * 100
        )

        # Calculate judge agreement
        judge_agreement = self._calculate_judge_agreement(all_response_sets)

        # Select representative explanation
        explanation = self._select_explanation(all_explanations, accuracy_score.indeterminacy)

        return IndeterminacyAwareScore(
            accuracy=accuracy_score,
            completeness=completeness_score,
            coherence=coherence_score,
            total=total,
            explanation=explanation,
            judge_agreement=judge_agreement,
        )

    def _evaluate_single_judge(self, question: str, answer: str) -> dict[str, Any]:
        """Run a single judge evaluation with response set elicitation.

        Args:
            question: Research question
            answer: Answer to evaluate

        Returns:
            Dict with ResponseSet for each dimension plus explanation
        """
        prompt = self._build_response_set_prompt(question, answer)

        response = completion(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slight temperature for response set diversity
        )

        content = response["choices"][0]["message"]["content"]

        return self._parse_response_set_evaluation(content)

    def _build_response_set_prompt(self, question: str, answer: str) -> str:
        """Build prompt for response set elicitation.

        This follows the CMU paper's recommendation: ask judges to select
        ALL options that could reasonably apply, not just one.
        """
        return f"""You are an expert evaluator assessing the quality of research answers.

For each dimension, select ALL rating ranges that could REASONABLY apply to this answer.
If multiple ranges seem valid given different interpretations, select all of them.
This helps capture genuine ambiguity in the rating task.

Research Question:
{question}

Answer to Evaluate:
{answer}

---

**ACCURACY** - Are the facts, explanations, and technical details correct?
Select ALL ranges that could reasonably apply:
A. [0-3] Major factual errors, significant misconceptions
B. [4-6] Some inaccuracies, minor errors present
C. [7-8] Mostly accurate, few minor issues
D. [9-10] Highly accurate, technically correct

**COMPLETENESS** - Does the answer address all aspects of the question?
Select ALL ranges that could reasonably apply:
A. [0-3] Missing most key points, very incomplete
B. [4-6] Covers some aspects, misses important details
C. [7-8] Covers most aspects, minor gaps
D. [9-10] Comprehensive, addresses all aspects

**COHERENCE** - Is the answer well-structured and easy to understand?
Select ALL ranges that could reasonably apply:
A. [0-3] Disorganized, hard to follow
B. [4-6] Some structure, but unclear flow
C. [7-8] Well-organized, mostly clear
D. [9-10] Excellent structure, very clear

---

RESPONSE FORMAT:
Accuracy: [letters, e.g., "CD" or "C" or "BCD"]
Completeness: [letters, e.g., "D" or "CD"]
Coherence: [letters, e.g., "CD" or "D"]

Explanation:
[2-3 sentences explaining your rating, especially noting any ambiguity]"""

    def _parse_response_set_evaluation(self, content: str) -> dict[str, Any]:
        """Parse response set evaluation from LLM output.

        Args:
            content: Raw LLM response

        Returns:
            Dict with ResponseSet for each dimension plus explanation
        """
        lines = content.strip().split("\n")

        result: dict[str, Any] = {
            "accuracy": ResponseSet(set(), "accuracy"),
            "completeness": ResponseSet(set(), "completeness"),
            "coherence": ResponseSet(set(), "coherence"),
            "explanation": "",
        }

        explanation_started = False

        for line in lines:
            line = line.strip()

            if line.lower().startswith("accuracy:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["accuracy"] = ResponseSet(letters, "accuracy")

            elif line.lower().startswith("completeness:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["completeness"] = ResponseSet(letters, "completeness")

            elif line.lower().startswith("coherence:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["coherence"] = ResponseSet(letters, "coherence")

            elif line.lower().startswith("explanation:"):
                explanation_started = True
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    result["explanation"] = parts[1].strip()

            elif explanation_started and line:
                result["explanation"] += " " + line

        # Validate: ensure at least one option selected per dimension
        for dim in ["accuracy", "completeness", "coherence"]:
            if not result[dim].selected_options:
                # Default to middle option if parsing fails
                result[dim] = ResponseSet({"C"}, dim)

        return result

    def _extract_letters(self, text: str) -> set[str]:
        """Extract valid option letters from text.

        Args:
            text: Text containing letter selections

        Returns:
            Set of valid option letters (A, B, C, D)
        """
        # Find all letters A-D in the text
        letters = set(re.findall(r"[A-D]", text.upper()))
        return letters if letters else {"C"}  # Default to C if none found

    def _aggregate_response_sets(
        self, response_sets: list[ResponseSet], dimension: str
    ) -> MultiLabelScore:
        """Aggregate multiple response sets into a MultiLabelScore.

        This computes the multi-label probability vector ω:
        ω_k = P(option k is in the response set)

        Args:
            response_sets: List of ResponseSet from multiple judges
            dimension: Quality dimension name

        Returns:
            MultiLabelScore with aggregated omega vector
        """
        n_judges = len(response_sets)
        options = self.OPTIONS

        # Compute omega: probability each option is deemed reasonable
        omega = np.zeros(len(options))
        for rs in response_sets:
            omega += rs.to_multilabel_vector(options)
        omega /= n_judges

        # Compute point estimate (weighted average of range midpoints)
        ranges = self._get_ranges_for_dimension(dimension)
        point_estimate = 0.0
        total_weight = 0.0
        for i, opt in enumerate(options):
            if omega[i] > 0:
                low, high, _ = ranges[opt]
                midpoint = (low + high) / 2
                point_estimate += omega[i] * midpoint
                total_weight += omega[i]
        if total_weight > 0:
            point_estimate /= total_weight

        # Compute indeterminacy (average response set size)
        indeterminacy = statistics.mean(rs.indeterminacy_level for rs in response_sets)

        return MultiLabelScore(
            dimension=dimension,
            omega=omega,
            point_estimate=point_estimate,
            indeterminacy=indeterminacy,
            raw_response_sets=response_sets,
        )

    def _get_ranges_for_dimension(self, dimension: str) -> dict[str, tuple[int, int, str]]:
        """Get rating ranges for a dimension."""
        dimension_ranges = {
            "accuracy": ACCURACY_RANGES,
            "completeness": COMPLETENESS_RANGES,
            "coherence": COHERENCE_RANGES,
        }
        return dimension_ranges.get(dimension, ACCURACY_RANGES)

    def _apply_hybrid_adjustment(
        self, score: MultiLabelScore, rule_score: float
    ) -> MultiLabelScore:
        """Apply hybrid adjustment (60% LLM + 40% rule-based).

        Args:
            score: Original MultiLabelScore from LLM judges
            rule_score: Rule-based score (0-10)

        Returns:
            Adjusted MultiLabelScore
        """
        adjusted_point = 0.6 * score.point_estimate + 0.4 * rule_score
        return MultiLabelScore(
            dimension=score.dimension,
            omega=score.omega,  # Keep omega unchanged
            point_estimate=adjusted_point,
            indeterminacy=score.indeterminacy,
            raw_response_sets=score.raw_response_sets,
        )

    def _calculate_judge_agreement(self, all_response_sets: dict[str, list[ResponseSet]]) -> float:
        """Calculate agreement across judges using Jaccard similarity.

        Higher agreement means judges selected similar response sets.
        Low agreement indicates genuine rating indeterminacy.

        Args:
            all_response_sets: Response sets from all judges for each dimension

        Returns:
            Average Jaccard similarity across all dimension pairs (0-1)
        """
        similarities = []

        for _dim, response_sets in all_response_sets.items():
            for i in range(len(response_sets)):
                for j in range(i + 1, len(response_sets)):
                    set_i = response_sets[i].selected_options
                    set_j = response_sets[j].selected_options

                    if set_i or set_j:
                        intersection = len(set_i & set_j)
                        union = len(set_i | set_j)
                        similarities.append(intersection / union if union > 0 else 1.0)

        return statistics.mean(similarities) if similarities else 1.0

    def _select_explanation(self, explanations: list[str], indeterminacy: float) -> str:
        """Select representative explanation based on indeterminacy.

        Args:
            explanations: All explanations from judges
            indeterminacy: Average indeterminacy level

        Returns:
            Selected explanation with indeterminacy note
        """
        # Use the longest explanation as it's likely most detailed
        selected = max(explanations, key=len) if explanations else ""

        if indeterminacy > 1.5:
            prefix = f"[High indeterminacy: {indeterminacy:.1f} options/dimension] "
        elif indeterminacy > 1.2:
            prefix = f"[Moderate indeterminacy: {indeterminacy:.1f}] "
        else:
            prefix = f"[Low indeterminacy: {indeterminacy:.1f}] "

        return prefix + selected.strip()

    def _calculate_rule_based_scores(self, question: str, answer: str) -> dict[str, float]:
        """Calculate rule-based quality metrics (same as original evaluator)."""
        # Accuracy proxy: Presence of numbers, technical terms, specifics
        numbers = len(re.findall(r"\d+\.?\d*%?", answer))
        technical_markers = len(
            re.findall(
                r"\b(algorithm|model|system|framework|approach|method|technique|"
                r"architecture|protocol|mechanism|implementation|optimization|"
                r"constraint|metric|analysis|evaluation|comparison|tradeoff|"
                r"advantage|disadvantage|limitation|challenge|benefit)\b",
                answer,
                re.IGNORECASE,
            )
        )
        accuracy = min(10.0, (numbers / 5 + technical_markers / 10) / 2 * 10)

        # Completeness proxy: Answer length and section structure
        words = len(answer.split())
        has_multiple_paragraphs = answer.count("\n\n") >= 2
        has_headings = bool(re.search(r"^#+\s+|\*\*.*\*\*", answer, re.MULTILINE))
        has_lists = bool(re.search(r"^\s*[-*\d]+[\.)]\s+", answer, re.MULTILINE))

        length_score = min(10.0, words / 500 * 10)
        structure_bonus = 2.0 * has_multiple_paragraphs + 1.0 * has_headings + 1.0 * has_lists
        completeness = min(10.0, (length_score + structure_bonus) / 2)

        # Coherence proxy: Average sentence length
        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = statistics.mean(len(s.split()) for s in sentences)
            if 15 <= avg_sentence_length <= 25:
                coherence = 10.0
            elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
                coherence = 8.0
            elif 5 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 40:
                coherence = 6.0
            else:
                coherence = 4.0
        else:
            coherence = 5.0

        return {"accuracy": accuracy, "completeness": completeness, "coherence": coherence}


# Metrics functions following the CMU paper
def mse_srs_srs(score1: IndeterminacyAwareScore, score2: IndeterminacyAwareScore) -> float:
    """Compute MSE between two IndeterminacyAwareScores on response sets.

    This is the recommended metric from the NeurIPS 2025 paper for
    comparing judge systems under rating indeterminacy.

    MSE(srs/srs) = E[||ω_judge - ω_human||²]

    Args:
        score1: First score (e.g., from judge)
        score2: Second score (e.g., from human or reference)

    Returns:
        Mean squared error across all dimensions
    """
    mse_total = 0.0
    for dim in ["accuracy", "completeness", "coherence"]:
        omega1 = getattr(score1, dim).omega
        omega2 = getattr(score2, dim).omega
        mse_total += np.sum((omega1 - omega2) ** 2)
    return mse_total / 3


def decision_consistency(
    score1: IndeterminacyAwareScore,
    score2: IndeterminacyAwareScore,
    tau: float = 0.5,
    positive_options: list[int] | None = None,
) -> float:
    """Compute decision consistency between two scores.

    Measures whether both scores would lead to the same downstream
    decision (e.g., pass/fail) at threshold τ.

    Args:
        score1: First score
        score2: Second score
        tau: Classification threshold
        positive_options: Indices of "positive" options (default: [2, 3] for C, D)

    Returns:
        Proportion of dimensions with matching decisions (0-1)
    """
    if positive_options is None:
        positive_options = [2, 3]  # C and D are "positive" (high quality)

    matches = 0
    for dim in ["accuracy", "completeness", "coherence"]:
        omega1 = getattr(score1, dim).omega
        omega2 = getattr(score2, dim).omega

        # Decision: sum of probabilities for positive options > tau
        decision1 = sum(omega1[i] for i in positive_options) > tau
        decision2 = sum(omega2[i] for i in positive_options) > tau

        if decision1 == decision2:
            matches += 1

    return matches / 3


def prevalence_bias(
    scores: list[IndeterminacyAwareScore],
    reference_scores: list[IndeterminacyAwareScore],
    tau: float = 0.5,
    positive_options: list[int] | None = None,
) -> float:
    """Compute prevalence estimation bias.

    Measures systematic over/underestimation of quality prevalence
    compared to reference (human) ratings.

    Args:
        scores: Judge scores
        reference_scores: Reference (human) scores
        tau: Classification threshold
        positive_options: Indices of "positive" options

    Returns:
        Bias (positive = overestimation, negative = underestimation)
    """
    if positive_options is None:
        positive_options = [2, 3]

    judge_positive = 0
    ref_positive = 0

    for score, ref in zip(scores, reference_scores, strict=False):
        for dim in ["accuracy", "completeness", "coherence"]:
            omega_j = getattr(score, dim).omega
            omega_r = getattr(ref, dim).omega

            judge_positive += sum(omega_j[i] for i in positive_options) > tau
            ref_positive += sum(omega_r[i] for i in positive_options) > tau

    total = len(scores) * 3
    return (judge_positive - ref_positive) / total if total > 0 else 0.0
