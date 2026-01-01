"""Email-specific indeterminacy-aware evaluator.

This module adapts the NeurIPS 2025 indeterminacy framework for email quality
evaluation. Instead of research report dimensions (accuracy, completeness,
coherence), we use email-specific dimensions:

- Purpose Clarity: Does the email clearly state its purpose?
- Professional Tone: Is the language appropriate for the situation?
- Completeness: Does it include all required information?
- Actionability: Is there a clear next step for the recipient?

Reference: "Validating LLM-as-a-Judge Systems under Rating Indeterminacy"
(Guerdan et al., NeurIPS 2025)
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from litellm import completion

from .scenarios import EmailScenario

# Rating ranges for each email quality dimension
PURPOSE_CLARITY_RANGES = {
    "A": (0, 3, "Purpose unclear or buried, reader confused about intent"),
    "B": (4, 6, "Purpose present but not immediately clear"),
    "C": (7, 8, "Purpose stated clearly within first 2-3 sentences"),
    "D": (9, 10, "Purpose immediately obvious, perfectly clear intent"),
}

PROFESSIONAL_TONE_RANGES = {
    "A": (0, 3, "Inappropriate, offensive, or very unprofessional"),
    "B": (4, 6, "Somewhat informal or slightly off-tone for context"),
    "C": (7, 8, "Professional and appropriate with minor issues"),
    "D": (9, 10, "Perfect tone for the situation, highly polished"),
}

COMPLETENESS_RANGES = {
    "A": (0, 3, "Missing most required information"),
    "B": (4, 6, "Some information present, important details missing"),
    "C": (7, 8, "Most information present, minor omissions"),
    "D": (9, 10, "All required information clearly included"),
}

ACTIONABILITY_RANGES = {
    "A": (0, 3, "No clear action or next step for recipient"),
    "B": (4, 6, "Action implied but not explicitly stated"),
    "C": (7, 8, "Clear action requested with some context"),
    "D": (9, 10, "Crystal clear call-to-action with deadline/specifics"),
}


@dataclass
class EmailResponseSet:
    """A response set representing all options deemed reasonable by a judge.

    In the indeterminacy framework, a response set S ⊆ O contains all options
    that a rater considers "reasonable" for an item.

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
        """Convert to binary multi-label vector."""
        return np.array([1.0 if opt in self.selected_options else 0.0 for opt in options])


@dataclass
class EmailMultiLabelScore:
    """Multi-label probability vector for an email quality dimension.

    This is the ω (omega) vector: P(option_k ∈ response_set) for each option k.

    Attributes:
        dimension: Quality dimension name
        omega: Probability vector [P(A reasonable), P(B), P(C), P(D)]
        point_estimate: Traditional single-point score (0-10)
        indeterminacy: Average number of options selected per judge
        raw_response_sets: Individual response sets from each judge
    """

    dimension: str
    omega: np.ndarray
    point_estimate: float
    indeterminacy: float
    raw_response_sets: list[EmailResponseSet] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        """Check if there's significant rating indeterminacy."""
        return self.indeterminacy > 1.2

    def to_hard_multilabel(self, tau: float = 0.5) -> np.ndarray:
        """Convert to hard multi-label using threshold τ."""
        return (self.omega >= tau).astype(float)


@dataclass
class EmailIndeterminacyScore:
    """Email quality assessment with full indeterminacy information.

    Attributes:
        purpose_clarity: Multi-label score for purpose clarity
        professional_tone: Multi-label score for professional tone
        completeness: Multi-label score for completeness
        actionability: Multi-label score for actionability
        length_score: Rule-based length score (0-1)
        weighted_score: Overall weighted score (0-1) for backward compatibility
        explanation: Aggregated explanation from judges
        judge_agreement: Measure of agreement across judges (0-1)
    """

    purpose_clarity: EmailMultiLabelScore
    professional_tone: EmailMultiLabelScore
    completeness: EmailMultiLabelScore
    actionability: EmailMultiLabelScore
    length_score: float
    weighted_score: float
    explanation: str
    judge_agreement: float = 1.0

    # Weights for each criterion (must sum to 1.0)
    WEIGHTS: ClassVar[dict[str, float]] = {
        "purpose_clarity": 0.25,
        "professional_tone": 0.20,
        "completeness": 0.25,
        "actionability": 0.15,
        "length": 0.15,
    }

    @property
    def point_estimates(self) -> dict[str, float]:
        """Get traditional point estimates for backward compatibility."""
        return {
            "purpose_clarity": self.purpose_clarity.point_estimate,
            "professional_tone": self.professional_tone.point_estimate,
            "completeness": self.completeness.point_estimate,
            "actionability": self.actionability.point_estimate,
            "length": self.length_score * 10,
            "weighted_score": self.weighted_score,
        }

    @property
    def indeterminacy_summary(self) -> dict[str, float]:
        """Summary of indeterminacy across dimensions."""
        return {
            "purpose_clarity": self.purpose_clarity.indeterminacy,
            "professional_tone": self.professional_tone.indeterminacy,
            "completeness": self.completeness.indeterminacy,
            "actionability": self.actionability.indeterminacy,
            "overall": (
                self.purpose_clarity.indeterminacy
                + self.professional_tone.indeterminacy
                + self.completeness.indeterminacy
                + self.actionability.indeterminacy
            )
            / 4,
        }

    @property
    def criteria_met(self) -> dict[str, bool]:
        """Return which criteria are met (point estimate >= 7)."""
        threshold = 7.0
        return {
            "purpose_clarity": self.purpose_clarity.point_estimate >= threshold,
            "professional_tone": self.professional_tone.point_estimate >= threshold,
            "completeness": self.completeness.point_estimate >= threshold,
            "actionability": self.actionability.point_estimate >= threshold,
            "length": self.length_score >= 0.7,
        }

    @property
    def gaps(self) -> list[str]:
        """Return list of criteria that are not met."""
        return [name for name, met in self.criteria_met.items() if not met]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "purpose_clarity": self.purpose_clarity.point_estimate,
            "professional_tone": self.professional_tone.point_estimate,
            "completeness": self.completeness.point_estimate,
            "actionability": self.actionability.point_estimate,
            "length_score": self.length_score,
            "weighted_score": self.weighted_score,
            "indeterminacy": self.indeterminacy_summary,
            "judge_agreement": self.judge_agreement,
            "criteria_met": self.criteria_met,
            "gaps": self.gaps,
            "explanation": self.explanation,
        }


class EmailIndeterminacyEvaluator:
    """Evaluates email quality using indeterminacy-aware LLM-as-judge.

    This evaluator implements the response set elicitation framework for
    email quality assessment, adapted from the NeurIPS 2025 paper.

    Attributes:
        judge_model: LLM model to use for evaluation
        num_judges: Number of evaluation passes (default 3)
        tau: Classification threshold for downstream decisions (default 0.5)
    """

    OPTIONS: ClassVar[list[str]] = ["A", "B", "C", "D"]

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash-lite",
        num_judges: int = 3,
        tau: float = 0.5,
    ) -> None:
        """Initialize email indeterminacy-aware evaluator.

        Args:
            judge_model: LLM model to use for evaluation
            num_judges: Number of independent evaluations to aggregate
            tau: Threshold for hard multi-label classification
        """
        self.judge_model = judge_model
        self.num_judges = num_judges
        self.tau = tau

    def evaluate(
        self,
        email: str,
        scenario: EmailScenario,
    ) -> EmailIndeterminacyScore:
        """Evaluate email quality with indeterminacy awareness.

        Args:
            email: The email text to evaluate
            scenario: The scenario the email was written for

        Returns:
            EmailIndeterminacyScore with full response set information
        """
        # Collect response sets from multiple judges
        all_response_sets: dict[str, list[EmailResponseSet]] = {
            "purpose_clarity": [],
            "professional_tone": [],
            "completeness": [],
            "actionability": [],
        }
        all_explanations: list[str] = []

        for _ in range(self.num_judges):
            judge_result = self._evaluate_single_judge(email, scenario)
            for dim in all_response_sets:
                all_response_sets[dim].append(judge_result[dim])
            all_explanations.append(judge_result["explanation"])

        # Aggregate response sets into multi-label scores
        purpose_score = self._aggregate_response_sets(
            all_response_sets["purpose_clarity"], "purpose_clarity"
        )
        tone_score = self._aggregate_response_sets(
            all_response_sets["professional_tone"], "professional_tone"
        )
        completeness_score = self._aggregate_response_sets(
            all_response_sets["completeness"], "completeness"
        )
        actionability_score = self._aggregate_response_sets(
            all_response_sets["actionability"], "actionability"
        )

        # Rule-based length evaluation
        word_count = len(email.split())
        length_score = self._evaluate_length(word_count)

        # Calculate weighted score (0-1 scale)
        weights = EmailIndeterminacyScore.WEIGHTS
        weighted_score = (
            weights["purpose_clarity"] * (purpose_score.point_estimate / 10)
            + weights["professional_tone"] * (tone_score.point_estimate / 10)
            + weights["completeness"] * (completeness_score.point_estimate / 10)
            + weights["actionability"] * (actionability_score.point_estimate / 10)
            + weights["length"] * length_score
        )

        # Calculate judge agreement
        judge_agreement = self._calculate_judge_agreement(all_response_sets)

        # Select representative explanation
        avg_indeterminacy = (
            purpose_score.indeterminacy
            + tone_score.indeterminacy
            + completeness_score.indeterminacy
            + actionability_score.indeterminacy
        ) / 4
        explanation = self._select_explanation(all_explanations, avg_indeterminacy)

        return EmailIndeterminacyScore(
            purpose_clarity=purpose_score,
            professional_tone=tone_score,
            completeness=completeness_score,
            actionability=actionability_score,
            length_score=length_score,
            weighted_score=weighted_score,
            explanation=explanation,
            judge_agreement=judge_agreement,
        )

    def _evaluate_single_judge(self, email: str, scenario: EmailScenario) -> dict[str, Any]:
        """Run a single judge evaluation with response set elicitation."""
        prompt = self._build_response_set_prompt(email, scenario)

        response = completion(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slight temperature for response set diversity
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_response_set_evaluation(content)

    def _build_response_set_prompt(self, email: str, scenario: EmailScenario) -> str:
        """Build prompt for response set elicitation."""
        key_info_str = "\n".join(f"  - {info}" for info in scenario.key_info)

        return f"""You are an expert email quality evaluator.

For each dimension, select ALL rating ranges that could REASONABLY apply to this email.
If multiple ranges seem valid given different interpretations, select all of them.
This helps capture genuine ambiguity in the rating task.

**CONTEXT:**
- Sender: {scenario.sender_role}
- Recipient: {scenario.recipient}
- Situation: {scenario.context}
- Goal: {scenario.goal}
- Required information:
{key_info_str}
- Expected tone: {scenario.tone_guidance}

**EMAIL TO EVALUATE:**
{email}

---

**PURPOSE CLARITY** - Does the email clearly state its purpose?
Select ALL ranges that could reasonably apply:
A. [0-3] Purpose unclear or buried, reader confused about intent
B. [4-6] Purpose present but not immediately clear
C. [7-8] Purpose stated clearly within first 2-3 sentences
D. [9-10] Purpose immediately obvious, perfectly clear intent

**PROFESSIONAL TONE** - Is the language appropriate for the situation?
Select ALL ranges that could reasonably apply:
A. [0-3] Inappropriate, offensive, or very unprofessional
B. [4-6] Somewhat informal or slightly off-tone for context
C. [7-8] Professional and appropriate with minor issues
D. [9-10] Perfect tone for the situation, highly polished

**COMPLETENESS** - Does it include all required information?
Select ALL ranges that could reasonably apply:
A. [0-3] Missing most required information
B. [4-6] Some information present, important details missing
C. [7-8] Most information present, minor omissions
D. [9-10] All required information clearly included

**ACTIONABILITY** - Is there a clear next step for the recipient?
Select ALL ranges that could reasonably apply:
A. [0-3] No clear action or next step for recipient
B. [4-6] Action implied but not explicitly stated
C. [7-8] Clear action requested with some context
D. [9-10] Crystal clear call-to-action with deadline/specifics

---

RESPONSE FORMAT:
Purpose Clarity: [letters, e.g., "CD" or "C" or "BCD"]
Professional Tone: [letters, e.g., "D" or "CD"]
Completeness: [letters, e.g., "CD" or "D"]
Actionability: [letters, e.g., "C" or "CD"]

Explanation:
[2-3 sentences explaining your rating, especially noting any ambiguity]"""

    def _parse_response_set_evaluation(self, content: str) -> dict[str, Any]:
        """Parse response set evaluation from LLM output."""
        lines = content.strip().split("\n")

        result: dict[str, Any] = {
            "purpose_clarity": EmailResponseSet(set(), "purpose_clarity"),
            "professional_tone": EmailResponseSet(set(), "professional_tone"),
            "completeness": EmailResponseSet(set(), "completeness"),
            "actionability": EmailResponseSet(set(), "actionability"),
            "explanation": "",
        }

        explanation_started = False

        for line in lines:
            line = line.strip()

            if line.lower().startswith("purpose clarity:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["purpose_clarity"] = EmailResponseSet(letters, "purpose_clarity")

            elif line.lower().startswith("professional tone:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["professional_tone"] = EmailResponseSet(letters, "professional_tone")

            elif line.lower().startswith("completeness:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["completeness"] = EmailResponseSet(letters, "completeness")

            elif line.lower().startswith("actionability:"):
                letters = self._extract_letters(line.split(":", 1)[-1])
                result["actionability"] = EmailResponseSet(letters, "actionability")

            elif line.lower().startswith("explanation:"):
                explanation_started = True
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    result["explanation"] = parts[1].strip()

            elif explanation_started and line:
                result["explanation"] += " " + line

        # Validate: ensure at least one option selected per dimension
        for dim in ["purpose_clarity", "professional_tone", "completeness", "actionability"]:
            if not result[dim].selected_options:
                result[dim] = EmailResponseSet({"C"}, dim)

        return result

    def _extract_letters(self, text: str) -> set[str]:
        """Extract valid option letters from text."""
        letters = set(re.findall(r"[A-D]", text.upper()))
        return letters if letters else {"C"}

    def _aggregate_response_sets(
        self, response_sets: list[EmailResponseSet], dimension: str
    ) -> EmailMultiLabelScore:
        """Aggregate multiple response sets into a MultiLabelScore."""
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

        return EmailMultiLabelScore(
            dimension=dimension,
            omega=omega,
            point_estimate=point_estimate,
            indeterminacy=indeterminacy,
            raw_response_sets=response_sets,
        )

    def _get_ranges_for_dimension(self, dimension: str) -> dict[str, tuple[int, int, str]]:
        """Get rating ranges for a dimension."""
        dimension_ranges = {
            "purpose_clarity": PURPOSE_CLARITY_RANGES,
            "professional_tone": PROFESSIONAL_TONE_RANGES,
            "completeness": COMPLETENESS_RANGES,
            "actionability": ACTIONABILITY_RANGES,
        }
        return dimension_ranges.get(dimension, PURPOSE_CLARITY_RANGES)

    def _evaluate_length(self, word_count: int) -> float:
        """Evaluate email length (rule-based).

        Ideal range: 50-300 words
        """
        if 50 <= word_count <= 300:
            return 1.0
        elif 30 <= word_count < 50 or 300 < word_count <= 400:
            return 0.7
        elif 20 <= word_count < 30 or 400 < word_count <= 500:
            return 0.5
        else:
            return 0.3

    def _calculate_judge_agreement(
        self, all_response_sets: dict[str, list[EmailResponseSet]]
    ) -> float:
        """Calculate agreement across judges using Jaccard similarity."""
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
        """Select representative explanation based on indeterminacy."""
        selected = max(explanations, key=len) if explanations else ""

        if indeterminacy > 1.5:
            prefix = f"[High indeterminacy: {indeterminacy:.1f}] "
        elif indeterminacy > 1.2:
            prefix = f"[Moderate indeterminacy: {indeterminacy:.1f}] "
        else:
            prefix = f"[Low indeterminacy: {indeterminacy:.1f}] "

        return prefix + selected.strip()

    def meets_threshold(
        self,
        score: EmailIndeterminacyScore,
        threshold: float = 0.80,
    ) -> bool:
        """Check if email meets quality threshold.

        Args:
            score: Evaluation results
            threshold: Q_min threshold (default 0.80)

        Returns:
            True if weighted_score >= threshold
        """
        return score.weighted_score >= threshold

    def get_feedback(self, score: EmailIndeterminacyScore) -> str:
        """Generate feedback for improving the email."""
        gaps = score.gaps
        if not gaps:
            return "Email meets all quality criteria. Ready to send."

        feedback_parts = ["The email needs improvement in:"]
        for gap in gaps:
            if gap == "purpose_clarity":
                feedback_parts.append("- State the purpose more clearly in the opening sentences")
            elif gap == "professional_tone":
                feedback_parts.append("- Adjust the tone to be more appropriate for the situation")
            elif gap == "completeness":
                feedback_parts.append("- Include all required information points")
            elif gap == "length":
                feedback_parts.append("- Adjust length (aim for 50-300 words)")
            elif gap == "actionability":
                feedback_parts.append("- Add a clear call-to-action or next step")

        return "\n".join(feedback_parts)
