"""Email quality evaluator using LLM-as-judge.

This module implements quality evaluation for emails against defined criteria.
The evaluator determines whether an email meets the Q_min threshold for
"good enough" to send.

Quality Criteria:
- clear_purpose: Email clearly states its purpose in first 2 sentences
- professional_tone: Language is professional, respectful, appropriate
- key_info_complete: All required information is included
- appropriate_length: Not too short (<50 words) or too long (>300 words)
- actionable: Clear next step or call-to-action for recipient
"""

import re
from dataclasses import dataclass, field
from typing import Any

from litellm import completion

from .scenarios import EmailScenario


@dataclass
class EmailQualityCriteria:
    """Quality criteria evaluation for an email.

    Each criterion is evaluated as a score from 0-1.

    Attributes:
        clear_purpose: Whether purpose is stated clearly (0-1)
        professional_tone: Whether tone is appropriate (0-1)
        key_info_complete: Whether all required info is present (0-1)
        appropriate_length: Whether length is appropriate (0-1)
        actionable: Whether there's a clear next step (0-1)
        explanation: LLM's explanation of the evaluation
        raw_response: Full LLM response for debugging
    """

    clear_purpose: float = 0.0
    professional_tone: float = 0.0
    key_info_complete: float = 0.0
    appropriate_length: float = 0.0
    actionable: float = 0.0
    explanation: str = ""
    raw_response: str = ""

    # Weights for each criterion (must sum to 1.0)
    WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "clear_purpose": 0.25,
            "professional_tone": 0.20,
            "key_info_complete": 0.25,
            "appropriate_length": 0.15,
            "actionable": 0.15,
        }
    )

    @property
    def weighted_score(self) -> float:
        """Calculate weighted quality score (0-1 scale)."""
        return (
            self.WEIGHTS["clear_purpose"] * self.clear_purpose
            + self.WEIGHTS["professional_tone"] * self.professional_tone
            + self.WEIGHTS["key_info_complete"] * self.key_info_complete
            + self.WEIGHTS["appropriate_length"] * self.appropriate_length
            + self.WEIGHTS["actionable"] * self.actionable
        )

    @property
    def criteria_met(self) -> dict[str, bool]:
        """Return which criteria are met (threshold: 0.7)."""
        threshold = 0.7
        return {
            "clear_purpose": self.clear_purpose >= threshold,
            "professional_tone": self.professional_tone >= threshold,
            "key_info_complete": self.key_info_complete >= threshold,
            "appropriate_length": self.appropriate_length >= threshold,
            "actionable": self.actionable >= threshold,
        }

    @property
    def gaps(self) -> list[str]:
        """Return list of criteria that are not met."""
        return [name for name, met in self.criteria_met.items() if not met]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clear_purpose": self.clear_purpose,
            "professional_tone": self.professional_tone,
            "key_info_complete": self.key_info_complete,
            "appropriate_length": self.appropriate_length,
            "actionable": self.actionable,
            "weighted_score": self.weighted_score,
            "criteria_met": self.criteria_met,
            "gaps": self.gaps,
            "explanation": self.explanation,
        }


class EmailQualityEvaluator:
    """LLM-as-judge evaluator for email quality.

    Uses an LLM to evaluate emails against quality criteria,
    determining whether the email meets Q_min threshold.

    Attributes:
        judge_model: LLM model to use for evaluation
        temperature: Sampling temperature (0 for deterministic)
    """

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash-lite",
        temperature: float = 0.0,
    ) -> None:
        """Initialize the email quality evaluator.

        Args:
            judge_model: LLM model to use as judge
            temperature: Sampling temperature (0 = deterministic)
        """
        self.judge_model = judge_model
        self.temperature = temperature

    def evaluate(
        self,
        email: str,
        scenario: EmailScenario,
    ) -> EmailQualityCriteria:
        """Evaluate an email against quality criteria.

        Args:
            email: The email text to evaluate
            scenario: The scenario the email was written for

        Returns:
            EmailQualityCriteria with scores and explanation
        """
        # First, do rule-based length check
        word_count = len(email.split())
        length_score = self._evaluate_length(word_count)

        # Build prompt for LLM evaluation
        key_info_str = "\n".join(f"  - {info}" for info in scenario.key_info)

        prompt = f"""You are an expert email quality evaluator. Evaluate this email against specific criteria.

**SCENARIO CONTEXT:**
- Sender's role: {scenario.sender_role}
- Recipient: {scenario.recipient}
- Situation: {scenario.context}
- Goal: {scenario.goal}
- Required information:
{key_info_str}
- Expected tone: {scenario.tone_guidance}

**EMAIL TO EVALUATE:**
{email}

**EVALUATION CRITERIA:**
Rate each criterion from 0.0 to 1.0:

1. **Clear Purpose (0-1)**: Does the email state its purpose clearly in the first 2 sentences?
   - 1.0: Purpose is immediately clear
   - 0.7: Purpose is clear but takes a moment to find
   - 0.5: Purpose is implied but not stated
   - 0.0: Purpose is unclear

2. **Professional Tone (0-1)**: Is the language appropriate for the situation?
   - 1.0: Perfect tone for the scenario
   - 0.7: Generally appropriate with minor issues
   - 0.5: Tone is off but not offensive
   - 0.0: Inappropriate or unprofessional

3. **Key Info Complete (0-1)**: Are all required information points included?
   - 1.0: All required info is present and clear
   - 0.8: Most info present, 1 minor omission
   - 0.5: Several important points missing
   - 0.0: Major required info is absent

4. **Actionable (0-1)**: Is there a clear next step or call-to-action?
   - 1.0: Very clear action for recipient
   - 0.7: Action is implied or general
   - 0.3: No clear action requested
   - 0.0: Completely unclear what to do

**RESPOND IN THIS EXACT FORMAT:**
Clear Purpose: [score]
Professional Tone: [score]
Key Info Complete: [score]
Actionable: [score]

Explanation: [2-3 sentences explaining the scores and any gaps]"""

        try:
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            content = response["choices"][0]["message"]["content"]

            # Parse scores from response
            scores = self._parse_evaluation(content)

            return EmailQualityCriteria(
                clear_purpose=scores["clear_purpose"],
                professional_tone=scores["professional_tone"],
                key_info_complete=scores["key_info_complete"],
                appropriate_length=length_score,  # Rule-based
                actionable=scores["actionable"],
                explanation=scores["explanation"],
                raw_response=content,
            )

        except Exception as e:
            # Return conservative scores on error
            return EmailQualityCriteria(
                clear_purpose=0.5,
                professional_tone=0.5,
                key_info_complete=0.5,
                appropriate_length=length_score,
                actionable=0.5,
                explanation=f"Evaluation error: {e}",
                raw_response="",
            )

    def _evaluate_length(self, word_count: int) -> float:
        """Evaluate email length (rule-based).

        Ideal range: 50-300 words
        - Too short (<50): Missing content
        - Too long (>300): Could be more concise

        Args:
            word_count: Number of words in email

        Returns:
            Score from 0-1
        """
        if 50 <= word_count <= 300:
            return 1.0
        elif 30 <= word_count < 50:
            return 0.7  # Slightly short
        elif 300 < word_count <= 400:
            return 0.7  # Slightly long
        elif 20 <= word_count < 30:
            return 0.5  # Too short
        elif 400 < word_count <= 500:
            return 0.5  # Too long
        else:
            return 0.3  # Way too short or way too long

    def _parse_evaluation(self, content: str) -> dict[str, Any]:
        """Parse evaluation scores from LLM response.

        Args:
            content: LLM response content

        Returns:
            Dict with scores for each criterion
        """
        scores: dict[str, Any] = {
            "clear_purpose": 0.5,
            "professional_tone": 0.5,
            "key_info_complete": 0.5,
            "actionable": 0.5,
            "explanation": "",
        }

        lines = content.strip().split("\n")
        explanation_started = False

        for line in lines:
            line = line.strip()

            if line.lower().startswith("clear purpose:"):
                scores["clear_purpose"] = self._extract_score(line)

            elif line.lower().startswith("professional tone:"):
                scores["professional_tone"] = self._extract_score(line)

            elif line.lower().startswith("key info complete:"):
                scores["key_info_complete"] = self._extract_score(line)

            elif line.lower().startswith("actionable:"):
                scores["actionable"] = self._extract_score(line)

            elif line.lower().startswith("explanation:"):
                explanation_started = True
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    scores["explanation"] = parts[1].strip()

            elif explanation_started and line:
                scores["explanation"] += " " + line

        return scores

    def _extract_score(self, line: str) -> float:
        """Extract numeric score from a line.

        Args:
            line: Line containing score (e.g., "Clear Purpose: 0.8")

        Returns:
            Score clamped to 0-1 range
        """
        try:
            # Find first number in the line after the colon
            match = re.search(r":\s*([\d.]+)", line)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except ValueError:
            pass
        return 0.5  # Default on parse failure

    def meets_threshold(
        self,
        criteria: EmailQualityCriteria,
        threshold: float = 0.80,
    ) -> bool:
        """Check if email meets quality threshold.

        Args:
            criteria: Evaluation results
            threshold: Q_min threshold (default 0.80)

        Returns:
            True if weighted_score >= threshold
        """
        return criteria.weighted_score >= threshold

    def get_feedback(self, criteria: EmailQualityCriteria) -> str:
        """Generate feedback for improving the email.

        Args:
            criteria: Evaluation results

        Returns:
            Actionable feedback string
        """
        gaps = criteria.gaps
        if not gaps:
            return "Email meets all quality criteria. Ready to send."

        feedback_parts = ["The email needs improvement in:"]
        for gap in gaps:
            if gap == "clear_purpose":
                feedback_parts.append("- State the purpose more clearly in the opening sentences")
            elif gap == "professional_tone":
                feedback_parts.append("- Adjust the tone to be more appropriate for the situation")
            elif gap == "key_info_complete":
                feedback_parts.append("- Include all required information points")
            elif gap == "appropriate_length":
                feedback_parts.append("- Adjust length (aim for 50-300 words)")
            elif gap == "actionable":
                feedback_parts.append("- Add a clear call-to-action or next step")

        return "\n".join(feedback_parts)
