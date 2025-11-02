"""LLM-as-judge evaluator for research quality assessment."""

from dataclasses import dataclass

from litellm import completion


@dataclass
class QualityScore:
    """Quality assessment for a research answer.

    Attributes:
        accuracy: Factual correctness (0-10)
        completeness: Coverage of all aspects (0-10)
        coherence: Logical structure and clarity (0-10)
        total: Total quality score (0-100)
        explanation: Explanation of the scores
    """

    accuracy: float
    completeness: float
    coherence: float
    total: float
    explanation: str


class QualityEvaluator:
    """Evaluates research quality using LLM-as-judge.

    Uses a separate LLM to assess:
    - Accuracy: Are the facts correct?
    - Completeness: Does it address all aspects?
    - Coherence: Is it well-structured and logical?
    """

    def __init__(self, judge_model: str = "gemini/gemini-2.5-flash-preview-09-2025") -> None:
        """Initialize quality evaluator.

        Args:
            judge_model: LLM model to use for evaluation
        """
        self.judge_model = judge_model

    def evaluate(self, question: str, answer: str) -> QualityScore:
        """Evaluate answer quality.

        Args:
            question: Original research question
            answer: Answer to evaluate

        Returns:
            QualityScore with detailed assessment
        """
        prompt = f"""You are an expert evaluator assessing the quality of research answers. Evaluate the following answer on three dimensions:

1. **Accuracy (0-10)**: Are the facts, explanations, and technical details correct? Are there any significant errors or misconceptions?

2. **Completeness (0-10)**: Does the answer address all aspects of the question? Are key points covered? Is important context included?

3. **Coherence (0-10)**: Is the answer well-structured, logically organized, and easy to understand? Does it flow naturally?

Research Question:
{question}

Answer to Evaluate:
{answer}

Provide your evaluation in the following format:

Accuracy: [score 0-10]
Completeness: [score 0-10]
Coherence: [score 0-10]

Explanation:
[2-3 sentences explaining the scores, highlighting strengths and weaknesses]"""

        response = completion(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",  # Use medium effort for evaluation
        )

        content = response["choices"][0]["message"]["content"]

        # Parse scores from response
        accuracy, completeness, coherence, explanation = self._parse_evaluation(content)

        # Calculate total (normalized to 0-100)
        total = (accuracy + completeness + coherence) / 30 * 100

        return QualityScore(
            accuracy=accuracy,
            completeness=completeness,
            coherence=coherence,
            total=total,
            explanation=explanation,
        )

    def _parse_evaluation(self, content: str) -> tuple[float, float, float, str]:
        """Parse evaluation from LLM response.

        Args:
            content: LLM response content

        Returns:
            Tuple of (accuracy, completeness, coherence, explanation)
        """
        lines = content.strip().split("\n")

        accuracy = 0.0
        completeness = 0.0
        coherence = 0.0
        explanation = ""

        explanation_started = False

        for line in lines:
            line = line.strip()

            if line.lower().startswith("accuracy:"):
                try:
                    accuracy = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    accuracy = 7.0  # Default if parsing fails

            elif line.lower().startswith("completeness:"):
                try:
                    completeness = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    completeness = 7.0

            elif line.lower().startswith("coherence:"):
                try:
                    coherence = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    coherence = 7.0

            elif line.lower().startswith("explanation:"):
                explanation_started = True
                # Get text after "Explanation:"
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    explanation = parts[1].strip()

            elif explanation_started and line:
                # Continue building explanation
                explanation += " " + line

        # Clamp scores to 0-10 range
        accuracy = max(0.0, min(10.0, accuracy))
        completeness = max(0.0, min(10.0, completeness))
        coherence = max(0.0, min(10.0, coherence))

        return accuracy, completeness, coherence, explanation.strip()
