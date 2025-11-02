"""LLM-as-judge evaluator for research quality assessment."""

import re
import statistics
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

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash-preview-09-2025",
        use_multiple_judges: bool = True,
        use_hybrid_scoring: bool = True,
    ) -> None:
        """Initialize quality evaluator.

        Args:
            judge_model: LLM model to use for evaluation
            use_multiple_judges: Whether to run 3 evaluations and take median (reduces variance)
            use_hybrid_scoring: Whether to combine LLM scores with rule-based metrics
        """
        self.judge_model = judge_model
        self.use_multiple_judges = use_multiple_judges
        self.use_hybrid_scoring = use_hybrid_scoring

    def evaluate(self, question: str, answer: str) -> QualityScore:
        """Evaluate answer quality.

        Args:
            question: Original research question
            answer: Answer to evaluate

        Returns:
            QualityScore with detailed assessment
        """
        # Get LLM-based scores (possibly multiple)
        if self.use_multiple_judges:
            llm_scores = self._evaluate_with_multiple_judges(question, answer)
        else:
            llm_scores = self._evaluate_single(question, answer)

        # Get rule-based scores if enabled
        if self.use_hybrid_scoring:
            rule_scores = self._calculate_rule_based_scores(question, answer)

            # Hybrid: 60% LLM + 40% rule-based (LLM is more important but rules reduce variance)
            accuracy = 0.6 * llm_scores.accuracy + 0.4 * rule_scores["accuracy"]
            completeness = 0.6 * llm_scores.completeness + 0.4 * rule_scores["completeness"]
            coherence = 0.6 * llm_scores.coherence + 0.4 * rule_scores["coherence"]

            explanation = f"[Hybrid Score: 60% LLM + 40% rule-based] {llm_scores.explanation}"
        else:
            accuracy = llm_scores.accuracy
            completeness = llm_scores.completeness
            coherence = llm_scores.coherence
            explanation = llm_scores.explanation

        # Calculate total (normalized to 0-100)
        total = (accuracy + completeness + coherence) / 30 * 100

        return QualityScore(
            accuracy=accuracy,
            completeness=completeness,
            coherence=coherence,
            total=total,
            explanation=explanation,
        )

    def _evaluate_single(self, question: str, answer: str) -> QualityScore:
        """Run a single LLM evaluation.

        Args:
            question: Research question
            answer: Answer to evaluate

        Returns:
            Quality score from single judge
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
            temperature=0,  # Deterministic for reproducibility
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

    def _evaluate_with_multiple_judges(self, question: str, answer: str) -> QualityScore:
        """Run 3 evaluations and take the median to reduce variance.

        Args:
            question: Research question
            answer: Answer to evaluate

        Returns:
            Median quality score across 3 evaluations
        """
        scores = []
        for _ in range(3):
            score = self._evaluate_single(question, answer)
            scores.append(score)

        # Take median of each component
        accuracy = statistics.median([s.accuracy for s in scores])
        completeness = statistics.median([s.completeness for s in scores])
        coherence = statistics.median([s.coherence for s in scores])

        # Use explanation from middle score
        median_total = statistics.median([s.total for s in scores])
        explanation = next(s.explanation for s in scores if abs(s.total - median_total) < 0.1)

        total = (accuracy + completeness + coherence) / 30 * 100

        return QualityScore(
            accuracy=accuracy,
            completeness=completeness,
            coherence=coherence,
            total=total,
            explanation=f"[Median of 3 evaluations] {explanation}",
        )

    def _calculate_rule_based_scores(self, question: str, answer: str) -> dict[str, float]:
        """Calculate rule-based quality metrics.

        These complement LLM judgment with deterministic measures:
        - Accuracy: Presence of technical terms and specificity
        - Completeness: Answer length and structure
        - Coherence: Logical organization and readability

        Args:
            question: Research question
            answer: Answer to evaluate

        Returns:
            Dict with accuracy, completeness, coherence scores (0-10)
        """
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

        # Normalize: expect ~5+ numbers and ~10+ technical terms for high accuracy
        accuracy = min(10.0, (numbers / 5 + technical_markers / 10) / 2 * 10)

        # Completeness proxy: Answer length and section structure
        words = len(answer.split())
        has_multiple_paragraphs = answer.count("\n\n") >= 2
        has_headings = bool(re.search(r"^#+\s+|\*\*.*\*\*", answer, re.MULTILINE))
        has_lists = bool(re.search(r"^\s*[-*\d]+[\.)]\s+", answer, re.MULTILINE))

        # Normalize: expect 500+ words, paragraphs, headings for completeness
        length_score = min(10.0, words / 500 * 10)
        structure_bonus = 2.0 * has_multiple_paragraphs + 1.0 * has_headings + 1.0 * has_lists
        completeness = min(10.0, (length_score + structure_bonus) / 2)

        # Coherence proxy: Average sentence length (too short = fragmented, too long = rambling)
        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = statistics.mean(len(s.split()) for s in sentences)
            # Ideal: 15-25 words per sentence
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
