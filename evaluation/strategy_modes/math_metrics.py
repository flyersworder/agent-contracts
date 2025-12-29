"""Metrics for math reasoning evaluation.

This module provides accuracy metrics for evaluating math problem solutions.
"""

from dataclasses import dataclass
from typing import Any

from .math_tasks import check_answer, extract_model_answer


@dataclass
class MathMetrics:
    """Metrics for a single math problem evaluation.

    Attributes:
        correct: Whether the answer was correct
        predicted_answer: The extracted predicted answer
        expected_answer: The ground truth answer
        answer_extracted: Whether an answer was successfully extracted
    """

    correct: bool = False
    predicted_answer: str = ""
    expected_answer: str = ""
    answer_extracted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "correct": self.correct,
            "predicted_answer": self.predicted_answer,
            "expected_answer": self.expected_answer,
            "answer_extracted": self.answer_extracted,
        }


def compute_math_metrics(
    response: str,
    expected_answer: str,
) -> MathMetrics:
    """Compute metrics for a math problem response.

    Args:
        response: The model's response text
        expected_answer: The ground truth answer

    Returns:
        MathMetrics with accuracy information
    """
    predicted = extract_model_answer(response)
    answer_extracted = bool(predicted)
    correct = check_answer(predicted, expected_answer)

    return MathMetrics(
        correct=correct,
        predicted_answer=predicted,
        expected_answer=expected_answer,
        answer_extracted=answer_extracted,
    )


@dataclass
class AccuracyStats:
    """Aggregate accuracy statistics.

    Attributes:
        total: Total number of problems
        correct: Number of correct answers
        extracted: Number of answers successfully extracted
        accuracy: Proportion correct (correct/total)
        extraction_rate: Proportion with extracted answers
    """

    total: int = 0
    correct: int = 0
    extracted: int = 0
    accuracy: float = 0.0
    extraction_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "correct": self.correct,
            "extracted": self.extracted,
            "accuracy": self.accuracy,
            "extraction_rate": self.extraction_rate,
        }


def compute_accuracy_stats(metrics_list: list[MathMetrics]) -> AccuracyStats:
    """Compute aggregate accuracy statistics.

    Args:
        metrics_list: List of MathMetrics from individual problems

    Returns:
        AccuracyStats with aggregate metrics
    """
    if not metrics_list:
        return AccuracyStats()

    total = len(metrics_list)
    correct = sum(1 for m in metrics_list if m.correct)
    extracted = sum(1 for m in metrics_list if m.answer_extracted)

    return AccuracyStats(
        total=total,
        correct=correct,
        extracted=extracted,
        accuracy=correct / total if total > 0 else 0.0,
        extraction_rate=extracted / total if total > 0 else 0.0,
    )
