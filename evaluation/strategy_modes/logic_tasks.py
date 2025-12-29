"""Task loading from OpenR1 Logic Puzzles dataset for strategy modes experiment.

This module handles loading logic and word problems from the OpenR1 dataset
for evaluating ContractExecutor strategy modes on reasoning-intensive tasks.

The dataset is from February 2025 - guaranteed uncontaminated in LLM training data.
It provides problems that require reasoning but are tractable for single LLM calls
(typically 3-10 seconds, ~1000-2000 tokens).

Example:
    >>> tasks = load_logic_tasks(limit=20)
    >>> print(f"Loaded {len(tasks)} tasks")
    >>> task = tasks[0]
    >>> print(f"Question: {task.question[:100]}...")
    >>> print(f"Answer: {task.answer}")

Dataset: sunyiyou/openr1_logic_and_puzzles_1k_nm
Source: OpenR1 project (Feb 2025)
Problems: 1000 total, ~672 with simple numeric answers
"""

import re
from dataclasses import dataclass
from typing import Any

# Dataset configuration
OPENR1_DATASET = "sunyiyou/openr1_logic_and_puzzles_1k_nm"


@dataclass
class LogicTask:
    """A logic/word problem task from OpenR1 dataset.

    Attributes:
        task_id: Unique identifier for the task
        question: The problem statement
        answer: The correct answer (numeric)
        source: Original source of the problem
        is_simple_numeric: Whether the answer is a simple integer or decimal
        correctness_count: Number of times model solved correctly (lower = harder)
    """

    task_id: str
    question: str
    answer: str
    source: str = ""
    is_simple_numeric: bool = True
    correctness_count: int = 0

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any], idx: int) -> "LogicTask":
        """Create a LogicTask from a dataset row.

        Args:
            row: Row from the OpenR1 dataset
            idx: Index in the dataset

        Returns:
            LogicTask instance
        """
        question = row.get("problem", "")
        answer = str(row.get("answer", ""))
        source = row.get("source", "")
        correctness_count = row.get("correctness_count", 0)

        # Check if answer is a simple numeric value
        is_simple = _is_simple_numeric(answer)

        return cls(
            task_id=f"logic_{idx:04d}",
            question=question,
            answer=answer,
            source=source,
            is_simple_numeric=is_simple,
            correctness_count=correctness_count,
        )

    def get_prompt(self) -> str:
        """Generate the reasoning prompt.

        Returns:
            Formatted prompt string for the LLM
        """
        return f"""Solve the following logic problem step by step.

Problem: {self.question}

Think through this carefully and show your reasoning.
Give your final answer as a number in \\boxed{{}} format (e.g., \\boxed{{42}})."""


def _is_simple_numeric(answer: str) -> bool:
    """Check if an answer is a simple numeric value.

    Args:
        answer: The answer string

    Returns:
        True if answer is a simple integer or decimal number
    """
    answer = answer.strip()
    # Match integers, decimals, and negative numbers
    return bool(re.match(r"^-?\d+(\.\d+)?$", answer))


def extract_logic_answer(response: str) -> str:
    """Extract the numerical answer from a model's response.

    Tries multiple patterns to find the final answer:
    1. \\boxed{number} (LaTeX boxed format)
    2. "The answer is [number]"
    3. "#### [number]" (GSM8K format)
    4. Last standalone number

    Args:
        response: The model's response text

    Returns:
        The extracted answer as a string, or empty string if not found
    """
    # Pattern 1: \boxed{number}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return _normalize_answer(match.group(1))

    # Pattern 2: "The answer is [number]"
    match = re.search(
        r"(?:the\s+)?answer\s+is[:\s]*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        response,
        re.IGNORECASE,
    )
    if match:
        return _normalize_answer(match.group(1))

    # Pattern 3: "#### [number]"
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", response)
    if match:
        return _normalize_answer(match.group(1))

    # Pattern 4: Last standalone number on its own line
    matches = re.findall(r"(?:^|\n)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(?:$|\n)", response)
    if matches:
        return _normalize_answer(matches[-1])

    # Pattern 5: Any number after "=" near the end
    match = re.search(
        r"=\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(?:\.|$)",
        response[-500:],
        re.IGNORECASE,
    )
    if match:
        return _normalize_answer(match.group(1))

    return ""


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string.

    Removes commas, extra whitespace, and handles common formats.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    answer = answer.strip()
    # Remove commas from numbers like "70,000"
    answer = answer.replace(",", "")
    # Remove leading zeros but keep "0" itself
    if re.match(r"^0+\d+$", answer):
        answer = str(int(answer))
    return answer


def check_logic_answer(predicted: str, expected: str) -> bool:
    """Check if the predicted answer matches the expected answer.

    Handles numerical comparison (e.g., "18" == "18.0", "025" == "25").

    Args:
        predicted: The model's predicted answer
        expected: The ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    if not predicted or not expected:
        return False

    # Normalize both
    predicted = _normalize_answer(predicted)
    expected = _normalize_answer(expected)

    # Try numeric comparison first
    try:
        # Handle both integer and float comparisons
        pred_float = float(predicted)
        exp_float = float(expected)
        # Allow small tolerance for floating point
        return abs(pred_float - exp_float) < 1e-9
    except ValueError:
        pass

    # Fall back to string comparison
    return predicted.strip() == expected.strip()


def load_logic_tasks(
    limit: int | None = None,
    numeric_only: bool = True,
    difficulty: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    sources: list[str] | None = None,
    random_seed: int = 42,
) -> list[LogicTask]:
    """Load logic reasoning tasks from OpenR1 dataset.

    Args:
        limit: Maximum number of tasks to return
        numeric_only: If True, only return problems with simple numeric answers
        difficulty: Filter by difficulty level based on correctness_count:
            - "hard": correctness_count=1 (model solved only once, 233 problems)
            - "medium": correctness_count=2 (model solved twice, 757 problems)
            - "easy": correctness_count>=4 (model solved multiple times, 10 problems)
            - None: All difficulties (default)
        min_length: Minimum problem length in characters
        max_length: Maximum problem length in characters (shorter = simpler)
        sources: Filter by source (e.g., ["olympiads"]). None = all sources.
        random_seed: Seed for random sampling

    Returns:
        List of LogicTask objects

    Example:
        # Load hard problems (sweet spot for mode differentiation)
        tasks = load_logic_tasks(limit=20, difficulty="hard")

        # Load medium difficulty problems
        tasks = load_logic_tasks(limit=20, difficulty="medium")
    """
    from datasets import load_dataset as hf_load_dataset

    # Load from HuggingFace
    ds = hf_load_dataset(OPENR1_DATASET, split="train")

    tasks: list[LogicTask] = []

    for idx, row in enumerate(ds):
        task = LogicTask.from_dataset_row(row, idx)

        # Filter by numeric answer if requested
        if numeric_only and not task.is_simple_numeric:
            continue

        # Filter by difficulty (based on correctness_count)
        if difficulty is not None and (
            (difficulty == "hard" and task.correctness_count != 1)
            or (difficulty == "medium" and task.correctness_count != 2)
            or (difficulty == "easy" and task.correctness_count < 4)
        ):
            continue

        # Filter by problem length
        if min_length is not None and len(task.question) < min_length:
            continue
        if max_length is not None and len(task.question) > max_length:
            continue

        # Filter by source
        if sources is not None and task.source not in sources:
            continue

        tasks.append(task)

    # Shuffle and limit
    if limit and len(tasks) > limit:
        import random

        random.seed(random_seed)
        random.shuffle(tasks)
        tasks = tasks[:limit]

    return tasks


def get_logic_task_statistics(tasks: list[LogicTask]) -> dict[str, Any]:
    """Get statistics about a list of logic tasks.

    Args:
        tasks: List of LogicTask objects

    Returns:
        Dictionary with statistics
    """
    if not tasks:
        return {"total": 0}

    question_lengths = [len(t.question) for t in tasks]

    # Count by source
    from collections import Counter

    source_counts = Counter(t.source for t in tasks)

    return {
        "total": len(tasks),
        "question_length": {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "avg": sum(question_lengths) / len(question_lengths),
        },
        "sources": dict(source_counts),
        "simple_numeric": sum(1 for t in tasks if t.is_simple_numeric),
    }
