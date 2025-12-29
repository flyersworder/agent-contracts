"""Task loading from MathArena datasets for math reasoning experiment.

This module handles loading math competition problems from MathArena datasets
(SMT 2025, AIME 2025, CMIMC 2025, etc.) for evaluating ContractExecutor
strategy modes on reasoning-intensive tasks.

These are 2025 competition problems - guaranteed uncontaminated in LLM training data.

Example:
    >>> tasks = load_math_tasks(dataset="smt_2025", limit=20)
    >>> print(f"Loaded {len(tasks)} tasks")
    >>> task = tasks[0]
    >>> print(f"Question: {task.question[:100]}...")
    >>> print(f"Answer: {task.answer}")

Available datasets:
    - smt_2025: Stanford Math Tournament 2025 (53 problems)
    - aime_2025: AIME 2025 I & II combined (30 problems)
    - aime_2025_I: AIME 2025 I only (15 problems)
    - aime_2025_II: AIME 2025 II only (15 problems)
    - cmimc_2025: Carnegie Mellon Math Competition 2025 (40 problems)
    - brumo_2025: BRUMO 2025 (30 problems)
"""

import re
from dataclasses import dataclass, field
from typing import Any

# Available MathArena datasets
MATHARENA_DATASETS = {
    "smt_2025": "MathArena/smt_2025",
    "aime_2025": "MathArena/aime_2025",
    "aime_2025_I": "MathArena/aime_2025_I",
    "aime_2025_II": "MathArena/aime_2025_II",
    "cmimc_2025": "MathArena/cmimc_2025",
    "brumo_2025": "MathArena/brumo_2025",
}


@dataclass
class MathTask:
    """A math reasoning task from MathArena competitions.

    Attributes:
        task_id: Unique identifier for the task
        question: The math problem statement
        answer: The correct answer (may be integer, fraction, or expression)
        problem_type: Category like "Algebra", "Geometry", "Combinatorics"
        dataset: Source dataset name
        is_integer_answer: Whether the answer is a simple integer
    """

    task_id: str
    question: str
    answer: str
    problem_type: list[str] = field(default_factory=list)
    dataset: str = ""
    is_integer_answer: bool = False

    @classmethod
    def from_matharena_row(cls, row: dict[str, Any], dataset_name: str) -> "MathTask":
        """Create a MathTask from a MathArena dataset row.

        Args:
            row: Row from a MathArena dataset
            dataset_name: Name of the source dataset

        Returns:
            MathTask instance
        """
        problem_idx = row.get("problem_idx", 0)
        question = row.get("problem", "")
        answer = str(row.get("answer", ""))
        problem_type = row.get("problem_type", [])

        # Handle problem_type as list or string
        if isinstance(problem_type, str):
            problem_type = [problem_type]

        # Check if answer is a simple integer
        is_integer = _is_integer_answer(answer)

        return cls(
            task_id=f"{dataset_name}_{problem_idx:03d}",
            question=question,
            answer=answer,
            problem_type=problem_type,
            dataset=dataset_name,
            is_integer_answer=is_integer,
        )

    def get_prompt(self) -> str:
        """Generate the math reasoning prompt.

        Returns:
            Formatted prompt string for the LLM
        """
        return f"""Solve the following math competition problem step by step.

Problem: {self.question}

Think through this carefully and show each step of your reasoning.
Give your final answer in \\boxed{{}} format (e.g., \\boxed{{42}})."""


def _is_integer_answer(answer: str) -> bool:
    """Check if an answer is a simple integer.

    Args:
        answer: The answer string

    Returns:
        True if answer is a simple integer (possibly negative)
    """
    # Remove whitespace
    answer = answer.strip()
    # Check for pure integer (possibly negative)
    return bool(re.match(r"^-?\d+$", answer))


def extract_model_answer(response: str) -> str:
    """Extract the numerical answer from a model's response.

    Tries multiple patterns to find the final answer:
    1. \\boxed{number} (LaTeX boxed format - common in competition math)
    2. "The answer is [number]" (with optional $ or currency symbols)
    3. "#### [number]" (GSM8K format)
    4. Last boxed content or standalone number

    Args:
        response: The model's response text

    Returns:
        The extracted answer as a string, or empty string if not found
    """
    # Pattern 1: \boxed{number} - most common in competition math
    # Handle both \\boxed and \boxed (escaped and unescaped)
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return _normalize_answer(match.group(1))

    # Pattern 2: "The answer is [number]" (handles $, currency symbols, spaces)
    match = re.search(
        r"(?:the\s+)?answer\s+is[:\s]*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        response,
        re.IGNORECASE,
    )
    if match:
        return _normalize_answer(match.group(1))

    # Pattern 3: "#### [number]" (GSM8K format)
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
        response[-500:],  # Look in last 500 chars
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


def check_answer(predicted: str, expected: str) -> bool:
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
        return float(predicted) == float(expected)
    except ValueError:
        pass

    # Fall back to string comparison (for expressions, fractions)
    return predicted.strip() == expected.strip()


def load_math_tasks(
    dataset: str = "smt_2025",
    limit: int | None = None,
    integer_only: bool = True,
    problem_types: list[str] | None = None,
    random_seed: int = 42,
) -> list[MathTask]:
    """Load math reasoning tasks from MathArena datasets.

    Args:
        dataset: Dataset name (see MATHARENA_DATASETS for options)
        limit: Maximum number of tasks to return
        integer_only: If True, only return problems with integer answers
        problem_types: Filter by problem types (e.g., ["Algebra", "Combinatorics"])
        random_seed: Seed for random sampling

    Returns:
        List of MathTask objects
    """
    from datasets import load_dataset as hf_load_dataset

    if dataset not in MATHARENA_DATASETS:
        available = ", ".join(MATHARENA_DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")

    hf_path = MATHARENA_DATASETS[dataset]

    # Load from HuggingFace
    ds = hf_load_dataset(hf_path, split="train")

    tasks: list[MathTask] = []

    for row in ds:
        task = MathTask.from_matharena_row(row, dataset)

        # Filter by integer answer if requested
        if integer_only and not task.is_integer_answer:
            continue

        # Filter by problem type if specified
        if problem_types and not any(pt in task.problem_type for pt in problem_types):
            continue

        tasks.append(task)

    # Shuffle and limit
    if limit and len(tasks) > limit:
        import random

        random.seed(random_seed)
        random.shuffle(tasks)
        tasks = tasks[:limit]

    return tasks


def get_task_statistics(tasks: list[MathTask]) -> dict[str, Any]:
    """Get statistics about a list of math tasks.

    Args:
        tasks: List of MathTask objects

    Returns:
        Dictionary with statistics
    """
    if not tasks:
        return {"total": 0}

    question_lengths = [len(t.question) for t in tasks]

    # Count by problem type
    from collections import Counter

    type_counts: Counter[str] = Counter()
    for task in tasks:
        for pt in task.problem_type:
            type_counts[pt] += 1

    # Count by dataset
    dataset_counts = Counter(t.dataset for t in tasks)

    return {
        "total": len(tasks),
        "question_length": {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "avg": sum(question_lengths) / len(question_lengths),
        },
        "problem_types": dict(type_counts),
        "datasets": dict(dataset_counts),
        "integer_answers": sum(1 for t in tasks if t.is_integer_answer),
    }


def list_available_datasets() -> dict[str, str]:
    """List available MathArena datasets.

    Returns:
        Dictionary mapping dataset names to HuggingFace paths
    """
    return MATHARENA_DATASETS.copy()
