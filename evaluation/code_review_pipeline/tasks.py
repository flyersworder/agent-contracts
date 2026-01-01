"""Task loading from LiveCodeBench dataset.

This module handles loading and filtering coding problems from LiveCodeBench
for the code review experiment.

Example:
    >>> tasks = load_tasks(difficulty="medium", after_date="2025-02-01", limit=50)
    >>> print(f"Loaded {len(tasks)} tasks")
    >>> task = tasks[0]
    >>> print(f"Task: {task.title} ({task.difficulty})")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskDifficulty(Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestCase:
    """A single test case for a coding problem.

    Attributes:
        input: Input to the program (stdin)
        expected_output: Expected output (stdout)
    """

    input: str
    expected_output: str


@dataclass
class CodeTask:
    """A coding task from LiveCodeBench.

    Attributes:
        task_id: Unique identifier (e.g., "1873_A")
        title: Problem title
        description: Full problem description
        difficulty: Easy/Medium/Hard
        contest_date: When the problem was released
        test_cases: List of test cases for validation
        starter_code: Optional starter code template
        metadata: Additional metadata from dataset
    """

    task_id: str
    title: str
    description: str
    difficulty: TaskDifficulty
    contest_date: datetime
    test_cases: list[TestCase]
    starter_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any]) -> "CodeTask":
        """Create a CodeTask from a LiveCodeBench dataset row.

        Args:
            row: Row from the LiveCodeBench dataset

        Returns:
            CodeTask instance
        """
        # Parse test cases from JSON
        public_tests = json.loads(row.get("public_test_cases", "[]"))
        test_cases = [
            TestCase(
                input=tc.get("input", ""),
                expected_output=tc.get("output", ""),
            )
            for tc in public_tests
        ]

        # Parse difficulty
        diff_str = row.get("difficulty", "medium").lower()
        try:
            difficulty = TaskDifficulty(diff_str)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM

        # Parse contest date
        date_str = row.get("contest_date", "")
        try:
            contest_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            contest_date = datetime.now()

        return cls(
            task_id=row.get("question_id", "unknown"),
            title=row.get("question_title", "Unknown Problem"),
            description=row.get("question_content", ""),
            difficulty=difficulty,
            contest_date=contest_date,
            test_cases=test_cases,
            starter_code=row.get("starter_code", ""),
            metadata={
                "platform": row.get("platform", ""),
                "contest_id": row.get("contest_id", ""),
            },
        )

    def get_prompt(self) -> str:
        """Generate a prompt for the coder agent.

        Returns:
            Formatted prompt string
        """
        prompt = f"""## Problem: {self.title}

{self.description}

## Test Cases

"""
        for i, tc in enumerate(self.test_cases[:3], 1):  # Show first 3 test cases
            prompt += f"**Example {i}:**\n"
            prompt += f"Input:\n```\n{tc.input}\n```\n"
            prompt += f"Expected Output:\n```\n{tc.expected_output}\n```\n\n"

        if self.starter_code:
            prompt += f"## Starter Code\n\n```python\n{self.starter_code}\n```\n"

        return prompt


def _load_livecodebench_jsonl() -> list[dict[str, Any]]:
    """Load LiveCodeBench data from JSONL files via HuggingFace Hub.

    Returns:
        List of problem dictionaries
    """
    from huggingface_hub import hf_hub_download

    # Download test6.jsonl (latest release with most problems)
    path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        filename="test6.jsonl",
        repo_type="dataset",
    )

    with open(path) as f:
        return [json.loads(line) for line in f]


def load_tasks(
    difficulty: str | TaskDifficulty | None = None,
    after_date: str | datetime | None = None,
    limit: int | None = None,
    random_seed: int = 42,
    exclude_hard: bool = False,
) -> list[CodeTask]:
    """Load coding tasks from LiveCodeBench.

    Args:
        difficulty: Filter by difficulty (easy/medium/hard)
        after_date: Only include problems after this date (for contamination-free eval)
        limit: Maximum number of tasks to return
        random_seed: Seed for random sampling
        exclude_hard: If True, exclude hard problems (useful for running easy+medium)

    Returns:
        List of CodeTask objects
    """
    dataset = _load_livecodebench_jsonl()

    # Convert to list for filtering
    tasks: list[CodeTask] = []

    # Parse filter parameters
    if isinstance(difficulty, str):
        difficulty = TaskDifficulty(difficulty.lower())

    if isinstance(after_date, str):
        after_date = datetime.fromisoformat(after_date)

    for row in dataset:
        task = CodeTask.from_dataset_row(row)

        # Apply filters
        if difficulty and task.difficulty != difficulty:
            continue

        # Exclude hard problems if requested
        if exclude_hard and task.difficulty == TaskDifficulty.HARD:
            continue

        if after_date and task.contest_date < after_date:
            continue

        # Skip tasks without test cases
        if not task.test_cases:
            continue

        tasks.append(task)

    # Random sampling if limit specified
    if limit and len(tasks) > limit:
        import random

        random.seed(random_seed)
        tasks = random.sample(tasks, limit)

    return tasks


def get_task_statistics(tasks: list[CodeTask]) -> dict[str, Any]:
    """Get statistics about a list of tasks.

    Args:
        tasks: List of CodeTask objects

    Returns:
        Dictionary with statistics
    """
    difficulties: dict[str, int] = {}
    for task in tasks:
        d = task.difficulty.value
        difficulties[d] = difficulties.get(d, 0) + 1

    dates = [t.contest_date for t in tasks]

    return {
        "total": len(tasks),
        "by_difficulty": difficulties,
        "date_range": {
            "earliest": min(dates).isoformat() if dates else None,
            "latest": max(dates).isoformat() if dates else None,
        },
        "avg_test_cases": sum(len(t.test_cases) for t in tasks) / len(tasks) if tasks else 0,
    }
