"""Task loading from CNN/DailyMail dataset for strategy modes experiment.

This module handles loading and filtering summarization tasks from the
CNN/DailyMail dataset for evaluating ContractExecutor strategy modes.

Example:
    >>> tasks = load_tasks(limit=100, random_seed=42)
    >>> print(f"Loaded {len(tasks)} tasks")
    >>> task = tasks[0]
    >>> print(f"Article: {task.article[:100]}...")
    >>> print(f"Reference: {task.reference_summary}")
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SummarizationTask:
    """A summarization task from CNN/DailyMail.

    Attributes:
        task_id: Unique identifier for the task
        article: The full article text to summarize
        reference_summary: The ground truth summary (highlights)
        article_length: Number of characters in the article
        source: Source of the article (cnn or dailymail)
    """

    task_id: str
    article: str
    reference_summary: str
    article_length: int
    source: str = "unknown"

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any], index: int) -> "SummarizationTask":
        """Create a SummarizationTask from a dataset row.

        Args:
            row: Row from the CNN/DailyMail dataset
            index: Index for generating task_id

        Returns:
            SummarizationTask instance
        """
        article = row.get("article", "")
        highlights = row.get("highlights", "")
        task_id = row.get("id", f"task_{index}")

        # Determine source from ID pattern
        source = "cnn" if task_id.startswith("c") else "dailymail"

        return cls(
            task_id=task_id,
            article=article,
            reference_summary=highlights,
            article_length=len(article),
            source=source,
        )

    def get_prompt(self) -> str:
        """Generate the summarization prompt.

        Returns:
            Formatted prompt string for the LLM
        """
        return f"""Summarize the following news article:

{self.article}

Provide a clear, concise summary capturing the key points."""


def load_tasks(
    limit: int | None = None,
    min_article_length: int = 500,
    max_article_length: int = 10000,
    random_seed: int = 42,
) -> list[SummarizationTask]:
    """Load summarization tasks from CNN/DailyMail dataset.

    Args:
        limit: Maximum number of tasks to return
        min_article_length: Minimum article length in characters
        max_article_length: Maximum article length in characters
        random_seed: Seed for random sampling

    Returns:
        List of SummarizationTask objects
    """
    from datasets import load_dataset

    # Load test split (streaming to avoid downloading entire dataset)
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)

    tasks: list[SummarizationTask] = []
    index = 0

    for row in dataset:
        article = row.get("article", "")
        article_len = len(article)

        # Filter by article length
        if article_len < min_article_length:
            continue
        if article_len > max_article_length:
            continue

        # Skip articles without highlights
        if not row.get("highlights"):
            continue

        task = SummarizationTask.from_dataset_row(row, index)
        tasks.append(task)
        index += 1

        # Check limit (before shuffling for efficiency with streaming)
        if limit and len(tasks) >= limit * 2:
            # Collect 2x limit, then shuffle and take limit
            break

    # Shuffle and limit
    if limit and len(tasks) > limit:
        import random

        random.seed(random_seed)
        random.shuffle(tasks)
        tasks = tasks[:limit]

    return tasks


def get_task_statistics(tasks: list[SummarizationTask]) -> dict[str, Any]:
    """Get statistics about a list of tasks.

    Args:
        tasks: List of SummarizationTask objects

    Returns:
        Dictionary with statistics
    """
    if not tasks:
        return {"total": 0}

    article_lengths = [t.article_length for t in tasks]
    reference_lengths = [len(t.reference_summary) for t in tasks]

    sources: dict[str, int] = {}
    for task in tasks:
        sources[task.source] = sources.get(task.source, 0) + 1

    return {
        "total": len(tasks),
        "by_source": sources,
        "article_length": {
            "min": min(article_lengths),
            "max": max(article_lengths),
            "avg": sum(article_lengths) / len(article_lengths),
        },
        "reference_length": {
            "min": min(reference_lengths),
            "max": max(reference_lengths),
            "avg": sum(reference_lengths) / len(reference_lengths),
        },
    }
