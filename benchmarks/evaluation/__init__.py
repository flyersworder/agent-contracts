"""Evaluation experiments for COINE 2026 paper.

This package contains the evaluation experiments designed to validate
the Agent Contracts framework for the COINE 2026 submission.

Primary Experiment: Multi-Agent Research Report Generation
- Validates core claims about budget enforcement and conservation laws
- Uses 25 research topics across 5 categories
- Compares UNCONTRACTED vs CONTRACTED conditions
"""

from .research_pipeline.topics import (
    ALL_TOPICS,
    TOPICS_BY_CATEGORY,
    TOPICS_BY_ID,
    ResearchTopic,
    get_topic,
    get_topics_by_category,
    get_topics_by_difficulty,
)

__all__ = [
    "ALL_TOPICS",
    "TOPICS_BY_CATEGORY",
    "TOPICS_BY_ID",
    "ResearchTopic",
    "get_topic",
    "get_topics_by_category",
    "get_topics_by_difficulty",
]
