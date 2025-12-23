"""Multi-agent research pipeline for evaluation experiment.

This module implements the multi-agent research report generation pipeline
as described in SUBMISSION_PLAN.md:

Parent Contract: B = 100,000 tokens, $2.00, 15 minutes
├── Orchestrator: 10,000 tokens (coordination, validation)
├── Researcher: 40,000 tokens (web search, data gathering)
├── Analyzer: 25,000 tokens (pattern identification, insights)
└── Reporter: 25,000 tokens (synthesis, writing)

Success Criteria Φ (from Section 8):
- All sections complete (weight = 0.4)
- ≥2,000 words (weight = 0.3)
- ≥5 citations (weight = 0.3)
- Threshold θ = 0.8
"""

from .topics import (
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
