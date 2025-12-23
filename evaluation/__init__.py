"""Evaluation experiments for Agent Contracts framework validation.

This package contains evaluation experiments designed to validate
the Agent Contracts framework through CONTRACTED vs UNCONTRACTED comparisons.

Evaluation Pipelines:

1. Research Pipeline (`research_pipeline/`)
   - Multi-agent report generation (Researcher → Analyzer → Reporter)
   - 25 research topics across 5 categories
   - Validates budget enforcement and conservation laws

2. Code Review Pipeline (`code_review_pipeline/`)
   - Coder ↔ Reviewer iterative loop
   - 175 LiveCodeBench problems (post-Feb 2025)
   - Validates iteration limits and runaway prevention

Indeterminacy-Aware Evaluator:
- Implements NeurIPS 2025 framework for rating ambiguity
- Response set elicitation captures genuine uncertainty
- MSE(srs/srs) metric for comparing judge systems
"""

from .indeterminacy_evaluator import (
    IndeterminacyAwareEvaluator,
    IndeterminacyAwareScore,
    MultiLabelScore,
    ResponseSet,
    decision_consistency,
    mse_srs_srs,
    prevalence_bias,
)
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
    "IndeterminacyAwareEvaluator",
    "IndeterminacyAwareScore",
    "MultiLabelScore",
    "ResearchTopic",
    "ResponseSet",
    "decision_consistency",
    "get_topic",
    "get_topics_by_category",
    "get_topics_by_difficulty",
    "mse_srs_srs",
    "prevalence_bias",
]
