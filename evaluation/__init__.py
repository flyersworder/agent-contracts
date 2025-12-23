"""Evaluation experiments for COINE 2026 paper.

This package contains the evaluation experiments designed to validate
the Agent Contracts framework for the COINE 2026 submission.

Primary Experiment: Multi-Agent Research Report Generation
- Validates core claims about budget enforcement and conservation laws
- Uses 25 research topics across 5 categories
- Compares UNCONTRACTED vs CONTRACTED conditions

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
