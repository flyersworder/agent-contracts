"""Code Review Pipeline Evaluation.

This module implements a code review experiment comparing CONTRACTED vs UNCONTRACTED
agent pipelines using LiveCodeBench problems.

The experiment demonstrates Agent Contracts' value:
- Runaway prevention via iteration limits
- Conservation laws for multi-agent budget delegation
- Predictable resource consumption

Architecture:
    UNCONTRACTED: Coder → Reviewer → Loop (no limits)
    CONTRACTED: Parent Contract → Coder (child) → Reviewer (child) → Loop (hard limits)
"""

from .execution import TestResult, execute_code
from .orchestrator import (
    ContractedPipeline,
    PipelineResult,
    UncontractedPipeline,
)
from .tasks import CodeTask, TaskDifficulty, load_tasks

__all__ = [
    "CodeTask",
    "ContractedPipeline",
    "PipelineResult",
    "TaskDifficulty",
    "TestResult",
    "UncontractedPipeline",
    "execute_code",
    "load_tasks",
]
