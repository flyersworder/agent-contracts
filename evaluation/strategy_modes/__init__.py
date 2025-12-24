"""Strategy Modes experiment for COINE 2026 evaluation.

This experiment demonstrates ContractExecutor, the core execution engine
that provides comprehensive Agent Contracts features for single LLM calls.

The experiment compares three strategic modes:
- URGENT: Optimize for speed, accept approximations
- ECONOMICAL: Minimize resource usage, be concise
- BALANCED: Standard thorough execution

Task: CNN/DailyMail summarization (quality-effort tradeoff)
"""

from .metrics import RougeMetrics, compute_rouge
from .orchestrator import StrategyModesRunner, TrialResult
from .tasks import SummarizationTask, load_tasks

__all__ = [
    "RougeMetrics",
    "StrategyModesRunner",
    "SummarizationTask",
    "TrialResult",
    "compute_rouge",
    "load_tasks",
]
