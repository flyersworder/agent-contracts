"""Strategy Modes experiment for COINE 2026 evaluation.

This experiment demonstrates ContractExecutor, the core execution engine
that provides comprehensive Agent Contracts features for single LLM calls.

The experiment compares three strategic modes:
- URGENT: Optimize for speed, accept approximations
- ECONOMICAL: Minimize resource usage, be concise
- BALANCED: Standard thorough execution

Available task types:
- Summarization: CNN/DailyMail (quality-effort tradeoff, LLM-as-judge evaluation)
- Math: MathArena 2025 (competition math, deterministic evaluation)
- Logic: OpenR1 Logic Puzzles (reasoning tasks, deterministic evaluation)
"""

from .logic_orchestrator import LogicModesRunner, LogicTrialResult, compute_logic_statistics
from .logic_tasks import LogicTask, load_logic_tasks
from .math_orchestrator import MathModesRunner, MathTrialResult, compute_mode_statistics
from .math_tasks import MathTask, load_math_tasks
from .metrics import RougeMetrics, compute_rouge
from .orchestrator import StrategyModesRunner, TrialResult
from .tasks import SummarizationTask, load_tasks

__all__ = [
    "LogicModesRunner",
    "LogicTask",
    "LogicTrialResult",
    "MathModesRunner",
    "MathTask",
    "MathTrialResult",
    "RougeMetrics",
    "StrategyModesRunner",
    "SummarizationTask",
    "TrialResult",
    "compute_logic_statistics",
    "compute_mode_statistics",
    "compute_rouge",
    "load_logic_tasks",
    "load_math_tasks",
    "load_tasks",
]
