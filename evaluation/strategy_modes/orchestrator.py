"""Orchestrator for strategy modes experiment.

This module wraps ContractExecutor to run summarization tasks with
different strategic modes (URGENT, ECONOMICAL, BALANCED).

Example:
    >>> runner = StrategyModesRunner(model="gpt-4o-mini")
    >>> result = runner.run_task(task, mode="economical")
    >>> print(f"Tokens used: {result.tokens_used}")
    >>> print(f"ROUGE-L: {result.rouge_metrics.rouge_l_f1:.3f}")
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from agent_contracts import Contract, ContractMode, ResourceConstraints, TemporalConstraints
from agent_contracts.core.executor import ContractExecutor, ExecutionResult

from .metrics import RougeMetrics, compute_rouge
from .tasks import SummarizationTask


@dataclass
class TrialResult:
    """Result from a single trial (task + mode).

    Attributes:
        task_id: ID of the summarization task
        mode: Strategy mode used (urgent/economical/balanced)
        success: Whether execution completed successfully
        generated_summary: The generated summary text
        reference_summary: The ground truth summary
        tokens_used: Total tokens consumed
        output_length: Length of generated summary (characters)
        word_count: Word count of generated summary
        execution_time: Wall clock time in seconds
        rouge_metrics: ROUGE evaluation scores
        strategy_recommendation: Strategy from recommend_strategy()
        contract_state: Final contract state
        error: Error message if failed
    """

    task_id: str
    mode: str
    success: bool = False
    generated_summary: str = ""
    reference_summary: str = ""
    tokens_used: int = 0
    output_length: int = 0
    word_count: int = 0
    execution_time: float = 0.0
    rouge_metrics: RougeMetrics = field(default_factory=RougeMetrics)
    strategy_recommendation: str = ""
    contract_state: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "success": self.success,
            "tokens_used": self.tokens_used,
            "output_length": self.output_length,
            "word_count": self.word_count,
            "execution_time": self.execution_time,
            "rouge_metrics": self.rouge_metrics.to_dict(),
            "strategy_recommendation": self.strategy_recommendation,
            "contract_state": self.contract_state,
            "error": self.error,
        }


class StrategyModesRunner:
    """Runner for strategy modes experiment using ContractExecutor.

    This class executes summarization tasks with ContractExecutor,
    comparing behavior across URGENT, ECONOMICAL, and BALANCED modes.

    Attributes:
        model: LLM model to use for summarization
        token_budget: Maximum tokens per task
        cost_budget: Maximum cost per task in USD
        time_budget: Maximum time per task
    """

    # Default budgets (generous to allow mode differences to show)
    DEFAULT_TOKEN_BUDGET = 4000
    DEFAULT_COST_BUDGET = 0.50
    DEFAULT_TIME_BUDGET = timedelta(minutes=5)

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        token_budget: int | None = None,
        cost_budget: float | None = None,
        time_budget: timedelta | None = None,
    ) -> None:
        """Initialize the strategy modes runner.

        Args:
            model: LLM model identifier (LiteLLM format, e.g., gemini/gemini-2.0-flash)
            token_budget: Maximum tokens per task
            cost_budget: Maximum cost per task in USD
            time_budget: Maximum time per task
        """
        self.model = model
        self.token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self.cost_budget = cost_budget or self.DEFAULT_COST_BUDGET
        self.time_budget = time_budget or self.DEFAULT_TIME_BUDGET

    def _get_contract_mode(self, mode: str) -> ContractMode:
        """Convert mode string to ContractMode enum.

        Args:
            mode: Mode string (urgent/economical/balanced)

        Returns:
            ContractMode enum value
        """
        mode_map = {
            "urgent": ContractMode.URGENT,
            "economical": ContractMode.ECONOMICAL,
            "balanced": ContractMode.BALANCED,
        }
        return mode_map.get(mode.lower(), ContractMode.BALANCED)

    def _create_contract(self, task: SummarizationTask, mode: str) -> Contract:
        """Create a contract for the task with specified mode.

        Args:
            task: The summarization task
            mode: Strategy mode

        Returns:
            Contract configured for the mode
        """
        from agent_contracts.core.contract import ExecutionConfig

        contract_mode = self._get_contract_mode(mode)

        return Contract(
            id=f"summarize-{task.task_id}-{mode}",
            name=f"Summarization: {task.task_id}",
            description=f"Summarize news article (mode: {mode})",
            mode=contract_mode,
            resources=ResourceConstraints(
                tokens=self.token_budget,
                cost_usd=self.cost_budget,
            ),
            temporal=TemporalConstraints(
                max_duration=self.time_budget,
            ),
            execution=ExecutionConfig(
                model=self.model,
                temperature=0.3,  # Lower temperature for summarization
            ),
        )

    def run_task(
        self,
        task: SummarizationTask,
        mode: str,
        verbose: bool = False,
    ) -> TrialResult:
        """Run a summarization task with specified mode.

        Args:
            task: The summarization task
            mode: Strategy mode (urgent/economical/balanced)
            verbose: If True, print progress

        Returns:
            TrialResult with execution details
        """
        result = TrialResult(
            task_id=task.task_id,
            mode=mode,
            reference_summary=task.reference_summary,
        )

        start_time = time.time()

        try:
            # Create contract with mode
            contract = self._create_contract(task, mode)

            if verbose:
                print(f"  [Contract] Mode: {mode.upper()}, Budget: {self.token_budget} tokens")

            # Create executor
            executor = ContractExecutor(
                contract=contract,
                strict_mode=False,  # Don't raise on violations, just report
            )

            # Run summarization
            prompt = task.get_prompt()
            execution_result: ExecutionResult = executor.run(query=prompt)

            # Extract results
            result.success = execution_result.success
            result.generated_summary = str(execution_result.output or "")
            result.tokens_used = execution_result.tokens_used
            result.output_length = len(result.generated_summary)
            result.word_count = len(result.generated_summary.split())
            result.contract_state = execution_result.contract_state.value

            # Get strategy recommendation
            if execution_result.strategy:
                result.strategy_recommendation = execution_result.strategy.recommended_approach

            # Compute ROUGE metrics
            if result.generated_summary:
                result.rouge_metrics = compute_rouge(
                    hypothesis=result.generated_summary,
                    reference=task.reference_summary,
                )

            if verbose:
                print(f"  [Result] Tokens: {result.tokens_used}, Words: {result.word_count}")
                print(f"  [Quality] ROUGE-L: {result.rouge_metrics.rouge_l_f1:.3f}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            if verbose:
                print(f"  [Error] {e}")

        result.execution_time = time.time() - start_time
        return result

    def run_all_modes(
        self,
        task: SummarizationTask,
        verbose: bool = False,
    ) -> dict[str, TrialResult]:
        """Run a task with all three modes.

        Args:
            task: The summarization task
            verbose: If True, print progress

        Returns:
            Dictionary mapping mode to TrialResult
        """
        results = {}
        for mode in ["urgent", "economical", "balanced"]:
            if verbose:
                print(f"\n  Running {mode.upper()} mode...")
            results[mode] = self.run_task(task, mode, verbose=verbose)
        return results


def compute_mode_statistics(results: list[TrialResult]) -> dict[str, Any]:
    """Compute statistics for a list of results from the same mode.

    Args:
        results: List of TrialResult objects

    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}

    successful = [r for r in results if r.success]
    n_total = len(results)
    n_success = len(successful)

    if not successful:
        return {
            "n_trials": n_total,
            "success_rate": 0.0,
            "avg_tokens": 0,
            "avg_word_count": 0,
            "avg_rouge_l_f1": 0.0,
        }

    tokens = [r.tokens_used for r in successful]
    word_counts = [r.word_count for r in successful]
    rouge_l_scores = [r.rouge_metrics.rouge_l_f1 for r in successful]
    execution_times = [r.execution_time for r in successful]

    return {
        "n_trials": n_total,
        "success_rate": n_success / n_total,
        "avg_tokens": sum(tokens) / len(tokens),
        "std_tokens": _std(tokens),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "avg_word_count": sum(word_counts) / len(word_counts),
        "std_word_count": _std(word_counts),
        "avg_rouge_l_f1": sum(rouge_l_scores) / len(rouge_l_scores),
        "std_rouge_l_f1": _std(rouge_l_scores),
        "avg_execution_time": sum(execution_times) / len(execution_times),
    }


def _std(values: list[int] | list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    float_values = [float(v) for v in values]
    mean = sum(float_values) / len(float_values)
    variance = sum((x - mean) ** 2 for x in float_values) / (len(float_values) - 1)
    return float(variance**0.5)
