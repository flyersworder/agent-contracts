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
from typing import Any, ClassVar

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
        reasoning_tokens: Tokens used for internal reasoning/thinking
        text_tokens: Tokens used for text output
        reasoning_content: The actual reasoning/thinking text from the model
        output_length: Length of generated summary (characters)
        word_count: Word count of generated summary
        execution_time: Wall clock time in seconds
        timeout_seconds: Timeout limit applied (None if no limit)
        timed_out: Whether the trial failed due to timeout
        truncated: Whether output was truncated due to soft cutoff (partial result)
        reasoning_effort: Reasoning effort level used (low/medium/high)
        rouge_metrics: ROUGE evaluation scores
        contract_state: Final contract state
        error: Error message if failed
    """

    task_id: str
    mode: str
    success: bool = False
    generated_summary: str = ""
    reference_summary: str = ""
    tokens_used: int = 0
    reasoning_tokens: int = 0
    text_tokens: int = 0
    reasoning_content: str = ""
    output_length: int = 0
    word_count: int = 0
    execution_time: float = 0.0
    timeout_seconds: float | None = None
    timed_out: bool = False
    truncated: bool = False  # Output was truncated due to soft cutoff
    reasoning_effort: str = ""
    rouge_metrics: RougeMetrics = field(default_factory=RougeMetrics)
    contract_state: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "success": self.success,
            "tokens_used": self.tokens_used,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "reasoning_content": self.reasoning_content,
            "generated_summary": self.generated_summary,
            "output_length": self.output_length,
            "word_count": self.word_count,
            "execution_time": self.execution_time,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "reasoning_effort": self.reasoning_effort,
            "rouge_metrics": self.rouge_metrics.to_dict(),
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
        enable_timeout: Whether to enforce mode-specific timeouts
        soft_cutoff: If True, return partial output on timeout instead of failing
    """

    # Default budgets (generous to allow mode differences to show)
    DEFAULT_TOKEN_BUDGET = 4000
    DEFAULT_COST_BUDGET = 0.50
    DEFAULT_TIME_BUDGET = timedelta(minutes=5)

    # Mode-specific timeout configuration (in seconds)
    # These values enforce the temporal aspect of contract modes:
    # - URGENT: Tight timeout enforces "speed is critical" (creates real pressure)
    # - ECONOMICAL: Tight timeout encourages brevity (shorter output = fewer tokens = lower cost)
    # - BALANCED: Generous timeout for thoroughness (quality requires time)
    # Values calibrated based on Gemini 2.5 Flash typical response times (1.5-5.5s)
    MODE_TIMEOUTS: ClassVar[dict[str, float]] = {
        "urgent": 8.0,  # 8s - speed pressure, forces fast completion
        "economical": 10.0,  # 10s - brevity pressure, encourages concise output
        "balanced": 30.0,  # 30s - ample time for thorough, quality responses
    }

    # Mode-specific reasoning effort configuration
    # Controls how deeply the LLM should think before responding:
    # - URGENT: No thinking for fastest possible response
    # - BALANCED: Moderate thinking for quality/speed balance
    # - ECONOMICAL: Shallow thinking to minimize token usage on reasoning
    MODE_REASONING_EFFORT: ClassVar[dict[str, str]] = {
        "urgent": "none",  # No thinking, fastest response (up to 96% cheaper)
        "balanced": "medium",  # Balanced thinking (~500-2000 tokens)
        "economical": "low",  # Minimal reasoning tokens for cost savings
    }

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        token_budget: int | None = None,
        cost_budget: float | None = None,
        time_budget: timedelta | None = None,
        enable_timeout: bool = True,
        soft_cutoff: bool = False,
    ) -> None:
        """Initialize the strategy modes runner.

        Args:
            model: LLM model identifier (LiteLLM format, e.g., gemini/gemini-2.0-flash)
            token_budget: Maximum tokens per task
            cost_budget: Maximum cost per task in USD
            time_budget: Maximum time per task
            enable_timeout: If True, enforce mode-specific API timeouts
            soft_cutoff: If True, use streaming and return partial output on timeout
                instead of failing completely. This enables graceful degradation.
        """
        self.model = model
        self.token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self.cost_budget = cost_budget or self.DEFAULT_COST_BUDGET
        self.time_budget = time_budget or self.DEFAULT_TIME_BUDGET
        self.enable_timeout = enable_timeout
        self.soft_cutoff = soft_cutoff

    def _get_timeout_for_mode(self, mode: str) -> float | None:
        """Get timeout value for a specific mode.

        Args:
            mode: Strategy mode (urgent/economical/balanced)

        Returns:
            Timeout in seconds, or None if timeouts disabled
        """
        if not self.enable_timeout:
            return None
        return self.MODE_TIMEOUTS.get(mode.lower(), self.MODE_TIMEOUTS["balanced"])

    def _get_reasoning_effort_for_mode(self, mode: str) -> str:
        """Get reasoning effort level for a specific mode.

        Args:
            mode: Strategy mode (urgent/economical/balanced)

        Returns:
            Reasoning effort level ("low", "medium", or "high")
        """
        return self.MODE_REASONING_EFFORT.get(mode.lower(), self.MODE_REASONING_EFFORT["balanced"])

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
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        return Contract(
            id=f"summarize-{task.task_id}-{mode}",
            name=f"Summarization: {task.task_id}",
            description=f"Summarize news article (mode: {mode})",
            mode=contract_mode,
            resources=ResourceConstraints(
                tokens=self.token_budget,
                cost_usd=self.cost_budget,
                reasoning_effort=reasoning_effort,
            ),
            temporal=TemporalConstraints(
                max_duration=self.time_budget,
            ),
            execution=ExecutionConfig(
                model=self.model,
                temperature=0.3,  # Lower temperature for summarization
                timeout_seconds=timeout,  # Mode-specific timeout enforcement
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
        # Get mode-specific configuration
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        result = TrialResult(
            task_id=task.task_id,
            mode=mode,
            reference_summary=task.reference_summary,
            timeout_seconds=timeout,
            reasoning_effort=reasoning_effort,
        )

        start_time = time.time()

        try:
            # Create contract with mode
            contract = self._create_contract(task, mode)

            if verbose:
                timeout_str = f", Timeout: {timeout}s" if timeout else ""
                print(
                    f"  [Contract] Mode: {mode.upper()}, Budget: {self.token_budget} tokens, "
                    f"Reasoning: {reasoning_effort}{timeout_str}"
                )

            # Create executor with soft cutoff support
            executor = ContractExecutor(
                contract=contract,
                strict_mode=False,  # Don't raise on violations, just report
                soft_cutoff=self.soft_cutoff,  # Enable graceful degradation on timeout
            )

            # Run summarization
            prompt = task.get_prompt()
            execution_result: ExecutionResult = executor.run(query=prompt)

            # Extract results
            result.success = execution_result.success
            result.generated_summary = str(execution_result.output or "")
            result.tokens_used = execution_result.tokens_used
            result.reasoning_tokens = execution_result.resource_usage.get("reasoning_tokens", 0)
            result.text_tokens = execution_result.resource_usage.get("text_tokens", 0)
            result.reasoning_content = execution_result.resource_usage.get("reasoning_content", "")
            result.output_length = len(result.generated_summary)
            result.word_count = len(result.generated_summary.split())
            result.contract_state = execution_result.contract_state.value
            result.truncated = execution_result.truncated  # Track soft cutoff truncation

            # Compute ROUGE metrics
            if result.generated_summary:
                result.rouge_metrics = compute_rouge(
                    hypothesis=result.generated_summary,
                    reference=task.reference_summary,
                )

            if verbose:
                reasoning_info = f"reasoning: {result.reasoning_tokens}, text: {result.text_tokens}"
                if result.reasoning_content:
                    reasoning_info += f", content: {len(result.reasoning_content)} chars"
                truncated_info = " [TRUNCATED]" if result.truncated else ""
                print(
                    f"  [Result] Tokens: {result.tokens_used} ({reasoning_info}), "
                    f"Words: {result.word_count}{truncated_info}"
                )
                print(f"  [Quality] ROUGE-L: {result.rouge_metrics.rouge_l_f1:.3f}")

        except Exception as e:
            result.success = False
            result.error = str(e)

            # Check if this was a timeout error
            # LiteLLM raises openai.APITimeoutError on timeout
            error_type = type(e).__name__
            if "Timeout" in error_type or "timeout" in str(e).lower():
                result.timed_out = True
                if verbose:
                    print(f"  [Timeout] Exceeded {timeout}s limit")
            elif verbose:
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
        Dictionary with aggregate statistics including timeout rates
    """
    if not results:
        return {}

    successful = [r for r in results if r.success]
    timed_out = [r for r in results if r.timed_out]
    n_total = len(results)
    n_success = len(successful)
    n_timed_out = len(timed_out)

    # Get mode-specific settings from first result (all same mode have same config)
    timeout_seconds = results[0].timeout_seconds if results else None
    reasoning_effort = results[0].reasoning_effort if results else ""

    if not successful:
        return {
            "n_trials": n_total,
            "success_rate": 0.0,
            "timeout_rate": n_timed_out / n_total if n_total > 0 else 0.0,
            "n_timed_out": n_timed_out,
            "timeout_seconds": timeout_seconds,
            "reasoning_effort": reasoning_effort,
            "avg_tokens": 0,
            "avg_reasoning_tokens": 0,
            "avg_text_tokens": 0,
            "avg_word_count": 0,
            "avg_rouge_l_f1": 0.0,
        }

    tokens = [r.tokens_used for r in successful]
    reasoning_tokens = [r.reasoning_tokens for r in successful]
    text_tokens = [r.text_tokens for r in successful]
    word_counts = [r.word_count for r in successful]
    rouge_l_scores = [r.rouge_metrics.rouge_l_f1 for r in successful]
    execution_times = [r.execution_time for r in successful]

    return {
        "n_trials": n_total,
        "success_rate": n_success / n_total,
        "timeout_rate": n_timed_out / n_total,
        "n_timed_out": n_timed_out,
        "timeout_seconds": timeout_seconds,
        "reasoning_effort": reasoning_effort,
        "avg_tokens": sum(tokens) / len(tokens),
        "std_tokens": _std(tokens),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "avg_reasoning_tokens": sum(reasoning_tokens) / len(reasoning_tokens),
        "std_reasoning_tokens": _std(reasoning_tokens),
        "avg_text_tokens": sum(text_tokens) / len(text_tokens),
        "std_text_tokens": _std(text_tokens),
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
