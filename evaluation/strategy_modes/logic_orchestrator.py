"""Orchestrator for logic reasoning strategy modes experiment.

This module wraps ContractExecutor to run logic/word problems with
different strategic modes (URGENT, ECONOMICAL, BALANCED).

Uses OpenR1 Logic Puzzles dataset (Feb 2025) which is guaranteed
uncontaminated in LLM training data.

Example:
    >>> runner = LogicModesRunner(model="gemini/gemini-2.5-flash")
    >>> result = runner.run_task(task, mode="balanced")
    >>> print(f"Correct: {result.correct}")
    >>> print(f"Tokens used: {result.tokens_used}")
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, ClassVar

from agent_contracts import Contract, ContractMode, ResourceConstraints, TemporalConstraints
from agent_contracts.core.executor import ContractExecutor, ExecutionResult

from .logic_tasks import LogicTask, check_logic_answer, extract_logic_answer


@dataclass
class LogicMetrics:
    """Metrics for a single logic problem evaluation.

    Attributes:
        correct: Whether the answer was correct
        predicted_answer: The extracted predicted answer
        expected_answer: The ground truth answer
        answer_extracted: Whether an answer was successfully extracted
    """

    correct: bool = False
    predicted_answer: str = ""
    expected_answer: str = ""
    answer_extracted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "correct": self.correct,
            "predicted_answer": self.predicted_answer,
            "expected_answer": self.expected_answer,
            "answer_extracted": self.answer_extracted,
        }


@dataclass
class LogicTrialResult:
    """Result from a single logic trial (task + mode).

    Attributes:
        task_id: ID of the logic task
        mode: Strategy mode used (urgent/economical/balanced)
        success: Whether execution completed successfully
        correct: Whether the answer was correct
        generated_response: The model's full response
        predicted_answer: Extracted numerical answer
        expected_answer: Ground truth answer
        source: Original source of the problem
        tokens_used: Total tokens consumed
        reasoning_tokens: Tokens used for internal reasoning/thinking
        text_tokens: Tokens used for text output
        execution_time: Wall clock time in seconds
        timeout_seconds: Timeout limit applied (None if no limit)
        timed_out: Whether the trial failed due to timeout
        reasoning_effort: Reasoning effort level used
        metrics: Detailed evaluation metrics
        contract_state: Final contract state
        error: Error message if failed
    """

    task_id: str
    mode: str
    success: bool = False
    correct: bool = False
    generated_response: str = ""
    predicted_answer: str = ""
    expected_answer: str = ""
    source: str = ""
    tokens_used: int = 0
    reasoning_tokens: int = 0
    text_tokens: int = 0
    execution_time: float = 0.0
    timeout_seconds: float | None = None
    timed_out: bool = False
    reasoning_effort: str = ""
    metrics: LogicMetrics = field(default_factory=LogicMetrics)
    contract_state: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "success": self.success,
            "correct": self.correct,
            "predicted_answer": self.predicted_answer,
            "expected_answer": self.expected_answer,
            "source": self.source,
            "tokens_used": self.tokens_used,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "generated_response": self.generated_response,
            "execution_time": self.execution_time,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "reasoning_effort": self.reasoning_effort,
            "metrics": self.metrics.to_dict(),
            "contract_state": self.contract_state,
            "error": self.error,
        }


class LogicModesRunner:
    """Runner for logic reasoning strategy modes experiment.

    This class executes logic tasks with ContractExecutor,
    comparing behavior across URGENT, ECONOMICAL, and BALANCED modes.
    """

    # Default budgets - moderate for logic problems
    DEFAULT_TOKEN_BUDGET = 8000
    DEFAULT_COST_BUDGET = 0.50
    DEFAULT_TIME_BUDGET = timedelta(minutes=5)

    # Mode-specific timeout configuration (shorter for logic vs competition math)
    MODE_TIMEOUTS: ClassVar[dict[str, float]] = {
        "urgent": 15.0,
        "economical": 30.0,
        "balanced": 45.0,
    }

    # Mode-specific reasoning effort
    MODE_REASONING_EFFORT: ClassVar[dict[str, str]] = {
        "urgent": "none",
        "balanced": "medium",
        "economical": "low",
    }

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        token_budget: int | None = None,
        cost_budget: float | None = None,
        time_budget: timedelta | None = None,
        enable_timeout: bool = True,
    ) -> None:
        """Initialize the logic modes runner."""
        self.model = model
        self.token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self.cost_budget = cost_budget or self.DEFAULT_COST_BUDGET
        self.time_budget = time_budget or self.DEFAULT_TIME_BUDGET
        self.enable_timeout = enable_timeout

    def _get_timeout_for_mode(self, mode: str) -> float | None:
        """Get timeout value for a specific mode."""
        if not self.enable_timeout:
            return None
        return self.MODE_TIMEOUTS.get(mode.lower(), self.MODE_TIMEOUTS["balanced"])

    def _get_reasoning_effort_for_mode(self, mode: str) -> str:
        """Get reasoning effort level for a specific mode."""
        return self.MODE_REASONING_EFFORT.get(mode.lower(), self.MODE_REASONING_EFFORT["balanced"])

    def _get_contract_mode(self, mode: str) -> ContractMode:
        """Convert mode string to ContractMode enum."""
        mode_map = {
            "urgent": ContractMode.URGENT,
            "economical": ContractMode.ECONOMICAL,
            "balanced": ContractMode.BALANCED,
        }
        return mode_map.get(mode.lower(), ContractMode.BALANCED)

    def _create_contract(self, task: LogicTask, mode: str) -> Contract:
        """Create a contract for the task with specified mode."""
        from agent_contracts.core.contract import ExecutionConfig

        contract_mode = self._get_contract_mode(mode)
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        return Contract(
            id=f"logic-{task.task_id}-{mode}",
            name=f"Logic: {task.task_id}",
            description=f"Solve logic problem (mode: {mode})",
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
                temperature=0.0,  # Deterministic for logic
                timeout_seconds=timeout,
            ),
        )

    def run_task(
        self,
        task: LogicTask,
        mode: str,
        verbose: bool = False,
    ) -> LogicTrialResult:
        """Run a logic task with specified mode."""
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        result = LogicTrialResult(
            task_id=task.task_id,
            mode=mode,
            expected_answer=task.answer,
            source=task.source,
            timeout_seconds=timeout,
            reasoning_effort=reasoning_effort,
        )

        start_time = time.time()

        try:
            contract = self._create_contract(task, mode)

            if verbose:
                timeout_str = f", Timeout: {timeout}s" if timeout else ""
                print(
                    f"  [Contract] Mode: {mode.upper()}, Budget: {self.token_budget} tokens, "
                    f"Reasoning: {reasoning_effort}{timeout_str}"
                )

            executor = ContractExecutor(
                contract=contract,
                strict_mode=False,
            )

            prompt = task.get_prompt()
            execution_result: ExecutionResult = executor.run(query=prompt)

            result.success = execution_result.success
            result.generated_response = str(execution_result.output or "")
            result.tokens_used = execution_result.tokens_used
            result.reasoning_tokens = execution_result.resource_usage.get("reasoning_tokens", 0)
            result.text_tokens = execution_result.resource_usage.get("text_tokens", 0)
            result.contract_state = execution_result.contract_state.value

            # Compute metrics
            if result.generated_response:
                predicted = extract_logic_answer(result.generated_response)
                correct = check_logic_answer(predicted, task.answer)
                result.metrics = LogicMetrics(
                    correct=correct,
                    predicted_answer=predicted,
                    expected_answer=task.answer,
                    answer_extracted=bool(predicted),
                )
                result.correct = correct
                result.predicted_answer = predicted

            if verbose:
                status = "CORRECT" if result.correct else "WRONG"
                print(
                    f"  [Result] {status} | Predicted: {result.predicted_answer} | "
                    f"Expected: {task.answer} | Tokens: {result.tokens_used}"
                )

        except Exception as e:
            result.success = False
            result.error = str(e)

            error_type = type(e).__name__
            if "Timeout" in error_type or "timeout" in str(e).lower():
                result.timed_out = True
                if verbose:
                    print(f"  [Timeout] Exceeded {timeout}s limit")
            elif verbose:
                print(f"  [Error] {e}")

        result.execution_time = time.time() - start_time
        return result


def compute_logic_statistics(results: list[LogicTrialResult]) -> dict[str, Any]:
    """Compute statistics for a list of results from the same mode."""
    if not results:
        return {}

    successful = [r for r in results if r.success]
    correct = [r for r in results if r.correct]
    timed_out = [r for r in results if r.timed_out]
    n_total = len(results)
    n_success = len(successful)
    n_correct = len(correct)
    n_timed_out = len(timed_out)

    timeout_seconds = results[0].timeout_seconds if results else None
    reasoning_effort = results[0].reasoning_effort if results else ""

    if not successful:
        return {
            "n_trials": n_total,
            "success_rate": 0.0,
            "accuracy": 0.0,
            "timeout_rate": n_timed_out / n_total if n_total > 0 else 0.0,
            "n_timed_out": n_timed_out,
            "timeout_seconds": timeout_seconds,
            "reasoning_effort": reasoning_effort,
            "avg_tokens": 0,
            "avg_reasoning_tokens": 0,
        }

    tokens = [r.tokens_used for r in successful]
    reasoning_tokens = [r.reasoning_tokens for r in successful]
    text_tokens = [r.text_tokens for r in successful]
    execution_times = [r.execution_time for r in successful]

    # Accuracy by source
    by_source: dict[str, dict[str, int]] = {}
    for r in successful:
        source = r.source or "Unknown"
        if source not in by_source:
            by_source[source] = {"total": 0, "correct": 0}
        by_source[source]["total"] += 1
        if r.correct:
            by_source[source]["correct"] += 1

    return {
        "n_trials": n_total,
        "success_rate": n_success / n_total,
        "accuracy": n_correct / n_success if n_success > 0 else 0.0,
        "n_correct": n_correct,
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
        "avg_execution_time": sum(execution_times) / len(execution_times),
        "accuracy_by_source": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in sorted(by_source.items())
        },
    }


def _std(values: list[int] | list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    float_values = [float(v) for v in values]
    mean = sum(float_values) / len(float_values)
    variance = sum((x - mean) ** 2 for x in float_values) / (len(float_values) - 1)
    return float(variance**0.5)
