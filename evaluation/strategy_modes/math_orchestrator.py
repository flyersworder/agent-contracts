"""Orchestrator for math reasoning strategy modes experiment.

This module wraps ContractExecutor to run math competition problems with
different strategic modes (URGENT, ECONOMICAL, BALANCED).

Uses MathArena 2025 datasets (SMT, AIME, CMIMC, etc.) which are
guaranteed uncontaminated in LLM training data.

Example:
    >>> runner = MathModesRunner(model="gemini/gemini-2.5-flash")
    >>> result = runner.run_task(task, mode="balanced")
    >>> print(f"Correct: {result.correct}")
    >>> print(f"Tokens used: {result.tokens_used}")
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, ClassVar

from agent_contracts import Contract, ContractMode, ResourceConstraints, TemporalConstraints
from agent_contracts.core.executor import ContractExecutionResult, ContractExecutor

from .math_metrics import MathMetrics, compute_math_metrics
from .math_tasks import MathTask


@dataclass
class MathTrialResult:
    """Result from a single math trial (task + mode).

    Attributes:
        task_id: ID of the math task
        mode: Strategy mode used (urgent/economical/balanced)
        success: Whether execution completed successfully
        correct: Whether the answer was correct
        generated_response: The model's full response
        predicted_answer: Extracted numerical answer
        expected_answer: Ground truth answer
        problem_type: Problem category (Algebra, Geometry, etc.)
        dataset: Source dataset name
        tokens_used: Total tokens consumed
        reasoning_tokens: Tokens used for internal reasoning/thinking
        text_tokens: Tokens used for text output
        execution_time: Wall clock time in seconds
        timeout_seconds: Timeout limit applied (None if no limit)
        timed_out: Whether the trial failed due to timeout
        reasoning_effort: Reasoning effort level used
        math_metrics: Detailed math evaluation metrics
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
    problem_type: list[str] = field(default_factory=list)
    dataset: str = ""
    tokens_used: int = 0
    reasoning_tokens: int = 0
    text_tokens: int = 0
    execution_time: float = 0.0
    timeout_seconds: float | None = None
    timed_out: bool = False
    reasoning_effort: str = ""
    math_metrics: MathMetrics = field(default_factory=MathMetrics)
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
            "problem_type": self.problem_type,
            "dataset": self.dataset,
            "tokens_used": self.tokens_used,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "generated_response": self.generated_response,
            "execution_time": self.execution_time,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "reasoning_effort": self.reasoning_effort,
            "math_metrics": self.math_metrics.to_dict(),
            "contract_state": self.contract_state,
            "error": self.error,
        }


class MathModesRunner:
    """Runner for math reasoning strategy modes experiment.

    This class executes math tasks with ContractExecutor,
    comparing behavior across URGENT, ECONOMICAL, and BALANCED modes.
    """

    # Default budgets - higher for competition math
    DEFAULT_TOKEN_BUDGET = 20000
    DEFAULT_COST_BUDGET = 1.00
    DEFAULT_TIME_BUDGET = timedelta(minutes=10)

    # Mode-specific timeout configuration (longer for competition math)
    MODE_TIMEOUTS: ClassVar[dict[str, float]] = {
        "urgent": 45.0,
        "economical": 60.0,
        "balanced": 120.0,
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
        """Initialize the math modes runner."""
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

    def _create_contract(self, task: MathTask, mode: str) -> Contract:
        """Create a contract for the task with specified mode."""
        from agent_contracts.core.capabilities import ExecutionConfig

        contract_mode = self._get_contract_mode(mode)
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        return Contract(
            id=f"math-{task.task_id}-{mode}",
            name=f"Math: {task.task_id}",
            description=f"Solve competition math problem (mode: {mode})",
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
                temperature=0.0,  # Deterministic for math
                timeout_seconds=timeout,
            ),
        )

    def run_task(
        self,
        task: MathTask,
        mode: str,
        verbose: bool = False,
    ) -> MathTrialResult:
        """Run a math task with specified mode."""
        timeout = self._get_timeout_for_mode(mode)
        reasoning_effort = self._get_reasoning_effort_for_mode(mode)

        result = MathTrialResult(
            task_id=task.task_id,
            mode=mode,
            expected_answer=task.answer,
            problem_type=task.problem_type,
            dataset=task.dataset,
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
            execution_result: ContractExecutionResult = executor.run(query=prompt)

            result.success = execution_result.success
            result.generated_response = str(execution_result.output or "")
            result.tokens_used = execution_result.tokens_used
            result.reasoning_tokens = execution_result.resource_usage.get("reasoning_tokens", 0)
            result.text_tokens = execution_result.resource_usage.get("text_tokens", 0)
            result.contract_state = execution_result.contract_state.value

            # Compute math metrics
            if result.generated_response:
                result.math_metrics = compute_math_metrics(
                    response=result.generated_response,
                    expected_answer=task.answer,
                )
                result.correct = result.math_metrics.correct
                result.predicted_answer = result.math_metrics.predicted_answer

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


def compute_mode_statistics(results: list[MathTrialResult]) -> dict[str, Any]:
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

    # Accuracy by problem type (for successful trials only)
    by_type: dict[str, dict[str, int]] = {}
    for r in successful:
        for pt in r.problem_type:
            if pt not in by_type:
                by_type[pt] = {"total": 0, "correct": 0}
            by_type[pt]["total"] += 1
            if r.correct:
                by_type[pt]["correct"] += 1

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
        "accuracy_by_type": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in sorted(by_type.items())
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
