"""Pipeline orchestrators for code review experiment.

This module implements two pipeline variants:
1. UncontractedPipeline: No governance, agents can loop indefinitely
2. ContractedPipeline: Agent Contracts governance with conservation laws

The comparison demonstrates Agent Contracts' value:
- Runaway prevention via iteration limits
- Conservation laws for budget delegation
- Predictable resource consumption
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .agents import (
    PipelineConfig,
    get_coder_instruction,
    get_reviewer_instruction,
)
from .execution import (
    FailureReason,
    execute_task,
    format_test_result,
)
from .tasks import CodeTask
from .usage_tracker import (
    CumulativeUsage,
    IterationUsage,
    generate_coder_status_prefix,
    generate_reviewer_status_prefix,
)


def _violation_token_count(error_str: str) -> int | None:
    """Recover an agent's cumulative token count from a contract-violation error.

    When a contract is violated, the agent's ``.run()`` raises before the
    orchestrator can read its usage, so the tokens already consumed would
    otherwise be lost (recorded as zero). The contract monitor records the
    count in the exception text, e.g. ``ViolationInfo(tokens: 66648 > 30000)``;
    this parses it back out so a halted trial is charged the resources it
    actually used rather than nothing.
    """
    match = re.search(r"tokens:\s*(\d+)", error_str)
    return int(match.group(1)) if match else None


@dataclass
class IterationDetail:
    """Details for a single Coder→Reviewer iteration.

    Attributes:
        iteration: Iteration number (1-indexed)
        coder_tokens: Tokens used by Coder
        coder_llm_calls: LLM calls made by Coder
        reviewer_tokens: Tokens used by Reviewer
        reviewer_llm_calls: LLM calls made by Reviewer
        test_passed: Whether tests passed this iteration
        code_snippet: First 200 chars of generated code
        feedback_snippet: First 200 chars of reviewer feedback
    """

    iteration: int
    coder_tokens: int = 0
    coder_llm_calls: int = 0
    reviewer_tokens: int = 0
    reviewer_llm_calls: int = 0
    test_passed: bool = False
    code_snippet: str = ""
    feedback_snippet: str = ""


@dataclass
class PipelineResult:
    """Result from a pipeline execution.

    Attributes:
        task: The coding task
        mode: "CONTRACTED" or "UNCONTRACTED"
        success: Whether the solution was approved
        num_iterations: Number of Coder→Reviewer rounds
        final_code: The final generated code
        total_tokens: Total tokens consumed
        tokens_by_agent: Token breakdown by agent
        total_llm_calls: Total LLM calls
        llm_calls_by_agent: LLM call breakdown by agent
        execution_time_seconds: Wall clock time
        budget_compliant: Whether stayed within budget
        runaway_prevented: Whether iteration limit stopped execution
        failure_reason: Why the task failed (if applicable)
        iteration_details: Per-iteration details
        error: Error message if failed
    """

    task: CodeTask
    mode: str
    success: bool = False
    num_iterations: int = 0
    final_code: str = ""
    total_tokens: int = 0
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    total_llm_calls: int = 0
    llm_calls_by_agent: dict[str, int] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    budget_compliant: bool = True
    runaway_prevented: bool = False
    failure_reason: FailureReason = FailureReason.NONE
    iteration_details: list[IterationDetail] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task.task_id,
            "task_title": self.task.title,
            "difficulty": self.task.difficulty.value,
            "mode": self.mode,
            "success": self.success,
            "num_iterations": self.num_iterations,
            "total_tokens": self.total_tokens,
            "tokens_by_agent": self.tokens_by_agent,
            "total_llm_calls": self.total_llm_calls,
            "llm_calls_by_agent": self.llm_calls_by_agent,
            "execution_time_seconds": self.execution_time_seconds,
            "budget_compliant": self.budget_compliant,
            "runaway_prevented": self.runaway_prevented,
            "failure_reason": self.failure_reason.value,
            "error": self.error,
        }


class UncontractedPipeline:
    """Code review pipeline WITHOUT Agent Contracts governance.

    This pipeline has no hard limits - agents can potentially loop forever
    (only limited by a safety max_iterations parameter).

    Used as baseline to compare against ContractedPipeline.
    """

    # Safety limit to prevent actual runaway (but higher than contracted)
    # Set to 6 to keep experiment runtime reasonable while still showing runaway behavior
    # (Contracted uses 3 iterations, matching the capstone project)
    SAFETY_MAX_ITERATIONS = 6

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize uncontracted pipeline.

        Args:
            config: Pipeline configuration (used for model selection only)
        """
        self.config = config or PipelineConfig()
        self.model = self.config.model

    def run(self, task: CodeTask) -> PipelineResult:
        """Run the code review pipeline on a task.

        Args:
            task: The coding task to solve

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()
        result = PipelineResult(task=task, mode="UNCONTRACTED")

        try:
            result = self._execute_loop(task, result)
        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time_seconds = time.time() - start_time
        return result

    def _execute_loop(self, task: CodeTask, result: PipelineResult) -> PipelineResult:
        """Execute the Coder→Reviewer loop.

        This uses Google ADK agents without Agent Contracts wrapping.
        """
        import asyncio

        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools import FunctionTool
        from google.genai import types

        # Track state
        current_code = ""
        feedback = ""
        iteration = 0

        # Create session service
        session_service = InMemorySessionService()

        # Helper to create sessions (async API requires asyncio.run)
        def create_session_sync(app_name: str, user_id: str, session_id: str) -> str:
            """Create a session synchronously, returning the session ID."""

            async def _create() -> str:
                session = await session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                return str(session.id)

            return asyncio.run(_create())

        # Storage for test results (closure)
        test_storage: dict[str, Any] = {"code": "", "task": task}

        def test_code() -> str:
            """Test the current code against test cases."""
            code = test_storage["code"]
            if not code:
                return "FAIL: No code provided"

            exec_result = execute_task(code, test_storage["task"])
            return format_test_result(test_storage["task"], exec_result)

        test_tool = FunctionTool(func=test_code)

        # Create agents (no budget limits)
        # Use same temperature as contracted pipeline to avoid confounding variables
        coder = LlmAgent(
            model=self.model,
            name="Coder",
            instruction=get_coder_instruction(task.get_prompt(), budget_aware=False),
            tools=[],
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        reviewer = LlmAgent(
            model=self.model,
            name="Reviewer",
            instruction=get_reviewer_instruction(budget_aware=False),
            tools=[test_tool],
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        # Run loop
        coder_runner = Runner(
            agent=coder,
            app_name="code_review_uncontracted",
            session_service=session_service,
        )
        reviewer_runner = Runner(
            agent=reviewer,
            app_name="code_review_uncontracted",
            session_service=session_service,
        )

        while iteration < self.SAFETY_MAX_ITERATIONS:
            iteration += 1
            iter_detail = IterationDetail(iteration=iteration)

            # Coder turn
            coder_message = (
                "Write code to solve the problem."
                if iteration == 1
                else f"Previous code failed. Feedback:\n{feedback}\n\nFix the code."
            )

            coder_session_id = create_session_sync(
                app_name="code_review_uncontracted",
                user_id="eval",
                session_id=f"coder_{task.task_id}_{iteration}",
            )

            coder_tokens = 0
            coder_calls = 0

            for event in coder_runner.run(
                user_id="eval",
                session_id=coder_session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=coder_message)]),
            ):
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    tokens = getattr(event.usage_metadata, "total_token_count", 0) or 0
                    if tokens > 0:
                        coder_tokens += tokens
                        coder_calls += 1
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            current_code = part.text

            iter_detail.coder_tokens = coder_tokens
            iter_detail.coder_llm_calls = coder_calls
            iter_detail.code_snippet = current_code[:200] if current_code else ""
            result.tokens_by_agent["coder"] = result.tokens_by_agent.get("coder", 0) + coder_tokens
            result.llm_calls_by_agent["coder"] = (
                result.llm_calls_by_agent.get("coder", 0) + coder_calls
            )

            # Update test storage
            test_storage["code"] = current_code

            # Reviewer turn
            reviewer_session_id = create_session_sync(
                app_name="code_review_uncontracted",
                user_id="eval",
                session_id=f"reviewer_{task.task_id}_{iteration}",
            )

            reviewer_tokens = 0
            reviewer_calls = 0
            reviewer_response = ""

            for event in reviewer_runner.run(
                user_id="eval",
                session_id=reviewer_session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text="Test the code and provide your decision.")],
                ),
            ):
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    tokens = getattr(event.usage_metadata, "total_token_count", 0) or 0
                    if tokens > 0:
                        reviewer_tokens += tokens
                        reviewer_calls += 1
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            reviewer_response = part.text

            iter_detail.reviewer_tokens = reviewer_tokens
            iter_detail.reviewer_llm_calls = reviewer_calls
            iter_detail.feedback_snippet = reviewer_response[:200]
            result.tokens_by_agent["reviewer"] = (
                result.tokens_by_agent.get("reviewer", 0) + reviewer_tokens
            )
            result.llm_calls_by_agent["reviewer"] = (
                result.llm_calls_by_agent.get("reviewer", 0) + reviewer_calls
            )

            # Check for approval
            if "APPROVE" in reviewer_response.upper():
                iter_detail.test_passed = True
                result.iteration_details.append(iter_detail)
                result.success = True
                result.final_code = current_code
                break

            feedback = reviewer_response
            result.iteration_details.append(iter_detail)

        # Finalize results
        result.num_iterations = iteration
        result.total_tokens = sum(result.tokens_by_agent.values())
        result.total_llm_calls = sum(result.llm_calls_by_agent.values())

        # Check if we hit safety limit (runaway)
        if iteration >= self.SAFETY_MAX_ITERATIONS and not result.success:
            result.runaway_prevented = True
            result.failure_reason = FailureReason.TIMEOUT

        return result


class ContractedPipeline:
    """Code review pipeline WITH Agent Contracts governance.

    This pipeline uses Agent Contracts to enforce:
    - Hard iteration limits per agent
    - Conservation laws for budget delegation
    - Predictable resource consumption

    Demonstrates the value of Agent Contracts over uncontracted execution.
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize contracted pipeline.

        Args:
            config: Pipeline configuration with budget constraints
        """
        self.config = config or PipelineConfig()

    def run(self, task: CodeTask) -> PipelineResult:
        """Run the contracted code review pipeline.

        Args:
            task: The coding task to solve

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()
        result = PipelineResult(task=task, mode="CONTRACTED")

        try:
            result = self._execute_with_contracts(task, result)
        except Exception as e:
            result.error = str(e)
            result.success = False

        result.execution_time_seconds = time.time() - start_time
        return result

    def _execute_with_contracts(self, task: CodeTask, result: PipelineResult) -> PipelineResult:
        """Execute with Agent Contracts governance."""
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool
        from google.genai import types

        from agent_contracts import Contract, ResourceConstraints
        from agent_contracts.integrations.google_adk import DelegatingAdkAgent

        # Create parent contract with conservation laws
        parent_contract = Contract(
            id=f"code-review-{task.task_id}",
            name=f"Code Review: {task.title}",
            description=f"Solve coding problem: {task.title}",
            resources=ResourceConstraints(
                tokens=self.config.parent_tokens,
                iterations=self.config.parent_iterations,
            ),
        )

        # Create delegating orchestrator
        orchestrator_agent = LlmAgent(
            model=self.config.model,
            name="Orchestrator",
            instruction="Coordinate the code review process.",
        )

        orchestrator = DelegatingAdkAgent(
            agent=orchestrator_agent,
            contract=parent_contract,
            reserve_ratio=self.config.reserve_ratio,
        )

        # Track state
        current_code = ""
        feedback = ""
        iteration = 0

        # Storage for test results
        test_storage: dict[str, Any] = {"code": "", "task": task}

        def test_code() -> str:
            """Test the current code against test cases."""
            code = test_storage["code"]
            if not code:
                return "FAIL: No code provided"
            exec_result = execute_task(code, test_storage["task"])
            return format_test_result(test_storage["task"], exec_result)

        test_tool = FunctionTool(func=test_code)

        # Create child agents
        coder_agent = LlmAgent(
            model=self.config.model,
            name="Coder",
            instruction=get_coder_instruction(task.get_prompt(), budget_aware=True),
            tools=[],
            generate_content_config=types.GenerateContentConfig(temperature=0.2),
        )

        reviewer_agent = LlmAgent(
            model=self.config.model,
            name="Reviewer",
            instruction=get_reviewer_instruction(budget_aware=True),
            tools=[test_tool],
            generate_content_config=types.GenerateContentConfig(temperature=0.2),
        )

        # Delegate with conservation laws
        contracted_coder = orchestrator.delegate(
            name="coder",
            agent=coder_agent,
            tokens=self.config.coder_tokens,
            iterations=self.config.coder_iterations,
            description="Write and revise code",
        )

        contracted_reviewer = orchestrator.delegate(
            name="reviewer",
            agent=reviewer_agent,
            tokens=self.config.reviewer_tokens,
            iterations=self.config.reviewer_iterations,
            description="Test and review code",
        )

        # Run loop with contract enforcement
        max_rounds = self.config.max_review_rounds

        # Track cumulative usage for dynamic status updates
        cumulative_usage = CumulativeUsage()

        while iteration < max_rounds:
            iteration += 1
            iter_detail = IterationDetail(iteration=iteration)
            iter_usage = IterationUsage(iteration=iteration)

            # Generate dynamic status prefix for Coder (iteration 2+)
            # This gives the agent visibility into resource consumption
            if iteration == 1:
                coder_message = "Write code to solve the problem."
            else:
                # Include status update showing iteration progress and token usage
                status_prefix = generate_coder_status_prefix(
                    iteration=iteration,
                    max_iterations=max_rounds,
                    tokens_used=cumulative_usage.coder_tokens,
                    token_budget=self.config.coder_tokens,
                )
                coder_message = (
                    f"{status_prefix}Previous code failed. Feedback:\n{feedback}\n\nFix the code."
                )

            try:
                coder_result = contracted_coder.run(
                    user_id="eval",
                    session_id=f"coder_{task.task_id}_{iteration}",
                    message=coder_message,
                )

                current_code = coder_result.get("response", "")
                iter_detail.coder_tokens = coder_result.get("total_tokens", 0)
                iter_detail.coder_llm_calls = coder_result.get("llm_calls", 0)
                iter_detail.code_snippet = current_code[:200]

                # Update iteration usage for status tracking
                iter_usage.coder_total_tokens = iter_detail.coder_tokens

                result.tokens_by_agent["coder"] = (
                    result.tokens_by_agent.get("coder", 0) + iter_detail.coder_tokens
                )
                result.llm_calls_by_agent["coder"] = (
                    result.llm_calls_by_agent.get("coder", 0) + iter_detail.coder_llm_calls
                )

            except RuntimeError as e:
                # Contract violation or limit reached (ADK raises RuntimeError)
                error_str = str(e)
                if "violated" in error_str.lower() or "limit" in error_str.lower():
                    result.error = f"Coder contract violation: {e}"
                    result.budget_compliant = False
                    # The Coder consumed tokens before the contract halted it;
                    # recover that count from the violation so the halted trial
                    # is not silently charged zero. The monitor count is
                    # cumulative for the trial, so overwrite rather than add.
                    consumed = _violation_token_count(error_str)
                    if consumed is not None:
                        result.tokens_by_agent["coder"] = consumed
                else:
                    result.error = f"Coder execution error: {e}"
                result.iteration_details.append(iter_detail)
                break
            except Exception as e:
                # Unexpected errors (not contract violations)
                result.error = f"Coder unexpected error: {e}"
                result.iteration_details.append(iter_detail)
                break

            # Update test storage
            test_storage["code"] = current_code

            # Generate status prefix for Reviewer
            # Reviewer also gets visibility into iteration progress
            reviewer_status_prefix = generate_reviewer_status_prefix(
                iteration=iteration,
                max_iterations=max_rounds,
                tokens_used=cumulative_usage.reviewer_tokens,
                token_budget=self.config.reviewer_tokens,
            )
            reviewer_message = f"{reviewer_status_prefix}Test the code and provide your decision."

            # Reviewer turn (with contract limits)
            try:
                reviewer_result = contracted_reviewer.run(
                    user_id="eval",
                    session_id=f"reviewer_{task.task_id}_{iteration}",
                    message=reviewer_message,
                )

                reviewer_response = reviewer_result.get("response", "")
                iter_detail.reviewer_tokens = reviewer_result.get("total_tokens", 0)
                iter_detail.reviewer_llm_calls = reviewer_result.get("llm_calls", 0)
                iter_detail.feedback_snippet = reviewer_response[:200]

                # Update iteration usage for status tracking
                iter_usage.reviewer_total_tokens = iter_detail.reviewer_tokens

                result.tokens_by_agent["reviewer"] = (
                    result.tokens_by_agent.get("reviewer", 0) + iter_detail.reviewer_tokens
                )
                result.llm_calls_by_agent["reviewer"] = (
                    result.llm_calls_by_agent.get("reviewer", 0) + iter_detail.reviewer_llm_calls
                )

                # Update cumulative usage for next iteration's status messages
                cumulative_usage.add_iteration(iter_usage)

            except RuntimeError as e:
                # Contract violation or limit reached (ADK raises RuntimeError)
                error_str = str(e)
                if "violated" in error_str.lower() or "limit" in error_str.lower():
                    result.error = f"Reviewer contract violation: {e}"
                    result.budget_compliant = False
                    # Recover tokens consumed before the halt (see Coder branch).
                    consumed = _violation_token_count(error_str)
                    if consumed is not None:
                        result.tokens_by_agent["reviewer"] = consumed
                else:
                    result.error = f"Reviewer execution error: {e}"
                result.iteration_details.append(iter_detail)
                break
            except Exception as e:
                # Unexpected errors (not contract violations)
                result.error = f"Reviewer unexpected error: {e}"
                result.iteration_details.append(iter_detail)
                break

            # Check for approval
            if "APPROVE" in reviewer_response.upper():
                iter_detail.test_passed = True
                result.iteration_details.append(iter_detail)
                result.success = True
                result.final_code = current_code
                break

            feedback = reviewer_response
            result.iteration_details.append(iter_detail)

        # Finalize results
        result.num_iterations = iteration
        result.total_tokens = sum(result.tokens_by_agent.values())
        result.total_llm_calls = sum(result.llm_calls_by_agent.values())

        # Check conservation law compliance
        summary = orchestrator.get_delegation_summary()
        result.budget_compliant = summary["conservation_satisfied"]

        # Check if iteration limit stopped us
        if iteration >= max_rounds and not result.success:
            result.runaway_prevented = True
            result.failure_reason = FailureReason.TIMEOUT

        return result
