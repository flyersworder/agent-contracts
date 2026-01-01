"""Google ADK agents for Good Enough experiment with Agent Contracts framework.

This module implements two agent variants using the Agent Contracts framework:

1. UnconstrainedAdkAgent: No SuccessCriterion, keeps improving
2. ContractedAdkAgent: Has SuccessCriterion with Q_min threshold

The key difference is that CONTRACTED uses a proper Contract with
SuccessCriterion that specifies when "good enough" is achieved.

This demonstrates:
- Agent Contracts as formal specification (not just prompt engineering)
- Agents that understand and respect their contracts
- True agentic behavior where the agent decides when to stop
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner, RunConfig
from google.genai import types

from agent_contracts import Contract, ResourceConstraints, SuccessCriterion

from .adk_tools import (
    EMAIL_TOOLS,
    get_execution_stats,
    set_scenario,
)
from .email_indeterminacy_evaluator import EmailIndeterminacyEvaluator
from .scenarios import EmailScenario

# Use AUTO mode - agent decides when to call tools and when to stop
# ANY mode causes infinite loops because agent can never stop calling tools
TOOL_USE_CONFIG = types.GenerateContentConfig(
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.AUTO,
        )
    ),
)

# Base instruction shared by both agents
# This is IDENTICAL for both - the difference is in the contract section
BASE_INSTRUCTION = """You are an expert email writer. Your task is to draft a professional email.

TOOLS:
- evaluate_quality: Assess your draft. Returns quality_score (0-1 scale) and feedback.
- submit_email: Submit final email. Calling this COMPLETES the task.

WORKFLOW:
1. Draft an email for the scenario
2. Call evaluate_quality with your complete email text
3. Review the quality_score and feedback
4. Either improve and re-evaluate, OR call submit_email to complete

Start by drafting an email, then call evaluate_quality."""


# Instruction for UNCONSTRAINED agent - no contract, uses subjective judgment
# Note: Sees urgency context but has no formal constraint to respect it
UNCONSTRAINED_INSTRUCTION = (
    BASE_INSTRUCTION
    + """

You have NO formal quality threshold or iteration limit.
Use your professional judgment to decide when the email is good enough to send.
Consider the feedback, quality_score, and any urgency mentioned in the scenario.
The final decision on when to submit is entirely yours."""
)


# Instruction for CONTRACTED agent - has explicit contract with SuccessCriterion
# Includes both quality threshold AND iteration constraints for crisis scenarios
CONTRACTED_INSTRUCTION_STANDARD = (
    BASE_INSTRUCTION
    + """

YOUR CONTRACT SPECIFIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SuccessCriterion: quality_score >= {quality_threshold}

  This is a BINDING requirement, not a suggestion.
  When quality_score meets the threshold, submit immediately.
  Continuing past the threshold wastes resources.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After each evaluation, compare quality_score to your threshold ({quality_threshold}).
If quality_score >= {quality_threshold}: call submit_email immediately.
If quality_score < {quality_threshold}: improve the email and evaluate again."""
)


# Instruction for CONTRACTED agent in CRISIS scenarios
# Has BOTH quality threshold AND max iteration constraint
CONTRACTED_INSTRUCTION_CRISIS = (
    BASE_INSTRUCTION
    + """

⚠️ CRISIS SCENARIO - TIME-SENSITIVE ⚠️

YOUR CONTRACT SPECIFIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SuccessCriterion: quality_score >= {quality_threshold}
  IterationLimit: maximum {max_iterations} evaluation cycles

  CRITICAL: In crisis situations, SPEED matters.
  - Waiting for "perfect" is WORSE than sending "good enough"
  - Submit when quality >= threshold OR after {max_iterations} iterations
  - Every additional iteration delays critical communication
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decision logic (check in order):
1. If quality_score >= {quality_threshold}: submit immediately
2. If you've already done {max_iterations} evaluations: submit your best version
3. Otherwise: improve once more and evaluate again

Remember: In a crisis, a "good enough" message sent NOW is better than
a "perfect" message sent too late."""
)


def get_contracted_instruction(
    quality_threshold: float,
    max_iterations: int | None = None,
) -> str:
    """Get the appropriate contracted instruction based on scenario type.

    Args:
        quality_threshold: The quality threshold for success
        max_iterations: If set, indicates a crisis scenario with iteration limit

    Returns:
        Formatted instruction string
    """
    if max_iterations is not None:
        return CONTRACTED_INSTRUCTION_CRISIS.format(
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )
    else:
        return CONTRACTED_INSTRUCTION_STANDARD.format(
            quality_threshold=quality_threshold,
        )


@dataclass
class AdkAgentResult:
    """Result from running an ADK email agent.

    Attributes:
        scenario_id: ID of the scenario
        condition: "UNCONSTRAINED" or "CONTRACTED"
        final_email: Final email output
        final_quality: Final quality score
        iterations: Number of evaluation iterations
        stopped_early: Whether agent stopped before max iterations
        stop_reason: Why the agent stopped
        total_tokens: Total tokens consumed (estimated)
        contract_id: ID of the contract (if any)
        events: Raw ADK events for debugging
    """

    scenario_id: str
    condition: str
    final_email: str = ""
    final_quality: float = 0.0
    iterations: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    total_tokens: int = 0
    contract_id: str | None = None
    events: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "final_email": self.final_email,
            "final_quality": self.final_quality,
            "iterations": self.iterations,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "total_tokens": self.total_tokens,
            "contract_id": self.contract_id,
        }


class BaseAdkEmailAgent:
    """Base class for ADK email agents.

    Provides common functionality for running ADK agents on email scenarios.
    Uses the simpler run_debug() API for reliability.
    """

    def __init__(
        self,
        instruction: str,
        condition_name: str,
        contract: Contract | None = None,
        model: str = "gemini-2.5-flash",  # Note: flash-lite has bugs with tool continuation
        max_llm_calls: int = 30,
        num_judges: int = 3,
    ) -> None:
        """Initialize the ADK email agent.

        Args:
            instruction: Agent instruction (defines behavior)
            condition_name: "UNCONSTRAINED" or "CONTRACTED"
            contract: Optional Contract for governance (Agent Contracts framework)
            model: LLM model to use for the agent
            max_llm_calls: Maximum LLM calls (safety limit)
            num_judges: Number of judges for quality evaluation
        """
        self.instruction = instruction
        self.condition_name = condition_name
        self.contract = contract
        self.model = model
        self.max_llm_calls = max_llm_calls

        # Create evaluator for quality assessment
        self.evaluator = EmailIndeterminacyEvaluator(
            judge_model="gemini/gemini-2.5-flash-lite",
            num_judges=num_judges,
        )

        # Create ADK agent with tools as simple functions
        # Use TOOL_USE_CONFIG to force tool calling (prevents text-only responses)
        self.agent = LlmAgent(
            name=f"email_agent_{condition_name.lower()}",
            model=model,
            instruction=instruction,
            tools=EMAIL_TOOLS,
            generate_content_config=TOOL_USE_CONFIG,
        )

        # Create runner
        self._app_name = f"good-enough-{condition_name.lower()}"
        self.runner = InMemoryRunner(
            agent=self.agent,
            app_name=self._app_name,
        )

    def run(
        self,
        scenario: EmailScenario,
        verbose: bool = False,
    ) -> AdkAgentResult:
        """Run the agent on an email scenario.

        Args:
            scenario: Email scenario to work on
            verbose: Print progress

        Returns:
            AdkAgentResult with execution details
        """
        # Initialize tool state with scenario
        set_scenario(scenario, self.evaluator)

        result = AdkAgentResult(
            scenario_id=scenario.id,
            condition=self.condition_name,
            contract_id=self.contract.id if self.contract else None,
        )

        # Run agent using the proper run_async() API
        run_config = RunConfig(max_llm_calls=self.max_llm_calls)
        prompt = f"Please draft an email for this scenario:\n\n{scenario.to_prompt()}"

        # Unique session per scenario and condition
        user_id = "experiment_user"
        session_id = f"session_{self.condition_name.lower()}_{scenario.id}"

        async def _run_agent() -> list[Any]:
            """Run the agent asynchronously using proper run_async pattern."""
            events_list: list[Any] = []

            # Step 1: Create session BEFORE running (required by ADK)
            session_service = self.runner.session_service
            await session_service.create_session(
                app_name=self._app_name,
                user_id=user_id,
                session_id=session_id,
            )

            # Step 2: Format message properly using types.Content
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )

            # Step 3: Run agent with run_async and collect events
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
                run_config=run_config,
            ):
                events_list.append(event)

                # Verbose output for debugging
                if verbose:
                    if event.get_function_calls():
                        for fc in event.get_function_calls():
                            print(f"    [Tool Call] {fc.name}({fc.args})")
                    if event.get_function_responses():
                        for fr in event.get_function_responses():
                            resp_str = str(fr.response)[:100]
                            print(f"    [Tool Result] {fr.name}: {resp_str}...")
                    if event.is_final_response() and event.content and event.content.parts:
                        text = event.content.parts[0].text if event.content.parts else ""
                        print(f"    [Final Response] {text[:100]}...")

            return events_list

        try:
            # Handle async execution - use asyncio.run for clean execution
            events = asyncio.run(_run_agent())
            result.events = events

            # Check events for submit_email call
            for event in events:
                if hasattr(event, "get_function_responses"):
                    try:
                        responses = event.get_function_responses()
                        if responses:
                            for response in responses:
                                if getattr(response, "name", "") == "submit_email":
                                    result.stopped_early = True
                                    result.stop_reason = "agent_submitted"
                    except (TypeError, AttributeError):
                        pass

        except Exception as e:
            result.stop_reason = f"error: {e}"
            import traceback

            if verbose:
                traceback.print_exc()

        # Get execution stats from tools
        stats = get_execution_stats()
        result.iterations = stats["iterations"]
        result.total_tokens = stats["total_tokens"]
        result.final_email = stats["final_email"]

        # Evaluate final quality
        if result.final_email:
            final_score = self.evaluator.evaluate(result.final_email, scenario)
            result.final_quality = final_score.weighted_score

        # Determine stop reason if not already set
        if not result.stop_reason:
            result.stop_reason = "max_llm_calls_reached"

        if verbose:
            print(f"    Iterations: {result.iterations}, Q: {result.final_quality:.2f}")

        return result


class UnconstrainedAdkAgent(BaseAdkEmailAgent):
    """ADK agent without a success criterion contract.

    This agent has NO quality threshold in its contract. It is instructed
    to create the best possible email, representing current AI behavior
    optimized for engagement/helpfulness rather than efficiency.

    Contract: ResourceConstraints only (iteration limit for safety)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_llm_calls: int = 30,
        num_judges: int = 3,
    ) -> None:
        """Initialize unconstrained ADK agent.

        Args:
            model: LLM model for email drafting
            max_llm_calls: Maximum LLM calls (safety limit)
            num_judges: Number of judges for evaluation
        """
        # Create contract WITHOUT success criterion
        contract = Contract(
            id="unconstrained-email-agent",
            name="Unconstrained Email Agent",
            description="Email agent with no quality threshold - keeps improving",
            resources=ResourceConstraints(
                iterations=max_llm_calls,  # Safety limit only
            ),
            # No success_criteria - agent has no "good enough" threshold
        )

        super().__init__(
            instruction=UNCONSTRAINED_INSTRUCTION,
            condition_name="UNCONSTRAINED",
            contract=contract,
            model=model,
            max_llm_calls=max_llm_calls,
            num_judges=num_judges,
        )


class ContractedAdkAgent(BaseAdkEmailAgent):
    """ADK agent with a quality-based success criterion contract.

    This agent has a proper Contract with SuccessCriterion that defines
    when the task is "good enough" (Q >= Q_min). The agent's instruction
    tells it to honor this contract and stop when the threshold is met.

    For CRISIS scenarios, the contract also includes an iteration limit,
    enforcing "good enough NOW" rather than "perfect later".

    Contract: ResourceConstraints + SuccessCriterion(quality >= threshold)
              + Optional IterationLimit for crisis scenarios

    This demonstrates the Agent Contracts value proposition: agents that
    recognize "good enough" and optimize for human benefit rather than
    endless improvement.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_llm_calls: int = 30,
        quality_threshold: float = 0.80,
        num_judges: int = 3,
        scenario_max_iterations: int | None = None,
    ) -> None:
        """Initialize contracted ADK agent.

        Args:
            model: LLM model for email drafting
            max_llm_calls: Maximum LLM calls (safety limit)
            quality_threshold: Q_min threshold (in contract's SuccessCriterion)
            num_judges: Number of judges for evaluation
            scenario_max_iterations: For crisis scenarios, max iterations before
                                    submitting (from scenario.max_iterations)
        """
        self.quality_threshold = quality_threshold
        self.scenario_max_iterations = scenario_max_iterations

        # Build success criteria
        success_criteria = [
            SuccessCriterion(
                name="quality_threshold",
                condition=f"quality_score >= {quality_threshold}",
                weight=1.0,
                required=True,  # This criterion MUST be met
            ),
        ]

        # For crisis scenarios, add iteration limit criterion
        if scenario_max_iterations is not None:
            success_criteria.append(
                SuccessCriterion(
                    name="iteration_limit",
                    condition=f"iterations <= {scenario_max_iterations}",
                    weight=0.5,
                    required=False,  # Soft constraint - submit if exceeded
                ),
            )

        # Create contract WITH success criterion
        contract = Contract(
            id="contracted-email-agent",
            name="Contracted Email Agent",
            description=f"Email agent with quality threshold Q_min={quality_threshold}"
            + (f" and max {scenario_max_iterations} iterations" if scenario_max_iterations else ""),
            resources=ResourceConstraints(
                iterations=max_llm_calls,  # Safety limit
            ),
            success_criteria=success_criteria,
        )

        # Get appropriate instruction based on scenario type
        instruction = get_contracted_instruction(
            quality_threshold=quality_threshold,
            max_iterations=scenario_max_iterations,
        )

        super().__init__(
            instruction=instruction,
            condition_name="CONTRACTED",
            contract=contract,
            model=model,
            max_llm_calls=max_llm_calls,
            num_judges=num_judges,
        )
