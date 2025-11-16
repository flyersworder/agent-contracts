"""Google Agent Development Kit (ADK) integration for Agent Contracts.

This module provides contract-aware wrappers for Google ADK agents,
enabling resource governance and budget enforcement for ADK applications.

Google ADK is a code-first Python framework for building sophisticated AI agents
with tools, multi-agent systems, and hierarchical coordination. This integration
wraps ADK agents with contract enforcement to prevent runaway costs.

Example:
    >>> from google.adk.agents import LlmAgent
    >>> from google.adk.runners import InMemoryRunner
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.google_adk import ContractedAdkAgent
    >>>
    >>> # Create standard ADK agent
    >>> agent = LlmAgent(
    ...     name="research_agent",
    ...     model="gemini-2.0-flash",
    ...     instruction="You are a research assistant.",
    ...     tools=[...]
    ... )
    >>>
    >>> # Wrap with contract
    >>> contract = Contract(
    ...     id="my-agent",
    ...     resources=ResourceConstraints(tokens=50000, cost_usd=2.0)
    ... )
    >>> contracted_agent = ContractedAdkAgent(
    ...     contract=contract,
    ...     agent=agent
    ... )
    >>>
    >>> # Execute with automatic budget enforcement
    >>> result = contracted_agent.run(
    ...     user_id="user123",
    ...     session_id="session456",
    ...     message="Research quantum computing"
    ... )
"""

from typing import Any

from agent_contracts.core.contract import Contract
from agent_contracts.core.wrapper import ContractAgent

# Type checking imports
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Event, InMemoryRunner

    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    LlmAgent = Any
    Event = Any
    InMemoryRunner = Any


class ContractedAdkAgent(ContractAgent[dict[str, Any], dict[str, Any]]):
    """Contract-aware wrapper for Google ADK agents.

    This class wraps any Google ADK LlmAgent and adds contract enforcement
    with automatic token tracking and budget monitoring.

    The wrapper:
    - Tracks token usage from all LLM calls and tool executions
    - Enforces budget constraints across multi-turn conversations
    - Logs execution for audit
    - Provides budget awareness to the agent
    - Supports both single and multi-agent systems

    Key Features:
    - Automatic token tracking via ADK's usage metadata
    - Multi-agent coordination with shared budget
    - Tool execution monitoring
    - Cached content tracking

    Attributes:
        contract: The contract governing execution
        agent: The underlying Google ADK LlmAgent
        runner: The ADK Runner for executing the agent
        enforcer: Contract enforcement engine
        resource_monitor: Resource consumption tracker
        temporal_monitor: Time constraint tracker

    Example:
        >>> from google.adk.agents import LlmAgent
        >>> from agent_contracts import Contract, ResourceConstraints
        >>>
        >>> contract = Contract(
        ...     id="research-agent",
        ...     resources=ResourceConstraints(tokens=50000, cost_usd=2.0)
        ... )
        >>> agent = LlmAgent(
        ...     name="researcher",
        ...     model="gemini-2.0-flash",
        ...     instruction="Research assistant",
        ...     tools=[search_tool, calculator_tool]
        ... )
        >>> contracted = ContractedAdkAgent(contract=contract, agent=agent)
        >>>
        >>> result = contracted.run(
        ...     user_id="user1",
        ...     session_id="session1",
        ...     message="What is quantum entanglement?"
        ... )
        >>> print(result.output["response"])
        >>> print(result.execution_log.resource_usage)
    """

    def __init__(
        self,
        contract: Contract,
        agent: Any,  # Google ADK LlmAgent type
        strict_mode: bool = True,
        enable_logging: bool = True,
        runner: Any | None = None,  # Optional custom Runner
    ) -> None:
        """Initialize contracted Google ADK agent.

        Args:
            contract: Contract to enforce
            agent: Google ADK LlmAgent to wrap
            strict_mode: If True, violations cause immediate termination
            enable_logging: If True, log execution for audit trail
            runner: Optional custom Runner (defaults to InMemoryRunner)

        Raises:
            ImportError: If google-adk is not installed
        """
        if not GOOGLE_ADK_AVAILABLE:
            raise ImportError(
                "google-adk is required for Google ADK integration. "
                "Install with: pip install google-adk"
            )

        # Initialize base ContractAgent with agent execution as callable
        super().__init__(
            contract=contract,
            agent=self._run_agent,
            strict_mode=strict_mode,
            enable_logging=enable_logging,
        )

        self.agent = agent

        # Set up runner (use provided or create InMemoryRunner)
        if runner is not None:
            self.runner = runner
        else:
            # Import here to avoid issues if google-adk not installed
            from google.adk.runners import InMemoryRunner

            # InMemoryRunner just needs the agent
            self.runner = InMemoryRunner(agent=agent)

    def _run_agent(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the Google ADK agent.

        This method is called by ContractAgent's execute() method.

        Args:
            inputs: Input dictionary containing:
                - user_id: User identifier
                - session_id: Session identifier
                - message: User message (string or Content)
                - run_config: Optional run configuration

        Returns:
            Dictionary containing:
                - response: Final agent response text
                - events: List of all events from execution
                - total_tokens: Total tokens used
                - usage_metadata: Detailed usage information
        """
        # Extract parameters
        user_id = inputs.get("user_id", "user")
        session_id = inputs.get("session_id", "session")
        message = inputs.get("message", "")
        run_config = inputs.get("run_config")

        # Convert message to Content if it's a string
        if isinstance(message, str):
            from google.genai.types import Content, Part

            content = Content(parts=[Part(text=message)])
        else:
            content = message

        # Run agent and collect events
        events: list[Any] = []
        final_response = ""
        cumulative_usage: dict[str, int] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "candidates_tokens": 0,
            "cached_tokens": 0,
            "thoughts_tokens": 0,
        }

        # Execute agent via runner
        event_generator = self.runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
            run_config=run_config,
        )

        # Process events and track usage
        for event in event_generator:
            events.append(event)

            # Track token usage from each event
            if event.usageMetadata:
                usage = event.usageMetadata
                event_tokens = usage.total_token_count or 0

                # Update cumulative tracking
                cumulative_usage["total_tokens"] += event_tokens
                cumulative_usage["prompt_tokens"] += usage.prompt_token_count or 0
                cumulative_usage["candidates_tokens"] += usage.candidates_token_count or 0
                cumulative_usage["cached_tokens"] += usage.cached_content_token_count or 0
                cumulative_usage["thoughts_tokens"] += usage.thoughts_token_count or 0

                # Track tokens in resource monitor
                if event_tokens > 0:
                    # Track tokens with breakdown
                    reasoning_tokens = usage.thoughts_token_count or 0
                    text_tokens = event_tokens - reasoning_tokens

                    self.resource_monitor.usage.add_tokens(
                        count=0,  # count not used when text/reasoning provided
                        reasoning=reasoning_tokens,
                        text=text_tokens,
                    )

                    # Track API call with cost estimate
                    # Gemini 2.0 Flash: ~$0.075 per 1M input, ~$0.30 per 1M output
                    # Use weighted average based on typical input/output ratio
                    prompt_cost = (usage.prompt_token_count or 0) * 0.000000075
                    output_cost = (usage.candidates_token_count or 0) * 0.00000030
                    total_cost = prompt_cost + output_cost

                    self.resource_monitor.usage.add_api_call(cost=total_cost, tokens=0)

            # Extract final response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response = part.text

            # Check constraints during execution
            is_violated, violations = self.enforcer.check_constraints()
            if is_violated and self.strict_mode:
                # Stop execution on violation
                raise RuntimeError(f"Contract violated during execution: {violations}")

        return {
            "response": final_response,
            "events": events,
            "total_tokens": cumulative_usage["total_tokens"],
            "usage_metadata": cumulative_usage,
        }

    def _monitored_execution(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute agent with monitoring.

        Overrides base method to add Google ADK-specific monitoring.

        Args:
            input_data: Input dictionary for the agent

        Returns:
            Agent's output dictionary
        """
        # Add budget awareness to inputs
        if "budget_info" not in input_data:
            input_data["budget_info"] = {
                "remaining_tokens": self.resource_monitor.get_remaining_tokens(),
                "remaining_cost": self.resource_monitor.get_remaining_cost(),
                "remaining_api_calls": self.resource_monitor.get_remaining_api_calls(),
                "time_pressure": self.temporal_monitor.get_time_pressure(),
            }

        # Execute agent
        return self._run_agent(input_data)

    def run(
        self,
        user_id: str,
        session_id: str,
        message: str,
        run_config: Any | None = None,
    ) -> dict[str, Any]:
        """Execute agent with contract enforcement (ADK-style API).

        This method provides a Google ADK-compatible interface that delegates
        to our execute() method.

        Args:
            user_id: User identifier
            session_id: Session identifier
            message: User message
            run_config: Optional run configuration

        Returns:
            Dictionary with response and metadata

        Raises:
            RuntimeError: If execution fails or contract is violated
        """
        # Build inputs dict
        inputs = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "run_config": run_config,
        }

        # Execute with contract enforcement
        result = self.execute(inputs)

        if result.success and result.output:
            return result.output
        else:
            raise RuntimeError(f"Agent execution failed: {result.violations}")

    def run_debug(
        self,
        message: str,
        user_id: str = "debug_user",
        session_id: str = "debug_session",
    ) -> dict[str, Any]:
        """Convenient debug execution with contract enforcement.

        Args:
            message: User message
            user_id: User identifier (defaults to "debug_user")
            session_id: Session identifier (defaults to "debug_session")

        Returns:
            Dictionary with response and metadata
        """
        return self.run(user_id=user_id, session_id=session_id, message=message)

    def __call__(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Make the contracted agent callable.

        Args:
            user_id: User identifier
            session_id: Session identifier
            message: User message

        Returns:
            Dictionary with response and metadata
        """
        return self.run(user_id=user_id, session_id=session_id, message=message)


class ContractedAdkMultiAgent(ContractedAdkAgent):
    """Contract-aware wrapper for Google ADK multi-agent systems.

    This class extends ContractedAdkAgent to support multi-agent hierarchies
    where a coordinator agent manages multiple sub-agents. The contract is
    enforced across the entire multi-agent system, tracking cumulative usage.

    Multi-agent systems can spiral out of control quickly as agents coordinate,
    delegate tasks, and iterate. This wrapper ensures the total budget is
    respected across all agents in the hierarchy.

    Example:
        >>> from google.adk.agents import LlmAgent
        >>> from agent_contracts import Contract, ResourceConstraints
        >>>
        >>> # Create sub-agents
        >>> researcher = LlmAgent(name="researcher", ...)
        >>> planner = LlmAgent(name="planner", ...)
        >>> executor = LlmAgent(name="executor", ...)
        >>>
        >>> # Create coordinator with sub-agents
        >>> coordinator = LlmAgent(
        ...     name="coordinator",
        ...     model="gemini-2.0-flash",
        ...     instruction="Coordinate research, planning, and execution",
        ...     sub_agents=[researcher, planner, executor]
        ... )
        >>>
        >>> # Wrap with contract (enforced across ALL agents)
        >>> contract = Contract(
        ...     id="multi-agent-workflow",
        ...     resources=ResourceConstraints(
        ...         tokens=100000,  # For ENTIRE multi-agent system
        ...         api_calls=50,
        ...         cost_usd=5.0
        ...     )
        ... )
        >>>
        >>> contracted = ContractedAdkMultiAgent(
        ...     contract=contract,
        ...     agent=coordinator
        ... )
        >>>
        >>> # Budget enforced across ALL agents and their interactions
        >>> result = contracted.run(
        ...     user_id="user1",
        ...     session_id="session1",
        ...     message="Research and plan a marketing campaign"
        ... )
    """

    pass  # Inherits all behavior from ContractedAdkAgent


# Convenience function for creating contracted ADK agents
def create_contracted_adk_agent(
    agent: Any,  # Google ADK LlmAgent type
    resources: dict[str, Any] | None = None,
    temporal: dict[str, Any] | None = None,
    contract_id: str | None = None,
    strict_mode: bool = True,
    runner: Any | None = None,
) -> ContractedAdkAgent:
    """Create a contracted ADK agent with simplified API.

    This is a convenience function for creating ContractedAdkAgent instances
    without manually creating Contract objects.

    Args:
        agent: Google ADK LlmAgent to wrap
        resources: Resource constraints dict (tokens, cost_usd, api_calls, etc.)
        temporal: Temporal constraints dict (deadline, max_duration, etc.)
        contract_id: Optional contract ID (auto-generated if not provided)
        strict_mode: If True, violations cause immediate termination
        runner: Optional custom Runner (defaults to InMemoryRunner)

    Returns:
        ContractedAdkAgent instance

    Example:
        >>> from google.adk.agents import LlmAgent
        >>> agent = LlmAgent(name="my_agent", model="gemini-2.0-flash", ...)
        >>>
        >>> contracted = create_contracted_adk_agent(
        ...     agent=agent,
        ...     resources={"tokens": 50000, "cost_usd": 2.0, "api_calls": 25},
        ...     temporal={"max_duration": "10 minutes"}
        ... )
    """
    from agent_contracts.core.contract import (
        ResourceConstraints,
        TemporalConstraints,
    )

    # Create contract
    contract_id_val = contract_id or f"adk-agent-{id(agent)}"
    contract = Contract(
        id=contract_id_val,
        name=contract_id_val,
        resources=ResourceConstraints(**resources) if resources else ResourceConstraints(),
        temporal=TemporalConstraints(**temporal) if temporal else TemporalConstraints(),
    )

    return ContractedAdkAgent(
        contract=contract, agent=agent, strict_mode=strict_mode, runner=runner
    )
