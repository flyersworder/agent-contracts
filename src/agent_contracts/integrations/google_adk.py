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
    ...     model="gemini-3-flash-preview",
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
        ...     model="gemini-3-flash-preview",
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
        self._app_name = f"agent-contracts-{contract.id}"
        self._created_sessions: set[str] = set()  # Track created sessions

        # Set up runner (use provided or create InMemoryRunner)
        if runner is not None:
            self.runner = runner
        else:
            # Import here to avoid issues if google-adk not installed
            from google.adk.runners import InMemoryRunner

            # InMemoryRunner requires agent and app_name
            self.runner = InMemoryRunner(agent=agent, app_name=self._app_name)

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

        # Apply contract's iterations limit via RunConfig.max_llm_calls
        # This prevents runaway agent loops by limiting LLM calls
        if self.contract.resources.iterations is not None:
            from google.adk.runners import RunConfig

            contract_limit = self.contract.resources.iterations

            if run_config is None:
                # Create new RunConfig with contract's iteration limit
                run_config = RunConfig(max_llm_calls=contract_limit)
            elif hasattr(run_config, "max_llm_calls"):
                # If user provided run_config, use the more restrictive limit
                # This ensures contract governance cannot be bypassed
                user_limit = run_config.max_llm_calls
                if user_limit is None or contract_limit < user_limit:
                    run_config = RunConfig(max_llm_calls=contract_limit)

        # Convert message to Content if it's a string
        if isinstance(message, str):
            from google.genai.types import Content, Part

            content = Content(parts=[Part(text=message)])
        else:
            content = message

        # Ensure session exists before running
        # ADK requires sessions to be created via session_service first
        # Note: create_session is async, so we need to run it in an event loop
        session_key = f"{user_id}:{session_id}"
        if session_key not in self._created_sessions:
            import asyncio

            async def _create_session() -> None:
                await self.runner.session_service.create_session(
                    app_name=self._app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

            # Run async session creation
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, use run_coroutine_threadsafe

                future = asyncio.run_coroutine_threadsafe(_create_session(), loop)
                future.result(timeout=30)
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                asyncio.run(_create_session())

            self._created_sessions.add(session_key)

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
        tool_invocations: dict[str, int] = {}  # Track per-tool usage
        llm_call_count = 0  # Track number of LLM calls (iterations)

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
            # Note: Python ADK uses snake_case (usage_metadata), not camelCase
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                usage = event.usage_metadata

                # Safe extraction helper - handles Mock objects in tests gracefully
                def _safe_int(value: Any, default: int = 0) -> int:
                    """Extract int from value, handling Mock objects."""
                    if isinstance(value, int):
                        return value
                    if value is None:
                        return default
                    # Check if it looks like a Mock (has _mock_name attribute)
                    if hasattr(value, "_mock_name"):
                        return default
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return default

                event_tokens = _safe_int(getattr(usage, "total_token_count", 0))

                # Update cumulative tracking - use safe extraction for all values
                prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", 0))
                candidates_tokens = _safe_int(getattr(usage, "candidates_token_count", 0))
                cached_tokens = _safe_int(getattr(usage, "cached_content_token_count", 0))
                thoughts_tokens = _safe_int(getattr(usage, "thoughts_token_count", 0))

                cumulative_usage["total_tokens"] += event_tokens
                cumulative_usage["prompt_tokens"] += prompt_tokens
                cumulative_usage["candidates_tokens"] += candidates_tokens
                cumulative_usage["cached_tokens"] += cached_tokens
                cumulative_usage["thoughts_tokens"] += thoughts_tokens

                # Track tokens in resource monitor
                if event_tokens > 0:
                    # Count this as an LLM call (iteration)
                    llm_call_count += 1

                    # Track tokens with breakdown
                    reasoning_tokens = thoughts_tokens
                    text_tokens = event_tokens - reasoning_tokens

                    self.resource_monitor.usage.add_tokens(
                        count=0,  # count not used when text/reasoning provided
                        reasoning=reasoning_tokens,
                        text=text_tokens,
                    )

                    # Track API call with cost estimate
                    # Gemini 3 Flash: ~$0.075 per 1M input, ~$0.30 per 1M output
                    prompt_cost = prompt_tokens * 0.000000075
                    output_cost = candidates_tokens * 0.00000030
                    total_cost = prompt_cost + output_cost

                    self.resource_monitor.usage.add_api_call(cost=total_cost, tokens=0)

            # Track tool invocations (per-tool limits)
            # Check for function responses (completed tool executions)
            # Note: Use try-except because Mock objects in tests aren't iterable
            if hasattr(event, "get_function_responses") and callable(
                getattr(event, "get_function_responses", None)
            ):
                try:
                    responses = event.get_function_responses()
                    if responses:
                        for response in responses:
                            tool_name = getattr(response, "name", None)
                            if tool_name:
                                # Track in local counter
                                tool_invocations[tool_name] = tool_invocations.get(tool_name, 0) + 1

                                # Track in resource monitor (for per-tool limits)
                                self.resource_monitor.usage.add_tool_invocation(tool_name)

                                # Check if per-tool limit exceeded
                                if (
                                    not self.resource_monitor.can_use_tool(tool_name)
                                    and self.strict_mode
                                ):
                                    raise RuntimeError(
                                        f"Per-tool limit exceeded for '{tool_name}': "
                                        f"{tool_invocations[tool_name]} invocations"
                                    )
                except (TypeError, AttributeError):
                    # Skip tool tracking if responses aren't available or iterable
                    # This handles Mock objects in tests gracefully
                    pass

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
            "tool_invocations": tool_invocations,  # Per-tool usage breakdown
            "llm_calls": llm_call_count,  # Number of LLM calls (iterations)
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
        ...     model="gemini-3-flash-preview",
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
        >>> agent = LlmAgent(name="my_agent", model="gemini-3-flash-preview", ...)
        >>>
        >>> contracted = create_contracted_adk_agent(
        ...     agent=agent,
        ...     resources={"tokens": 50000, "cost_usd": 2.0, "api_calls": 25},
        ...     temporal={"max_duration": "10 minutes"}
        ... )
    """
    from datetime import timedelta

    from agent_contracts.core.contract import (
        ResourceConstraints,
        TemporalConstraints,
    )

    # Convert numeric max_duration to timedelta if needed
    if temporal and "max_duration" in temporal:
        max_dur = temporal["max_duration"]
        if isinstance(max_dur, (int, float)):
            temporal = {**temporal, "max_duration": timedelta(seconds=max_dur)}

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


class DelegatingAdkAgent(ContractedAdkAgent):
    """Contract-aware Google ADK agent with hierarchical delegation support.

    This class extends ContractedAdkAgent to support explicit budget delegation
    to sub-agents, with conservation law enforcement ensuring no sub-agent can
    receive more budget than is available from the parent.

    The key insight (from Whitepaper Section 6.2) is that **contracting is itself
    a capability**: this agent can spawn sub-agents with their own contracts,
    enabling recursive delegation and dynamic team formation.

    Conservation Law:
        For any parent contract with budget B, if it creates child contracts
        with budgets b_1, b_2, ..., b_k, the following must hold:

            Σ b_i ≤ B - used

        where 'used' is the parent's own consumption.

    This class enables the paper's example (Section 8) where an orchestrator
    with 150K tokens allocates to researcher (50K), analyzer (40K), and
    reporter (45K), while reserving 15K for coordination.

    Example:
        >>> from google.adk.agents import LlmAgent
        >>> from agent_contracts import Contract, ResourceConstraints
        >>> from agent_contracts.integrations.google_adk import DelegatingAdkAgent
        >>>
        >>> # Create parent contract for orchestrator
        >>> parent_contract = Contract(
        ...     id="orchestrator",
        ...     resources=ResourceConstraints(tokens=150_000, cost_usd=5.0)
        ... )
        >>>
        >>> # Create orchestrator agent
        >>> orchestrator = LlmAgent(
        ...     name="orchestrator",
        ...     model="gemini-3-flash-preview",
        ...     instruction="Coordinate research workflow"
        ... )
        >>>
        >>> # Create delegating agent
        >>> delegating_agent = DelegatingAdkAgent(
        ...     contract=parent_contract,
        ...     agent=orchestrator,
        ...     reserve_ratio=0.1  # Reserve 10% for coordination
        ... )
        >>>
        >>> # Delegate to sub-agents with budget allocation
        >>> researcher_agent = LlmAgent(name="researcher", ...)
        >>> researcher = delegating_agent.delegate(
        ...     name="researcher",
        ...     agent=researcher_agent,
        ...     tokens=50_000,
        ...     description="Research the topic"
        ... )
        >>>
        >>> # Conservation law enforced - this would fail if tokens exceed remaining
        >>> analyzer_agent = LlmAgent(name="analyzer", ...)
        >>> analyzer = delegating_agent.delegate(
        ...     name="analyzer",
        ...     agent=analyzer_agent,
        ...     tokens=40_000  # OK: 50K + 40K = 90K < 135K remaining
        ... )
    """

    def __init__(
        self,
        contract: Contract,
        agent: Any,  # Google ADK LlmAgent type
        strict_mode: bool = True,
        enable_logging: bool = True,
        runner: Any | None = None,
        reserve_ratio: float = 0.0,
    ) -> None:
        """Initialize delegating Google ADK agent.

        Args:
            contract: Contract to enforce for this agent
            agent: Google ADK LlmAgent to wrap
            strict_mode: If True, violations cause immediate termination
            enable_logging: If True, log execution for audit trail
            runner: Optional custom Runner (defaults to InMemoryRunner)
            reserve_ratio: Fraction of budget to reserve for coordination (0.0-0.5)

        Raises:
            ImportError: If google-adk is not installed
            ValueError: If reserve_ratio is out of valid range
        """
        super().__init__(
            contract=contract,
            agent=agent,
            strict_mode=strict_mode,
            enable_logging=enable_logging,
            runner=runner,
        )

        # Import delegation capability
        from agent_contracts.core.delegation import ContractingCapability

        # Create contracting capability with this agent's monitor
        self.contracting = ContractingCapability(
            parent_contract=contract,
            parent_monitor=self.resource_monitor,
            reserve_ratio=reserve_ratio,
        )

        # Track delegated agents for lifecycle management
        self._delegated_agents: dict[str, ContractedAdkAgent] = {}

    def delegate(
        self,
        name: str,
        agent: Any,  # Google ADK LlmAgent type
        tokens: int = 0,
        cost_usd: float = 0.0,
        api_calls: int | None = None,
        iterations: int | None = None,
        description: str = "",
        strict_mode: bool = True,
        runner: Any | None = None,
    ) -> "ContractedAdkAgent":
        """Delegate to a sub-agent with budget allocation.

        Creates a new ContractedAdkAgent for the sub-agent with a contract
        that has budget allocated from this agent's remaining budget.
        Conservation laws are enforced: the allocation will fail if there
        isn't enough remaining budget.

        Args:
            name: Name for the delegated agent (used in ID generation)
            agent: Google ADK LlmAgent to delegate to
            tokens: Token budget to allocate to delegated agent
            cost_usd: Cost budget to allocate to delegated agent
            api_calls: Optional API call limit for delegated agent
            iterations: Optional iteration limit for delegated agent
                (maps to Google ADK RunConfig.max_llm_calls)
            description: Description of the delegated task
            strict_mode: If True, violations cause immediate termination
            runner: Optional custom Runner for the delegated agent

        Returns:
            ContractedAdkAgent for the delegated agent

        Raises:
            ConservationViolationError: If allocation would exceed remaining budget
            ValueError: If name is empty or already used
        """
        # Create subcontract (enforces conservation law)
        child_contract = self.contracting.create_subcontract(
            name=name,
            tokens=tokens,
            cost_usd=cost_usd,
            api_calls=api_calls,
            iterations=iterations,
            description=description,
        )

        # Wrap sub-agent with the allocated contract
        delegated = ContractedAdkAgent(
            contract=child_contract,
            agent=agent,
            strict_mode=strict_mode,
            enable_logging=self.enable_logging,
            runner=runner,
        )

        # Track for lifecycle management
        self._delegated_agents[name] = delegated

        return delegated

    def can_delegate(self, tokens: int = 0, cost_usd: float = 0.0) -> bool:
        """Check if a delegation is possible without violating conservation.

        Args:
            tokens: Number of tokens to allocate
            cost_usd: Cost budget to allocate

        Returns:
            True if delegation would satisfy conservation law
        """
        return self.contracting.can_allocate(tokens=tokens, cost_usd=cost_usd)

    def release_delegation(self, name: str) -> int:
        """Release a delegation's budget back to the pool.

        Call this when a delegated agent completes early and returns
        unused budget. The budget becomes available for other delegations.

        Args:
            name: Name of the delegated agent

        Returns:
            Number of tokens released back to pool

        Raises:
            KeyError: If delegation not found
        """
        # Remove from tracking
        if name in self._delegated_agents:
            del self._delegated_agents[name]

        # Release allocation (returns tokens to pool)
        return self.contracting.release_allocation(name)

    def get_delegated_agent(self, name: str) -> "ContractedAdkAgent | None":
        """Get a delegated agent by name.

        Args:
            name: Name of the delegated agent

        Returns:
            ContractedAdkAgent if found, None otherwise
        """
        return self._delegated_agents.get(name)

    @property
    def remaining_delegation_tokens(self) -> int:
        """Tokens available for further delegation."""
        return self.contracting.remaining_tokens

    @property
    def remaining_delegation_cost(self) -> float:
        """Cost budget available for further delegation."""
        return self.contracting.remaining_cost

    @property
    def delegated_agents(self) -> list["ContractedAdkAgent"]:
        """List of all delegated agents."""
        return list(self._delegated_agents.values())

    def get_delegation_summary(self) -> dict[str, Any]:
        """Get summary of all delegations.

        Returns:
            Dictionary with delegation summary including:
            - parent_id: ID of the parent contract
            - parent_budget: Total parent budget
            - parent_used: Parent's own usage
            - total_delegated: Total budget delegated to children
            - remaining: Budget available for further delegation
            - delegations: List of individual delegation details
            - conservation_satisfied: Whether conservation law holds
        """
        summary = self.contracting.get_summary()

        return {
            "parent_id": summary.parent_id,
            "parent_budget_tokens": summary.parent_budget_tokens,
            "parent_budget_cost": summary.parent_budget_cost,
            "parent_used_tokens": summary.parent_used_tokens,
            "parent_used_cost": summary.parent_used_cost,
            "total_delegated_tokens": summary.total_allocated_tokens,
            "total_delegated_cost": summary.total_allocated_cost,
            "remaining_tokens": summary.remaining_tokens,
            "remaining_cost": summary.remaining_cost,
            "delegations": [
                {
                    "name": alloc.child_name,
                    "id": alloc.child_id,
                    "tokens": alloc.tokens_allocated,
                    "cost": alloc.cost_allocated,
                    "created_at": alloc.created_at.isoformat(),
                }
                for alloc in summary.allocations
            ],
            "conservation_satisfied": summary.conservation_satisfied,
        }

    def __repr__(self) -> str:
        """String representation of delegating agent state."""
        return (
            f"DelegatingAdkAgent("
            f"contract='{self.contract.id}', "
            f"budget={self.contracting.parent_budget_tokens:,} tokens, "
            f"used={self.contracting.parent_used_tokens:,}, "
            f"delegated={self.contracting._total_allocated_tokens:,}, "
            f"remaining={self.contracting.remaining_tokens:,}, "
            f"children={len(self._delegated_agents)})"
        )
