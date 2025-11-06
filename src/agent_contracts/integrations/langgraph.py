"""LangGraph integration for Agent Contracts.

This module provides contract-aware wrappers for LangGraph state machines,
enabling resource governance and budget enforcement for complex multi-agent
workflows with cycles, conditional branching, and parallel execution.

LangGraph is specifically designed for complex agentic workflows where:
- Multiple agents coordinate through message passing
- Workflows have cycles, retries, and conditional branches
- Budget can spiral out of control without governance

This is where Agent Contracts provides the most value - proactive budget
enforcement across the entire graph execution, not just individual nodes.

Example:
    >>> from langgraph.graph import StateGraph, END
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.langgraph import ContractedGraph
    >>>
    >>> # Define multi-agent workflow
    >>> workflow = StateGraph(AgentState)
    >>> workflow.add_node("researcher", research_agent)
    >>> workflow.add_node("planner", planning_agent)
    >>> workflow.add_node("executor", execution_agent)
    >>> workflow.add_edge("researcher", "planner")
    >>> workflow.add_edge("planner", "executor")
    >>> workflow.add_edge("executor", END)
    >>> workflow.set_entry_point("researcher")
    >>>
    >>> # Wrap entire graph with contract
    >>> contract = Contract(
    ...     id="research-workflow",
    ...     resources=ResourceConstraints(
    ...         tokens=50000,  # For ENTIRE workflow
    ...         api_calls=25,
    ...         cost_usd=2.0
    ...     )
    ... )
    >>>
    >>> contracted_workflow = ContractedGraph(
    ...     contract=contract,
    ...     graph=workflow.compile()
    ... )
    >>>
    >>> # Budget enforced across ALL nodes and cycles!
    >>> result = contracted_workflow.invoke({"query": "..."})
"""

from typing import Any, TypeVar

from agent_contracts.core.contract import Contract
from agent_contracts.core.wrapper import ContractAgent

# Type checking imports
try:
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = Any

# Type variable for state
TState = TypeVar("TState")


class ContractedGraph(ContractAgent[dict[str, Any], dict[str, Any]]):
    """Contract-aware wrapper for LangGraph state machines.

    This class wraps any LangGraph CompiledGraph and adds contract enforcement
    with cumulative budget tracking across all nodes, cycles, and branches.

    The wrapper:
    - Tracks token usage across ALL node executions
    - Enforces budget constraints for the entire workflow
    - Handles cycles, retries, and conditional branches
    - Logs execution for audit
    - Prevents runaway costs in complex multi-agent loops

    Key Difference from ContractedChain:
    - ContractedChain: Simple sequential/branching chains (3-10 calls)
    - ContractedGraph: Complex stateful workflows with cycles (30+ calls possible!)

    This is where budget governance becomes CRITICAL - a graph with retries
    can easily spiral into hundreds of LLM calls without proper constraints.

    Attributes:
        contract: The contract governing execution
        graph: The underlying LangGraph CompiledGraph
        enforcer: Contract enforcement engine
        resource_monitor: Resource consumption tracker (shared across all nodes)
        temporal_monitor: Time constraint tracker

    Example:
        >>> from langgraph.graph import StateGraph, END
        >>> from agent_contracts import Contract, ResourceConstraints
        >>>
        >>> # Complex workflow with cycles
        >>> workflow = StateGraph(AgentState)
        >>> workflow.add_node("research", research_node)
        >>> workflow.add_node("validate", validate_node)
        >>> workflow.add_conditional_edges(
        ...     "validate",
        ...     should_continue,
        ...     {True: "research", False: END}  # Can loop back!
        ... )
        >>>
        >>> # Compile graph
        >>> app = workflow.compile()
        >>>
        >>> # Wrap with contract to prevent runaway loops
        >>> contract = Contract(
        ...     id="research-loop",
        ...     resources=ResourceConstraints(
        ...         tokens=50000,
        ...         api_calls=25,  # Limit iterations!
        ...         cost_usd=2.0
        ...     )
        ... )
        >>>
        >>> contracted = ContractedGraph(contract=contract, graph=app)
        >>>
        >>> # Budget tracked cumulatively - stops if limits exceeded
        >>> result = contracted.invoke({"query": "Research topic"})
        >>> print(f"Used {result.execution_log.resource_usage['api_calls']} calls")
    """

    def __init__(
        self,
        contract: Contract,
        graph: Any,  # CompiledGraph type (varies by LangGraph version)
        strict_mode: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """Initialize contracted LangGraph workflow.

        Args:
            contract: Contract to enforce
            graph: LangGraph CompiledGraph to wrap
            strict_mode: If True, violations cause immediate termination
            enable_logging: If True, log execution for audit trail

        Raises:
            ImportError: If langgraph is not installed
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required for LangGraph integration. "
                "Install with: pip install langgraph"
            )

        # Initialize base ContractAgent with graph invocation as callable
        super().__init__(
            contract=contract,
            agent=self._run_graph,
            strict_mode=strict_mode,
            enable_logging=enable_logging,
        )

        self.graph = graph

        # Set up interception for node-level token tracking
        self._setup_node_tracking()

    def _run_graph(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the LangGraph workflow.

        This method is called by ContractAgent's execute() method.
        It invokes the graph and returns the final state.

        Args:
            inputs: Initial state dictionary for the graph

        Returns:
            Final state dictionary after graph execution
        """
        # LangGraph uses invoke() method
        if hasattr(self.graph, "invoke"):
            # Pass config for callbacks if needed
            config = self._build_config()
            result = self.graph.invoke(inputs, config=config)
            return result  # type: ignore[no-any-return]
        else:
            # Fallback for older API
            return self.graph(inputs)  # type: ignore[no-any-return]

    def _build_config(self) -> dict[str, Any]:
        """Build configuration for graph execution with callbacks.

        Returns:
            Configuration dict with callbacks for tracking
        """
        config: dict[str, Any] = {}

        # Try to set up callbacks for token tracking
        try:
            # Try LangChain 1.0+ callback system (LangGraph uses LangChain callbacks)
            try:
                from langchain_core.callbacks import BaseCallbackHandler
            except ImportError:
                from langchain.callbacks.base import BaseCallbackHandler

            class GraphTokenTrackingCallback(BaseCallbackHandler):  # type: ignore[misc]
                """Callback to track token usage across all graph nodes."""

                def __init__(self, monitor: Any) -> None:
                    """Initialize with resource monitor."""
                    self.monitor = monitor

                def on_llm_end(self, response: Any, **kwargs: Any) -> None:
                    """Track tokens when any LLM call completes in any node."""
                    total_tokens = 0

                    # Try multiple locations for token usage
                    if response.llm_output and "token_usage" in response.llm_output:
                        usage = response.llm_output["token_usage"]
                        total_tokens = usage.get("total_tokens", 0)
                    elif response.llm_output and "usage_metadata" in response.llm_output:
                        usage = response.llm_output["usage_metadata"]
                        total_tokens = usage.get("total_tokens", 0)
                    elif (
                        response.generations
                        and len(response.generations) > 0
                        and len(response.generations[0]) > 0
                    ):
                        gen = response.generations[0][0]
                        if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                            metadata = gen.message.response_metadata
                            if "usage_metadata" in metadata:
                                usage = metadata["usage_metadata"]
                                total_tokens = usage.get("total_tokens", 0)

                    # Track tokens cumulatively across all nodes
                    if total_tokens > 0:
                        self.monitor.usage.add_tokens(count=total_tokens)

                        # Track API call with cost estimate
                        cost_estimate = total_tokens * 0.00000015
                        self.monitor.usage.add_api_call(cost=cost_estimate, tokens=0)

            # Add callback to config
            callback = GraphTokenTrackingCallback(self.resource_monitor)
            config["callbacks"] = [callback]

        except ImportError:
            # Callbacks not available, will need manual tracking
            pass

        return config

    def _setup_node_tracking(self) -> None:
        """Set up tracking for individual node executions.

        This wraps node functions to track budget at the node level,
        enabling fine-grained monitoring and early termination if needed.
        """
        # For now, rely on callbacks. Future enhancement could wrap individual nodes.
        pass

    def _monitored_execution(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute graph with monitoring.

        Overrides base method to add LangGraph-specific monitoring.

        Args:
            input_data: Initial state dictionary for the graph

        Returns:
            Final state dictionary
        """
        # Add budget awareness to state if graph expects it
        if "budget_info" not in input_data:
            input_data["budget_info"] = {
                "remaining_tokens": self.resource_monitor.get_remaining_tokens(),
                "remaining_cost": self.resource_monitor.get_remaining_cost(),
                "remaining_api_calls": self.resource_monitor.get_remaining_api_calls(),
                "time_pressure": self.temporal_monitor.get_time_pressure(),
            }

        # Execute graph
        return self._run_graph(input_data)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute graph with contract enforcement (LangGraph-style API).

        This method provides a LangGraph-compatible interface that delegates
        to our execute() method.

        Args:
            inputs: Initial state dictionary for the graph

        Returns:
            Final state dictionary

        Raises:
            RuntimeError: If execution fails or contract is violated
        """
        result = self.execute(inputs)

        if result.success and result.output:
            return result.output
        else:
            raise RuntimeError(f"Graph execution failed: {result.violations}")

    def stream(self, inputs: dict[str, Any]) -> Any:
        """Stream graph execution with contract enforcement.

        Note: Streaming support is experimental. Budget tracking occurs
        at checkpoint boundaries, not per-token.

        Args:
            inputs: Initial state dictionary for the graph

        Yields:
            State updates as they occur

        Raises:
            RuntimeError: If execution fails or contract is violated
        """
        # For streaming, we need to wrap the graph's stream method
        if not hasattr(self.graph, "stream"):
            raise NotImplementedError("Graph does not support streaming")

        # Start monitoring

        self.temporal_monitor.start()
        self.enforcer.start()

        try:
            config = self._build_config()

            # Stream execution
            for chunk in self.graph.stream(inputs, config=config):
                # Check constraints at each chunk
                is_violated, _violations = self.enforcer.check_constraints()

                if is_violated and self.strict_mode:
                    raise RuntimeError("Contract violated during streaming execution")

                yield chunk

            # Final constraint check
            self.enforcer.check_constraints()
            self.enforcer.check_temporal_constraints()

        except Exception as e:
            self.contract.state = self.contract.state  # Keep current state
            raise RuntimeError(f"Streaming execution failed: {e}") from e

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Make the contracted graph callable like a regular graph.

        Args:
            inputs: Initial state dictionary for the graph

        Returns:
            Final state dictionary
        """
        return self.invoke(inputs)


# Convenience function for creating contracted graphs
def create_contracted_graph(
    graph: Any,  # CompiledGraph type (varies by version)
    resources: dict[str, Any] | None = None,
    temporal: dict[str, Any] | None = None,
    contract_id: str | None = None,
    strict_mode: bool = True,
) -> ContractedGraph:
    """Create a contracted graph with simplified API.

    This is a convenience function for creating ContractedGraph instances
    without manually creating Contract objects.

    Args:
        graph: LangGraph CompiledGraph to wrap
        resources: Resource constraints dict (tokens, cost_usd, api_calls, etc.)
        temporal: Temporal constraints dict (deadline, max_duration, etc.)
        contract_id: Optional contract ID (auto-generated if not provided)
        strict_mode: If True, violations cause immediate termination

    Returns:
        ContractedGraph instance

    Example:
        >>> from langgraph.graph import StateGraph, END
        >>> workflow = StateGraph(MyState)
        >>> # ... add nodes and edges ...
        >>> app = workflow.compile()
        >>>
        >>> contracted = create_contracted_graph(
        ...     graph=app,
        ...     resources={"tokens": 50000, "api_calls": 25, "cost_usd": 2.0},
        ...     temporal={"max_duration": "10 minutes"}
        ... )
    """
    from agent_contracts.core.contract import (
        ResourceConstraints,
        TemporalConstraints,
    )

    # Create contract
    contract_id_val = contract_id or f"graph-{id(graph)}"
    contract = Contract(
        id=contract_id_val,
        name=contract_id_val,
        resources=ResourceConstraints(**resources) if resources else ResourceConstraints(),
        temporal=TemporalConstraints(**temporal) if temporal else TemporalConstraints(),
    )

    return ContractedGraph(contract=contract, graph=graph, strict_mode=strict_mode)
