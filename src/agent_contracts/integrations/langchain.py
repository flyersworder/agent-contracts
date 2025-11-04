"""LangChain integration for Agent Contracts.

This module provides contract-aware wrappers for LangChain chains and agents,
enabling resource governance and budget enforcement for LangChain applications.

Example:
    >>> from langchain.chains import LLMChain
    >>> from langchain.llms import OpenAI
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.langchain import ContractedChain
    >>>
    >>> # Create standard LangChain chain
    >>> llm = OpenAI(temperature=0)
    >>> chain = LLMChain(llm=llm, prompt=prompt_template)
    >>>
    >>> # Wrap with contract
    >>> contract = Contract(
    ...     id="my-chain",
    ...     resources=ResourceConstraints(tokens=10000, cost_usd=1.0)
    ... )
    >>> contracted_chain = ContractedChain(contract=contract, chain=chain)
    >>>
    >>> # Execute with automatic budget enforcement
    >>> result = contracted_chain.execute({"input": "Write a report"})
"""

from typing import Any

from agent_contracts.core.contract import Contract
from agent_contracts.core.wrapper import ContractAgent, ExecutionResult

# Type checking imports
try:
    from langchain.chains.base import Chain
    from langchain.schema import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Chain = Any  # type: ignore
    LLMResult = Any  # type: ignore


class ContractedChain(ContractAgent[dict[str, Any], dict[str, Any]]):
    """Contract-aware wrapper for LangChain chains.

    This class wraps any LangChain Chain (LLMChain, SequentialChain, etc.) and
    adds contract enforcement with automatic token tracking and budget monitoring.

    The wrapper:
    - Tracks token usage from LLM calls
    - Enforces budget constraints
    - Logs execution for audit
    - Provides budget awareness to the chain

    Attributes:
        contract: The contract governing execution
        chain: The underlying LangChain chain
        enforcer: Contract enforcement engine
        resource_monitor: Resource consumption tracker
        temporal_monitor: Time constraint tracker

    Example:
        >>> from langchain.chains import LLMChain
        >>> from langchain.llms import OpenAI
        >>> from agent_contracts import Contract, ResourceConstraints
        >>>
        >>> contract = Contract(
        ...     id="summarize",
        ...     resources=ResourceConstraints(tokens=5000)
        ... )
        >>> chain = LLMChain(llm=OpenAI(), prompt=prompt)
        >>> contracted = ContractedChain(contract=contract, chain=chain)
        >>>
        >>> result = contracted.execute({"text": "Long article..."})
        >>> print(result.output["text"])  # Summarized text
        >>> print(result.execution_log.resource_usage)  # Audit log
    """

    def __init__(
        self,
        contract: Contract,
        chain: "Chain",
        strict_mode: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """Initialize contracted LangChain chain.

        Args:
            contract: Contract to enforce
            chain: LangChain Chain to wrap
            strict_mode: If True, violations cause immediate termination
            enable_logging: If True, log execution for audit trail

        Raises:
            ImportError: If langchain is not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is required for LangChain integration. "
                "Install with: pip install langchain"
            )

        # Initialize base ContractAgent with chain as callable
        super().__init__(
            contract=contract,
            agent=self._run_chain,  # type: ignore
            strict_mode=strict_mode,
            enable_logging=enable_logging,
        )

        self.chain = chain

        # Set up callback for token tracking
        self._setup_callbacks()

    def _run_chain(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the LangChain chain.

        This method is called by ContractAgent's execute() method.

        Args:
            inputs: Input dictionary for the chain

        Returns:
            Chain's output dictionary
        """
        return self.chain(inputs)

    def _setup_callbacks(self) -> None:
        """Set up LangChain callbacks for token tracking.

        This adds a callback handler that tracks token usage from LLM calls
        and updates the resource monitor automatically.
        """
        # Try to set up callback handler for automatic token tracking
        try:
            from langchain.callbacks.base import BaseCallbackHandler

            class TokenTrackingCallback(BaseCallbackHandler):
                """Callback to track token usage and update monitor."""

                def __init__(self, monitor: Any) -> None:
                    """Initialize with resource monitor."""
                    self.monitor = monitor

                def on_llm_end(self, response: "LLMResult", **kwargs: Any) -> None:
                    """Track tokens when LLM call completes."""
                    if response.llm_output and "token_usage" in response.llm_output:
                        usage = response.llm_output["token_usage"]

                        # Extract token counts
                        total_tokens = usage.get("total_tokens", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                        # Update monitor
                        self.monitor.update_resource("tokens", total_tokens)
                        self.monitor.update_resource("api_calls", 1)

                        # Estimate cost (rough approximation)
                        # Real implementation should use model-specific pricing
                        cost_estimate = total_tokens * 0.00002  # ~$0.02 per 1K tokens
                        self.monitor.update_resource("cost_usd", cost_estimate)

            # Add callback to chain
            callback = TokenTrackingCallback(self.resource_monitor)

            # Inject callback into chain if it supports callbacks
            if hasattr(self.chain, "callbacks"):
                if self.chain.callbacks is None:
                    self.chain.callbacks = []
                self.chain.callbacks.append(callback)

        except ImportError:
            # Callback setup failed, will need manual token tracking
            pass

    def _monitored_execution(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute chain with monitoring.

        Overrides base method to add LangChain-specific monitoring.

        Args:
            input_data: Input dictionary for the chain

        Returns:
            Chain's output dictionary
        """
        # Add budget awareness to inputs if chain expects it
        if "budget_info" not in input_data:
            input_data["budget_info"] = {
                "remaining_tokens": self.resource_monitor.get_remaining_tokens(),
                "remaining_cost": self.resource_monitor.get_remaining_cost(),
                "time_pressure": self.temporal_monitor.get_time_pressure(),
            }

        # Execute chain
        return self._run_chain(input_data)

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute chain with contract enforcement (LangChain-style API).

        This method provides a LangChain-compatible interface that delegates
        to our execute() method.

        Args:
            *args: Positional arguments for the chain
            **kwargs: Keyword arguments for the chain

        Returns:
            Chain's output dictionary
        """
        # Convert args/kwargs to input dict
        if args:
            inputs = args[0] if isinstance(args[0], dict) else {"input": args[0]}
        else:
            inputs = kwargs

        # Execute with contract enforcement
        result = self.execute(inputs)

        # Return just the output (match LangChain's API)
        if result.success and result.output:
            return result.output
        else:
            raise RuntimeError(f"Chain execution failed: {result.violations}")

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Make the contracted chain callable like a regular chain.

        Args:
            inputs: Input dictionary for the chain

        Returns:
            Chain's output dictionary
        """
        return self.run(inputs)


class ContractedLLM:
    """Contract-aware wrapper for standalone LLM calls.

    This class wraps individual LLM calls (not full chains) with contract
    enforcement. Useful for simple use cases where you don't need a full chain.

    Example:
        >>> from langchain.llms import OpenAI
        >>> from agent_contracts import Contract, ResourceConstraints
        >>>
        >>> contract = Contract(
        ...     id="simple-call",
        ...     resources=ResourceConstraints(tokens=1000)
        ... )
        >>>
        >>> llm = OpenAI()
        >>> contracted_llm = ContractedLLM(contract=contract, llm=llm)
        >>>
        >>> response = contracted_llm("What is 2+2?")
        >>> print(response)  # "4"
    """

    def __init__(
        self,
        contract: Contract,
        llm: Any,
        strict_mode: bool = True,
    ) -> None:
        """Initialize contracted LLM.

        Args:
            contract: Contract to enforce
            llm: LangChain LLM instance
            strict_mode: If True, violations cause immediate termination
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is required for LangChain integration. "
                "Install with: pip install langchain"
            )

        self.contract = contract
        self.llm = llm
        self.strict_mode = strict_mode

        # Create a simple chain wrapper
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        # Simple pass-through prompt
        prompt = PromptTemplate(input_variables=["input"], template="{input}")
        self.chain = LLMChain(llm=llm, prompt=prompt)

        # Wrap with ContractedChain
        self.contracted_chain = ContractedChain(
            contract=contract,
            chain=self.chain,
            strict_mode=strict_mode,
        )

    def __call__(self, prompt: str) -> str:
        """Execute LLM call with contract enforcement.

        Args:
            prompt: Input prompt for the LLM

        Returns:
            LLM's response text
        """
        result = self.contracted_chain.execute({"input": prompt})

        if result.success and result.output:
            return result.output.get("text", "")
        else:
            raise RuntimeError(f"LLM call failed: {result.violations}")

    def execute(self, prompt: str) -> ExecutionResult[dict[str, Any]]:
        """Execute LLM call and return full execution result.

        Args:
            prompt: Input prompt for the LLM

        Returns:
            ExecutionResult with output and audit log
        """
        return self.contracted_chain.execute({"input": prompt})


# Convenience function for creating contracted chains
def create_contracted_chain(
    chain: "Chain",
    resources: dict[str, Any] | None = None,
    temporal: dict[str, Any] | None = None,
    contract_id: str | None = None,
    strict_mode: bool = True,
) -> ContractedChain:
    """Create a contracted chain with simplified API.

    This is a convenience function for creating ContractedChain instances
    without manually creating Contract objects.

    Args:
        chain: LangChain Chain to wrap
        resources: Resource constraints dict (tokens, cost_usd, etc.)
        temporal: Temporal constraints dict (deadline, max_duration, etc.)
        contract_id: Optional contract ID (auto-generated if not provided)
        strict_mode: If True, violations cause immediate termination

    Returns:
        ContractedChain instance

    Example:
        >>> from langchain.chains import LLMChain
        >>> contracted = create_contracted_chain(
        ...     chain=my_chain,
        ...     resources={"tokens": 10000, "cost_usd": 1.0},
        ...     temporal={"deadline": "5 minutes"}
        ... )
    """
    from agent_contracts.core.contract import (
        ResourceConstraints,
        TemporalConstraints,
    )

    # Create contract
    contract = Contract(
        id=contract_id or f"chain-{id(chain)}",
        resources=ResourceConstraints(**resources) if resources else None,
        temporal=TemporalConstraints(**temporal) if temporal else None,
    )

    return ContractedChain(contract=contract, chain=chain, strict_mode=strict_mode)
