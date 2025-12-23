"""Contract execution engine.

This module provides the ContractExecutor class that orchestrates contract execution
by integrating all core modules:
- prompts.py: Budget-aware prompt generation
- planning.py: Resource allocation and strategy
- monitor.py: Resource tracking
- enforcement.py: Constraint enforcement
- LiteLLM: LLM execution

The executor is the "conductor" that enables Contract.execute() to work seamlessly.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_contracts.core.contract import (
    Capabilities,
    Contract,
    ContractState,
    ExecutionConfig,
)
from agent_contracts.core.enforcement import ContractEnforcer
from agent_contracts.core.monitor import TemporalMonitor
from agent_contracts.core.planning import StrategyRecommendation, recommend_strategy
from agent_contracts.core.prompts import generate_adaptive_instruction, generate_budget_prompt


@dataclass
class ExecutionResult:
    """Result of contract execution.

    This is the unified return type for Contract.execute(), providing
    comprehensive information about the execution outcome.

    Attributes:
        success: Whether execution completed successfully within constraints
        output: The agent's response/output (str, dict, or other)
        resource_usage: Resources consumed during execution
        violations: List of constraint violations (empty if success)
        execution_log: Detailed trace of execution events
        strategy: Strategic recommendation used during execution
        contract_state: Final state of the contract
        started_at: When execution started
        completed_at: When execution completed
        error: Error message if execution failed
    """

    success: bool
    output: Any
    resource_usage: dict[str, Any]
    violations: list[str] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    strategy: StrategyRecommendation | None = None
    contract_state: ContractState = ContractState.DRAFTED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def tokens_used(self) -> int:
        """Get total tokens used."""
        return int(self.resource_usage.get("tokens", 0))

    @property
    def cost_usd(self) -> float:
        """Get total cost in USD."""
        return float(self.resource_usage.get("cost_usd", 0.0))


class ContractExecutor:
    """Executes contracts by orchestrating all core modules.

    The ContractExecutor is the central execution engine that:
    1. Analyzes the contract and generates a strategy (planning.py)
    2. Creates budget-aware prompts (prompts.py)
    3. Executes the LLM call with monitoring
    4. Enforces constraints and handles violations
    5. Returns a comprehensive ExecutionResult

    This enables the simple API: contract.execute(query="...")

    Attributes:
        contract: The contract being executed
        capabilities: Model-agnostic agent capabilities (tools, skills, etc.)
        execution_config: Model-specific execution settings (model, temperature)
        resource_monitor: Tracks resource consumption
        temporal_monitor: Tracks time constraints
        enforcer: Enforces contract constraints
        strict_mode: Whether to raise on violations (vs. return with violations)

    Example:
        >>> contract = Contract(
        ...     id="qa",
        ...     name="Q&A",
        ...     resources=ResourceConstraints(tokens=1000),
        ...     capabilities=Capabilities(tools=["calculator"]),
        ...     execution=ExecutionConfig(model="gpt-4o")
        ... )
        >>> executor = ContractExecutor(contract)
        >>> result = executor.run(query="What is 2+2?")
        >>> print(result.output)
    """

    def __init__(
        self,
        contract: Contract,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the executor.

        Args:
            contract: The contract to execute
            strict_mode: If True, violations raise RuntimeError; else return result with violations
        """
        self.contract = contract
        self.capabilities: Capabilities = contract.capabilities  # type: ignore[assignment]
        self.strict_mode = strict_mode

        # Get execution config (required for execution)
        if contract.execution is not None:
            self.execution_config = contract.execution
        else:
            # Use default execution config
            self.execution_config = ExecutionConfig()

        # Initialize enforcer (which creates its own monitor)
        self.enforcer = ContractEnforcer(
            contract=contract,
            strict_mode=strict_mode,
        )

        # Use the enforcer's monitor for resource tracking
        self.resource_monitor = self.enforcer.monitor

        # Initialize temporal monitor separately
        self.temporal_monitor = TemporalMonitor(contract)

        # Execution log
        self._execution_log: list[dict[str, Any]] = []
        self._violations: list[str] = []

    def run(self, **kwargs: Any) -> ExecutionResult:
        """Execute the contract with provided inputs.

        This is the main execution method that orchestrates the full
        contract execution lifecycle.

        Args:
            **kwargs: Input arguments. Common ones:
                - query: Text query/prompt
                - messages: List of message dicts for chat
                - context: Additional context

        Returns:
            ExecutionResult with output, usage, and status
        """
        started_at = datetime.now()
        self._log("execution_started", {"inputs": list(kwargs.keys())})

        try:
            # Step 1: Start monitoring and activate contract
            # Note: enforcer.start() activates the contract
            self.temporal_monitor.start()
            self.enforcer.start()
            self._log("contract_activated", {"state": self.contract.state.value})

            # Step 3: Generate strategy recommendation
            strategy = recommend_strategy(self.contract, self.resource_monitor.usage)
            self._log(
                "strategy_generated",
                {
                    "mode": strategy.mode.value,
                    "approach": strategy.recommended_approach,
                    "risk_level": strategy.risk_level,
                },
            )

            # Step 4: Generate budget-aware prompt
            task_description = self._extract_task_description(**kwargs)
            system_prompt = generate_budget_prompt(
                self.contract,
                task_description,
                self.resource_monitor.usage,
            )
            self._log("prompt_generated", {"prompt_length": len(system_prompt)})

            # Step 5: Execute LLM call
            output = self._execute_llm(system_prompt, **kwargs)
            self._log("llm_executed", {"output_type": type(output).__name__})

            # Step 6: Check constraints
            is_violated, violations = self.enforcer.check_constraints()
            if is_violated:
                for v in violations:
                    self._violations.append(str(v))
                    self._log("violation_detected", {"violation": str(v)})

            # Step 7: Finalize contract state
            if is_violated and self.strict_mode:
                # In strict mode, enforcer may have already violated the contract
                if self.contract.state != ContractState.VIOLATED:
                    self.contract.violate("; ".join(self._violations))
                raise RuntimeError(f"Contract violated: {self._violations}")
            elif is_violated:
                if self.contract.state != ContractState.VIOLATED:
                    self.contract.violate("; ".join(self._violations))
            else:
                if self.contract.state == ContractState.ACTIVE:
                    self.contract.fulfill()

            completed_at = datetime.now()
            self._log(
                "execution_completed",
                {
                    "success": not is_violated,
                    "duration_seconds": (completed_at - started_at).total_seconds(),
                },
            )

            return ExecutionResult(
                success=not is_violated,
                output=output,
                resource_usage=self._get_usage_dict(),
                violations=self._violations,
                execution_log=self._execution_log,
                strategy=strategy,
                contract_state=self.contract.state,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            completed_at = datetime.now()
            self._log("execution_failed", {"error": str(e)})

            # Mark contract as violated on error
            if self.contract.state == ContractState.ACTIVE:
                self.contract.violate(str(e))

            if self.strict_mode:
                raise

            return ExecutionResult(
                success=False,
                output=None,
                resource_usage=self._get_usage_dict(),
                violations=[*self._violations, str(e)],
                execution_log=self._execution_log,
                contract_state=self.contract.state,
                started_at=started_at,
                completed_at=completed_at,
                error=str(e),
            )

    def _extract_task_description(self, **kwargs: Any) -> str:
        """Extract task description from inputs.

        Args:
            **kwargs: Input arguments

        Returns:
            Task description string
        """
        # Try common input patterns
        if "query" in kwargs:
            return str(kwargs["query"])
        elif "prompt" in kwargs:
            return str(kwargs["prompt"])
        elif "task" in kwargs:
            return str(kwargs["task"])
        elif kwargs.get("messages"):
            # Extract from last user message
            messages = kwargs["messages"]
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return str(msg.get("content", ""))
            # Fallback to last message
            return str(messages[-1].get("content", ""))
        elif "input" in kwargs:
            return str(kwargs["input"])
        else:
            # Use contract description as fallback
            return self.contract.description or self.contract.name

    def _execute_llm(self, system_prompt: str, **kwargs: Any) -> Any:
        """Execute the LLM call.

        This method handles the actual LLM invocation, selecting the
        appropriate backend based on capabilities.

        Args:
            system_prompt: Budget-aware system prompt
            **kwargs: Input arguments

        Returns:
            LLM response
        """
        # Determine execution mode based on input format
        if "messages" in kwargs:
            return self._execute_chat(system_prompt, kwargs["messages"])
        else:
            # Convert to simple completion
            query = self._extract_task_description(**kwargs)
            return self._execute_completion(system_prompt, query)

    def _execute_completion(self, system_prompt: str, query: str) -> str:
        """Execute a simple completion request.

        Args:
            system_prompt: System prompt with budget awareness
            query: User query

        Returns:
            Completion response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return self._execute_chat(system_prompt, messages)

    def _execute_chat(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
        """Execute a chat completion request.

        Uses LiteLLM as the universal LLM interface.

        Args:
            system_prompt: System prompt (may be prepended to messages)
            messages: Chat messages

        Returns:
            Chat response text
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for contract execution. Install with: pip install litellm"
            ) from e

        # Prepare messages with system prompt
        full_messages = self._prepare_messages(system_prompt, messages)

        # Build LiteLLM parameters
        params = self._build_llm_params()

        # Execute with tracking
        self._log(
            "llm_call_started",
            {
                "model": self.execution_config.model,
                "message_count": len(full_messages),
            },
        )

        response = litellm.completion(
            model=self.execution_config.model,
            messages=full_messages,
            **params,
        )

        # Track usage
        self._track_usage(response)

        # Extract response text
        content = response.choices[0].message.content or ""
        self._log("llm_call_completed", {"response_length": len(content)})

        return content

    def _prepare_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepare messages for LLM call.

        Args:
            system_prompt: Budget-aware system prompt
            messages: Original messages

        Returns:
            Messages with system prompt integrated
        """
        # Check if messages already have a system message
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system:
            # Prepend to existing system message
            result = []
            for msg in messages:
                if msg.get("role") == "system":
                    combined_content = system_prompt + "\n\n" + msg.get("content", "")
                    result.append({"role": "system", "content": combined_content})
                else:
                    result.append(msg)
            return result
        else:
            # Add system message at the beginning
            return [{"role": "system", "content": system_prompt}, *messages]

    def _build_llm_params(self) -> dict[str, Any]:
        """Build parameters for LiteLLM call.

        Returns:
            Dict of LiteLLM parameters
        """
        params: dict[str, Any] = {
            "temperature": self.execution_config.temperature,
        }

        # Add max_tokens if specified
        if self.contract.resources.tokens is not None:
            # Use remaining tokens as max
            remaining = self.resource_monitor.get_remaining_tokens()
            if remaining > 0:
                params["max_tokens"] = min(remaining, 4096)  # Cap at reasonable max

        return params

    def _track_usage(self, response: Any) -> None:
        """Track resource usage from LLM response.

        Args:
            response: LiteLLM response object
        """
        usage = getattr(response, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

            # Track tokens
            self.resource_monitor.usage.add_tokens(total_tokens)

            # Estimate cost (simplified)
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            self.resource_monitor.usage.add_api_call(cost=cost, tokens=0)

            self._log(
                "usage_tracked",
                {
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost,
                },
            )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on model and token counts.

        Args:
            prompt_tokens: Input token count
            completion_tokens: Output token count

        Returns:
            Estimated cost in USD
        """
        # Simplified pricing (per 1M tokens)
        model = self.execution_config.model.lower()

        if "gpt-4o" in model:
            prompt_price = 2.50 / 1_000_000  # $2.50/1M input
            completion_price = 10.00 / 1_000_000  # $10/1M output
        elif "gpt-4" in model:
            prompt_price = 30.00 / 1_000_000
            completion_price = 60.00 / 1_000_000
        elif "gpt-3.5" in model:
            prompt_price = 0.50 / 1_000_000
            completion_price = 1.50 / 1_000_000
        elif "claude" in model:
            prompt_price = 3.00 / 1_000_000
            completion_price = 15.00 / 1_000_000
        elif "gemini" in model:
            prompt_price = 0.075 / 1_000_000
            completion_price = 0.30 / 1_000_000
        else:
            # Default conservative estimate
            prompt_price = 10.00 / 1_000_000
            completion_price = 30.00 / 1_000_000

        return (prompt_tokens * prompt_price) + (completion_tokens * completion_price)

    def _get_usage_dict(self) -> dict[str, Any]:
        """Get current resource usage as a dictionary.

        Returns:
            Dict with usage metrics
        """
        usage = self.resource_monitor.usage
        return {
            "tokens": usage.tokens,
            "api_calls": usage.api_calls,
            "web_searches": usage.web_searches,
            "tool_invocations": usage.tool_invocations,
            "cost_usd": usage.cost_usd,
        }

    def _log(self, event: str, data: dict[str, Any]) -> None:
        """Add entry to execution log.

        Args:
            event: Event name
            data: Event data
        """
        self._execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                **data,
            }
        )

    def get_adaptive_instruction(self) -> str:
        """Get current adaptive instruction based on budget state.

        This is useful for multi-step execution where instructions
        need to adapt as budget is consumed.

        Returns:
            Adaptive instruction text
        """
        utilization = 0.0
        if self.contract.resources.tokens:
            utilization = self.resource_monitor.usage.tokens / self.contract.resources.tokens

        return generate_adaptive_instruction(utilization, self.contract.mode)
