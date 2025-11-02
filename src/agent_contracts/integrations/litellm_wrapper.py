"""LiteLLM integration for contract-enforced LLM calls.

This module provides a wrapper around litellm that automatically enforces
contract constraints during LLM API calls.
"""

from typing import Any

from litellm import completion

from agent_contracts.core import Contract, ContractEnforcer, EnforcementEvent, TokenCounter


class ContractViolationError(Exception):
    """Raised when a contract constraint is violated during LLM execution."""

    pass


class ContractedLLM:
    """LLM wrapper with automatic contract enforcement.

    This class wraps litellm's completion API and automatically enforces
    contract constraints, tracking tokens, costs, and API calls in real-time.

    Attributes:
        contract: The contract to enforce
        enforcer: Contract enforcer instance
        auto_start: Whether to automatically start enforcement on first call
    """

    def __init__(
        self,
        contract: Contract,
        strict_mode: bool = True,
        auto_start: bool = True,
    ) -> None:
        """Initialize contracted LLM wrapper.

        Args:
            contract: Contract to enforce
            strict_mode: If True, violations immediately raise errors
            auto_start: If True, automatically start enforcement on first call
        """
        self.contract = contract
        self.enforcer = ContractEnforcer(contract, strict_mode=strict_mode)
        self.auto_start = auto_start
        self._started = False

    def start(self) -> None:
        """Start contract enforcement.

        Raises:
            RuntimeError: If enforcement is already active
        """
        if self._started:
            raise RuntimeError("Enforcement already started")

        self.enforcer.start()
        self._started = True

    def stop(self) -> None:
        """Stop contract enforcement."""
        if self._started:
            self.enforcer.stop()
            self._started = False

    def completion(self, **kwargs: Any) -> Any:
        """Make a completion call with contract enforcement.

        This wraps litellm.completion() and automatically:
        - Checks constraints before the call
        - Tracks tokens and costs
        - Updates resource usage
        - Checks constraints after the call
        - Raises ContractViolationError if violated in strict mode

        Args:
            **kwargs: Arguments to pass to litellm.completion()

        Returns:
            litellm completion response

        Raises:
            ContractViolationError: If contract is violated in strict mode
        """
        # Auto-start if needed
        if self.auto_start and not self._started:
            self.start()

        # Check constraints before call
        self._check_constraints_before_call()

        # Auto-apply reasoning_effort from contract if not already specified
        if "reasoning_effort" not in kwargs:
            effort = self._get_reasoning_effort()
            if effort is not None:
                kwargs["reasoning_effort"] = effort

        # Make the LLM call
        try:
            response = completion(**kwargs)
        except Exception as e:
            # Track failed API call
            self.enforcer.monitor.usage.add_api_call()
            raise e

        # Extract token usage from response
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Extract reasoning vs text tokens for reasoning models (e.g., Gemini 2.5, o1)
        reasoning_tokens = 0
        text_tokens = 0
        completion_tokens_details = usage.get("completion_tokens_details")
        if completion_tokens_details:
            # Handle both dict and Pydantic object formats
            if isinstance(completion_tokens_details, dict):
                reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
                text_tokens = completion_tokens_details.get("text_tokens", 0)
            else:
                # Pydantic object - use attribute access
                reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", 0) or 0
                text_tokens = getattr(completion_tokens_details, "text_tokens", 0) or 0

        # Estimate cost using our token counter or litellm's tracking
        model = kwargs.get("model", "unknown")
        try:
            # Try to use litellm's cost tracking if available
            cost = response.get("_hidden_params", {}).get("response_cost", 0)
            if cost == 0:
                # Fallback to our cost estimation
                from agent_contracts.core.tokens import TokenCount

                token_count = TokenCount(input_tokens=input_tokens, output_tokens=output_tokens)
                cost_estimate = TokenCounter.calculate_cost(token_count, model)
                cost = cost_estimate.total_cost
        except Exception:
            # If cost calculation fails, use 0
            cost = 0

        # Update resource usage with separate reasoning/text tracking
        # Note: Don't pass tokens to add_api_call since we track them separately below
        self.enforcer.monitor.usage.add_api_call(cost=cost, tokens=0)

        # Track tokens based on whether model provides reasoning/text breakdown
        if reasoning_tokens > 0 or text_tokens > 0:
            # Models with breakdown (Gemini 2.5, o1, etc.) - use detailed tracking
            self.enforcer.monitor.usage.add_tokens(
                count=0, reasoning=reasoning_tokens, text=text_tokens
            )
        else:
            # Models without breakdown (GPT-4, Claude, etc.) - treat all as text
            # This allows fine-grained mode to work with non-reasoning models
            self.enforcer.monitor.usage.add_tokens(count=0, reasoning=0, text=output_tokens)

        # Also track input tokens
        self.enforcer.monitor.usage.add_tokens(count=input_tokens, reasoning=0, text=0)

        # Emit completion event
        self.enforcer._emit_event(
            EnforcementEvent(
                event_type="llm_completion",
                contract=self.contract,
                message=f"LLM completion: {model}",
                data={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                },
            )
        )

        # Check constraints after call
        self._check_constraints_after_call()

        return response

    def streaming_completion(self, **kwargs: Any) -> Any:
        """Make a streaming completion call with contract enforcement.

        This wraps litellm.completion(stream=True) and checks constraints
        periodically during streaming.

        Args:
            **kwargs: Arguments to pass to litellm.completion()

        Yields:
            Streaming response chunks

        Raises:
            ContractViolationError: If contract is violated in strict mode
        """
        # Force streaming mode
        kwargs["stream"] = True

        # Auto-start if needed
        if self.auto_start and not self._started:
            self.start()

        # Check constraints before call
        self._check_constraints_before_call()

        # Auto-apply reasoning_effort from contract if not already specified
        if "reasoning_effort" not in kwargs:
            effort = self._get_reasoning_effort()
            if effort is not None:
                kwargs["reasoning_effort"] = effort

        # Track that we made an API call
        self.enforcer.monitor.usage.add_api_call()

        # Stream the response
        response = completion(**kwargs)

        # Track tokens as we stream
        total_input_tokens = 0
        total_output_tokens = 0
        chunk_count = 0
        model = kwargs.get("model", "unknown")

        try:
            for chunk in response:
                chunk_count += 1

                # Extract token usage if available
                usage = chunk.get("usage")
                if usage:
                    total_input_tokens = usage.get("prompt_tokens", total_input_tokens)
                    total_output_tokens += usage.get("completion_tokens", 0)

                # Check constraints periodically (every 10 chunks)
                if chunk_count % 10 == 0:
                    # Update estimated usage
                    estimated_tokens = total_input_tokens + total_output_tokens
                    if estimated_tokens > 0:
                        # Update token count
                        current_tokens = self.enforcer.monitor.usage.tokens
                        self.enforcer.monitor.usage.tokens = (
                            current_tokens - (chunk_count - 10) + estimated_tokens
                        )

                    # Check constraints
                    is_violated, violations = self.enforcer.check_constraints()
                    if is_violated and self.enforcer.strict_mode:
                        raise ContractViolationError(
                            f"Contract violated during streaming: {violations}"
                        )

                yield chunk

        finally:
            # Final token count update
            total_tokens = total_input_tokens + total_output_tokens
            if total_tokens > 0:
                # Set final token count
                self.enforcer.monitor.usage.tokens = (
                    self.enforcer.monitor.usage.tokens - chunk_count + total_tokens
                )

            # Estimate final cost
            try:
                from agent_contracts.core.tokens import TokenCount

                token_count = TokenCount(
                    input_tokens=total_input_tokens, output_tokens=total_output_tokens
                )
                cost_estimate = TokenCounter.calculate_cost(token_count, model)
                cost = cost_estimate.total_cost

                # Update cost
                self.enforcer.monitor.usage.add_cost(cost)
            except Exception:
                pass

            # Emit completion event
            self.enforcer._emit_event(
                EnforcementEvent(
                    event_type="llm_streaming_completion",
                    contract=self.contract,
                    message=f"LLM streaming completion: {model}",
                    data={
                        "model": model,
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_tokens,
                        "chunks": chunk_count,
                    },
                )
            )

            # Final constraint check
            self._check_constraints_after_call()

    def _get_reasoning_effort(self) -> str | None:
        """Get reasoning effort level to use for the call.

        Priority:
        1. Explicit reasoning_effort in contract (if specified)
        2. Auto-selected based on reasoning_tokens budget
        3. None if no reasoning constraints

        Returns:
            Reasoning effort level ("low"/"medium"/"high") or None
        """
        # If explicitly specified in contract, use that
        if self.contract.resources.reasoning_effort:
            return self.contract.resources.reasoning_effort

        # Otherwise, auto-select based on budget
        return self.contract.resources.recommended_reasoning_effort

    def _check_constraints_before_call(self) -> None:
        """Check constraints before making an LLM call.

        Raises:
            ContractViolationError: If already violated in strict mode
        """
        if not self._started:
            return

        # Check if already violated
        is_violated, violations = self.enforcer.check_constraints()
        if is_violated and self.enforcer.strict_mode:
            raise ContractViolationError(f"Contract already violated: {violations}")

        # Check temporal constraints
        is_exceeded = self.enforcer.check_temporal_constraints()
        if is_exceeded and self.enforcer.strict_mode:
            raise ContractViolationError("Temporal constraints exceeded")

    def _check_constraints_after_call(self) -> None:
        """Check constraints after making an LLM call.

        Raises:
            ContractViolationError: If violated in strict mode
        """
        if not self._started:
            return

        # Check resource constraints
        is_violated, violations = self.enforcer.check_constraints()
        if is_violated and self.enforcer.strict_mode:
            raise ContractViolationError(f"Contract violated: {violations}")

        # Check temporal constraints
        is_exceeded = self.enforcer.check_temporal_constraints()
        if is_exceeded and self.enforcer.strict_mode:
            raise ContractViolationError("Temporal constraints exceeded")

    def get_usage_summary(self) -> dict[str, Any]:
        """Get current resource usage summary.

        Returns:
            Dictionary with usage statistics
        """
        return self.enforcer.get_usage_summary()

    def add_callback(self, callback: Any) -> None:
        """Add event callback.

        Args:
            callback: Callback function for enforcement events
        """
        self.enforcer.add_callback(callback)

    def __enter__(self) -> "ContractedLLM":
        """Context manager entry."""
        if not self._started:
            self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        status = "STARTED" if self._started else "NOT_STARTED"
        return f"ContractedLLM(contract='{self.contract.id}', status={status})"
