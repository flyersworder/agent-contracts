"""Contract-aware agent wrapper (Whitepaper Section 5.3).

This module implements the ContractAgent pattern from the whitepaper, providing
a generic wrapper that adds contract enforcement to any agent or callable.

The wrapper handles:
- Contract lifecycle management (DRAFTED → ACTIVE → FULFILLED/VIOLATED)
- Resource and temporal monitoring
- Budget awareness injection
- Violation detection and handling
- Execution logging for audit trails

Example:
    >>> from agent_contracts import Contract, ContractAgent
    >>>
    >>> # Define contract
    >>> contract = Contract(
    ...     id="my-task",
    ...     resources=ResourceConstraints(tokens=10000, cost_usd=1.0)
    ... )
    >>>
    >>> # Wrap any callable
    >>> def my_agent(task: str) -> str:
    ...     return f"Completed: {task}"
    >>>
    >>> wrapped = ContractAgent(contract=contract, agent=my_agent)
    >>> result = wrapped.execute("Write a report")
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_contracts.core.contract import Contract, ContractState
from agent_contracts.core.enforcement import ContractEnforcer, EnforcementEvent
from agent_contracts.core.monitor import ResourceMonitor, TemporalMonitor


@dataclass
class ExecutionResult[TOutput]:
    """Result of contract-bounded agent execution.

    Attributes:
        output: The agent's output (if successful)
        contract: The contract that governed execution
        success: Whether the contract was fulfilled
        violations: List of any violations that occurred
        execution_log: Detailed execution log for audit
        metadata: Additional execution metadata
    """

    output: TOutput | None
    contract: Contract
    success: bool
    violations: list[str]
    execution_log: "ExecutionLog"
    metadata: dict[str, Any]


@dataclass
class ExecutionLog:
    """Detailed log of contract execution for audit trails.

    Attributes:
        contract_id: ID of the contract
        start_time: When execution started
        end_time: When execution ended
        final_state: Final contract state
        resource_usage: Actual resource consumption
        temporal_metrics: Time-related metrics
        events: List of enforcement events
        metadata: Additional log metadata
    """

    contract_id: str
    start_time: datetime
    end_time: datetime | None
    final_state: ContractState
    resource_usage: dict[str, Any]
    temporal_metrics: dict[str, Any]
    events: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert log to dictionary for JSON serialization."""
        return {
            "contract_id": self.contract_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "final_state": self.final_state.value,
            "resource_usage": self.resource_usage,
            "temporal_metrics": self.temporal_metrics,
            "events": self.events,
            "metadata": self.metadata,
        }


class ContractAgent[TInput, TOutput]:
    """Contract-aware agent wrapper (Whitepaper Section 5.3).

    This class wraps any callable (function, agent, chain) and adds contract
    enforcement, monitoring, and audit logging.

    The pattern follows the whitepaper's ContractAgent design:
    1. Activate contract
    2. Inject budget/time awareness into agent
    3. Execute with monitoring
    4. Verify success criteria
    5. Update contract state
    6. Return results with audit log

    Attributes:
        contract: The contract governing execution
        agent: The underlying agent/callable to wrap
        enforcer: Contract enforcement engine
        resource_monitor: Resource consumption tracker
        temporal_monitor: Time constraint tracker
        execution_log: Current execution log
        strict_mode: If True, violations immediately stop execution
    """

    def __init__(
        self,
        contract: Contract,
        agent: Callable[[TInput], TOutput],
        strict_mode: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """Initialize contract-aware agent wrapper.

        Args:
            contract: Contract to enforce
            agent: Underlying agent/callable to wrap
            strict_mode: If True, violations cause immediate termination
            enable_logging: If True, log execution for audit trail
        """
        self.contract = contract
        self.agent = agent
        self.strict_mode = strict_mode
        self.enable_logging = enable_logging

        # Initialize monitoring and enforcement
        self.resource_monitor = ResourceMonitor(contract.resources)
        self.temporal_monitor = TemporalMonitor(contract)
        self.enforcer = ContractEnforcer(
            contract=contract,
            strict_mode=strict_mode,
            callbacks=[self._on_enforcement_event] if enable_logging else None,
        )
        # IMPORTANT: Make enforcer use the same resource monitor for tracking
        self.enforcer.monitor = self.resource_monitor

        # Execution state
        self.execution_log: ExecutionLog | None = None
        self._events: list[dict[str, Any]] = []

    def execute(self, input_data: TInput) -> ExecutionResult[TOutput]:
        """Execute agent within contract constraints.

        This method implements the contract execution pattern from the whitepaper:
        1. Activate contract (DRAFTED → ACTIVE)
        2. Set up monitoring
        3. Execute agent with constraints
        4. Check success criteria
        5. Update contract state (FULFILLED/VIOLATED)
        6. Return results with audit log

        Args:
            input_data: Input to pass to the agent

        Returns:
            ExecutionResult with output, success status, and audit log

        Raises:
            ContractViolationError: If strict_mode=True and violation occurs
        """
        # Start execution
        start_time = datetime.now()
        self._events = []

        # Initialize execution log
        if self.enable_logging:
            self.execution_log = ExecutionLog(
                contract_id=self.contract.id,
                start_time=start_time,
                end_time=None,
                final_state=ContractState.ACTIVE,
                resource_usage={},
                temporal_metrics={},
                events=[],
                metadata={},
            )

        # Start monitoring
        self.temporal_monitor.start()

        # Start enforcement (only if not already active)
        if not self.enforcer._enforcement_active:
            self.enforcer.start()

        try:
            # Inject budget/time awareness (optional - agent may use these)
            # For now, agents need to manually check these via get_remaining_budget()
            # Future: Could use decorators or context managers for automatic injection

            # Execute agent with monitoring
            output = self._monitored_execution(input_data)

            # Check constraints after execution
            is_violated, _constraint_violations = self.enforcer.check_constraints()

            # Check temporal constraints
            self.enforcer.check_temporal_constraints()

            # Check success criteria
            success = self._check_success_criteria(output) and not is_violated

            # Update contract state based on violations
            # Note: We keep it ACTIVE if successful to allow cumulative tracking
            # Only mark as VIOLATED if there were actual violations
            if is_violated:
                self.contract.state = ContractState.VIOLATED
            elif not success:
                self.contract.state = ContractState.VIOLATED
                self._events.append(
                    {
                        "type": "incomplete",
                        "message": "Success criteria not met",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            # else: keep contract in ACTIVE state for cumulative tracking

            violations = [
                event["message"]
                for event in self._events
                if event["type"] in ("violation", "constraint_violated")
            ]

            # Note: We don't stop the enforcer here to allow cumulative tracking
            # across multiple execute() calls. The enforcer stays active.

            # Finalize log
            end_time = datetime.now()
            if self.enable_logging and self.execution_log:
                self.execution_log.end_time = end_time
                self.execution_log.final_state = self.contract.state
                self.execution_log.resource_usage = {
                    "tokens": self.resource_monitor.usage.tokens,
                    "reasoning_tokens": self.resource_monitor.usage.reasoning_tokens,
                    "text_tokens": self.resource_monitor.usage.text_tokens,
                    "api_calls": self.resource_monitor.usage.api_calls,
                    "cost_usd": self.resource_monitor.usage.cost_usd,
                }
                self.execution_log.temporal_metrics = {
                    "elapsed_seconds": (end_time - start_time).total_seconds(),
                    "deadline_met": not self.temporal_monitor.is_past_deadline(),
                }
                self.execution_log.events = self._events

            return ExecutionResult(
                output=output,
                contract=self.contract,
                success=success,
                violations=violations,
                execution_log=self.execution_log,  # type: ignore
                metadata={"elapsed_seconds": (end_time - start_time).total_seconds()},
            )

        except Exception as e:
            # Note: Don't stop enforcer to allow recovery and cumulative tracking

            # Handle execution failure
            self.contract.state = ContractState.VIOLATED
            self._events.append(
                {"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()}
            )

            # Finalize log
            end_time = datetime.now()
            if self.enable_logging and self.execution_log:
                self.execution_log.end_time = end_time
                self.execution_log.final_state = self.contract.state
                self.execution_log.events = self._events

            return ExecutionResult(
                output=None,
                contract=self.contract,
                success=False,
                violations=[str(e)],
                execution_log=self.execution_log,  # type: ignore
                metadata={
                    "error": str(e),
                    "elapsed_seconds": (end_time - start_time).total_seconds(),
                },
            )

    def _monitored_execution(self, input_data: TInput) -> TOutput:
        """Execute agent with active monitoring.

        This is a hook for subclasses to implement custom monitoring logic.
        The base implementation simply calls the agent.

        Args:
            input_data: Input to pass to the agent

        Returns:
            Agent's output
        """
        # For base implementation, just call the agent
        # Subclasses (like ContractedChain) will override to add monitoring
        return self.agent(input_data)

    def _check_success_criteria(self, output: TOutput) -> bool:
        """Check if success criteria are met.

        Args:
            output: Agent's output

        Returns:
            True if success criteria met, False otherwise
        """
        # Default: consider successful if output is not None
        # Subclasses can override for custom success criteria
        return output is not None

    def _on_enforcement_event(self, event: EnforcementEvent) -> None:
        """Handle enforcement events for logging.

        Args:
            event: Enforcement event from ContractEnforcer
        """
        if self.enable_logging:
            self._events.append(
                {
                    "type": event.event_type,
                    "message": event.message,
                    "data": event.data,
                    "timestamp": event.timestamp.isoformat(),
                }
            )

    def get_remaining_budget(self) -> dict[str, float]:
        """Get remaining resource budget.

        Agents can call this to check how much budget remains.

        Returns:
            Dictionary with remaining budget for each resource
        """
        return {
            "tokens": self.resource_monitor.get_remaining_tokens(),
            "cost_usd": self.resource_monitor.get_remaining_cost(),
            "api_calls": self.resource_monitor.get_remaining_api_calls(),
        }

    def get_time_pressure(self) -> float:
        """Get current time pressure (0.0 = no pressure, 1.0 = deadline reached).

        Agents can call this to adjust strategy based on time remaining.

        Returns:
            Time pressure value between 0.0 and 1.0
        """
        return self.temporal_monitor.get_time_pressure()

    def to_json(self) -> dict[str, Any]:
        """Export execution log as JSON for audit.

        Returns:
            Dictionary representation of execution log
        """
        if not self.execution_log:
            return {"error": "No execution log available"}
        return self.execution_log.to_dict()


class ContractViolationError(Exception):
    """Raised when contract constraints are violated in strict mode.

    Attributes:
        contract: The violated contract
        violation_type: Type of violation (e.g., "budget", "deadline")
        message: Human-readable error message
    """

    def __init__(self, contract: Contract, violation_type: str, message: str) -> None:
        """Initialize violation error.

        Args:
            contract: The violated contract
            violation_type: Type of violation
            message: Error message
        """
        self.contract = contract
        self.violation_type = violation_type
        super().__init__(f"Contract {contract.id} violated ({violation_type}): {message}")
