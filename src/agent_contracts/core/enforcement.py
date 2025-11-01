"""Contract enforcement mechanisms.

This module implements the enforcement layer that actively monitors and enforces
contract constraints during agent execution.
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from agent_contracts.core.contract import Contract, ContractState
from agent_contracts.core.monitor import ResourceMonitor, ViolationInfo


class EnforcementAction(Enum):
    """Actions that can be taken when constraints are violated.

    Attributes:
        WARN: Log a warning but continue execution
        SOFT_STOP: Request graceful termination
        HARD_STOP: Immediately terminate execution
        THROTTLE: Slow down execution
    """

    WARN = "warn"
    SOFT_STOP = "soft_stop"
    HARD_STOP = "hard_stop"
    THROTTLE = "throttle"


class EnforcementEvent:
    """An event triggered during contract enforcement.

    Attributes:
        event_type: Type of enforcement event
        contract: The contract being enforced
        message: Human-readable description
        data: Additional event data
        timestamp: When the event occurred
    """

    def __init__(
        self,
        event_type: str,
        contract: Contract,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize enforcement event.

        Args:
            event_type: Type of event (e.g., "violation", "warning", "completion")
            contract: Contract being enforced
            message: Human-readable description
            data: Additional event data
        """
        self.event_type = event_type
        self.contract = contract
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        """String representation of event."""
        return (
            f"EnforcementEvent({self.event_type}, contract={self.contract.id}, "
            f"message='{self.message}', timestamp={self.timestamp})"
        )


# Type alias for enforcement callbacks
EnforcementCallback = Callable[[EnforcementEvent], None]


class ContractEnforcer:
    """Enforces contract constraints during agent execution.

    This class combines contracts with resource monitoring to provide active
    enforcement, handling violations, warnings, and termination conditions.

    Attributes:
        contract: The contract being enforced
        monitor: Resource monitor tracking actual usage
        callbacks: List of callback functions for enforcement events
        strict_mode: If True, violations immediately terminate execution
    """

    def __init__(
        self,
        contract: Contract,
        strict_mode: bool = True,
        callbacks: list[EnforcementCallback] | None = None,
    ) -> None:
        """Initialize contract enforcer.

        Args:
            contract: Contract to enforce
            strict_mode: If True, violations cause immediate termination
            callbacks: Optional list of callback functions for events
        """
        self.contract = contract
        self.monitor = ResourceMonitor(contract.resources)
        self.strict_mode = strict_mode
        self.callbacks = callbacks or []
        self._enforcement_active = False

    def start(self) -> None:
        """Start enforcement (activate contract).

        Raises:
            RuntimeError: If enforcement is already active
            ValueError: If contract cannot be activated
        """
        if self._enforcement_active:
            raise RuntimeError("Enforcement is already active")

        # Activate the contract
        self.contract.activate()
        self._enforcement_active = True

        # Emit start event
        self._emit_event(
            EnforcementEvent(
                event_type="contract_started",
                contract=self.contract,
                message=f"Contract '{self.contract.name}' enforcement started",
            )
        )

    def stop(self, reason: str = "") -> None:
        """Stop enforcement.

        Args:
            reason: Optional reason for stopping
        """
        if not self._enforcement_active:
            return

        self._enforcement_active = False

        # Emit stop event
        self._emit_event(
            EnforcementEvent(
                event_type="contract_stopped",
                contract=self.contract,
                message=f"Contract '{self.contract.name}' enforcement stopped",
                data={"reason": reason} if reason else {},
            )
        )

    def check_constraints(self) -> tuple[bool, list[ViolationInfo]]:
        """Check if current usage violates any constraints.

        Returns:
            Tuple of (is_violated, violations_list)
        """
        violations = self.monitor.check_constraints()
        is_violated = len(violations) > 0

        if is_violated:
            # Record violations
            for violation in violations:
                self.monitor.record_violation(violation)

            # Emit violation event
            self._emit_event(
                EnforcementEvent(
                    event_type="constraint_violated",
                    contract=self.contract,
                    message=f"Constraint violation: {len(violations)} resource(s) exceeded",
                    data={
                        "violations": [
                            {
                                "resource": v.resource,
                                "limit": v.limit,
                                "actual": v.actual,
                            }
                            for v in violations
                        ]
                    },
                )
            )

            # Handle violation based on strict mode
            if self.strict_mode:
                self._handle_violation(violations)

        return is_violated, violations

    def check_temporal_constraints(self) -> bool:
        """Check if temporal constraints are violated.

        Returns:
            True if time limit exceeded, False otherwise
        """
        if (
            self.contract.temporal.deadline is not None
            and datetime.now() > self.contract.temporal.deadline
        ):
            self._emit_event(
                EnforcementEvent(
                    event_type="deadline_exceeded",
                    contract=self.contract,
                    message="Contract deadline exceeded",
                    data={"deadline": self.contract.temporal.deadline.isoformat()},
                )
            )
            if self.strict_mode:
                self._handle_deadline_exceeded()
            return True

        if self.contract.temporal.max_duration is not None:
            elapsed = self.monitor.usage.elapsed_time()
            if elapsed > self.contract.temporal.max_duration:
                self._emit_event(
                    EnforcementEvent(
                        event_type="duration_exceeded",
                        contract=self.contract,
                        message="Contract max duration exceeded",
                        data={
                            "max_duration": self.contract.temporal.max_duration.total_seconds(),
                            "elapsed": elapsed.total_seconds(),
                        },
                    )
                )
                if self.strict_mode:
                    self._handle_duration_exceeded()
                return True

        return False

    def get_usage_summary(self) -> dict[str, Any]:
        """Get current resource usage summary.

        Returns:
            Dictionary with usage statistics and percentages
        """
        return {
            "usage": self.monitor.usage.to_dict(),
            "percentages": self.monitor.get_usage_percentage(),
            "violations": len(self.monitor.violations),
            "contract_state": self.contract.state.value,
            "is_violated": self.monitor.is_violated(),
        }

    def add_callback(self, callback: EnforcementCallback) -> None:
        """Add a callback function for enforcement events.

        Args:
            callback: Function that takes an EnforcementEvent
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: EnforcementCallback) -> None:
        """Remove a callback function.

        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _emit_event(self, event: EnforcementEvent) -> None:
        """Emit an enforcement event to all callbacks.

        Args:
            event: Event to emit
        """
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                # Don't let callback errors crash enforcement
                print(f"Error in enforcement callback: {e}")

    def _handle_violation(self, violations: list[ViolationInfo]) -> None:
        """Handle constraint violations in strict mode.

        Args:
            violations: List of violations that occurred
        """
        # Build violation reason message
        violation_details = ", ".join([f"{v.resource} ({v.actual}/{v.limit})" for v in violations])
        reason = f"Resource constraints violated: {violation_details}"

        # Mark contract as violated
        self.contract.violate(reason=reason)

        # Stop enforcement
        self.stop(reason=reason)

        # Emit termination event
        self._emit_event(
            EnforcementEvent(
                event_type="contract_terminated",
                contract=self.contract,
                message=f"Contract terminated due to violations: {violation_details}",
                data={"violations": violations, "reason": reason},
            )
        )

    def _handle_deadline_exceeded(self) -> None:
        """Handle deadline exceeded in strict mode."""
        reason = f"Deadline exceeded: {self.contract.temporal.deadline}"
        self.contract.expire()
        self.stop(reason=reason)

        self._emit_event(
            EnforcementEvent(
                event_type="contract_expired",
                contract=self.contract,
                message="Contract expired due to deadline",
                data={"reason": reason},
            )
        )

    def _handle_duration_exceeded(self) -> None:
        """Handle max duration exceeded in strict mode."""
        elapsed = self.monitor.usage.elapsed_time()
        reason = f"Max duration exceeded: {elapsed} > {self.contract.temporal.max_duration}"
        self.contract.expire()
        self.stop(reason=reason)

        self._emit_event(
            EnforcementEvent(
                event_type="contract_expired",
                contract=self.contract,
                message="Contract expired due to duration limit",
                data={"reason": reason},
            )
        )

    def is_active(self) -> bool:
        """Check if enforcement is currently active.

        Returns:
            True if enforcement is active, False otherwise
        """
        return self._enforcement_active and self.contract.state == ContractState.ACTIVE

    def __repr__(self) -> str:
        """String representation of enforcer."""
        status = "ACTIVE" if self.is_active() else "INACTIVE"
        mode = "STRICT" if self.strict_mode else "LENIENT"
        return f"ContractEnforcer(contract='{self.contract.id}', status={status}, mode={mode})"
