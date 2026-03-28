"""Contract enforcement mechanisms.

This module implements the enforcement layer that actively monitors and enforces
contract constraints during agent execution.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from agent_contracts.core.contract import Contract, ContractState
from agent_contracts.core.monitor import ResourceMonitor, ViolationInfo

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CheckContext:
    """Context passed to pre/post-check hooks.

    Provides hooks with the contract, resource monitor, current phase,
    and integration-specific metadata for making allow/block decisions.

    Attributes:
        contract: The contract being enforced
        monitor: Resource monitor with current usage state
        phase: Which phase triggered the hook ("pre_check" or "post_check")
        metadata: Integration-specific data (e.g. {"integration": "litellm", "model": "gpt-4"})
    """

    contract: Contract
    monitor: ResourceMonitor
    phase: Literal["pre_check", "post_check"]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HookResult:
    """Result from a check hook execution.

    Attributes:
        allow: If True, execution proceeds. If False, action determines severity.
        reason: Human-readable explanation (used in enforcement events when allow=False)
        action: What to do when allow=False. WARN/THROTTLE emit events but don't block.
                SOFT_STOP/HARD_STOP emit events and block execution.
    """

    allow: bool = True
    reason: str = ""
    action: EnforcementAction = EnforcementAction.WARN  # Only consulted when allow=False


# Type alias for check hooks
CheckHook = Callable[[CheckContext], HookResult]


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
        monitor: ResourceMonitor | None = None,
        pre_check_hooks: list[CheckHook] | None = None,
        post_check_hooks: list[CheckHook] | None = None,
    ) -> None:
        """Initialize contract enforcer.

        Args:
            contract: Contract to enforce
            strict_mode: If True, violations cause immediate termination
            callbacks: Optional list of callback functions for events
            monitor: Optional pre-existing monitor to use (creates one if None)
            pre_check_hooks: Optional hooks to run before constraint checking
            post_check_hooks: Optional hooks to run after constraint checking
        """
        self.contract = contract
        self.monitor = monitor or ResourceMonitor(contract.resources)
        self.strict_mode = strict_mode
        self.callbacks = callbacks or []
        self.pre_check_hooks: list[CheckHook] = pre_check_hooks or []
        self.post_check_hooks: list[CheckHook] = post_check_hooks or []
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

    def check_constraints(
        self, metadata: dict[str, Any] | None = None
    ) -> tuple[bool, list[ViolationInfo]]:
        """Check if current usage violates any constraints.

        Runs pre-check hooks before and post-check hooks after constraint checking.

        Args:
            metadata: Optional integration-specific data passed to hooks

        Returns:
            Tuple of (is_violated, violations_list)
        """
        resolved_metadata = metadata or {}

        # 1. Run pre-check hooks
        blocked = self._run_hooks(self.pre_check_hooks, "pre_check", resolved_metadata)
        if blocked:
            return True, [ViolationInfo(resource="hook", limit=0, actual=0)]

        # 2. Existing constraint checking (unchanged)
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

        # 3. Run post-check hooks
        self._run_hooks(self.post_check_hooks, "post_check", resolved_metadata)

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

    def add_pre_check_hook(self, hook: CheckHook) -> None:
        """Add a pre-check hook."""
        self.pre_check_hooks.append(hook)

    def remove_pre_check_hook(self, hook: CheckHook) -> None:
        """Remove a pre-check hook."""
        if hook in self.pre_check_hooks:
            self.pre_check_hooks.remove(hook)

    def add_post_check_hook(self, hook: CheckHook) -> None:
        """Add a post-check hook."""
        self.post_check_hooks.append(hook)

    def remove_post_check_hook(self, hook: CheckHook) -> None:
        """Remove a post-check hook."""
        if hook in self.post_check_hooks:
            self.post_check_hooks.remove(hook)

    def _run_hooks(
        self,
        hooks: list[CheckHook],
        phase: Literal["pre_check", "post_check"],
        metadata: dict[str, Any],
    ) -> bool:
        """Run hooks, emit events, return True if any hook blocked."""
        context = CheckContext(
            contract=self.contract,
            monitor=self.monitor,
            phase=phase,
            metadata=metadata,
        )
        for hook in hooks:
            try:
                result = hook(context)
            except Exception as e:
                logger.warning("Error in check hook: %s", e, exc_info=True)
                continue
            if not result.allow:
                self._emit_event(
                    EnforcementEvent(
                        event_type="hook_blocked",
                        contract=self.contract,
                        message=f"Hook blocked execution: {result.reason}",
                        data={
                            "phase": phase,
                            "action": result.action.value,
                            "reason": result.reason,
                        },
                    )
                )
                if result.action in (EnforcementAction.HARD_STOP, EnforcementAction.SOFT_STOP):
                    return True
        return False

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
                logger.warning("Error in enforcement callback: %s", e, exc_info=True)

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
