"""Unit tests for contract enforcement mechanisms."""

import time
from datetime import datetime, timedelta

import pytest

from agent_contracts.core import (
    Contract,
    ContractEnforcer,
    ContractState,
    DeadlineType,
    EnforcementEvent,
    ResourceConstraints,
    TemporalConstraints,
)


class TestEnforcementEvent:
    """Tests for EnforcementEvent."""

    def test_create_event(self) -> None:
        """Test creating an enforcement event."""
        contract = Contract(id="test", name="Test")
        event = EnforcementEvent(
            event_type="test_event",
            contract=contract,
            message="Test message",
            data={"key": "value"},
        )

        assert event.event_type == "test_event"
        assert event.contract == contract
        assert event.message == "Test message"
        assert event.data == {"key": "value"}
        assert isinstance(event.timestamp, datetime)

    def test_event_without_data(self) -> None:
        """Test creating event without data."""
        contract = Contract(id="test", name="Test")
        event = EnforcementEvent(event_type="test_event", contract=contract, message="Test")

        assert event.data == {}

    def test_repr(self) -> None:
        """Test string representation."""
        contract = Contract(id="test-123", name="Test")
        event = EnforcementEvent(event_type="violation", contract=contract, message="Test")
        repr_str = repr(event)

        assert "violation" in repr_str
        assert "test-123" in repr_str


class TestContractEnforcer:
    """Tests for ContractEnforcer."""

    def test_create_enforcer(self) -> None:
        """Test creating a contract enforcer."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        enforcer = ContractEnforcer(contract)

        assert enforcer.contract == contract
        assert enforcer.strict_mode is True
        assert len(enforcer.callbacks) == 0
        assert not enforcer.is_active()

    def test_create_enforcer_with_callbacks(self) -> None:
        """Test creating enforcer with callbacks."""
        contract = Contract(id="test", name="Test")
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, callbacks=[callback])
        assert len(enforcer.callbacks) == 1

    def test_start_enforcement(self) -> None:
        """Test starting enforcement."""
        contract = Contract(id="test", name="Test")
        enforcer = ContractEnforcer(contract)

        enforcer.start()

        assert enforcer.is_active()
        assert contract.state == ContractState.ACTIVE

    def test_start_enforcement_emits_event(self) -> None:
        """Test that starting enforcement emits an event."""
        contract = Contract(id="test", name="Test")
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, callbacks=[callback])
        enforcer.start()

        assert len(events) == 1
        assert events[0].event_type == "contract_started"

    def test_start_already_active_raises_error(self) -> None:
        """Test that starting already active enforcement raises error."""
        contract = Contract(id="test", name="Test")
        enforcer = ContractEnforcer(contract)
        enforcer.start()

        with pytest.raises(RuntimeError, match="already active"):
            enforcer.start()

    def test_stop_enforcement(self) -> None:
        """Test stopping enforcement."""
        contract = Contract(id="test", name="Test")
        enforcer = ContractEnforcer(contract)
        enforcer.start()

        enforcer.stop(reason="Test complete")

        assert not enforcer.is_active()

    def test_stop_emits_event(self) -> None:
        """Test that stopping enforcement emits an event."""
        contract = Contract(id="test", name="Test")
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, callbacks=[callback])
        enforcer.start()
        events.clear()  # Clear start event

        enforcer.stop(reason="Test")

        assert len(events) == 1
        assert events[0].event_type == "contract_stopped"
        assert events[0].data["reason"] == "Test"

    def test_stop_inactive_enforcer_is_safe(self) -> None:
        """Test that stopping inactive enforcer doesn't error."""
        contract = Contract(id="test", name="Test")
        enforcer = ContractEnforcer(contract)

        enforcer.stop()  # Should not raise

    def test_check_constraints_no_violations(self) -> None:
        """Test checking constraints with no violations."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        enforcer = ContractEnforcer(contract)
        enforcer.start()

        # Add some usage below limits
        enforcer.monitor.usage.add_tokens(500)

        is_violated, violations = enforcer.check_constraints()

        assert not is_violated
        assert len(violations) == 0

    def test_check_constraints_with_violations_strict(self) -> None:
        """Test checking constraints with violations in strict mode."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        enforcer = ContractEnforcer(contract, strict_mode=True)
        enforcer.start()

        # Exceed token limit
        enforcer.monitor.usage.add_tokens(1500)

        is_violated, violations = enforcer.check_constraints()

        assert is_violated
        assert len(violations) == 1
        # Contract should be violated and stopped
        assert contract.state == ContractState.VIOLATED
        assert not enforcer.is_active()

    def test_check_constraints_with_violations_lenient(self) -> None:
        """Test checking constraints with violations in lenient mode."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        enforcer = ContractEnforcer(contract, strict_mode=False)
        enforcer.start()

        # Exceed token limit
        enforcer.monitor.usage.add_tokens(1500)

        is_violated, violations = enforcer.check_constraints()

        assert is_violated
        assert len(violations) == 1
        # Contract should still be active in lenient mode
        assert contract.state == ContractState.ACTIVE
        assert enforcer.is_active()

    def test_check_constraints_emits_violation_event(self) -> None:
        """Test that constraint violations emit events."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, strict_mode=False, callbacks=[callback])
        enforcer.start()
        events.clear()  # Clear start event

        enforcer.monitor.usage.add_tokens(1500)
        enforcer.check_constraints()

        # Should have violation event
        violation_events = [e for e in events if e.event_type == "constraint_violated"]
        assert len(violation_events) == 1
        assert "violations" in violation_events[0].data

    def test_check_temporal_no_violations(self) -> None:
        """Test checking temporal constraints with no violations."""
        future_deadline = datetime.now() + timedelta(hours=1)
        contract = Contract(
            id="test",
            name="Test",
            temporal=TemporalConstraints(
                deadline=future_deadline, max_duration=timedelta(minutes=10)
            ),
        )
        enforcer = ContractEnforcer(contract)
        enforcer.start()

        is_exceeded = enforcer.check_temporal_constraints()

        assert not is_exceeded

    def test_check_temporal_deadline_exceeded(self) -> None:
        """Test checking temporal constraints with deadline exceeded."""
        past_deadline = datetime.now() - timedelta(hours=1)
        contract = Contract(
            id="test", name="Test", temporal=TemporalConstraints(deadline=past_deadline)
        )
        enforcer = ContractEnforcer(contract, strict_mode=True)
        enforcer.start()

        is_exceeded = enforcer.check_temporal_constraints()

        assert is_exceeded
        assert contract.state == ContractState.EXPIRED
        assert not enforcer.is_active()

    def test_check_temporal_duration_exceeded(self) -> None:
        """Test checking temporal constraints with duration exceeded."""
        contract = Contract(
            id="test",
            name="Test",
            temporal=TemporalConstraints(max_duration=timedelta(seconds=0.01)),
        )
        enforcer = ContractEnforcer(contract, strict_mode=True)
        enforcer.start()

        time.sleep(0.02)  # Exceed duration

        is_exceeded = enforcer.check_temporal_constraints()

        assert is_exceeded
        assert contract.state == ContractState.EXPIRED

    def test_check_temporal_lenient_mode(self) -> None:
        """Test temporal checking in lenient mode."""
        past_deadline = datetime.now() - timedelta(hours=1)
        contract = Contract(
            id="test", name="Test", temporal=TemporalConstraints(deadline=past_deadline)
        )
        enforcer = ContractEnforcer(contract, strict_mode=False)
        enforcer.start()

        is_exceeded = enforcer.check_temporal_constraints()

        assert is_exceeded
        # Contract should still be active in lenient mode
        assert contract.state == ContractState.ACTIVE

    def test_get_usage_summary(self) -> None:
        """Test getting usage summary."""
        contract = Contract(id="test", name="Test", resources=ResourceConstraints(tokens=1000))
        enforcer = ContractEnforcer(contract)
        enforcer.start()

        enforcer.monitor.usage.add_tokens(500)

        summary = enforcer.get_usage_summary()

        assert "usage" in summary
        assert "percentages" in summary
        assert "violations" in summary
        assert "contract_state" in summary
        assert "is_violated" in summary
        assert summary["usage"]["tokens"] == 500
        assert summary["percentages"]["tokens"] == 50.0

    def test_add_callback(self) -> None:
        """Test adding callbacks dynamically."""
        contract = Contract(id="test", name="Test")
        enforcer = ContractEnforcer(contract)
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer.add_callback(callback)
        enforcer.start()

        assert len(events) == 1  # Start event

    def test_remove_callback(self) -> None:
        """Test removing callbacks."""
        contract = Contract(id="test", name="Test")
        events: list[EnforcementEvent] = []

        def callback(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, callbacks=[callback])
        enforcer.remove_callback(callback)
        enforcer.start()

        assert len(events) == 0  # Callback was removed

    def test_callback_error_handling(self) -> None:
        """Test that callback errors don't crash enforcement."""
        contract = Contract(id="test", name="Test")

        def bad_callback(event: EnforcementEvent) -> None:
            raise ValueError("Callback error")

        enforcer = ContractEnforcer(contract, callbacks=[bad_callback])
        enforcer.start()  # Should not raise despite callback error

        assert enforcer.is_active()

    def test_repr(self) -> None:
        """Test string representation."""
        contract = Contract(id="test-123", name="Test")
        enforcer = ContractEnforcer(contract, strict_mode=True)

        repr_str = repr(enforcer)
        assert "test-123" in repr_str
        assert "INACTIVE" in repr_str
        assert "STRICT" in repr_str

        enforcer.start()
        repr_str = repr(enforcer)
        assert "ACTIVE" in repr_str


class TestEnforcementIntegration:
    """Integration tests for enforcement system."""

    def test_full_enforcement_workflow(self) -> None:
        """Test complete enforcement workflow."""
        # Create contract with constraints
        contract = Contract(
            id="code-review",
            name="Code Review Agent",
            resources=ResourceConstraints(tokens=10000, api_calls=20, cost_usd=0.5),
            temporal=TemporalConstraints(max_duration=timedelta(minutes=5)),
        )

        # Track events
        events: list[EnforcementEvent] = []

        def event_logger(event: EnforcementEvent) -> None:
            events.append(event)

        # Create enforcer
        enforcer = ContractEnforcer(contract, strict_mode=True, callbacks=[event_logger])

        # Start enforcement
        enforcer.start()
        assert len(events) == 1
        assert events[0].event_type == "contract_started"

        # Simulate agent work
        enforcer.monitor.usage.add_api_call(cost=0.01, tokens=500)
        enforcer.monitor.usage.add_api_call(cost=0.02, tokens=750)
        enforcer.check_constraints()  # Should pass

        assert enforcer.is_active()
        assert contract.state == ContractState.ACTIVE

        # Check usage summary
        summary = enforcer.get_usage_summary()
        assert summary["usage"]["tokens"] == 1250
        assert summary["usage"]["api_calls"] == 2

        # Stop enforcement
        enforcer.stop(reason="Task complete")
        stop_events = [e for e in events if e.event_type == "contract_stopped"]
        assert len(stop_events) == 1

    def test_enforcement_with_violation(self) -> None:
        """Test enforcement handling violations."""
        contract = Contract(
            id="test", name="Test", resources=ResourceConstraints(tokens=1000, api_calls=5)
        )
        events: list[EnforcementEvent] = []

        def event_logger(event: EnforcementEvent) -> None:
            events.append(event)

        enforcer = ContractEnforcer(contract, strict_mode=True, callbacks=[event_logger])
        enforcer.start()
        events.clear()

        # Violate multiple constraints
        enforcer.monitor.usage.add_tokens(2000)
        for _ in range(10):
            enforcer.monitor.usage.add_api_call()

        enforcer.check_constraints()

        # Contract should be violated and stopped
        assert contract.state == ContractState.VIOLATED
        assert not enforcer.is_active()

        # Check events
        violation_events = [e for e in events if e.event_type == "constraint_violated"]
        termination_events = [e for e in events if e.event_type == "contract_terminated"]

        assert len(violation_events) == 1
        assert len(termination_events) == 1
        assert len(violation_events[0].data["violations"]) == 2  # tokens and api_calls

    def test_soft_deadline_workflow(self) -> None:
        """Test workflow with soft deadline."""
        future_deadline = datetime.now() + timedelta(hours=1)
        contract = Contract(
            id="test",
            name="Test",
            temporal=TemporalConstraints(deadline=future_deadline, deadline_type=DeadlineType.SOFT),
        )

        enforcer = ContractEnforcer(contract, strict_mode=False)
        enforcer.start()

        # Work should continue even if deadline passes (in lenient mode)
        is_exceeded = enforcer.check_temporal_constraints()
        assert not is_exceeded
        assert enforcer.is_active()

    def test_multiple_callbacks(self) -> None:
        """Test multiple callbacks receiving events."""
        contract = Contract(id="test", name="Test")

        events1: list[EnforcementEvent] = []
        events2: list[EnforcementEvent] = []

        def callback1(event: EnforcementEvent) -> None:
            events1.append(event)

        def callback2(event: EnforcementEvent) -> None:
            events2.append(event)

        enforcer = ContractEnforcer(contract, callbacks=[callback1, callback2])
        enforcer.start()

        # Both callbacks should receive the event
        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0].event_type == events2[0].event_type

    def test_realistic_budget_scenario(self) -> None:
        """Test realistic budget management scenario."""
        # Set a realistic budget
        contract = Contract(
            id="research-task",
            name="Research Task",
            resources=ResourceConstraints(
                tokens=50000,  # ~$1-2 for most models
                api_calls=100,
                cost_usd=5.0,
            ),
            temporal=TemporalConstraints(max_duration=timedelta(minutes=30)),
        )

        enforcer = ContractEnforcer(contract, strict_mode=False)
        enforcer.start()

        # Simulate incremental work
        for i in range(20):
            enforcer.monitor.usage.add_api_call(cost=0.10, tokens=2000)
            is_violated, _ = enforcer.check_constraints()

            if i < 10:
                # First half should be fine
                assert not is_violated
            else:
                # After 10 calls, cost exceeds budget
                if is_violated:
                    break

        # Check final state
        summary = enforcer.get_usage_summary()
        assert summary["usage"]["api_calls"] <= 100
        assert summary["usage"]["cost_usd"] >= 2.0  # At least some cost accumulated
