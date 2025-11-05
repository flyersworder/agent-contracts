"""Tests for TemporalMonitor (Phase 2B)."""

import time
from datetime import datetime, timedelta

from agent_contracts.core.contract import (
    Contract,
    DeadlineType,
    ResourceConstraints,
    TemporalConstraints,
)
from agent_contracts.core.monitor import TemporalMonitor


class TestTemporalMonitor:
    """Test TemporalMonitor functionality."""

    def test_create_monitor_without_temporal_constraints(self) -> None:
        """Test creating monitor with contract that has no temporal constraints."""
        contract = Contract(
            id="test-no-temporal",
            name="test-no-temporal",
            resources=ResourceConstraints(tokens=1000),
        )

        monitor = TemporalMonitor(contract)

        assert monitor.contract == contract
        assert monitor.start_time is None
        assert monitor.deadline is None
        assert monitor.max_duration is None

    def test_create_monitor_with_max_duration_timedelta(self) -> None:
        """Test creating monitor with max_duration as timedelta."""
        contract = Contract(
            id="test-duration-td",
            name="test-duration-td",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=60)),
        )

        monitor = TemporalMonitor(contract)

        assert monitor.max_duration == 60.0
        assert monitor.deadline is None  # Not set until start()

    def test_create_monitor_with_max_duration_numeric(self) -> None:
        """Test creating monitor with max_duration as number."""
        contract = Contract(
            id="test-duration-num",
            name="test-duration-num",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=30),  # seconds
        )

        monitor = TemporalMonitor(contract)

        assert monitor.max_duration == 30.0

    def test_create_monitor_with_absolute_deadline(self) -> None:
        """Test creating monitor with absolute deadline datetime."""
        deadline = datetime.now() + timedelta(minutes=5)
        contract = Contract(
            id="test-absolute-deadline",
            name="test-absolute-deadline",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(deadline=deadline),
        )

        monitor = TemporalMonitor(contract)

        assert monitor.deadline == deadline

    def test_create_monitor_with_relative_deadline_timedelta(self) -> None:
        """Test creating monitor with deadline as timedelta."""
        contract = Contract(
            id="test-relative-deadline",
            name="test-relative-deadline",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(deadline=timedelta(minutes=10)),
        )

        monitor = TemporalMonitor(contract)

        assert monitor.max_duration == 600.0  # 10 minutes
        assert monitor.deadline is None  # Not set until start()

    def test_start_monitoring(self) -> None:
        """Test starting time monitoring."""
        contract = Contract(
            id="test-start",
            name="test-start",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)
        assert monitor.start_time is None

        monitor.start()

        assert monitor.start_time is not None
        assert isinstance(monitor.start_time, datetime)

    def test_start_sets_absolute_deadline(self) -> None:
        """Test that start() sets absolute deadline from max_duration."""
        contract = Contract(
            id="test-deadline-set",
            name="test-deadline-set",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=30)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.deadline is not None
        # Deadline should be approximately start_time + 30 seconds
        expected_deadline = monitor.start_time + timedelta(seconds=30)
        diff = abs((monitor.deadline - expected_deadline).total_seconds())
        assert diff < 0.1  # Within 100ms

    def test_get_elapsed_seconds(self) -> None:
        """Test getting elapsed seconds."""
        contract = Contract(
            id="test-elapsed",
            name="test-elapsed",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)

        # Before start, should return 0
        assert monitor.get_elapsed_seconds() == 0.0

        # After start, should return actual elapsed time
        monitor.start()
        time.sleep(0.1)
        elapsed = monitor.get_elapsed_seconds()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Shouldn't take that long

    def test_get_remaining_seconds(self) -> None:
        """Test getting remaining seconds until deadline."""
        contract = Contract(
            id="test-remaining",
            name="test-remaining",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)

        # Before start, should return None
        assert monitor.get_remaining_seconds() is None

        # After start, should return remaining time
        monitor.start()
        time.sleep(0.1)
        remaining = monitor.get_remaining_seconds()
        assert remaining is not None
        assert 9.0 < remaining < 10.0  # Should have ~9.9 seconds left

    def test_get_remaining_seconds_without_deadline(self) -> None:
        """Test get_remaining_seconds when no deadline set."""
        contract = Contract(
            id="test-no-deadline",
            name="test-no-deadline",
            resources=ResourceConstraints(tokens=1000),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.get_remaining_seconds() is None

    def test_get_time_pressure_without_duration(self) -> None:
        """Test time pressure when no max_duration set."""
        contract = Contract(
            id="test-no-pressure",
            name="test-no-pressure",
            resources=ResourceConstraints(tokens=1000),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.get_time_pressure() == 0.0

    def test_get_time_pressure_at_start(self) -> None:
        """Test time pressure immediately after start."""
        contract = Contract(
            id="test-pressure-start",
            name="test-pressure-start",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        pressure = monitor.get_time_pressure()
        assert pressure >= 0.0
        assert pressure < 0.1  # Should be very low at start

    def test_get_time_pressure_increases_over_time(self) -> None:
        """Test that time pressure increases as time passes."""
        contract = Contract(
            id="test-pressure-increase",
            name="test-pressure-increase",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=1)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        pressure1 = monitor.get_time_pressure()
        time.sleep(0.3)
        pressure2 = monitor.get_time_pressure()
        time.sleep(0.3)
        pressure3 = monitor.get_time_pressure()

        # Pressure should increase
        assert pressure1 < pressure2 < pressure3
        assert 0.0 <= pressure1 <= 1.0
        assert 0.0 <= pressure2 <= 1.0
        assert 0.0 <= pressure3 <= 1.0

    def test_get_time_pressure_clamped_at_one(self) -> None:
        """Test that time pressure is clamped to 1.0."""
        contract = Contract(
            id="test-pressure-clamp",
            name="test-pressure-clamp",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(milliseconds=100)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()
        time.sleep(0.2)  # Exceed the duration

        pressure = monitor.get_time_pressure()
        assert pressure == 1.0  # Should be clamped

    def test_is_past_deadline_false_initially(self) -> None:
        """Test that deadline is not past initially."""
        contract = Contract(
            id="test-deadline-false",
            name="test-deadline-false",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.is_past_deadline() is False

    def test_is_past_deadline_true_after_duration(self) -> None:
        """Test that deadline is past after duration exceeds."""
        contract = Contract(
            id="test-deadline-true",
            name="test-deadline-true",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(milliseconds=100)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()
        time.sleep(0.2)  # Exceed the duration

        assert monitor.is_past_deadline() is True

    def test_is_past_deadline_without_deadline(self) -> None:
        """Test is_past_deadline when no deadline set."""
        contract = Contract(
            id="test-no-deadline-check",
            name="test-no-deadline-check",
            resources=ResourceConstraints(tokens=1000),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.is_past_deadline() is False

    def test_is_over_duration_false_initially(self) -> None:
        """Test that duration is not exceeded initially."""
        contract = Contract(
            id="test-duration-false",
            name="test-duration-false",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.is_over_duration() is False

    def test_is_over_duration_true_after_exceeding(self) -> None:
        """Test that duration is exceeded after time passes."""
        contract = Contract(
            id="test-duration-true",
            name="test-duration-true",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(milliseconds=100)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()
        time.sleep(0.2)  # Exceed the duration

        assert monitor.is_over_duration() is True

    def test_is_over_duration_without_max_duration(self) -> None:
        """Test is_over_duration when no max_duration set."""
        contract = Contract(
            id="test-no-duration",
            name="test-no-duration",
            resources=ResourceConstraints(tokens=1000),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        assert monitor.is_over_duration() is False

    def test_is_over_duration_before_start(self) -> None:
        """Test is_over_duration before monitoring starts."""
        contract = Contract(
            id="test-before-start",
            name="test-before-start",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)

        assert monitor.is_over_duration() is False

    def test_repr_before_start(self) -> None:
        """Test string representation before start."""
        contract = Contract(
            id="test-repr-before",
            name="test-repr-before",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)

        repr_str = repr(monitor)
        assert "not started" in repr_str

    def test_repr_after_start_ok(self) -> None:
        """Test string representation after start with OK status."""
        contract = Contract(
            id="test-repr-ok",
            name="test-repr-ok",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=10)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        repr_str = repr(monitor)
        assert "elapsed=" in repr_str
        assert "status=OK" in repr_str

    def test_repr_after_exceeding_duration(self) -> None:
        """Test string representation after exceeding duration."""
        contract = Contract(
            id="test-repr-exceeded",
            name="test-repr-exceeded",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(milliseconds=100)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()
        time.sleep(0.2)

        repr_str = repr(monitor)
        assert "elapsed=" in repr_str
        assert "status=EXCEEDED" in repr_str


class TestTemporalMonitorIntegration:
    """Integration tests for TemporalMonitor."""

    def test_realistic_deadline_scenario(self) -> None:
        """Test realistic scenario with deadline tracking."""
        # 5-second deadline
        contract = Contract(
            id="realistic-deadline",
            name="realistic-deadline",
            resources=ResourceConstraints(tokens=10000),
            temporal=TemporalConstraints(
                max_duration=timedelta(seconds=5), deadline_type=DeadlineType.SOFT
            ),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        # Simulate work
        time.sleep(0.1)

        # Check status
        elapsed = monitor.get_elapsed_seconds()
        remaining = monitor.get_remaining_seconds()
        pressure = monitor.get_time_pressure()

        assert elapsed >= 0.1
        assert remaining >= 4.8  # Should have ~4.9 seconds left
        assert 0.01 < pressure < 0.05  # Low pressure
        assert not monitor.is_past_deadline()
        assert not monitor.is_over_duration()

    def test_deadline_exceeded_scenario(self) -> None:
        """Test scenario where deadline is exceeded."""
        # Very short deadline
        contract = Contract(
            id="exceeded-deadline",
            name="exceeded-deadline",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(milliseconds=50)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()
        time.sleep(0.1)  # Exceed deadline

        assert monitor.is_past_deadline()
        assert monitor.is_over_duration()
        assert monitor.get_time_pressure() == 1.0

    def test_monitor_with_absolute_and_relative_deadline(self) -> None:
        """Test when contract has both absolute deadline and max_duration."""
        # Absolute deadline in 10 seconds
        absolute = datetime.now() + timedelta(seconds=10)

        contract = Contract(
            id="both-deadlines",
            name="both-deadlines",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(
                deadline=absolute,
                max_duration=timedelta(seconds=5),  # Shorter duration
            ),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        # max_duration should take precedence for duration checks
        assert monitor.max_duration == 5.0
        # But absolute deadline is also set
        assert monitor.deadline == absolute

    def test_time_pressure_ranges(self) -> None:
        """Test time pressure in different ranges."""
        contract = Contract(
            id="pressure-ranges",
            name="pressure-ranges",
            resources=ResourceConstraints(tokens=1000),
            temporal=TemporalConstraints(max_duration=timedelta(seconds=1)),
        )

        monitor = TemporalMonitor(contract)
        monitor.start()

        # Initial: Low pressure (0.0-0.3)
        time.sleep(0.1)
        pressure1 = monitor.get_time_pressure()
        assert 0.0 <= pressure1 <= 0.3

        # Mid: Moderate pressure (0.3-0.7)
        time.sleep(0.3)
        pressure2 = monitor.get_time_pressure()
        assert 0.3 <= pressure2 <= 0.7

        # Late: High pressure (0.7-1.0)
        time.sleep(0.4)
        pressure3 = monitor.get_time_pressure()
        assert 0.7 <= pressure3 <= 1.0
