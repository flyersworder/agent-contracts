"""Unit tests for resource monitoring system."""

import time
from datetime import datetime

import pytest

from agent_contracts.core import ResourceConstraints, ResourceMonitor, ResourceUsage, ViolationInfo


class TestResourceUsage:
    """Tests for ResourceUsage tracking."""

    def test_create_empty_usage(self) -> None:
        """Test creating usage tracker with defaults."""
        usage = ResourceUsage()
        assert usage.tokens == 0
        assert usage.api_calls == 0
        assert usage.web_searches == 0
        assert usage.tool_invocations == 0
        assert usage.memory_mb == 0.0
        assert usage.compute_seconds == 0.0
        assert usage.cost_usd == 0.0

    def test_create_with_initial_values(self) -> None:
        """Test creating usage with initial values."""
        usage = ResourceUsage(tokens=100, api_calls=5, cost_usd=0.5)
        assert usage.tokens == 100
        assert usage.api_calls == 5
        assert usage.cost_usd == 0.5

    def test_negative_values_raise_error(self) -> None:
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            ResourceUsage(tokens=-100)

        with pytest.raises(ValueError, match="cost_usd must be non-negative"):
            ResourceUsage(cost_usd=-1.0)

    def test_add_tokens(self) -> None:
        """Test adding tokens."""
        usage = ResourceUsage()
        usage.add_tokens(100)
        assert usage.tokens == 100

        usage.add_tokens(50)
        assert usage.tokens == 150

    def test_add_tokens_negative_raises_error(self) -> None:
        """Test that adding negative tokens raises error."""
        usage = ResourceUsage()
        with pytest.raises(ValueError, match="Token count must be non-negative"):
            usage.add_tokens(-10)

    def test_add_api_call(self) -> None:
        """Test recording API calls."""
        usage = ResourceUsage()
        usage.add_api_call(cost=0.01, tokens=500)

        assert usage.api_calls == 1
        assert usage.cost_usd == 0.01
        assert usage.tokens == 500

    def test_add_api_call_accumulates(self) -> None:
        """Test that multiple API calls accumulate."""
        usage = ResourceUsage()
        usage.add_api_call(cost=0.01, tokens=100)
        usage.add_api_call(cost=0.02, tokens=200)

        assert usage.api_calls == 2
        assert usage.cost_usd == 0.03
        assert usage.tokens == 300

    def test_add_api_call_negative_cost_raises_error(self) -> None:
        """Test that negative cost raises error."""
        usage = ResourceUsage()
        with pytest.raises(ValueError, match="Cost must be non-negative"):
            usage.add_api_call(cost=-0.01)

    def test_add_web_search(self) -> None:
        """Test recording web searches."""
        usage = ResourceUsage()
        usage.add_web_search()
        assert usage.web_searches == 1

        usage.add_web_search()
        assert usage.web_searches == 2

    def test_add_tool_invocation(self) -> None:
        """Test recording tool invocations."""
        usage = ResourceUsage()
        usage.add_tool_invocation()
        assert usage.tool_invocations == 1

        usage.add_tool_invocation()
        assert usage.tool_invocations == 2

    def test_update_memory_tracks_peak(self) -> None:
        """Test that memory tracking keeps the peak value."""
        usage = ResourceUsage()
        usage.update_memory(100.0)
        assert usage.memory_mb == 100.0

        usage.update_memory(150.0)
        assert usage.memory_mb == 150.0

        # Lower value doesn't reduce peak
        usage.update_memory(120.0)
        assert usage.memory_mb == 150.0

    def test_update_memory_negative_raises_error(self) -> None:
        """Test that negative memory raises error."""
        usage = ResourceUsage()
        with pytest.raises(ValueError, match="Memory must be non-negative"):
            usage.update_memory(-10.0)

    def test_add_compute_time(self) -> None:
        """Test adding compute time."""
        usage = ResourceUsage()
        usage.add_compute_time(1.5)
        assert usage.compute_seconds == 1.5

        usage.add_compute_time(2.5)
        assert usage.compute_seconds == 4.0

    def test_add_compute_time_negative_raises_error(self) -> None:
        """Test that negative compute time raises error."""
        usage = ResourceUsage()
        with pytest.raises(ValueError, match="Compute time must be non-negative"):
            usage.add_compute_time(-1.0)

    def test_add_cost(self) -> None:
        """Test adding cost."""
        usage = ResourceUsage()
        usage.add_cost(0.5)
        assert usage.cost_usd == 0.5

        usage.add_cost(0.3)
        assert usage.cost_usd == 0.8

    def test_add_cost_negative_raises_error(self) -> None:
        """Test that negative cost raises error."""
        usage = ResourceUsage()
        with pytest.raises(ValueError, match="Cost must be non-negative"):
            usage.add_cost(-0.5)

    def test_elapsed_time(self) -> None:
        """Test elapsed time calculation."""
        usage = ResourceUsage()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = usage.elapsed_time()
        assert elapsed.total_seconds() >= 0.01

    def test_to_dict(self) -> None:
        """Test converting usage to dictionary."""
        usage = ResourceUsage(tokens=100, api_calls=5, cost_usd=0.5)
        usage_dict = usage.to_dict()

        assert usage_dict["tokens"] == 100
        assert usage_dict["api_calls"] == 5
        assert usage_dict["cost_usd"] == 0.5
        assert "elapsed_seconds" in usage_dict
        assert "start_time" in usage_dict
        assert "last_updated" in usage_dict

    def test_repr(self) -> None:
        """Test string representation."""
        usage = ResourceUsage(tokens=100, api_calls=5, cost_usd=0.5)
        repr_str = repr(usage)
        assert "tokens=100" in repr_str
        assert "api_calls=5" in repr_str
        assert "cost_usd=0.5" in repr_str


class TestViolationInfo:
    """Tests for ViolationInfo."""

    def test_create_violation(self) -> None:
        """Test creating violation info."""
        violation = ViolationInfo(resource="tokens", limit=1000, actual=1500)
        assert violation.resource == "tokens"
        assert violation.limit == 1000
        assert violation.actual == 1500
        assert isinstance(violation.timestamp, datetime)

    def test_repr(self) -> None:
        """Test string representation."""
        violation = ViolationInfo(resource="tokens", limit=1000, actual=1500)
        repr_str = repr(violation)
        assert "tokens" in repr_str
        assert "1500" in repr_str
        assert "1000" in repr_str


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_create_monitor(self) -> None:
        """Test creating a resource monitor."""
        constraints = ResourceConstraints(tokens=1000, api_calls=10)
        monitor = ResourceMonitor(constraints)

        assert monitor.constraints == constraints
        assert monitor.usage.tokens == 0
        assert len(monitor.violations) == 0

    def test_check_constraints_no_violations(self) -> None:
        """Test checking constraints with no violations."""
        constraints = ResourceConstraints(tokens=1000, api_calls=10, cost_usd=1.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(500)
        monitor.usage.add_api_call(cost=0.5)

        violations = monitor.check_constraints()
        assert len(violations) == 0
        assert not monitor.is_violated()

    def test_check_constraints_token_violation(self) -> None:
        """Test detecting token constraint violation."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(1500)
        violations = monitor.check_constraints()

        assert len(violations) == 1
        assert violations[0].resource == "tokens"
        assert violations[0].limit == 1000
        assert violations[0].actual == 1500
        assert monitor.is_violated()

    def test_check_constraints_api_call_violation(self) -> None:
        """Test detecting API call constraint violation."""
        constraints = ResourceConstraints(api_calls=5)
        monitor = ResourceMonitor(constraints)

        for _ in range(6):
            monitor.usage.add_api_call()

        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "api_calls"

    def test_check_constraints_cost_violation(self) -> None:
        """Test detecting cost constraint violation."""
        constraints = ResourceConstraints(cost_usd=1.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_cost(1.5)
        violations = monitor.check_constraints()

        assert len(violations) == 1
        assert violations[0].resource == "cost_usd"

    def test_check_constraints_multiple_violations(self) -> None:
        """Test detecting multiple constraint violations."""
        constraints = ResourceConstraints(tokens=1000, api_calls=5, cost_usd=1.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(1500)
        for _ in range(6):
            monitor.usage.add_api_call(cost=0.3)

        violations = monitor.check_constraints()
        assert len(violations) == 3  # tokens, api_calls, and cost_usd

    def test_check_constraints_web_search_violation(self) -> None:
        """Test detecting web search constraint violation."""
        constraints = ResourceConstraints(web_searches=3)
        monitor = ResourceMonitor(constraints)

        for _ in range(4):
            monitor.usage.add_web_search()

        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "web_searches"

    def test_check_constraints_tool_invocation_violation(self) -> None:
        """Test detecting tool invocation constraint violation."""
        constraints = ResourceConstraints(tool_invocations=10)
        monitor = ResourceMonitor(constraints)

        for _ in range(11):
            monitor.usage.add_tool_invocation()

        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "tool_invocations"

    def test_check_constraints_memory_violation(self) -> None:
        """Test detecting memory constraint violation."""
        constraints = ResourceConstraints(memory_mb=500.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.update_memory(600.0)
        violations = monitor.check_constraints()

        assert len(violations) == 1
        assert violations[0].resource == "memory_mb"

    def test_check_constraints_compute_time_violation(self) -> None:
        """Test detecting compute time constraint violation."""
        constraints = ResourceConstraints(compute_seconds=10.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_compute_time(12.0)
        violations = monitor.check_constraints()

        assert len(violations) == 1
        assert violations[0].resource == "compute_seconds"

    def test_record_violation(self) -> None:
        """Test recording violations."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        violation = ViolationInfo(resource="tokens", limit=1000, actual=1500)
        monitor.record_violation(violation)

        assert len(monitor.violations) == 1
        assert monitor.violations[0] == violation

    def test_get_usage_percentage(self) -> None:
        """Test calculating usage percentages."""
        constraints = ResourceConstraints(tokens=1000, api_calls=10, cost_usd=1.0)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(500)
        monitor.usage.add_api_call(cost=0.25)

        percentages = monitor.get_usage_percentage()

        assert percentages["tokens"] == 50.0
        assert percentages["api_calls"] == 10.0
        assert percentages["cost_usd"] == 25.0

    def test_get_usage_percentage_over_100(self) -> None:
        """Test usage percentage can exceed 100%."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(1500)
        percentages = monitor.get_usage_percentage()

        assert percentages["tokens"] == 150.0

    def test_get_usage_percentage_excludes_unconstrained(self) -> None:
        """Test that unconstrained resources are excluded from percentages."""
        constraints = ResourceConstraints(tokens=1000)  # Only tokens constrained
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(500)
        monitor.usage.add_api_call()  # API calls not constrained

        percentages = monitor.get_usage_percentage()

        assert "tokens" in percentages
        assert "api_calls" not in percentages

    def test_get_usage_percentage_zero_constraint(self) -> None:
        """Test that zero constraints are excluded from percentages."""
        constraints = ResourceConstraints(tokens=0)  # Zero budget
        monitor = ResourceMonitor(constraints)

        percentages = monitor.get_usage_percentage()

        # Zero constraints are excluded to avoid division errors
        assert "tokens" not in percentages

    def test_reset(self) -> None:
        """Test resetting monitor state."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        monitor.usage.add_tokens(500)
        violation = ViolationInfo(resource="tokens", limit=1000, actual=1500)
        monitor.record_violation(violation)

        monitor.reset()

        assert monitor.usage.tokens == 0
        assert len(monitor.violations) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        constraints = ResourceConstraints(tokens=1000)
        monitor = ResourceMonitor(constraints)

        repr_str = repr(monitor)
        assert "ResourceMonitor" in repr_str
        assert "OK" in repr_str

        monitor.usage.add_tokens(1500)
        repr_str = repr(monitor)
        assert "VIOLATED" in repr_str

    def test_integration_realistic_scenario(self) -> None:
        """Test realistic usage scenario with multiple resource types."""
        # Create contract with realistic limits
        constraints = ResourceConstraints(
            tokens=10000, api_calls=50, web_searches=5, cost_usd=5.0, memory_mb=1024.0
        )
        monitor = ResourceMonitor(constraints)

        # Simulate agent execution
        for i in range(10):
            monitor.usage.add_api_call(cost=0.1, tokens=500)
            if i % 3 == 0:
                monitor.usage.add_web_search()
            monitor.usage.add_tool_invocation()
            monitor.usage.update_memory(256.0 + i * 10)

        # Check state
        assert monitor.usage.tokens == 5000  # 10 calls * 500 tokens
        assert monitor.usage.api_calls == 10
        assert monitor.usage.web_searches == 4  # floor(10/3) + 1
        assert monitor.usage.tool_invocations == 10
        assert abs(monitor.usage.cost_usd - 1.0) < 0.0001  # Floating point comparison
        assert monitor.usage.memory_mb == 346.0  # Peak memory

        # Should not be violated
        assert not monitor.is_violated()

        # Get usage percentages
        percentages = monitor.get_usage_percentage()
        assert percentages["tokens"] == 50.0
        assert percentages["api_calls"] == 20.0
        assert percentages["cost_usd"] == 20.0


class TestPerToolLimits:
    """Tests for per-tool limit functionality."""

    def test_resource_constraints_with_per_tool_limits(self) -> None:
        """Test creating ResourceConstraints with per-tool limits."""
        constraints = ResourceConstraints(
            tool_invocations=20,
            per_tool_limits={
                "web_search": 5,
                "calculator": 10,
            },
        )
        assert constraints.per_tool_limits["web_search"] == 5
        assert constraints.per_tool_limits["calculator"] == 10
        assert constraints.tool_invocations == 20

    def test_per_tool_limits_validation_non_integer(self) -> None:
        """Test that non-integer per-tool limits raise ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            ResourceConstraints(
                per_tool_limits={"web_search": 5.5}  # type: ignore
            )

    def test_per_tool_limits_validation_negative(self) -> None:
        """Test that negative per-tool limits raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ResourceConstraints(per_tool_limits={"web_search": -1})

    def test_add_tool_invocation_with_name(self) -> None:
        """Test adding tool invocation with a specific tool name."""
        usage = ResourceUsage()
        usage.add_tool_invocation("web_search")
        usage.add_tool_invocation("web_search")
        usage.add_tool_invocation("calculator")

        assert usage.tool_invocations == 3
        assert usage.get_tool_usage("web_search") == 2
        assert usage.get_tool_usage("calculator") == 1
        assert usage.get_tool_usage("unknown_tool") == 0

    def test_add_tool_invocation_without_name(self) -> None:
        """Test adding tool invocation without a tool name (backward compat)."""
        usage = ResourceUsage()
        usage.add_tool_invocation()
        usage.add_tool_invocation()

        assert usage.tool_invocations == 2
        assert usage.tool_usage_by_name == {}  # No per-tool tracking

    def test_add_tool_invocation_mixed(self) -> None:
        """Test mixing named and unnamed tool invocations."""
        usage = ResourceUsage()
        usage.add_tool_invocation("web_search")
        usage.add_tool_invocation()  # No name
        usage.add_tool_invocation("calculator")

        assert usage.tool_invocations == 3
        assert usage.get_tool_usage("web_search") == 1
        assert usage.get_tool_usage("calculator") == 1

    def test_to_dict_includes_tool_usage_by_name(self) -> None:
        """Test that to_dict includes per-tool usage."""
        usage = ResourceUsage()
        usage.add_tool_invocation("web_search")
        usage.add_tool_invocation("calculator")

        usage_dict = usage.to_dict()
        assert "tool_usage_by_name" in usage_dict
        assert usage_dict["tool_usage_by_name"]["web_search"] == 1
        assert usage_dict["tool_usage_by_name"]["calculator"] == 1

    def test_check_constraints_per_tool_no_violation(self) -> None:
        """Test per-tool limit checking with no violations."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 5, "calculator": 10})
        monitor = ResourceMonitor(constraints)

        for _ in range(3):
            monitor.usage.add_tool_invocation("web_search")
        for _ in range(5):
            monitor.usage.add_tool_invocation("calculator")

        violations = monitor.check_constraints()
        assert len(violations) == 0

    def test_check_constraints_per_tool_violation(self) -> None:
        """Test detecting per-tool limit violation."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 3})
        monitor = ResourceMonitor(constraints)

        for _ in range(5):
            monitor.usage.add_tool_invocation("web_search")

        violations = monitor.check_constraints()
        assert len(violations) == 1
        assert violations[0].resource == "tool:web_search"
        assert violations[0].limit == 3
        assert violations[0].actual == 5

    def test_check_constraints_per_tool_and_aggregate_violation(self) -> None:
        """Test both per-tool and aggregate violations."""
        constraints = ResourceConstraints(tool_invocations=10, per_tool_limits={"web_search": 3})
        monitor = ResourceMonitor(constraints)

        # Use web_search 5 times (violates per-tool limit of 3)
        for _ in range(5):
            monitor.usage.add_tool_invocation("web_search")
        # Use calculator 8 times (total now 13, violates aggregate of 10)
        for _ in range(8):
            monitor.usage.add_tool_invocation("calculator")

        violations = monitor.check_constraints()
        violation_resources = {v.resource for v in violations}

        assert "tool:web_search" in violation_resources
        assert "tool_invocations" in violation_resources
        assert len(violations) == 2

    def test_get_remaining_tool_calls_with_limit(self) -> None:
        """Test getting remaining calls for a limited tool."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 5})
        monitor = ResourceMonitor(constraints)

        assert monitor.get_remaining_tool_calls("web_search") == 5

        monitor.usage.add_tool_invocation("web_search")
        monitor.usage.add_tool_invocation("web_search")

        assert monitor.get_remaining_tool_calls("web_search") == 3

    def test_get_remaining_tool_calls_without_limit(self) -> None:
        """Test getting remaining calls for an unlimited tool."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 5})
        monitor = ResourceMonitor(constraints)

        # calculator has no per-tool limit
        remaining = monitor.get_remaining_tool_calls("calculator")
        assert remaining == float("inf")

    def test_get_remaining_tool_calls_exhausted(self) -> None:
        """Test remaining calls when limit is exhausted."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 3})
        monitor = ResourceMonitor(constraints)

        for _ in range(5):
            monitor.usage.add_tool_invocation("web_search")

        assert monitor.get_remaining_tool_calls("web_search") == 0

    def test_can_use_tool_per_tool_limit(self) -> None:
        """Test can_use_tool with per-tool limits."""
        constraints = ResourceConstraints(per_tool_limits={"web_search": 2})
        monitor = ResourceMonitor(constraints)

        assert monitor.can_use_tool("web_search") is True
        assert monitor.can_use_tool("calculator") is True  # No limit

        monitor.usage.add_tool_invocation("web_search")
        assert monitor.can_use_tool("web_search") is True

        monitor.usage.add_tool_invocation("web_search")
        assert monitor.can_use_tool("web_search") is False  # Limit reached
        assert monitor.can_use_tool("calculator") is True  # Still OK

    def test_can_use_tool_aggregate_limit(self) -> None:
        """Test can_use_tool with aggregate limit only."""
        constraints = ResourceConstraints(tool_invocations=3)
        monitor = ResourceMonitor(constraints)

        assert monitor.can_use_tool("web_search") is True

        for _ in range(3):
            monitor.usage.add_tool_invocation("web_search")

        # Aggregate limit reached
        assert monitor.can_use_tool("web_search") is False
        assert monitor.can_use_tool("calculator") is False  # All tools blocked

    def test_can_use_tool_combined_limits(self) -> None:
        """Test can_use_tool with both per-tool and aggregate limits."""
        constraints = ResourceConstraints(tool_invocations=10, per_tool_limits={"web_search": 2})
        monitor = ResourceMonitor(constraints)

        # Use web_search until per-tool limit
        monitor.usage.add_tool_invocation("web_search")
        monitor.usage.add_tool_invocation("web_search")

        assert monitor.can_use_tool("web_search") is False  # Per-tool limit
        assert monitor.can_use_tool("calculator") is True  # Still OK

        # Use other tools until aggregate limit
        for _ in range(8):
            monitor.usage.add_tool_invocation("calculator")

        assert monitor.can_use_tool("calculator") is False  # Aggregate limit

    def test_per_tool_zero_limit(self) -> None:
        """Test per-tool limit of zero (tool disabled)."""
        constraints = ResourceConstraints(
            per_tool_limits={"web_search": 0}  # Tool is disabled
        )
        monitor = ResourceMonitor(constraints)

        assert monitor.can_use_tool("web_search") is False  # Immediately blocked
        assert monitor.get_remaining_tool_calls("web_search") == 0

    def test_realistic_multi_tool_scenario(self) -> None:
        """Test realistic scenario with multiple tools and limits."""
        constraints = ResourceConstraints(
            tool_invocations=20,
            per_tool_limits={
                "web_search": 5,
                "code_execution": 3,
                "file_read": 10,
            },
        )
        monitor = ResourceMonitor(constraints)

        # Simulate agent using tools
        for _ in range(4):
            monitor.usage.add_tool_invocation("web_search")
        for _ in range(2):
            monitor.usage.add_tool_invocation("code_execution")
        for _ in range(6):
            monitor.usage.add_tool_invocation("file_read")

        # No violations yet
        assert not monitor.is_violated()
        assert monitor.can_use_tool("web_search") is True  # 1 left
        assert monitor.can_use_tool("code_execution") is True  # 1 left
        assert monitor.can_use_tool("file_read") is True  # 4 left

        # Use one more of each
        monitor.usage.add_tool_invocation("web_search")
        monitor.usage.add_tool_invocation("code_execution")

        # Now web_search and code_execution are at limit
        assert monitor.can_use_tool("web_search") is False
        assert monitor.can_use_tool("code_execution") is False
        assert monitor.can_use_tool("file_read") is True

        # Verify violation detection
        monitor.usage.add_tool_invocation("web_search")  # Over limit
        violations = monitor.check_constraints()
        assert any(v.resource == "tool:web_search" for v in violations)
