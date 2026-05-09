"""Tests for contract delegation with conservation laws.

Tests cover:
- Basic subcontract creation
- Conservation law enforcement
- Budget tracking and remaining calculations
- Multiple allocations
- Allocation release (budget pooling)
- Edge cases and error handling
"""

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import (
    ConservationViolationError,
    ContractingCapability,
)
from agent_contracts.core.monitor import ResourceMonitor


class TestContractingCapabilityBasic:
    """Basic functionality tests."""

    def test_create_capability_with_contract(self):
        """Test creating a contracting capability with a parent contract."""
        parent = Contract(
            id="parent",
            name="Parent Agent",
            resources=ResourceConstraints(tokens=100_000),
        )

        capability = ContractingCapability(parent)

        assert capability.parent_contract == parent
        assert capability.parent_budget_tokens == 100_000
        assert capability.remaining_tokens == 100_000
        assert len(capability.allocations) == 0

    def test_create_capability_with_reserve(self):
        """Test creating capability with reserve ratio."""
        parent = Contract(
            id="parent",
            name="Parent Agent",
            resources=ResourceConstraints(tokens=100_000),
        )

        capability = ContractingCapability(parent, reserve_ratio=0.1)

        assert capability.reserved_tokens == 10_000
        assert capability.remaining_tokens == 90_000  # 100K - 10K reserve

    def test_create_capability_invalid_reserve_ratio(self):
        """Test that invalid reserve ratios are rejected."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )

        with pytest.raises(ValueError, match="reserve_ratio must be between"):
            ContractingCapability(parent, reserve_ratio=0.6)

        with pytest.raises(ValueError, match="reserve_ratio must be between"):
            ContractingCapability(parent, reserve_ratio=-0.1)


class TestSubcontractCreation:
    """Tests for creating subcontracts."""

    def test_create_simple_subcontract(self):
        """Test creating a simple subcontract."""
        parent = Contract(
            id="orchestrator",
            name="Orchestrator",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        child = capability.create_subcontract(
            name="researcher",
            tokens=40_000,
            description="Research the topic",
        )

        assert child.id == "orchestrator/researcher"
        assert child.name == "researcher"
        assert child.resources.tokens == 40_000
        assert child.description == "Research the topic"
        assert child.metadata["parent_id"] == "orchestrator"

    def test_create_subcontract_with_cost(self):
        """Test creating subcontract with cost budget."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=5.0),
        )
        capability = ContractingCapability(parent)

        child = capability.create_subcontract(
            name="worker",
            tokens=50_000,
            cost_usd=2.0,
        )

        assert child.resources.tokens == 50_000
        assert child.resources.cost_usd == 2.0
        assert capability.remaining_tokens == 50_000
        assert capability.remaining_cost == 3.0

    def test_create_multiple_subcontracts(self):
        """Test creating multiple subcontracts (paper example)."""
        parent = Contract(
            id="orchestrator",
            name="Report Generation",
            resources=ResourceConstraints(tokens=150_000),
        )
        capability = ContractingCapability(parent)

        # Allocate as per paper Section 8
        capability.create_subcontract(
            name="orchestrator_reserve",
            tokens=15_000,
        )
        capability.create_subcontract(
            name="researcher",
            tokens=50_000,
        )
        capability.create_subcontract(
            name="analyzer",
            tokens=40_000,
        )
        capability.create_subcontract(
            name="reporter",
            tokens=45_000,
        )

        # Verify all contracts created
        assert len(capability.allocations) == 4
        assert capability.remaining_tokens == 0  # 150K - 15K - 50K - 40K - 45K = 0

        # Verify conservation: sum of children = parent budget
        total_allocated = sum(a.tokens_allocated for a in capability.allocations)
        assert total_allocated == 150_000

    def test_create_subcontract_empty_name_rejected(self):
        """Test that empty names are rejected."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        with pytest.raises(ValueError, match="cannot be empty"):
            capability.create_subcontract(name="", tokens=10_000)

    def test_create_subcontract_duplicate_name_rejected(self):
        """Test that duplicate names are rejected."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        capability.create_subcontract(name="worker", tokens=10_000)

        with pytest.raises(ValueError, match="already exists"):
            capability.create_subcontract(name="worker", tokens=10_000)


class TestConservationLaw:
    """Tests for conservation law enforcement."""

    def test_conservation_violation_tokens(self):
        """Test that allocating more tokens than available raises error."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        # First allocation succeeds
        capability.create_subcontract(name="first", tokens=60_000)

        # Second allocation exceeds remaining (40K)
        with pytest.raises(ConservationViolationError) as exc_info:
            capability.create_subcontract(name="second", tokens=50_000)

        assert exc_info.value.requested == 50_000
        assert exc_info.value.available == 40_000
        assert exc_info.value.parent_id == "parent"

    def test_conservation_violation_cost(self):
        """Test that allocating more cost than available raises error."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=1.0),
        )
        capability = ContractingCapability(parent)

        capability.create_subcontract(name="first", tokens=10_000, cost_usd=0.7)

        with pytest.raises(ConservationViolationError, match="Cannot allocate"):
            capability.create_subcontract(name="second", tokens=10_000, cost_usd=0.5)

    def test_conservation_with_parent_usage(self):
        """Test conservation accounts for parent's own usage."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        monitor = ResourceMonitor(parent.resources)

        # Parent uses 30K tokens itself
        monitor.usage.add_tokens(30_000)

        capability = ContractingCapability(parent, parent_monitor=monitor)

        # Only 70K remaining for delegation
        assert capability.remaining_tokens == 70_000

        # This should work (60K < 70K)
        capability.create_subcontract(name="worker", tokens=60_000)

        # Now only 10K remaining
        assert capability.remaining_tokens == 10_000

    def test_conservation_with_reserve(self):
        """Test conservation accounts for reserve."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )

        capability = ContractingCapability(parent, reserve_ratio=0.2)

        # Only 80K available (100K - 20K reserve)
        assert capability.remaining_tokens == 80_000

        # This should fail (90K > 80K available)
        with pytest.raises(ConservationViolationError):
            capability.create_subcontract(name="worker", tokens=90_000)

    def test_can_allocate_check(self):
        """Test can_allocate returns correct boolean."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=1.0),
        )
        capability = ContractingCapability(parent)

        assert capability.can_allocate(tokens=50_000) is True
        assert capability.can_allocate(tokens=150_000) is False
        assert capability.can_allocate(cost_usd=0.5) is True
        assert capability.can_allocate(cost_usd=1.5) is False
        assert capability.can_allocate(tokens=50_000, cost_usd=0.5) is True
        assert capability.can_allocate(tokens=50_000, cost_usd=1.5) is False

    def test_exact_budget_allocation(self):
        """Test allocating exactly the remaining budget."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        # Allocate exactly 100K - should succeed
        child = capability.create_subcontract(name="worker", tokens=100_000)

        assert child.resources.tokens == 100_000
        assert capability.remaining_tokens == 0


class TestBudgetTracking:
    """Tests for budget tracking and remaining calculations."""

    def test_remaining_budget_updates(self):
        """Test remaining budget updates after allocations."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=5.0),
        )
        capability = ContractingCapability(parent)

        assert capability.remaining_budget == {"tokens": 100_000, "cost_usd": 5.0}

        capability.create_subcontract(name="a", tokens=30_000, cost_usd=1.5)
        assert capability.remaining_budget == {"tokens": 70_000, "cost_usd": 3.5}

        capability.create_subcontract(name="b", tokens=20_000, cost_usd=1.0)
        assert capability.remaining_budget == {"tokens": 50_000, "cost_usd": 2.5}

    def test_get_allocation_by_name(self):
        """Test retrieving allocation by name."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        capability.create_subcontract(name="worker", tokens=50_000)

        allocation = capability.get_allocation("worker")
        assert allocation is not None
        assert allocation.tokens_allocated == 50_000

        missing = capability.get_allocation("nonexistent")
        assert missing is None

    def test_get_child_contract_by_name(self):
        """Test retrieving child contract by name."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        created = capability.create_subcontract(name="worker", tokens=50_000)

        retrieved = capability.get_child_contract("worker")
        assert retrieved is created

        missing = capability.get_child_contract("nonexistent")
        assert missing is None

    def test_child_contracts_list(self):
        """Test listing all child contracts."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        c1 = capability.create_subcontract(name="a", tokens=30_000)
        c2 = capability.create_subcontract(name="b", tokens=30_000)

        children = capability.child_contracts
        assert len(children) == 2
        assert c1 in children
        assert c2 in children


class TestAllocationRelease:
    """Tests for releasing allocations (budget pooling)."""

    def test_release_allocation(self):
        """Test releasing an allocation returns budget to pool."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        capability.create_subcontract(name="worker", tokens=60_000)
        assert capability.remaining_tokens == 40_000

        released = capability.release_allocation("worker")
        assert released == 60_000
        assert capability.remaining_tokens == 100_000
        assert len(capability.allocations) == 0

    def test_release_nonexistent_allocation(self):
        """Test releasing nonexistent allocation raises error."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        with pytest.raises(KeyError, match="No allocation found"):
            capability.release_allocation("nonexistent")

    def test_budget_pooling_scenario(self):
        """Test budget pooling: efficient agent subsidizes struggling one."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        # Allocate to two workers
        capability.create_subcontract(name="worker_a", tokens=50_000)
        capability.create_subcontract(name="worker_b", tokens=40_000)
        assert capability.remaining_tokens == 10_000

        # Worker A finishes early, release its allocation
        capability.release_allocation("worker_a")
        assert capability.remaining_tokens == 60_000  # 10K + 50K returned

        # Can now allocate more to worker C
        capability.create_subcontract(name="worker_c", tokens=55_000)
        assert capability.remaining_tokens == 5_000


class TestDelegationSummary:
    """Tests for delegation summary."""

    def test_get_summary_empty(self):
        """Test summary with no allocations."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=5.0),
        )
        capability = ContractingCapability(parent)

        summary = capability.get_summary()

        assert summary.parent_id == "parent"
        assert summary.parent_budget_tokens == 100_000
        assert summary.parent_budget_cost == 5.0
        assert summary.parent_used_tokens == 0
        assert summary.total_allocated_tokens == 0
        assert summary.remaining_tokens == 100_000
        assert len(summary.allocations) == 0
        assert summary.conservation_satisfied is True

    def test_get_summary_with_allocations(self):
        """Test summary with allocations."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        monitor = ResourceMonitor(parent.resources)
        monitor.usage.add_tokens(10_000)  # Parent used 10K

        capability = ContractingCapability(parent, parent_monitor=monitor)
        capability.create_subcontract(name="a", tokens=30_000)
        capability.create_subcontract(name="b", tokens=20_000)

        summary = capability.get_summary()

        assert summary.parent_used_tokens == 10_000
        assert summary.total_allocated_tokens == 50_000
        assert summary.remaining_tokens == 40_000  # 100K - 10K - 50K
        assert len(summary.allocations) == 2
        assert summary.conservation_satisfied is True

    def test_conservation_satisfied_check(self):
        """Test that conservation_satisfied correctly identifies violations."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        capability.create_subcontract(name="worker", tokens=50_000)

        summary = capability.get_summary()
        assert summary.conservation_satisfied is True

        # Conservation: used (0) + allocated (50K) = 50K ≤ 100K budget


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        parent = Contract(
            id="orchestrator",
            name="Orchestrator",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)
        capability.create_subcontract(name="worker", tokens=40_000)

        repr_str = repr(capability)

        assert "orchestrator" in repr_str
        assert "100,000" in repr_str or "100000" in repr_str
        assert "children=1" in repr_str


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_budget_parent(self):
        """Test with parent having zero budget."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=0),
        )
        capability = ContractingCapability(parent)

        assert capability.remaining_tokens == 0

        with pytest.raises(ConservationViolationError):
            capability.create_subcontract(name="worker", tokens=1)

    def test_subcontract_with_zero_tokens(self):
        """Test creating subcontract with zero tokens (cost-only)."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000, cost_usd=5.0),
        )
        capability = ContractingCapability(parent)

        child = capability.create_subcontract(
            name="worker",
            tokens=0,
            cost_usd=1.0,
        )

        assert child.resources.tokens is None
        assert child.resources.cost_usd == 1.0

    def test_metadata_preserved(self):
        """Test that custom metadata is preserved in child."""
        parent = Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(tokens=100_000),
        )
        capability = ContractingCapability(parent)

        child = capability.create_subcontract(
            name="worker",
            tokens=50_000,
            metadata={"custom_key": "custom_value"},
        )

        assert child.metadata["custom_key"] == "custom_value"
        assert child.metadata["parent_id"] == "parent"
        assert "delegation_time" in child.metadata


# ---------------------------------------------------------------------------
# Per-tool conservation (added with chamber pillar M3c support).
#
# Mirror the token / cost conservation classes above but for the
# `per_tool_limits` axis. The framework should:
#   - Enforce parent_used + Σ child_allocations ≤ parent_limit per tool
#   - Treat tools the parent doesn't constrain as unbounded for children
#   - Record each child's per-tool allocation in AllocationRecord
#   - Sum constrained-tool allocations into total_allocated_per_tool
#   - Surface conservation_satisfied=False when any tool is over-allocated
# ---------------------------------------------------------------------------


class TestPerToolConservation:
    """Per-tool conservation enforced by `create_subcontract`."""

    def _parent(self, intervene_limit: int = 10) -> Contract:
        return Contract(
            id="parent",
            name="Parent",
            resources=ResourceConstraints(per_tool_limits={"intervene": intervene_limit}),
        )

    def test_within_budget_succeeds(self):
        """A=2, B=3, k=10 — well under budget, both allocations succeed."""
        cap = ContractingCapability(self._parent(intervene_limit=10))
        cap.create_subcontract(name="planner", per_tool_limits={"intervene": 2})
        cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 3})
        assert cap.total_allocated_per_tool == {"intervene": 5}

    def test_exact_match_succeeds(self):
        """A + B == k — boundary case, must be allowed."""
        cap = ContractingCapability(self._parent(intervene_limit=4))
        cap.create_subcontract(name="planner", per_tool_limits={"intervene": 2})
        cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 2})
        assert cap.remaining_per_tool("intervene") == 0

    def test_oversubscription_raises(self):
        """A + B > k — second create_subcontract must raise."""
        cap = ContractingCapability(self._parent(intervene_limit=2))
        cap.create_subcontract(name="planner", per_tool_limits={"intervene": 2})
        with pytest.raises(ConservationViolationError) as exc:
            cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 1})

        # The error names the offending tool and the actual numbers so
        # callers can debug allocation issues without reading the
        # capability's internal state.
        msg = str(exc.value)
        assert "intervene" in msg
        assert "reasoner" in msg
        assert exc.value.requested == 1
        assert exc.value.available == 0
        assert exc.value.parent_id == "parent"

    def test_first_oversubscribed_raises(self):
        """Single child requesting more than parent's full limit raises immediately."""
        cap = ContractingCapability(self._parent(intervene_limit=3))
        with pytest.raises(ConservationViolationError, match="intervene"):
            cap.create_subcontract(name="bigchild", per_tool_limits={"intervene": 5})
        # No partial state — capability stays empty.
        assert cap.allocations == []
        assert cap.total_allocated_per_tool == {}

    def test_unconstrained_tool_is_unbounded(self):
        """If parent has no constraint on a tool, child requests are allowed."""
        # Parent constrains only `intervene`; child requests `observe`.
        cap = ContractingCapability(self._parent(intervene_limit=2))
        # Massive request on an unconstrained tool — should not raise.
        cap.create_subcontract(name="watcher", per_tool_limits={"observe": 9999})
        # And it's recorded in the allocation, but NOT in the
        # constrained-tools running total.
        record = cap.get_allocation("watcher")
        assert record is not None
        assert record.per_tool_limits_allocated == {"observe": 9999}
        assert "observe" not in cap.total_allocated_per_tool

    def test_zero_request_does_not_consume(self):
        """A zero per-tool request shouldn't bump allocated totals."""
        cap = ContractingCapability(self._parent(intervene_limit=2))
        cap.create_subcontract(name="zero", per_tool_limits={"intervene": 0})
        assert cap.total_allocated_per_tool == {}
        assert cap.remaining_per_tool("intervene") == 2

    def test_parent_used_counts_against_remaining(self):
        """Parent's own per-tool spend reduces what's available to children."""
        parent = self._parent(intervene_limit=5)
        monitor = ResourceMonitor(parent.resources)
        # Parent burned 2 itself.
        monitor.usage.add_tool_invocation("intervene")
        monitor.usage.add_tool_invocation("intervene")

        cap = ContractingCapability(parent, parent_monitor=monitor)
        # Only 3 left for children; requesting 4 must raise.
        with pytest.raises(ConservationViolationError, match="intervene") as exc:
            cap.create_subcontract(name="child", per_tool_limits={"intervene": 4})
        assert exc.value.available == 3

    def test_release_allocation_returns_per_tool_to_pool(self):
        """release_allocation must restore per-tool budget for re-allocation."""
        cap = ContractingCapability(self._parent(intervene_limit=4))
        cap.create_subcontract(name="planner", per_tool_limits={"intervene": 3})
        # No room for B=2 → 3+2=5 > 4.
        with pytest.raises(ConservationViolationError):
            cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 2})
        # Release planner; now B=2 fits.
        cap.release_allocation("planner")
        assert cap.total_allocated_per_tool == {}
        cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 2})
        assert cap.total_allocated_per_tool == {"intervene": 2}

    def test_can_allocate_checks_per_tool(self):
        """can_allocate's new per_tool_limits parameter mirrors the create check."""
        cap = ContractingCapability(self._parent(intervene_limit=3))
        cap.create_subcontract(name="a", per_tool_limits={"intervene": 2})

        assert cap.can_allocate(per_tool_limits={"intervene": 1}) is True
        assert cap.can_allocate(per_tool_limits={"intervene": 2}) is False
        # Unconstrained tools are always allowed.
        assert cap.can_allocate(per_tool_limits={"observe": 1000}) is True

    def test_summary_reports_total_allocated_per_tool(self):
        cap = ContractingCapability(self._parent(intervene_limit=10))
        cap.create_subcontract(name="planner", per_tool_limits={"intervene": 2})
        cap.create_subcontract(name="reasoner", per_tool_limits={"intervene": 3})
        summary = cap.get_summary()
        assert summary.total_allocated_per_tool == {"intervene": 5}
        assert summary.conservation_satisfied is True

    def test_remaining_per_tool_returns_none_for_unconstrained(self):
        cap = ContractingCapability(self._parent(intervene_limit=3))
        assert cap.remaining_per_tool("observe") is None
        assert cap.remaining_per_tool("intervene") == 3

    def test_reserve_ratio_floors_per_tool(self):
        """reserve_ratio applies to per-tool limits too (floor-rounded)."""
        cap = ContractingCapability(self._parent(intervene_limit=10), reserve_ratio=0.3)
        # int(10 * 0.3) = 3 reserved → 7 available for children.
        assert cap.reserved_per_tool("intervene") == 3
        assert cap.remaining_per_tool("intervene") == 7
        # Confirm the conservation check uses the reserved amount.
        with pytest.raises(ConservationViolationError):
            cap.create_subcontract(name="greedy", per_tool_limits={"intervene": 8})
        cap.create_subcontract(name="ok", per_tool_limits={"intervene": 7})

    def test_check_conservation_detects_per_tool_overrun(self):
        """If parent burns past budget AFTER allocation, conservation_satisfied flips."""
        parent = self._parent(intervene_limit=5)
        monitor = ResourceMonitor(parent.resources)
        cap = ContractingCapability(parent, parent_monitor=monitor)
        cap.create_subcontract(name="child", per_tool_limits={"intervene": 3})
        # Parent then burns 3 itself; 3 (allocated) + 3 (used) = 6 > 5.
        for _ in range(3):
            monitor.usage.add_tool_invocation("intervene")
        summary = cap.get_summary()
        assert summary.conservation_satisfied is False
