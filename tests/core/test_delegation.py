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
