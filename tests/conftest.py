"""Shared test fixtures for agent_contracts tests."""

import pytest

from agent_contracts import (
    Contract,
    ContractEnforcer,
    ResourceConstraints,
    ResourceMonitor,
)


@pytest.fixture
def basic_constraints() -> ResourceConstraints:
    """Standard resource constraints for testing."""
    return ResourceConstraints(tokens=1000, api_calls=10)


@pytest.fixture
def basic_contract(basic_constraints: ResourceConstraints) -> Contract:
    """Standard contract for testing."""
    return Contract(
        id="test-contract",
        name="Test Contract",
        resources=basic_constraints,
    )


@pytest.fixture
def strict_contract() -> Contract:
    """Tight-budget contract for testing constraint violations."""
    return Contract(
        id="test-strict",
        name="Strict Contract",
        resources=ResourceConstraints(tokens=100, api_calls=2),
    )


@pytest.fixture
def basic_monitor(basic_constraints: ResourceConstraints) -> ResourceMonitor:
    """Standard resource monitor for testing."""
    return ResourceMonitor(basic_constraints)


@pytest.fixture
def basic_enforcer(basic_contract: Contract) -> ContractEnforcer:
    """Standard enforcer for testing."""
    return ContractEnforcer(contract=basic_contract, strict_mode=True)
