"""Regression tests for correctness bugs found in the 2026-05-20 framework review.

Each test pins one verified bug. They are grouped here (rather than scattered
across the per-module test files) so the review's findings stay traceable.
"""

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ContractingCapability
from agent_contracts.core.skillspec import SkillSpec
from agent_contracts.core.tokens import TokenCounter
from agent_contracts.core.wrapper import ContractAgent, ContractViolationError

# --- Bug 1: frozen ResourceConstraints was mutable via its per_tool_limits dict ---


def test_resource_constraints_per_tool_limits_is_immutable() -> None:
    """A frozen ResourceConstraints must not be mutable through per_tool_limits."""
    rc = ResourceConstraints(tool_invocations=20, per_tool_limits={"web_search": 5})
    with pytest.raises(TypeError):
        rc.per_tool_limits["web_search"] = 999  # type: ignore[index]


def test_resource_constraints_does_not_alias_caller_dict() -> None:
    """Mutating the dict passed to the constructor must not change the contract."""
    source = {"web_search": 5}
    rc = ResourceConstraints(per_tool_limits=source)
    source["web_search"] = 999
    assert rc.per_tool_limits["web_search"] == 5


# --- Bug 2: get_model_pricing returned the wrong (first) prefix for versioned ids ---


def test_get_model_pricing_versioned_id_uses_longest_prefix() -> None:
    """A versioned model id must resolve to the most specific known model."""
    versioned = TokenCounter.get_model_pricing("gpt-4o-2024-08-06")
    assert versioned == TokenCounter.get_model_pricing("gpt-4o")
    assert versioned != TokenCounter.get_model_pricing("gpt-4")


# --- Bug 3: SkillSpec name regex was off by one (rejected valid 64-char names) ---


def test_skillspec_accepts_64_character_name() -> None:
    """The agentskills.io limit is 1-64 chars; a 64-char name must be valid."""
    name64 = "a" + "b" * 62 + "c"
    assert len(name64) == 64
    assert SkillSpec(name=name64, description="x").name == name64


def test_skillspec_rejects_65_character_name() -> None:
    with pytest.raises(ValueError):
        SkillSpec(name="a" * 65, description="x")


# --- Bug 4: SkillSpec regex accepted consecutive hyphens its error message forbids ---


def test_skillspec_rejects_consecutive_hyphens() -> None:
    with pytest.raises(ValueError):
        SkillSpec(name="a--b", description="x")


def test_skillspec_accepts_single_hyphens() -> None:
    assert SkillSpec(name="a-b-c", description="x").name == "a-b-c"


# --- Bug 5: ResourceConstraints accepted bool as a valid int budget ---


def test_resource_constraints_rejects_bool_token_budget() -> None:
    """bool is an int subclass; a boolean budget must be rejected, not coerced."""
    with pytest.raises(ValueError):
        ResourceConstraints(tokens=True)  # type: ignore[arg-type]


def test_resource_constraints_rejects_bool_per_tool_limit() -> None:
    with pytest.raises(ValueError):
        ResourceConstraints(per_tool_limits={"web_search": True})  # type: ignore[dict-item]


# --- Bug 6: cost-axis conservation used an unguarded float comparison ---


def test_cost_conservation_tolerates_floating_point_error() -> None:
    """Allocating 0.1 three times against a 0.3 cost budget must not spuriously fail."""
    parent = Contract(id="p", name="p", resources=ResourceConstraints(cost_usd=0.3))
    cap = ContractingCapability(parent)
    cap.create_subcontract(name="a", cost_usd=0.1)
    cap.create_subcontract(name="b", cost_usd=0.1)
    cap.create_subcontract(name="c", cost_usd=0.1)  # must not raise
    assert cap.get_summary().conservation_satisfied


# --- Bug 7: ContractAgent strict_mode never raised, despite its docstring ---


def test_contract_agent_strict_mode_raises_on_violation() -> None:
    """strict_mode=True must raise ContractViolationError when a constraint is breached."""
    contract = Contract(id="t", name="t", resources=ResourceConstraints(tokens=10))
    wrapped = ContractAgent(contract=contract, agent=lambda x: "done", strict_mode=True)
    wrapped.resource_monitor.usage.add_tokens(100)  # exceed the 10-token budget
    with pytest.raises(ContractViolationError):
        wrapped.execute("x")


def test_contract_agent_lenient_mode_returns_result_on_violation() -> None:
    """lenient mode must still report the violation via a result, not an exception."""
    contract = Contract(id="t2", name="t2", resources=ResourceConstraints(tokens=10))
    wrapped = ContractAgent(contract=contract, agent=lambda x: "done", strict_mode=False)
    wrapped.resource_monitor.usage.add_tokens(100)
    result = wrapped.execute("x")
    assert result.success is False
