import pytest

from agent_contracts.core.contract import ResourceConstraints
from agent_contracts.core.monitor import ResourceUsage
from agent_contracts.core.resource_vector import ResourceVector


def test_add_finite_scalars():
    a = ResourceVector(tokens=10, cost_usd=1.0, tool_invocations=2, iterations=1)
    b = ResourceVector(tokens=5, cost_usd=0.5, tool_invocations=3, iterations=4)
    total = a + b
    assert total.tokens == 15
    assert total.cost_usd == 1.5
    assert total.tool_invocations == 5
    assert total.iterations == 5


def test_add_unbounded_absorbs():
    a = ResourceVector(tokens=None)
    b = ResourceVector(tokens=5)
    assert (a + b).tokens is None


def test_subtract_from_unbounded_stays_unbounded():
    a = ResourceVector(tokens=None)
    b = ResourceVector(tokens=5)
    assert (a - b).tokens is None


def test_subtract_unbounded_from_finite_raises():
    a = ResourceVector(tokens=5)
    b = ResourceVector(tokens=None)
    with pytest.raises(ValueError, match="unbounded"):
        a - b


def test_le_anything_under_unbounded():
    assert ResourceVector(tokens=10**9) <= ResourceVector(tokens=None)


def test_le_unbounded_not_under_finite():
    assert not (ResourceVector(tokens=None) <= ResourceVector(tokens=5))


def test_per_tool_add_unions_keys():
    a = ResourceVector(per_tool={"exp": 3})
    b = ResourceVector(per_tool={"exp": 2, "web": 1})
    total = a + b
    assert total.per_tool == {"exp": 5, "web": 1}


def test_per_tool_le_ignores_tools_budget_does_not_constrain():
    used = ResourceVector(per_tool={"exp": 3, "unconstrained": 99})
    budget = ResourceVector(per_tool={"exp": 5})
    assert used <= budget


def test_per_tool_le_detects_overrun():
    used = ResourceVector(per_tool={"exp": 6})
    budget = ResourceVector(per_tool={"exp": 5})
    assert not (used <= budget)


def test_from_constraints_reads_all_dimensions():
    rc = ResourceConstraints(
        tokens=100, cost_usd=2.0, tool_invocations=7, iterations=3, per_tool_limits={"exp": 4}
    )
    v = ResourceVector.from_constraints(rc)
    assert v.tokens == 100
    assert v.cost_usd == 2.0
    assert v.tool_invocations == 7
    assert v.iterations == 3
    assert v.per_tool == {"exp": 4}


def test_from_usage_reads_all_dimensions():
    usage = ResourceUsage(tokens=50, cost_usd=1.0, tool_invocations=2)
    usage.add_tool_invocation("exp")
    usage.add_iteration()
    v = ResourceVector.from_usage(usage)
    assert v.tokens == 50
    assert v.cost_usd == 1.0
    assert v.tool_invocations == 3
    assert v.iterations == 1
    assert v.per_tool == {"exp": 1}


def test_to_constraints_round_trips():
    rc = ResourceConstraints(tokens=100, cost_usd=2.0, per_tool_limits={"exp": 4})
    assert ResourceVector.from_constraints(rc).to_constraints() == rc


def test_is_finite():
    assert ResourceVector(tokens=1, cost_usd=0.0, tool_invocations=0, iterations=0).is_finite()
    assert not ResourceVector(tokens=None).is_finite()


def test_zero_is_all_zeros():
    assert ResourceVector.ZERO.tokens == 0
    assert ResourceVector.ZERO.per_tool == {}


# --------------------------------------------------------------- finding 1
# Cost-axis float tolerance (PR 77 review). Mirrors the 0.3.2 fix already
# applied to `delegation.py`'s tree conservation law.


def test_le_cost_axis_tolerates_float_noise_at_exact_budget():
    used = (
        ResourceVector(cost_usd=0.1) + ResourceVector(cost_usd=0.1) + ResourceVector(cost_usd=0.1)
    )
    budget = ResourceVector(cost_usd=0.3)
    assert used.cost_usd != 0.3  # confirm the float noise this test guards against
    assert used <= budget


def test_le_cost_axis_rejects_genuine_overrun():
    assert not (ResourceVector(cost_usd=0.4) <= ResourceVector(cost_usd=0.3))


def test_le_token_axis_stays_exact_no_tolerance():
    """The integer axes must not inherit the cost-axis tolerance."""
    assert not (ResourceVector(tokens=11) <= ResourceVector(tokens=10))


def test_cost_epsilon_matches_delegation_module():
    """`delegation.py` is intentionally untouched (see CHANGELOG 0.4.0's
    "core/delegation.py is untouched"), so `resource_vector.py` cannot import
    its private `_COST_EPSILON` without coupling the two modules. Each
    module defines its own copy instead; this test pins them equal so the
    tree and flow conservation laws cannot silently drift apart.
    """
    from agent_contracts.core import delegation, resource_vector

    assert resource_vector._COST_EPSILON == delegation._COST_EPSILON
