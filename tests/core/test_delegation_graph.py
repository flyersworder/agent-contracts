import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import (
    CycleError,
    DelegationGraph,
    FlowConservationError,
)


def make_root(**kwargs) -> Contract:
    defaults = {"tokens": 100_000, "per_tool_limits": {"exp": 59}}
    defaults.update(kwargs)
    return Contract(
        id="root-contract",
        name="Root",
        resources=ResourceConstraints(**defaults),
    )


def test_root_node_registered_on_construction():
    graph = DelegationGraph(make_root())
    assert graph.in_flow(DelegationGraph.ROOT).tokens == 100_000


def test_add_node_then_allocate_sets_in_flow():
    graph = DelegationGraph(make_root())
    graph.add_node("scout_a")
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    assert graph.in_flow("scout_a").tokens == 40_000
    assert graph.out_flow(DelegationGraph.ROOT).tokens == 40_000


def test_fan_in_accumulates_budget():
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    assert graph.in_flow("aggregator").tokens == 30_000


def test_over_allocation_raises_flow_conservation_error():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("child")
    with pytest.raises(FlowConservationError) as excinfo:
        graph.allocate(DelegationGraph.ROOT, "child", tokens=101)
    assert excinfo.value.dimension == "tokens"
    assert excinfo.value.node_id == DelegationGraph.ROOT


def test_flow_conservation_error_is_catchable_as_conservation_violation():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("child")
    with pytest.raises(ConservationViolationError):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=101)


def test_cumulative_over_allocation_across_two_edges_raises():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("a")
    graph.add_node("b")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=60)
    with pytest.raises(FlowConservationError):
        graph.allocate(DelegationGraph.ROOT, "b", tokens=60)


def test_per_tool_over_allocation_raises():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    with pytest.raises(FlowConservationError):
        graph.allocate(DelegationGraph.ROOT, "child", per_tool={"exp": 60})


def test_unbounded_parent_allows_any_finite_allocation():
    root = Contract(id="r", name="R", resources=ResourceConstraints())
    graph = DelegationGraph(root)
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10**9)
    assert graph.in_flow("child").tokens == 10**9


def test_unbounded_allocation_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    with pytest.raises(ValueError, match="finite"):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=None)


def test_cycle_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    graph.add_node("b")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=100)
    graph.allocate("a", "b", tokens=50)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=10)


def test_self_edge_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    with pytest.raises(CycleError):
        graph.allocate("a", "a", tokens=10)


def test_duplicate_node_name_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node("a")


def test_allocate_to_unknown_node_rejected():
    graph = DelegationGraph(make_root())
    with pytest.raises(KeyError):
        graph.allocate(DelegationGraph.ROOT, "nope", tokens=10)
