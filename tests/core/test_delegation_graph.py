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


def test_seal_accepts_valid_graph():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    assert graph.is_sealed


def test_seal_rejects_orphan_node():
    from agent_contracts.core.delegation_graph import GraphLintError

    graph = DelegationGraph(make_root())
    graph.add_node("orphan")
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert any("orphan" in problem for problem in excinfo.value.problems)


def test_seal_reports_all_problems_not_just_first():
    from agent_contracts.core.delegation_graph import GraphLintError

    graph = DelegationGraph(make_root())
    graph.add_node("orphan_a")
    graph.add_node("orphan_b")
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert len(excinfo.value.problems) == 2


def test_allocate_after_seal_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        graph.add_node("late")
    with pytest.raises(RuntimeError, match="sealed"):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=1)


def test_double_seal_is_idempotent():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    graph.seal()
    assert graph.is_sealed


def test_contract_for_sums_in_flow():
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    graph.seal()
    contract = graph.contract_for("aggregator")
    assert contract.resources.tokens == 30_000
    assert contract.id == "root-contract/aggregator"


def test_contract_for_before_seal_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    with pytest.raises(RuntimeError, match="seal"):
        graph.contract_for("child")


def test_contract_for_is_stable_across_calls():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    assert graph.contract_for("child") is graph.contract_for("child")


def test_residual_is_in_flow_minus_consumption_minus_out_flow():
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1000)
    graph.allocate("mid", "leaf", tokens=400)
    graph.seal()
    graph.monitor_for("mid").usage.add_tokens(100)
    assert graph.residual("mid").tokens == 500


def test_check_node_raises_when_consumption_breaks_invariant():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    graph.monitor_for("child").usage.add_tokens(101)
    with pytest.raises(FlowConservationError) as excinfo:
        graph.check_node("child")
    assert excinfo.value.node_id == "child"


def test_verify_checks_every_node():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    graph.verify()  # clean
    graph.monitor_for("child").usage.add_tokens(101)
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_node_monitor_enforces_summed_budget_via_existing_machinery():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    monitor = graph.monitor_for("child")
    monitor.usage.add_tokens(101)
    assert any(v.resource == "tokens" for v in monitor.check_constraints())


def _diamond() -> DelegationGraph:
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    graph.seal()
    return graph


def test_release_returns_proportional_share():
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(20_000)  # residual 10_000
    refund = graph.release("scout_a", "aggregator")
    assert refund.tokens == 5_000


def test_release_is_order_independent():
    first = _diamond()
    first.monitor_for("aggregator").usage.add_tokens(20_000)
    a_then_b = (
        first.release("scout_a", "aggregator").tokens,
        first.release("scout_b", "aggregator").tokens,
    )

    second = _diamond()
    second.monitor_for("aggregator").usage.add_tokens(20_000)
    b_then_a = (
        second.release("scout_b", "aggregator").tokens,
        second.release("scout_a", "aggregator").tokens,
    )

    # Both siblings funded the aggregator equally, so both reclaim half the
    # 10_000 unused tokens regardless of which released first.
    assert a_then_b == (5_000, 5_000)
    assert b_then_a == (5_000, 5_000)
    assert first.residual("scout_a").tokens == second.residual("scout_a").tokens
    assert first.residual("aggregator").tokens == second.residual("aggregator").tokens


def test_release_restores_parent_residual():
    graph = _diamond()
    before = graph.residual("scout_a").tokens
    graph.monitor_for("aggregator").usage.add_tokens(20_000)
    graph.release("scout_a", "aggregator")
    assert graph.residual("scout_a").tokens == before + 5_000


def test_release_of_fully_consumed_edge_refunds_nothing():
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(30_000)
    assert graph.release("scout_a", "aggregator").tokens == 0


def test_double_release_rejected():
    graph = _diamond()
    graph.release("scout_a", "aggregator")
    with pytest.raises(ValueError, match="already released"):
        graph.release("scout_a", "aggregator")


def test_abandon_refunds_unconsumed_budget_to_parents():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(1_000)
    graph.abandon("scout_a")
    # scout_a held 40_000, spent 1_000, passed 15_000 downstream; root reclaims 24_000
    assert graph.residual(DelegationGraph.ROOT).tokens == 100_000 - 40_000 - 40_000 + 24_000


def test_abandon_marks_downstream_unreachable():
    graph = _diamond()
    graph.abandon("scout_a")
    assert not graph.is_reachable("scout_a")
    assert graph.is_reachable("scout_b")
    assert graph.is_reachable("aggregator")  # scout_b still funds it


def test_abandoned_node_excluded_from_verify():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(40_000)
    graph.abandon("scout_a")
    graph.verify()
