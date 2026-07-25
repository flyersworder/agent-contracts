import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import (
    CycleError,
    DelegationGraph,
    FlowConservationError,
)
from agent_contracts.core.resource_vector import ResourceVector


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


def test_per_tool_grant_of_a_tool_the_source_lacks_is_rejected():
    """The conserved resource in M6 *is* the per-tool dimension.

    ``ResourceVector.__le__`` compares only the keys the budget side names, so
    a node whose in-flow omits a tool would otherwise be free to hand that tool
    out without limit.
    """
    graph = DelegationGraph(make_root())  # root constrains exp to 59
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1_000, tool_invocations=150)

    with pytest.raises(FlowConservationError) as excinfo:
        graph.allocate("mid", "leaf", per_tool={"exp": 150})
    assert "exp" in str(excinfo.value)
    assert excinfo.value.node_id == "mid"


def test_consuming_a_tool_the_in_flow_does_not_fund_is_a_violation():
    """Conserving per-tool on the grant path alone left the consumption path
    vacuously compliant: a node whose in-flow omits a tool could spend it
    without limit, which is the M6 conserved resource."""
    graph = DelegationGraph(make_root())  # root constrains exp to 59
    graph.add_node("mid")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1_000, tool_invocations=1_000)
    graph.seal()
    for _ in range(150):
        graph.monitor_for("mid").usage.add_tool_invocation("exp")

    with pytest.raises(FlowConservationError) as excinfo:
        graph.verify()
    error = excinfo.value
    assert error.node_id == "mid"
    assert error.dimension == "tool:exp"
    assert error.consumed == 150
    # Undeclared is not a budget of zero, and the audit trail must say so.
    assert error.in_flow is None
    assert "does not fund" in str(error)


def test_root_may_consume_a_tool_it_leaves_unconstrained():
    root = Contract(id="r", name="R", resources=ResourceConstraints(tokens=1_000))
    graph = DelegationGraph(root)
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    for _ in range(5):
        graph.monitor_for(DelegationGraph.ROOT).usage.add_tool_invocation("exp")
    graph.verify()


def test_consuming_a_funded_tool_within_budget_is_fine():
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.allocate(DelegationGraph.ROOT, "mid", tool_invocations=10, per_tool={"exp": 10})
    graph.seal()
    for _ in range(10):
        graph.monitor_for("mid").usage.add_tool_invocation("exp")
    graph.verify()
    graph.monitor_for("mid").usage.add_tool_invocation("exp")
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_zero_grant_of_an_undeclared_tool_is_allowed():
    """A grant of zero tightens the child; it cannot make anything unbounded.

    After per-tool conservation on the consumption path, a zero grant is the
    only way a node with no budget for a tool can constrain its child at all —
    and M6 arm 3's aggregator is exactly that node.
    """
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("aggregator")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1_000, tool_invocations=10)
    graph.allocate("mid", "aggregator", tokens=100, per_tool={"exp": 0})
    graph.seal()
    assert graph.contract_for("aggregator").resources.per_tool_limits == {"exp": 0}
    graph.verify()


def test_propagation_error_reports_the_tool_as_undeclared_not_zero():
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1_000)
    with pytest.raises(FlowConservationError) as excinfo:
        graph.allocate("mid", "leaf", per_tool={"exp": 5})
    error = excinfo.value
    assert error.in_flow is None  # undeclared, not a budget of zero
    assert error.deficit == 5
    assert "does not constrain" in str(error)


def test_root_may_grant_a_tool_it_leaves_unconstrained():
    """The root's budget is exogenous: an unconstrained tool at the root is
    unbounded, not absent."""
    root = Contract(id="r", name="R", resources=ResourceConstraints(tokens=1_000))
    graph = DelegationGraph(root)
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", per_tool={"exp": 10})
    assert graph.in_flow("child").per_tool == {"exp": 10}


def test_per_tool_propagates_down_a_chain():
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", per_tool={"exp": 30})
    graph.allocate("mid", "leaf", per_tool={"exp": 10})
    graph.seal()
    assert graph.contract_for("leaf").resources.per_tool_limits == {"exp": 10}


def test_zero_per_tool_grant_is_constrained_not_unconstrained():
    """M6 arm 3 gives the aggregator zero experiments; that must materialize as
    a limit of 0, never as an absent (i.e. unlimited) key."""
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("aggregator")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=10, per_tool={"exp": 30})
    graph.allocate("mid", "aggregator", tokens=5, per_tool={"exp": 0})
    graph.seal()
    assert graph.contract_for("aggregator").resources.per_tool_limits == {"exp": 0}
    monitor = graph.monitor_for("aggregator")
    monitor.usage.add_tool_invocation("exp")
    assert any(v.resource == "tool:exp" for v in monitor.check_constraints())


def test_seal_lints_out_edges_naming_a_tool_the_in_flow_does_not_constrain():
    from agent_contracts.core.delegation_graph import EdgeAllocation, GraphLintError

    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1_000)
    graph.allocate("mid", "leaf", tokens=10)
    # allocate() rejects this at build time; the seal() lint is the independent
    # whole-graph check, so exercise it by planting the edge directly.
    graph._edges["mid->leaf"] = EdgeAllocation(
        source="mid",
        target="leaf",
        amount=ResourceVector(
            tokens=10, cost_usd=0.0, tool_invocations=0, iterations=0, per_tool={"exp": 5}
        ),
    )
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert any("exp" in problem for problem in excinfo.value.problems)


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


def test_seal_flags_a_node_funded_with_nothing():
    """An all-zero edge is legal but funds nothing; testing edge *existence*
    alone would seal such a node clean."""
    from agent_contracts.core.delegation_graph import GraphLintError

    graph = DelegationGraph(make_root())
    graph.add_node("z")
    graph.allocate(DelegationGraph.ROOT, "z")
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert any("z" in problem for problem in excinfo.value.problems)


def test_seal_accepts_a_node_funded_only_with_a_tool_budget():
    graph = DelegationGraph(make_root())
    graph.add_node("z")
    graph.allocate(DelegationGraph.ROOT, "z", per_tool={"exp": 3})
    graph.seal()
    assert graph.is_sealed


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


def test_consumption_overrun_is_reported_as_consumption_not_out_flow():
    graph = DelegationGraph(make_root(tokens=100_000))
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100_000)
    graph.seal()
    graph.monitor_for("child").usage.add_tokens(140_000)

    with pytest.raises(FlowConservationError) as excinfo:
        graph.check_node("child")

    error = excinfo.value
    assert error.dimension == "tokens"
    assert error.consumed == 140_000
    assert error.out_flow == 0  # the node delegated nothing
    assert error.in_flow == 100_000
    assert error.deficit == 40_000
    assert "out-flow 140000" not in str(error)
    assert "consumption 140000" in str(error)


def test_allocation_overrun_still_reports_out_flow():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("child")
    with pytest.raises(FlowConservationError) as excinfo:
        graph.allocate(DelegationGraph.ROOT, "child", tokens=101)
    error = excinfo.value
    assert error.out_flow == 101
    assert error.consumed == 0
    assert error.deficit == 1
    assert "over-allocate" in str(error)


def test_float_dimension_violation_does_not_round_away():
    """``requested``/``available`` are inherited as ints; truncating a float
    overrun to equality would read as no violation at all."""
    graph = DelegationGraph(make_root(tokens=None, cost_usd=1.5))
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", cost_usd=1.5)
    graph.seal()
    graph.monitor_for("child").usage.add_cost(1.9)

    with pytest.raises(FlowConservationError) as excinfo:
        graph.check_node("child")
    error = excinfo.value
    assert error.dimension == "cost_usd"
    assert error.requested > error.available


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


def test_release_is_exact_when_the_target_consumes_no_further():
    """Pins ``release()``'s documented precondition.

    With the target done consuming, the sibling refunds sum exactly to the
    unused budget, the target retains exactly what it spent, and the result is
    the same in either order.
    """
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(20_000)

    refunded = (
        graph.release("scout_b", "aggregator").tokens
        + graph.release("scout_a", "aggregator").tokens
    )

    assert refunded == 10_000  # exactly the unused budget, nothing invented
    assert graph.in_flow("aggregator").tokens == 20_000  # exactly what it spent
    assert graph.residual("aggregator").tokens == 0
    graph.verify()


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


def test_abandoned_node_still_checked_by_verify():
    """Abandonment is the timeout case, and a timed-out node is the likeliest
    to have overspent. Marking it dead must not launder its overspend."""
    graph = _diamond()
    # scout_a holds 40_000 and has already passed 15_000 downstream, so
    # consuming 40_000 puts its commitment at 55_000 > 40_000.
    graph.monitor_for("scout_a").usage.add_tokens(40_000)
    with pytest.raises(FlowConservationError):
        graph.verify()
    graph.abandon("scout_a")
    with pytest.raises(FlowConservationError) as excinfo:
        graph.verify()
    assert excinfo.value.node_id == "scout_a"


def test_abandon_cannot_certify_a_budget_busting_graph():
    graph = DelegationGraph(make_root(tokens=100_000))
    graph.add_node("a")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=100_000)
    graph.seal()
    graph.monitor_for("a").usage.add_tokens(150_000)

    with pytest.raises(FlowConservationError):
        graph.verify()

    graph.abandon("a")

    # Total consumption is 150_000 against a root budget of 100_000; no
    # amount of abandonment may make verify() certify that.
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_abandoned_honest_node_still_passes_verify():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(1_000)
    graph.abandon("scout_a")
    graph.verify()


def test_overspent_abandoned_node_stays_flagged_after_downstream_release():
    """The snapshot is what makes the flag permanent.

    Releasing the dead node's out-edge shrinks its live out-flow, which would
    let a live-values check quietly clear an overspend that really happened.
    """
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(40_000)
    graph.abandon("scout_a")
    graph.release("scout_a", "aggregator")
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_consumption_recorded_after_abandonment_is_still_caught():
    """Freezing consumption at the moment of death reopened C1 by another door:
    a node could keep spending post-mortem and verify() would certify it."""
    graph = DelegationGraph(make_root(tokens=100_000))
    graph.add_node("a")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=100_000)
    graph.seal()
    graph.monitor_for("a").usage.add_tokens(10_000)
    graph.abandon("a")
    graph.verify()  # honest so far

    graph.monitor_for("a").usage.add_tokens(140_000)  # total 150_000 vs a 100_000 budget
    with pytest.raises(FlowConservationError) as excinfo:
        graph.verify()
    assert excinfo.value.node_id == "a"
    assert excinfo.value.consumed == 150_000


def test_abandon_snapshot_records_pre_refund_state():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(1_000)
    graph.abandon("scout_a")
    snapshot = graph.abandon_snapshot("scout_a")
    assert snapshot is not None
    assert snapshot.in_flow.tokens == 40_000  # pre-refund
    assert snapshot.consumed.tokens == 1_000
    assert snapshot.out_flow.tokens == 15_000


def test_release_then_abandon_does_not_double_refund():
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(20_000)
    first = graph.release("scout_a", "aggregator").tokens
    assert first == 5_000

    scout_a_before = graph.residual("scout_a").tokens
    graph.abandon("aggregator")

    # scout_a's edge was already refunded; abandon must not refund it again.
    assert graph.residual("scout_a").tokens == scout_a_before
    edge = next(e for e in graph.edges() if e.key == "scout_a->aggregator")
    assert edge.amount.tokens >= 0


def test_release_of_unknown_edge_rejected():
    graph = _diamond()
    with pytest.raises(KeyError, match="unknown edge"):
        graph.release("scout_a", "scout_b")


def test_abandon_root_rejected():
    graph = _diamond()
    with pytest.raises(ValueError, match="cannot abandon the root"):
        graph.abandon(DelegationGraph.ROOT)


def test_abandon_twice_rejected():
    graph = _diamond()
    graph.abandon("scout_a")
    with pytest.raises(ValueError, match="already abandoned"):
        graph.abandon("scout_a")


def test_abandon_unknown_node_rejected():
    graph = _diamond()
    with pytest.raises(KeyError):
        graph.abandon("nope")


def test_public_api_exports():
    from agent_contracts.core import (
        CycleError,
        DelegationGraph,
        FlowConservationError,
        GraphLintError,
        ResourceVector,
    )
    from agent_contracts.core import delegation_graph as submodule

    assert DelegationGraph.ROOT == "root"
    assert DelegationGraph is submodule.DelegationGraph
    assert issubclass(FlowConservationError, ConservationViolationError)
    assert ResourceVector.ZERO.tokens == 0
    assert CycleError is submodule.CycleError
    assert issubclass(CycleError, Exception)
    assert GraphLintError is submodule.GraphLintError
    assert issubclass(GraphLintError, Exception)
