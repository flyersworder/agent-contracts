"""Executable artifacts for whitepaper §4.6 propositions P1-P6.

Each proposition is falsified or confirmed by running code *before* its prose
proof is written. Two of the six are suspected false as currently stated; a
test says so in minutes where a proof attempt can absorb a day.

See docs/superpowers/plans/2026-08-23-m6-theory-propositions.md
"""

import random

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import CycleError, DelegationGraph

ROOT = DelegationGraph.ROOT


def make_root(tokens: int = 100) -> Contract:
    return Contract(id="p-root", name="Root", resources=ResourceConstraints(tokens=tokens))


def _in_flow(edges, node, root_budget):
    if node == ROOT:
        return root_budget
    return sum(a for _s, d, a in edges if d == node)


def _out_flow(edges, node):
    return sum(a for s, _d, a in edges if s == node)


def permitted_total(edges, root_budget):
    """Maximum total consumption the allocation physically permits.

    A node can spend whatever arrives minus whatever it forwards, and never
    less than zero -- an over-committed node cannot offset its neighbours by
    spending negatively. That asymmetry is exactly what a tree accountant
    misses when it cannot see one of a node's out-edges.
    """
    nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
    return sum(max(0, _in_flow(edges, n, root_budget) - _out_flow(edges, n)) for n in nodes)


def tree_admits(edges, root_budget):
    """Drop-policy tree accountant: keep one in-edge per node, ignore the rest.

    The dropped edge is invisible to the accountant but real to the agents --
    the receiving node still holds that budget.
    """
    parents = {}
    for src, dst, _amt in edges:
        parents.setdefault(dst, []).append(src)
    kept = [(s, d, a) for s, d, a in edges if len(parents[d]) == 1 or s == parents[d][0]]
    nodes = {s for s, _d, _a in kept} | {d for _s, d, _a in kept}
    return all(_out_flow(kept, n) <= _in_flow(kept, n, root_budget) for n in nodes)


def test_p2_tree_admits_an_allocation_permitting_more_than_the_root_budget():
    edges = [
        (ROOT, "a", 50),
        (ROOT, "b", 50),
        ("a", "d", 30),
        ("b", "d", 30),  # unrepresentable: d already has parent "a"
        ("b", "e", 30),  # b re-grants budget it has already committed to d
    ]
    assert tree_admits(edges, 100) is True
    assert permitted_total(edges, 100) == 110  # a:20 + b:0 + d:60 + e:30


def test_p2_dag_law_rejects_the_over_commitment():
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d", "e"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "d", tokens=30)
    graph.allocate("b", "d", tokens=30)
    with pytest.raises(ConservationViolationError):
        graph.allocate("b", "e", tokens=30)


def test_p2_dag_law_accepts_the_same_graph_without_the_over_commitment():
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "d", tokens=30)
    graph.allocate("b", "d", tokens=30)
    graph.seal()
    assert graph.in_flow("d").tokens == 60
    edges = [(ROOT, "a", 50), (ROOT, "b", 50), ("a", "d", 30), ("b", "d", 30)]
    assert permitted_total(edges, 100) == 100  # exactly the root budget


def test_p3_static_bound_survives_budget_cycles():
    """If this PASSES, P3 as specified is false and must be restated."""
    edges = [(ROOT, "a", 100), ("a", "b", 50), ("b", "a", 50)]
    assert permitted_total(edges, 100) == 100


def _is_valid_allocation(edges, root_budget):
    """Every node forwards no more than it received."""
    nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
    return all(_out_flow(edges, n) <= _in_flow(edges, n, root_budget) for n in nodes)


def _generate(rng, force_cycle):
    names = ["a", "b", "c"]
    edges = [(ROOT, rng.choice(names), rng.randint(1, 60))]
    for _ in range(rng.randint(1, 4)):
        src, dst = rng.sample(names, 2)
        edges.append((src, dst, rng.randint(1, 30)))
    pairs = {(x, y) for x, y, _a in edges}
    has_cycle = any((d, s) in pairs for s, d, _a in edges)
    return edges if has_cycle == force_cycle else None


def test_p3_cyclic_and_acyclic_allocations_saturate_identically():
    """Valid allocations saturate B(root) exactly, cycles or not.

    Assert the IDENTITY, not `<= root_budget`. The inequality is implied by
    `_is_valid_allocation` alone -- the filter asserts the very invariant whose
    consequence is under test -- so an inequality here can never fail and proves
    nothing. The identity plus the acyclic control is what carries the claim:
    the two populations behave the same, so acyclicity is nowhere used.
    """
    rng = random.Random(20260823)
    counts = {True: 0, False: 0}
    for force_cycle in (True, False):
        for _ in range(4000):
            edges = _generate(rng, force_cycle)
            if edges is None or not _is_valid_allocation(edges, 100):
                continue
            counts[force_cycle] += 1
            assert permitted_total(edges, 100) == 100, edges
    assert counts[True] >= 50, f"only {counts[True]} cyclic allocations"
    assert counts[False] >= 50, f"only {counts[False]} acyclic allocations"


def test_p3_mutation_dropping_the_clamp_breaks_the_identity():
    """Mutation check: without max(0, ...) an over-committed node offsets its
    neighbours, so the identity stops detecting over-commitment."""

    def unclamped(edges, rb):
        nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
        return sum(_in_flow(edges, n, rb) - _out_flow(edges, n) for n in nodes)

    over = [
        (ROOT, "a", 50),
        (ROOT, "b", 50),
        ("a", "d", 30),
        ("b", "d", 30),
        ("b", "e", 30),
    ]
    assert permitted_total(over, 100) == 110  # clamped: over-commitment visible
    assert unclamped(over, 100) == 100  # unclamped: hidden


def test_p3_budget_cycles_are_refused_at_allocation_time():
    """The graph cannot be built, so cyclic reclamation can never arise.

    This is what acyclicity actually buys: not the static bound (see
    test_p3_static_bound_survives_budget_cycles), but a guarantee that
    refund propagation is well-founded, enforced structurally.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "b", tokens=10)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=10)  # would close the budget cycle


def test_p3_zero_amount_does_not_exempt_a_cycle():
    """The cycle check precedes the amount check, so zero is refused too."""
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "b", tokens=10)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=0)


def test_p4_abandonment_bound_is_tight():
    graph = DelegationGraph(make_root(100))
    for name in ("live", "doomed"):
        graph.add_node(name)
    graph.allocate(ROOT, "live", tokens=40)
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()

    graph.monitor_for("live").usage.add_tokens(40)  # spends all of its share
    graph.monitor_for("doomed").usage.add_tokens(10)
    refund = graph.abandon("doomed")
    assert refund.tokens == 50  # 60 granted - 10 spent

    # The abandoned node keeps spending, up to exactly the refunded amount.
    graph.monitor_for("doomed").usage.add_tokens(50)
    total = sum(graph.monitor_for(n).usage.tokens for n in ("live", "doomed"))
    assert total == 100  # == B(root); refunds unusable in v1
    graph.verify()  # doomed sits exactly on its frozen in-flow

    # One token past the frozen in-flow breaks it.
    graph.monitor_for("doomed").usage.add_tokens(1)
    with pytest.raises(ConservationViolationError):
        graph.verify()
