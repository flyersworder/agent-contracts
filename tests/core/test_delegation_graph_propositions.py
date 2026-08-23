"""Executable artifacts for whitepaper §4.6 propositions P1-P6.

Each proposition is falsified or confirmed by running code *before* its prose
proof is written. Two of the six are suspected false as currently stated; a
test says so in minutes where a proof attempt can absorb a day.

See docs/superpowers/plans/2026-08-23-m6-theory-propositions.md
"""

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import DelegationGraph

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
