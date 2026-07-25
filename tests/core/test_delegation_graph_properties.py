"""Property-style tests using a seeded generator (no hypothesis dependency)."""

import random

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ContractingCapability
from agent_contracts.core.delegation_graph import DelegationGraph
from agent_contracts.core.resource_vector import ResourceVector

SEEDS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def _root(tokens: int = 1_000_000) -> Contract:
    return Contract(id="root", name="Root", resources=ResourceConstraints(tokens=tokens))


@pytest.mark.parametrize("seed", SEEDS)
def test_tree_topology_matches_contracting_capability(seed):
    """On a tree, DelegationGraph and ContractingCapability must agree exactly."""
    rng = random.Random(seed)
    child_budgets = {f"child_{i}": rng.randint(1, 50_000) for i in range(rng.randint(1, 6))}

    tree = ContractingCapability(_root())
    for name, tokens in child_budgets.items():
        tree.create_subcontract(name, tokens=tokens)

    graph = DelegationGraph(_root())
    for name, tokens in child_budgets.items():
        graph.add_node(name)
        graph.allocate(DelegationGraph.ROOT, name, tokens=tokens)

    assert graph.residual(DelegationGraph.ROOT).tokens == tree.remaining_tokens
    for name, tokens in child_budgets.items():
        assert graph.in_flow(name).tokens == tokens
        assert tree.get_allocation(name).tokens_allocated == tokens


def _random_dag(rng: random.Random, root_tokens: int) -> tuple[DelegationGraph, list[str]]:
    """Build a random DAG in which every node is funded.

    Node ``i`` only ever receives edges from nodes ``< i``, so the result is
    acyclic by construction. Each target is offered candidates in random order
    and takes the first one or two with headroom; the root is always a
    candidate and each allocation takes at most half the funder's headroom, so
    with at most 7 nodes the root retains at least ``root_tokens / 2**7``
    tokens and can always fund. Every node therefore gets at least one in-edge,
    the graph always seals, and the test never skips.
    """
    graph = DelegationGraph(_root(root_tokens))
    names = [DelegationGraph.ROOT]
    for i in range(rng.randint(2, 7)):
        name = f"n{i}"
        graph.add_node(name)
        names.append(name)

    for target_index in range(1, len(names)):
        target = names[target_index]
        candidates = names[:target_index]
        rng.shuffle(candidates)
        wanted = rng.randint(1, min(2, len(candidates)))
        funded = 0
        for source in candidates:
            headroom = graph.residual(source).tokens
            if headroom is None or headroom < 2:
                continue
            graph.allocate(source, target, tokens=rng.randint(1, headroom // 2))
            funded += 1
            if funded == wanted:
                break
        assert funded > 0, f"generator left '{target}' unfunded; root should always have headroom"
    return graph, names


@pytest.mark.parametrize("seed", SEEDS)
def test_local_invariants_imply_global_bound(seed):
    """If every node satisfies its local invariant, total consumption <= root budget."""
    rng = random.Random(seed)
    root_tokens = 100_000
    graph, names = _random_dag(rng, root_tokens)

    for name in names:
        if name != DelegationGraph.ROOT:
            assert graph.contributing_edges(name), f"'{name}' must be funded"
    graph.seal()

    # Saturate: every node consumes its ENTIRE residual. Consuming a random
    # fraction instead leaves roughly 2x slack under the bound, which lets a
    # diamond that double-counts its in-flow pass undetected on all 10 seeds
    # (verified by mutation testing). Saturation makes the telescoping identity
    # tight, so the assertion below is an equality, not a loose inequality.
    for name in names:
        headroom = graph.residual(name).tokens
        if headroom and headroom > 0:
            graph.monitor_for(name).usage.add_tokens(headroom)

    graph.verify()  # every local invariant holds

    total_consumed = sum(
        ResourceVector.from_usage(graph.monitor_for(name).usage).tokens for name in names
    )
    assert total_consumed == root_tokens
