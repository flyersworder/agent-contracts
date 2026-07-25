"""Property-style tests using a seeded generator (no hypothesis dependency)."""

import random

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ContractingCapability
from agent_contracts.core.delegation_graph import DelegationGraph, FlowConservationError
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.core.resource_vector import ResourceVector

SEEDS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]

ROOT_TOKENS = 100_000
ROOT_COST = 100.0
ROOT_EXP = 200
ROOT_TOOLS = 2 * ROOT_EXP
# 1/64 of a dollar: exactly representable in binary, unlike a cent.
COST_UNIT = 1 / 64


def _root(tokens: int = 1_000_000) -> Contract:
    return Contract(id="root", name="Root", resources=ResourceConstraints(tokens=tokens))


def _multi_root() -> Contract:
    """Root budget spanning four conserved dimensions, not tokens alone."""
    return Contract(
        id="root",
        name="Root",
        resources=ResourceConstraints(
            tokens=ROOT_TOKENS,
            cost_usd=ROOT_COST,
            tool_invocations=ROOT_TOOLS,
            per_tool_limits={"exp": ROOT_EXP},
        ),
    )


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


@pytest.mark.parametrize("seed", SEEDS)
def test_tree_equivalence_holds_with_root_consumption(seed):
    """The root spending its own budget must reduce both laws identically.

    The depth-1 star with an idle root never exercised
    ``remaining_tokens``'s ``- parent_used_tokens`` term.
    """
    rng = random.Random(seed)
    root_used = rng.randint(1, 100_000)
    child_budgets = {f"child_{i}": rng.randint(1, 50_000) for i in range(rng.randint(1, 6))}

    tree_contract = _root()
    tree_monitor = ResourceMonitor(tree_contract.resources)
    tree_monitor.usage.add_tokens(root_used)
    tree = ContractingCapability(tree_contract, tree_monitor)

    graph_contract = _root()
    graph_monitor = ResourceMonitor(graph_contract.resources)
    graph_monitor.usage.add_tokens(root_used)
    graph = DelegationGraph(graph_contract, graph_monitor)

    for name, tokens in child_budgets.items():
        tree.create_subcontract(name, tokens=tokens)
        graph.add_node(name)
        graph.allocate(DelegationGraph.ROOT, name, tokens=tokens)

    assert graph.residual(DelegationGraph.ROOT).tokens == tree.remaining_tokens


@pytest.mark.parametrize("seed", SEEDS)
def test_multi_level_tree_matches_nested_contracting_capabilities(seed):
    """Depth 2: the graph's mid node must agree with a child's own capability."""
    rng = random.Random(seed)
    child_tokens = rng.randint(50_000, 200_000)
    grandchildren = {
        f"grand_{i}": rng.randint(1, child_tokens // 4) for i in range(rng.randint(1, 3))
    }
    child_used = rng.randint(0, child_tokens // 8)

    tree = ContractingCapability(_root())
    child_contract = tree.create_subcontract("child", tokens=child_tokens)
    child_monitor = ResourceMonitor(child_contract.resources)
    child_monitor.usage.add_tokens(child_used)
    child_capability = ContractingCapability(child_contract, child_monitor)

    graph = DelegationGraph(_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=child_tokens)
    for name, tokens in grandchildren.items():
        child_capability.create_subcontract(name, tokens=tokens)
        graph.add_node(name)
        graph.allocate("child", name, tokens=tokens)
    graph.seal()
    graph.monitor_for("child").usage.add_tokens(child_used)

    assert graph.residual(DelegationGraph.ROOT).tokens == tree.remaining_tokens
    assert graph.residual("child").tokens == child_capability.remaining_tokens
    for name, tokens in grandchildren.items():
        assert graph.in_flow(name).tokens == tokens
        assert child_capability.get_allocation(name).tokens_allocated == tokens


def test_reserve_ratio_is_the_only_divergence_from_the_tree_law():
    """ "Strict generalization" is exact at ``reserve_ratio=0``.

    ``ContractingCapability`` withholds a coordination reserve that the flow
    graph has no analogue for, so the two laws differ by exactly that reserve.
    """
    tree = ContractingCapability(_root(), reserve_ratio=0.2)
    tree.create_subcontract("child", tokens=100_000)

    graph = DelegationGraph(_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100_000)

    assert tree.reserved_tokens == 200_000
    assert graph.residual(DelegationGraph.ROOT).tokens - tree.remaining_tokens == 200_000


def test_residual_may_go_negative_where_remaining_tokens_clamps():
    """The second, deliberate divergence: ``remaining_tokens`` clamps at zero.

    A graph residual is signed so an overrun is visible; the tree law reports
    zero for both "exactly spent" and "overspent".
    """
    contract = _root(1_000)
    monitor = ResourceMonitor(contract.resources)
    monitor.usage.add_tokens(1_500)
    tree = ContractingCapability(contract, monitor)

    graph_contract = _root(1_000)
    graph_monitor = ResourceMonitor(graph_contract.resources)
    graph_monitor.usage.add_tokens(1_500)
    graph = DelegationGraph(graph_contract, graph_monitor)

    assert tree.remaining_tokens == 0
    assert graph.residual(DelegationGraph.ROOT).tokens == -500


def _grant(graph: DelegationGraph, rng: random.Random, source: str, target: str) -> bool:
    """Fund ``target`` from ``source`` with at most half of ``source``'s headroom.

    Returns False when the source is too poor to fund anything. Every grant
    keeps ``tool_invocations == 2 * per_tool["exp"]``, an invariant the root
    budget also satisfies, so both tool dimensions saturate exactly and the
    aggregate is never a mere copy of the per-tool count.

    Cost is granted in whole ``COST_UNIT`` (1/64 dollar) steps. Decimal cents
    are not representable in binary floating point, so a generator using them
    makes ``in_flow - out_flow`` land a few ulps *above* the true residual;
    saturating on that value then trips the invariant by ~1e-15 and the test
    fails on the generator's rounding rather than on the law. Powers of two are
    exact, which keeps the telescoping equality meaningful.
    """
    headroom = graph.residual(source)
    exp_room = headroom.per_tool.get("exp", 0)
    cost_units = round(headroom.cost_usd / COST_UNIT)
    if headroom.tokens < 2 or cost_units < 2 or exp_room < 2:
        return False
    experiments = rng.randint(1, exp_room // 2)
    graph.allocate(
        source,
        target,
        tokens=rng.randint(1, headroom.tokens // 2),
        cost_usd=rng.randint(1, cost_units // 2) * COST_UNIT,
        tool_invocations=2 * experiments,
        per_tool={"exp": experiments},
    )
    return True


def _random_dag(rng: random.Random) -> tuple[DelegationGraph, list[str]]:
    """Build a random DAG in which every node is funded and at least one fans in.

    Node ``i`` only ever receives edges from nodes ``< i``, so the result is
    acyclic by construction. Each target is offered every earlier node as a
    candidate, in random order, and takes as many as it wants among those with
    headroom; each allocation takes at most half the funder's headroom, so with
    at most 7 nodes the root retains at least ``ROOT_TOKENS / 2**7`` and can
    always fund.

    Fan-in is guaranteed *structurally*, not by luck of the seed: ``n0`` gets a
    fixed quarter of the root — large enough to fund anything downstream — and
    ``n1`` is then funded by both the root and ``n0``. A generator that relied
    on a random grant to leave ``n0`` solvent would fail on an unlucky seed,
    which breaks the generator rather than the invariant under test. A tree
    would make the telescoping test vacuous: a tree is the case the pre-existing
    law already covered.

    Allocations span tokens, cost, tool invocations and a per-tool experiment
    budget together: conserving only tokens leaves the other dimensions'
    arithmetic — and, for per-tool, the whole M6 conserved resource — untested.
    """
    graph = DelegationGraph(_multi_root())
    names = [DelegationGraph.ROOT]
    for i in range(rng.randint(3, 7)):
        name = f"n{i}"
        graph.add_node(name)
        names.append(name)

    graph.allocate(
        DelegationGraph.ROOT,
        "n0",
        tokens=ROOT_TOKENS // 4,
        cost_usd=ROOT_COST / 4,
        tool_invocations=2 * (ROOT_EXP // 4),
        per_tool={"exp": ROOT_EXP // 4},
    )
    for source in (DelegationGraph.ROOT, "n0"):
        assert _grant(graph, rng, source, "n1"), f"'{source}' must be able to fund the fan-in node"

    for target_index in range(3, len(names)):
        target = names[target_index]
        candidates = names[:target_index]
        rng.shuffle(candidates)
        wanted = rng.randint(1, len(candidates))
        funded = 0
        for source in candidates:
            if _grant(graph, rng, source, target):
                funded += 1
                if funded == wanted:
                    break
        assert funded > 0, f"generator left '{target}' unfunded; root should always have headroom"
    return graph, names


def _saturate(graph: DelegationGraph, names: list[str]) -> None:
    """Every node consumes its entire residual in every dimension.

    Experiments are consumed by name and the rest of the aggregate tool budget
    anonymously. Because every grant keeps ``tool_invocations == 2 * exp``, and
    a named invocation counts against both, consuming ``e`` experiments and
    then ``e`` unnamed invocations saturates both dimensions exactly.
    """
    for name in names:
        headroom = graph.residual(name)
        usage = graph.monitor_for(name).usage
        if headroom.tokens > 0:
            usage.add_tokens(headroom.tokens)
        if headroom.cost_usd > 0:
            usage.add_cost(headroom.cost_usd)
        experiments = headroom.per_tool.get("exp", 0)
        for _ in range(max(0, experiments)):
            usage.add_tool_invocation("exp")
        for _ in range(max(0, graph.residual(name).tool_invocations)):
            usage.add_tool_invocation()


@pytest.mark.parametrize("seed", SEEDS)
def test_local_invariants_imply_global_bound(seed):
    """If every node satisfies its local invariant, total consumption == root budget."""
    rng = random.Random(seed)
    graph, names = _random_dag(rng)

    for name in names:
        if name != DelegationGraph.ROOT:
            assert graph.contributing_edges(name), f"'{name}' must be funded"
    assert any(len(graph.contributing_edges(name)) > 1 for name in names), "no fan-in generated"
    graph.seal()

    # Saturate: every node consumes its ENTIRE residual. Consuming a random
    # fraction instead leaves roughly 2x slack under the bound, which lets a
    # diamond that double-counts its in-flow pass undetected on all 10 seeds
    # (verified by mutation testing). Saturation makes the telescoping identity
    # tight, so the assertions below are equalities, not loose inequalities.
    _saturate(graph, names)

    graph.verify()  # every local invariant holds

    consumed = [ResourceVector.from_usage(graph.monitor_for(name).usage) for name in names]
    assert sum(vector.tokens for vector in consumed) == ROOT_TOKENS
    assert sum(vector.tool_invocations for vector in consumed) == ROOT_TOOLS
    assert sum(vector.per_tool.get("exp", 0) for vector in consumed) == ROOT_EXP
    # Exact even for the float dimension, because every grant is a whole
    # COST_UNIT: see `_grant`. Verified exact across 500 seeds, not just these.
    assert sum(vector.cost_usd for vector in consumed) == ROOT_COST


@pytest.mark.parametrize("seed", SEEDS)
def test_consuming_an_unfunded_tool_is_caught_anywhere_in_the_graph(seed):
    """Per-tool conservation on the consumption path, at every node.

    A tool a node's in-flow never named is unfunded, not unconstrained — the
    scalar dimensions still have headroom here, so only the per-tool clause can
    catch this.
    """
    rng = random.Random(seed)
    graph, names = _random_dag(rng)
    graph.seal()

    victim = rng.choice([name for name in names if name != DelegationGraph.ROOT])
    graph.monitor_for(victim).usage.add_tool_invocation("unfunded")

    with pytest.raises(FlowConservationError) as excinfo:
        graph.verify()
    assert excinfo.value.node_id == victim
    assert excinfo.value.dimension == "tool:unfunded"
    assert excinfo.value.in_flow is None


@pytest.mark.parametrize("seed", SEEDS)
def test_abandoning_an_honest_node_keeps_the_graph_verifiable(seed):
    """Abandonment interleaved with consumption must not break honest graphs."""
    rng = random.Random(seed)
    graph, names = _random_dag(rng)
    graph.seal()

    # Partial, honest consumption everywhere, then a mid-graph node dies.
    for name in names:
        headroom = graph.residual(name).tokens
        if headroom > 1:
            graph.monitor_for(name).usage.add_tokens(rng.randint(1, headroom // 2))

    victim = rng.choice([name for name in names if name != DelegationGraph.ROOT])
    graph.abandon(victim)

    graph.verify()
    assert not graph.is_reachable(victim)


@pytest.mark.parametrize("seed", SEEDS)
def test_abandoning_an_overspent_node_still_fails_verification(seed):
    """The same interleaving, but the dying node overspent: it stays flagged."""
    rng = random.Random(seed)
    graph, names = _random_dag(rng)
    graph.seal()

    victim = rng.choice([name for name in names if name != DelegationGraph.ROOT])
    overspend = graph.in_flow(victim).tokens + graph.out_flow(victim).tokens + 1
    graph.monitor_for(victim).usage.add_tokens(overspend)

    with pytest.raises(FlowConservationError):
        graph.verify()

    graph.abandon(victim)

    with pytest.raises(FlowConservationError) as excinfo:
        graph.verify()
    assert excinfo.value.node_id == victim
