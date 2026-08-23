"""Executable artifacts for whitepaper §4.6 propositions P1-P6.

Each proposition is falsified or confirmed by running code *before* its prose
proof is written. Three of the six were restated after their first artifact
contradicted the statement they were written to support.

Two rules this module learned the hard way, both from artifacts that passed
while proving nothing:

- A guard is worthless unless the population contains a case that would trip
  it. Where a test relies on a clamp, a filter, or a branch, it must also show
  that branch is *reached* -- see `test_p3_the_clamp_binds_on_the_invalid_half`.
- Reference models must be the real implementation, not a hand-rolled stand-in.
  P2's first artifact modelled a tree accountant that no real system implements
  and whose verdict flipped with edge insertion order.

See docs/superpowers/plans/2026-08-23-m6-theory-propositions.md
"""

import contextlib
import itertools
import random
import sys
import threading

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import (
    ConservationViolationError,
    ContractingCapability,
)
from agent_contracts.core.delegation_graph import (
    CycleError,
    DelegationGraph,
    FlowConservationError,
)
from agent_contracts.core.monitor import ResourceMonitor

ROOT = DelegationGraph.ROOT


def make_root(tokens: int = 100) -> Contract:
    return Contract(id="p-root", name="Root", resources=ResourceConstraints(tokens=tokens))


def _in_flow(edges, node, root_budget):
    if node == ROOT:
        return root_budget
    return sum(a for _s, d, a in edges if d == node)


def _out_flow(edges, node):
    return sum(a for s, _d, a in edges if s == node)


def _nodes_of(edges):
    return {s for s, _d, _a in edges} | {d for _s, d, _a in edges}


def permitted_total(edges, root_budget):
    """Maximum total consumption the allocation physically permits.

    A node can spend whatever arrives minus whatever it forwards, and never
    less than zero -- an over-committed node cannot offset its neighbours by
    spending negatively.
    """
    return sum(
        max(0, _in_flow(edges, n, root_budget) - _out_flow(edges, n)) for n in _nodes_of(edges)
    )


def _is_valid_allocation(edges, root_budget):
    """Every node forwards no more than it received."""
    return all(_out_flow(edges, n) <= _in_flow(edges, n, root_budget) for n in _nodes_of(edges))


def _has_cycle(edges):
    """True if the edge set contains a cycle of ANY length.

    A reversed-pair test finds only 2-cycles and silently admits 3-cycles into
    what is supposed to be an acyclic control group.
    """
    adjacency: dict[str, set[str]] = {}
    for src, dst, _amt in edges:
        adjacency.setdefault(src, set()).add(dst)

    def reachable(start):
        seen, stack = set(), [start]
        while stack:
            for nxt in adjacency.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    return any(n in reachable(n) for n in _nodes_of(edges))


# --------------------------------------------------------------------------
# P2 -- tree encodings of a fan-in node are incomplete
# --------------------------------------------------------------------------
#
# The claim is NOT that tree accounting is unsound. Checked against the real
# `ContractingCapability`, the natural (split) encoding is sound: it refuses
# the over-commitment a hand-rolled drop-policy accountant admits. What no
# tree encoding can do is admit every execution the DAG's local invariant
# admits.


def _split_encoded_fan_in():
    """Fan-in on a tree, encoded the only sound way: split the node.

    root(100) -> a(50), b(50); a -> d_from_a(30), b -> d_from_b(30).
    Agent `d` is one process holding 60 across two contracts.
    """
    root = make_root(100)
    cap = ContractingCapability(root)
    a = cap.create_subcontract("a", tokens=50)
    b = cap.create_subcontract("b", tokens=50)
    d_from_a = ContractingCapability(a).create_subcontract("d_from_a", tokens=30)
    d_from_b = ContractingCapability(b).create_subcontract("d_from_b", tokens=30)
    return d_from_a, d_from_b


def test_p2_merge_encoding_is_refused_no_parent_can_fund_the_node():
    """Encoding 1 of 3: give `d` a single contract for its true budget."""
    cap = ContractingCapability(make_root(100))
    a = cap.create_subcontract("a", tokens=50)
    cap.create_subcontract("b", tokens=50)
    with pytest.raises(ConservationViolationError):
        ContractingCapability(a).create_subcontract("d", tokens=60)


def test_p2_split_encoding_is_sound_the_real_tree_law_refuses_over_commitment():
    """Encoding 2 of 3 is SOUND -- the unsoundness claim was a strawman.

    An earlier artifact asserted that tree accounting certifies an execution
    exceeding B(root). It reached that verdict only by modelling an accountant
    that *drops* one in-edge. The real tree law sees both of `b`'s out-edges
    and refuses the second.
    """
    cap = ContractingCapability(make_root(100))
    cap.create_subcontract("a", tokens=50)
    b = cap.create_subcontract("b", tokens=50)
    cap_b = ContractingCapability(b)
    cap_b.create_subcontract("d_from_b", tokens=30)
    with pytest.raises(ConservationViolationError):
        cap_b.create_subcontract("e", tokens=30)  # b has 50, has committed 30


def test_p2_split_encoding_cannot_execute_an_indivisible_call_within_budget():
    """Encoding 2's real cost: incompleteness, not unsoundness.

    Agent `d` holds 60 and must make one indivisible 40-token call. The call
    respects its true in-flow, and no fragment can absorb it.
    """
    for fragment in _split_encoded_fan_in():
        monitor = ResourceMonitor(fragment.resources)
        monitor.usage.add_tokens(40)
        assert monitor.check_constraints() != [], (
            f"{fragment.name} should refuse a 40-token charge against 30"
        )


def test_p2_the_dag_admits_the_same_call_and_stays_globally_bounded():
    """The DAG pools the two grants, so the identical call is executable."""
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "d", tokens=30)
    graph.allocate("b", "d", tokens=30)
    graph.seal()

    assert graph.in_flow("d").tokens == 60
    monitor = graph.monitor_for("d")
    monitor.usage.add_tokens(40)
    assert monitor.check_constraints() == []  # executable under the DAG law
    graph.verify()  # and the global bound is intact


def test_p2_fragmentation_penalty_scales_with_the_number_of_parents():
    """Under a uniform split across m parents the largest indivisible call is
    B/m, not B. This is the quantitative form of the incompleteness."""
    budget = 60
    for parents in (2, 3, 6):
        share = budget // parents
        cap = ContractingCapability(make_root(1000))
        fragments = [
            ContractingCapability(cap.create_subcontract(f"p{i}", tokens=share)).create_subcontract(
                f"d_from_p{i}", tokens=share
            )
            for i in range(parents)
        ]
        largest = max(f.resources.tokens for f in fragments)
        assert largest == budget // parents

        graph = DelegationGraph(make_root(1000))
        graph.add_node("d")
        for i in range(parents):
            graph.add_node(f"p{i}")
            graph.allocate(ROOT, f"p{i}", tokens=share)
            graph.allocate(f"p{i}", "d", tokens=share)
        graph.seal()
        assert graph.in_flow("d").tokens == share * parents  # the DAG pools


def test_p2_a_drop_policy_accountant_would_be_unsound_but_nobody_builds_one():
    """Recorded for completeness: encoding 3 is the unsound one.

    Dropping `b->d` hides 30 tokens that `d` still holds, so the accountant
    approves grants permitting 110 against a root budget of 100. No real
    implementation drops edges -- `ContractingCapability` splits -- so this
    is the encoding the proposition rules out rather than the one it indicts.
    """
    edges = [
        (ROOT, "a", 50),
        (ROOT, "b", 50),
        ("a", "d", 30),
        ("b", "d", 30),
        ("b", "e", 30),
    ]
    kept = [e for e in edges if e != ("b", "d", 30)]
    assert _is_valid_allocation(kept, 100) is True  # the accountant approves
    assert permitted_total(edges, 100) == 110  # what the agents actually hold


# --------------------------------------------------------------------------
# P3 -- what acyclicity is actually necessary for
# --------------------------------------------------------------------------


def test_p3_static_bound_survives_budget_cycles():
    """P3 as originally specified is false: cycles telescope like any edge."""
    edges = [(ROOT, "a", 100), ("a", "b", 50), ("b", "a", 50)]
    assert permitted_total(edges, 100) == 100


def _generate(rng, force_cycle):
    names = ["a", "b", "c"]
    edges = [(ROOT, rng.choice(names), rng.randint(1, 60))]
    for _ in range(rng.randint(1, 4)):
        src, dst = rng.sample(names, 2)
        edges.append((src, dst, rng.randint(1, 30)))
    return edges if _has_cycle(edges) == force_cycle else None


def _population(seed=20260823, trials=4000):
    """Valid and invalid allocations, split by whether they contain a cycle.

    Invalid allocations are kept deliberately. A population of valid ones only
    is where P3's first artifact went wrong: validity forces in - out >= 0 at
    every node, so the clamp in `permitted_total` never binds and the identity
    holds by telescoping alone, whatever the clamp does.
    """
    rng = random.Random(seed)
    buckets = {(c, v): [] for c in (True, False) for v in (True, False)}
    for force_cycle in (True, False):
        for _ in range(trials):
            edges = _generate(rng, force_cycle)
            if edges is None:
                continue
            buckets[(force_cycle, _is_valid_allocation(edges, 100))].append(edges)
    return buckets


def test_p3_cyclic_and_acyclic_allocations_saturate_identically():
    """Valid allocations saturate B(root) exactly, cycles or not.

    Cycle membership is decided by reachability, so the acyclic bucket is a
    genuine control -- a reversed-pair test admits 3-cycles into it.
    """
    buckets = _population()
    for force_cycle in (True, False):
        sample = buckets[(force_cycle, True)]
        assert len(sample) >= 50, f"only {len(sample)} valid, cycle={force_cycle}"
        for edges in sample:
            assert permitted_total(edges, 100) == 100, edges


def test_p3_the_clamp_binds_on_the_invalid_half():
    """The guard must be shown to be reached, not merely present.

    `permitted_total` equals B(root) exactly when the allocation is locally
    valid, and exceeds it precisely when some node is over-committed. The
    unclamped variant reports B(root) in both cases, so this is the assertion
    that distinguishes them -- and it is only reachable because the population
    admits invalid allocations.
    """

    def unclamped(edges, root_budget):
        return sum(_in_flow(edges, n, root_budget) - _out_flow(edges, n) for n in _nodes_of(edges))

    buckets = _population()
    invalid = buckets[(True, False)] + buckets[(False, False)]
    assert len(invalid) >= 50, f"only {len(invalid)} invalid allocations"

    bound_by_clamp = 0
    for edges in invalid:
        assert permitted_total(edges, 100) > 100, edges
        assert unclamped(edges, 100) == 100, edges  # the mutant cannot tell
        bound_by_clamp += 1
    assert bound_by_clamp >= 50

    valid = buckets[(True, True)] + buckets[(False, True)]
    for edges in valid:
        assert permitted_total(edges, 100) == unclamped(edges, 100) == 100


def test_p3_budget_cycles_are_refused_at_allocation_time():
    """What acyclicity actually buys: cyclic reclamation is unreachable.

    Not the static bound (see test_p3_static_bound_survives_budget_cycles) but
    a guarantee that refund propagation is well-founded, enforced structurally.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "b", tokens=10)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=10)


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


# --------------------------------------------------------------------------
# P4 -- the certified bound under abandonment, and its tightness
# --------------------------------------------------------------------------


def test_p4_the_certified_bound_is_exactly_saturable():
    """`B(root) + Σ refunds` is reachable, and one unit past it is caught.

    An earlier artifact claimed this bound was unsaturable because reclaimed
    budget is not re-delegatable in v1. That conflated *re-delegatable* with
    *re-spendable*: the refund lands at the parent, and the parent may consume
    it itself. The artifact hid this by summing over ("live", "doomed") and
    omitting ROOT -- the node that received the refund.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("live", "doomed"):
        graph.add_node(name)
    graph.allocate(ROOT, "live", tokens=40)
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()

    graph.monitor_for("live").usage.add_tokens(40)
    graph.monitor_for("doomed").usage.add_tokens(10)
    refund = graph.abandon("doomed")
    assert refund.tokens == 50

    # The abandoned node spends up to its frozen pre-refund in-flow of 60 ...
    graph.monitor_for("doomed").usage.add_tokens(50)
    # ... and ROOT spends the 50 it was refunded, its live out-flow having
    # fallen from 100 to 50.
    graph.monitor_for(ROOT).usage.add_tokens(50)

    total = sum(graph.monitor_for(n).usage.tokens for n in graph.node_names())
    assert total == 150  # == B(root) + Σ refunds, exactly
    graph.verify()

    graph.monitor_for(ROOT).usage.add_tokens(1)
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_p4_summing_over_the_wrong_node_set_hides_the_saturation():
    """Regression guard for how the false claim survived review.

    Restricting the sum to the non-root nodes reports 100 for the same
    execution that in fact consumes 150. Any total must range over
    `graph.node_names()`.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("live", "doomed"):
        graph.add_node(name)
    graph.allocate(ROOT, "live", tokens=40)
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()
    graph.monitor_for("live").usage.add_tokens(40)
    graph.monitor_for("doomed").usage.add_tokens(10)
    graph.abandon("doomed")
    graph.monitor_for("doomed").usage.add_tokens(50)
    graph.monitor_for(ROOT).usage.add_tokens(50)

    partial = sum(graph.monitor_for(n).usage.tokens for n in ("live", "doomed"))
    complete = sum(graph.monitor_for(n).usage.tokens for n in graph.node_names())
    assert partial == 100
    assert complete == 150
    assert ROOT in graph.node_names()


def test_p4_per_node_tightness_at_the_frozen_in_flow():
    """The per-node half: an abandoned node is caught one unit past its
    frozen pre-refund in-flow, independently of what its parent does."""
    graph = DelegationGraph(make_root(100))
    graph.add_node("doomed")
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()
    graph.monitor_for("doomed").usage.add_tokens(10)
    graph.abandon("doomed")

    graph.monitor_for("doomed").usage.add_tokens(50)  # exactly 60
    graph.verify()
    graph.monitor_for("doomed").usage.add_tokens(1)
    with pytest.raises(FlowConservationError):
        graph.verify()


# --------------------------------------------------------------------------
# P5 -- reclamation is confluent
# --------------------------------------------------------------------------


def _two_parent_graph(a=40, b=40, ad=10, bd=10, consumed=0):
    """Fan-in graph with consumption recorded at the shared child.

    `consumed` is not decoration. With zero consumption the refund pool equals
    the sum of the originals and every share collapses to the edge's own
    amount under *either* refund rule, so the artifact cannot tell the two
    apart. Consumption at `d` is what makes the proportional split observable.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=a)
    graph.allocate(ROOT, "b", tokens=b)
    graph.allocate("a", "d", tokens=ad)
    graph.allocate("b", "d", tokens=bd)
    graph.seal()
    if consumed:
        graph.monitor_for("d").usage.add_tokens(consumed)
    return graph


def test_p5_release_order_does_not_change_residuals():
    results = []
    for order in itertools.permutations([("a", "d"), ("b", "d")]):
        graph = _two_parent_graph(ad=15, bd=5, consumed=5)
        for src, dst in order:
            graph.release(src, dst)
        results.append({n: graph.residual(n).tokens for n in graph.node_names()})
    assert all(r == results[0] for r in results), results


def test_p5_the_asymmetric_case_discriminates_between_refund_rules():
    """Guard that the fixture reaches the regime where the rules differ.

    With unequal in-edges and nonzero consumption the two parents' refunds are
    unequal; with the degenerate ad == bd, consumed == 0 fixture they are not,
    and a live-value implementation passes the confluence test unchanged.
    """
    graph = _two_parent_graph(ad=15, bd=5, consumed=5)
    before = {n: graph.residual(n).tokens for n in graph.node_names()}
    graph.release("a", "d")
    graph.release("b", "d")
    after = {n: graph.residual(n).tokens for n in graph.node_names()}
    gained = {n: after[n] - before[n] for n in ("a", "b")}
    # Pool = 20 granted - 5 consumed = 15, split 15:5 -> 11.25 and 3.75, each
    # truncated to int. The unequal shares are the point; the missing token is
    # truncation, and it is asserted rather than rounded away.
    assert gained == {"a": 11, "b": 3}, gained
    assert gained["a"] != gained["b"]


def test_p5_live_value_refunds_are_order_dependent():
    """The converse half, simulated with a self-consistent live rule.

    An earlier version prorated a live numerator against a frozen pool of 20,
    which is not a rule anyone would implement; against a consistent live rule
    the symmetric case is order-INDEPENDENT, so the counterexample has to use
    the asymmetric one.
    """

    def simulate(order, originals, consumed):
        """`_refund_share` with `in_flow`/`edge.amount` for `original_*`.

        A release shrinks its edge by the refunded amount rather than removing
        it, exactly as `release()` does; the earlier version deleted the edge,
        which made both orders coincide and the test vacuous.
        """
        live = dict(originals)
        refunds = {}
        for edge in order:
            in_flow = sum(live.values())
            pool = in_flow - consumed
            share = live[edge] / in_flow * pool if in_flow else 0.0
            refunds[edge] = share
            live[edge] -= share
        return refunds

    originals = {("a", "d"): 15, ("b", "d"): 5}
    forward = simulate([("a", "d"), ("b", "d")], originals, 5)
    reverse = simulate([("b", "d"), ("a", "d")], originals, 5)
    assert forward != reverse, (forward, reverse)
    assert round(forward[("b", "d")], 2) == 2.14
    assert round(reverse[("b", "d")], 2) == 3.75


def test_p5_confluence_holds_on_random_two_parent_graphs():
    rng = random.Random(20260824)
    for trial in range(200):
        a = rng.randint(10, 60)
        b = rng.randint(10, 100 - a)
        ad = rng.randint(2, a)
        bd = rng.randint(2, b)
        params = {
            "a": a,
            "b": b,
            "ad": ad,
            "bd": bd,
            # Consume a real fraction of d's pooled in-flow so the refund
            # shares are proportional rather than degenerate.
            "consumed": rng.randint(1, ad + bd - 1),
        }
        base = None
        for order in itertools.permutations([("a", "d"), ("b", "d")]):
            graph = _two_parent_graph(**params)
            for src, dst in order:
                graph.release(src, dst)
            snapshot = {n: graph.residual(n).tokens for n in graph.node_names()}
            if base is None:
                base = snapshot
            assert snapshot == base, (trial, params, order, snapshot, base)


# --------------------------------------------------------------------------
# P6 -- the two halves of the invariant separate by locality
# --------------------------------------------------------------------------


def test_p6_locality_separation():
    """A node-local check may read only:
      - its own contract (materialized from its summed in-flow), and
      - its own recorded usage.
    It may NOT enumerate its out-edges, read another node's state, or
    consult the graph. Claim: under this definition, `consumption <=
    in-flow` is decidable and `consumption + out-flow <= in-flow` is not.
    """
    graph = DelegationGraph(make_root(100))
    graph.add_node("w")
    graph.allocate(ROOT, "w", tokens=40)
    graph.seal()
    monitor = graph.monitor_for("w")
    monitor.usage.add_tokens(41)
    # check_constraints() -> list[ViolationInfo] (monitor.py:311). It NEVER
    # returns None, so assert on emptiness, not identity.
    assert monitor.check_constraints() != []  # local knowledge suffices

    graph2 = DelegationGraph(make_root(100))
    for name in ("w", "child"):
        graph2.add_node(name)
    graph2.allocate(ROOT, "w", tokens=40)
    graph2.allocate("w", "child", tokens=35)
    graph2.seal()
    m = graph2.monitor_for("w")
    m.usage.add_tokens(30)  # 30 consumed + 35 delegated = 65 > 40
    assert m.check_constraints() == []  # the monitor sees no violation
    with pytest.raises(FlowConservationError):
        graph2.check_node("w")  # only the graph can see it


def test_p6_materializing_from_residual_would_collapse_the_separation():
    """The separation is a property of the materialization choice.

    `allocate()` is build-phase only and `release`/`abandon` can only shrink an
    edge, so out-flow is a static upper bound the instant `seal()` returns. Had
    `contract_for` used the residual, w's own monitor would flag the very spend
    it currently misses. P6 is therefore a claim about materializing from
    in-flow, not an unconditional impossibility -- and in-flow is the right
    choice because a node's grant should not shrink as it delegates.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("w", "child"):
        graph.add_node(name)
    graph.allocate(ROOT, "w", tokens=40)
    graph.allocate("w", "child", tokens=35)
    graph.seal()

    hypothetical = ResourceMonitor(ResourceConstraints(tokens=graph.residual("w").tokens))
    assert graph.residual("w").tokens == 5
    hypothetical.usage.add_tokens(30)
    assert hypothetical.check_constraints() != []  # would be locally decidable

    actual = graph.monitor_for("w")
    actual.usage.add_tokens(30)
    assert actual.check_constraints() == []  # as materialized, it is not


# --------------------------------------------------------------------------
# P7 -- the abandonment trilemma
# --------------------------------------------------------------------------
#
# P4 says the certified bound degrades to `B(root) + Σ refunds`, and P6 says a
# node cannot see its own graph state. Together they are not two limitations
# but one: under an unreliable failure detector, no scheme keeps an abandoned
# node's budget both safe and reusable without waiting on the node it just
# declared dead.
#
# Assumptions, both load-bearing and both tested below:
#   A1 (asynchrony)  abandonment cannot distinguish a crashed node from a slow
#                    one, so an abandoned node may still be running.
#   A2 (in-flight)   consumption may be committed externally before it is
#                    recorded locally, so a local gate cannot retract it.


def _abandonment_fixture():
    """root(100) -> live(40), doomed(60); live spends 40, doomed spends 10."""
    graph = DelegationGraph(make_root(100))
    for name in ("live", "doomed"):
        graph.add_node(name)
    graph.allocate(ROOT, "live", tokens=40)
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()
    graph.monitor_for("live").usage.add_tokens(40)
    graph.monitor_for("doomed").usage.add_tokens(10)
    return graph


def _total(graph):
    return sum(graph.monitor_for(n).usage.tokens for n in graph.node_names())


def test_p7_horn_a_safety_and_independence_strands_the_budget():
    """Never refund: safe and independent, but the 50 is unreachable.

    Nothing waits on `doomed` and nothing can overspend, because its unspent
    budget stays committed to its own in-edge -- and therefore stays unusable
    by anyone else. Liveness is what this horn gives up.
    """
    graph = _abandonment_fixture()
    assert graph.residual(ROOT).tokens == 0  # the parent cannot reach it
    graph.monitor_for("doomed").usage.add_tokens(50)
    assert _total(graph) == 100
    graph.verify()


def test_p7_horn_b_safety_and_liveness_requires_the_abandoned_node_to_stop():
    """Refund and reuse: safe and live only because `doomed` truly stopped.

    Establishing that premise means obtaining an acknowledgement from a node
    that was abandoned precisely because it stopped answering. Independence is
    what this horn gives up.
    """
    graph = _abandonment_fixture()
    refund = graph.abandon("doomed").tokens
    assert refund == 50
    assert graph.residual(ROOT).tokens == 50  # now reusable
    graph.monitor_for(ROOT).usage.add_tokens(refund)
    assert _total(graph) == 100  # doomed spends nothing more
    graph.verify()


def test_p7_horn_c_liveness_and_independence_admits_exactly_the_refund():
    """The shipped scheme: nothing blocks, budget is reusable, and a node that
    was slow rather than dead spends its frozen allowance anyway."""
    graph = _abandonment_fixture()
    refund = graph.abandon("doomed").tokens
    graph.monitor_for("doomed").usage.add_tokens(50)  # A1: it was only slow
    graph.monitor_for(ROOT).usage.add_tokens(refund)
    assert _total(graph) == 150
    graph.verify()  # and the over-spend is *certified*


def test_p7_the_overspend_is_linear_in_the_fraction_refunded():
    """The trilemma is a continuous frontier, not a binary choice.

    Refunding a fraction of the abandoned budget buys exactly that much
    reusability and admits exactly that much over-spend: the exchange rate is
    1:1, so there is no partial-refund policy that escapes the tradeoff.
    """
    observed = []
    for abandoned in range(5):
        graph = DelegationGraph(make_root(100))
        graph.add_node("live")
        graph.allocate(ROOT, "live", tokens=40)
        for i in range(4):
            graph.add_node(f"d{i}")
            graph.allocate(ROOT, f"d{i}", tokens=15)
        graph.seal()
        graph.monitor_for("live").usage.add_tokens(40)
        refunded = sum(graph.abandon(f"d{i}").tokens for i in range(abandoned))
        for i in range(4):
            graph.monitor_for(f"d{i}").usage.add_tokens(15)
        graph.monitor_for(ROOT).usage.add_tokens(refunded)
        graph.verify()
        observed.append((refunded, _total(graph) - 100))
    assert observed == [(0, 0), (15, 15), (30, 30), (45, 45), (60, 60)]


def test_p7_the_gap_is_bounded_by_the_refund_not_unbounded():
    """The positive half: the damage is exactly Σ refunds and no more."""
    graph = _abandonment_fixture()
    refund = graph.abandon("doomed").tokens
    graph.monitor_for("doomed").usage.add_tokens(50)
    graph.monitor_for(ROOT).usage.add_tokens(refund)
    graph.verify()
    graph.monitor_for(ROOT).usage.add_tokens(1)
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_p7_in_flight_consumption_is_load_bearing():
    """A2 is an assumption the result needs, not decoration.

    Drop it -- assume every unit of consumption passes a local gate before
    being committed -- and re-materializing the abandoned node's contract at
    its consumed-so-far restores safety with liveness and independence intact.
    The trilemma holds because an LLM call already dispatched is billed
    whatever the monitor subsequently says.
    """
    graph = _abandonment_fixture()
    refund = graph.abandon("doomed").tokens

    gated = ResourceMonitor(ResourceConstraints(tokens=10))
    gated.usage.add_tokens(10)  # exactly what doomed had consumed
    gated.usage.add_tokens(1)
    assert gated.check_constraints() != []  # the gate refuses the next unit
    assert 40 + 10 + refund == 100  # ... and safety is restored

    # With A2 the same execution is unbounded by that gate.
    graph.monitor_for("doomed").usage.add_tokens(50)
    graph.monitor_for(ROOT).usage.add_tokens(refund)
    assert _total(graph) == 150


# --------------------------------------------------------------------------
# P8 -- what "no global lock" does and does not cover
# --------------------------------------------------------------------------


def test_p8_concurrent_consumption_at_distinct_nodes_needs_no_graph_lock():
    """The corollary of P1, exercised rather than asserted.

    Monitors are per node and disjoint, and `ResourceUsage` guards its own
    counters, so agents record usage concurrently with no cross-node
    coordination and no central accountant.
    """
    adds = 5000
    names = [f"n{i}" for i in range(8)]
    graph = DelegationGraph(make_root(100_000))
    for name in names:
        graph.add_node(name)
        graph.allocate(ROOT, name, tokens=10_000)
    graph.seal()

    def consume(name):
        monitor = graph.monitor_for(name)
        for _ in range(adds):
            monitor.usage.add_tokens(1)

    threads = [threading.Thread(target=consume, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [graph.monitor_for(n).usage.tokens for n in names] == [adds] * 8
    graph.verify()


def test_p8_concurrent_allocation_cannot_breach_the_root_budget():
    """Topology mutation IS synchronized, and must be.

    `allocate` validates conservation and then mutates. Unsynchronized, that
    check-then-act over-granted in 190 of 300 trials, reaching 140 tokens
    against a root budget of 100 -- a breach of the framework's central
    guarantee at build time. Regression guard for the per-graph lock.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)  # force preemption inside the critical section
    try:
        for _ in range(50):
            graph = DelegationGraph(make_root(100))
            for i in range(8):
                graph.add_node(f"n{i}")
            barrier = threading.Barrier(8)

            def allocate(i, g=graph, b=barrier):
                b.wait()
                # Refusals are the correct outcome for the losers of the race;
                # the assertion below is on what was actually granted.
                with contextlib.suppress(ConservationViolationError):
                    g.allocate(ROOT, f"n{i}", tokens=20)  # 8 x 20 = 160 > 100

            threads = [threading.Thread(target=allocate, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            granted = sum(graph.in_flow(f"n{i}").tokens for i in range(8))
            assert granted <= 100, granted
    finally:
        sys.setswitchinterval(previous)


def test_p8_the_graph_lock_is_reentrant():
    """`abandon` releases the edges it unwinds, so a non-reentrant lock would
    self-deadlock on the very path abandonment exists to serve."""
    graph = DelegationGraph(make_root(100))
    for name in ("a", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=60)
    graph.allocate("a", "d", tokens=30)
    graph.seal()
    graph.monitor_for("d").usage.add_tokens(10)
    assert graph.abandon("d").tokens == 20
    graph.verify()
