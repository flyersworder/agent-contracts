"""Coordination metrics for the M6 ladder.

Rungs 1 and 2 both run two scouts against a shared budget. What separates
them is whether role differentiation buys exploration diversity that plain
ensembling does not, so the ladder needs a measure of how much the two scouts
actually diverged. That is what this module provides.

See docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md
"""

from __future__ import annotations

import math

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation_graph import DelegationGraph


def overlap_fraction(chosen_a: list[str], chosen_b: list[str]) -> float | None:
    """Fraction of the smaller selection that also appears in the other.

    ``|A ∩ B| / min(|A|, |B|)``. The minimum is the denominator so that a
    subset relationship reads as total overlap: a scout whose every pick was
    also made by its partner contributed no new experiments, whatever the
    partner did besides.

    Returns ``None`` -- not ``0.0`` -- when either side is empty. Zero is the
    H-B success case, two perfectly disjoint scouts; a cell where a scout got
    no picks at all is a degenerate run and must stay distinguishable from it,
    or the analyzer averages the artifact into the headline.

    Compares sets, so a scout repeating itself within its own loop does not
    register as agreement with its partner.
    """
    if not chosen_a or not chosen_b:
        return None
    shared = len(set(chosen_a) & set(chosen_b))
    return shared / min(len(set(chosen_a)), len(set(chosen_b)))


def mean_pairwise_overlap(chosen: list[list[str]]) -> float | None:
    """`overlap_fraction` averaged over every pair of scouts.

    Reduces to `overlap_fraction` at two scouts. ``None`` if any scout has
    no picks, for the same reason as the pairwise version: a degenerate cell
    must not read as perfect divergence.
    """
    if len(chosen) < 2:
        return None
    pairs: list[float] = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            o = overlap_fraction(chosen[i], chosen[j])
            if o is None:
                return None
            pairs.append(o)
    return sum(pairs) / len(pairs)


def scout_names(n: int) -> tuple[str, ...]:
    """``("scout_a", "scout_b", ...)`` -- lettered so two-scout columns keep their names.

    At least two: a fan-in with one scout is a loop, and the reconcile
    prompt, the calibration and the record all read `scout_b`. Rejected here
    so it fails before any budget is spent, not after.
    """
    if n < 2 or n > 26:
        raise ValueError(f"n_scouts must be in 2..26, got {n}")
    return tuple(f"scout_{chr(ord('a') + i)}" for i in range(n))


def split_budget(k: int, n: int) -> tuple[int, ...]:
    """Near-equal split of ``k`` over ``n`` scouts, remainder to the earlier ones.

    ``(ceil(k/2), k//2)`` at ``n=2``, which is the convention every two-scout
    arm and the planner/reasoner split already use.
    """
    base, rem = divmod(k, n)
    return tuple(base + (1 if i < rem else 0) for i in range(n))


def build_fan_in_graph(
    k: int,
    c95: int,
    a95: int,
    c95_b: int | None = None,
    fixed_overhead: int = 0,
    multiple: int = 2,
    *,
    scout_c95s: tuple[int, ...] | None = None,
) -> DelegationGraph:
    """Budget graph for the fan-in rungs: two scouts feeding one aggregator.

    This is the topology a tree cannot express (whitepaper §4.6 P2). The
    aggregator is funded by both scouts, and its reconciliation call is a
    single indivisible request larger than either scout's forward -- exactly
    the ``max_i a_i < c <= sum_i a_i`` regime that no tree encoding admits.

    Args:
        k: Total intervention budget, split between the scouts. An odd budget
            gives the remainder to ``scout_a``.
        c95: 95th-percentile tokens for one of ``scout_a``'s selection calls.
        a95: 95th-percentile tokens for one aggregation call.
        c95_b: Same for ``scout_b``; defaults to ``c95``. The roles are NOT
            interchangeable — measured through the production provider order,
            the targeted role costs 4.7x the plain one (10,379 median tokens
            against 2,205), so one shared figure under-budgets whichever scout
            reasons harder and produces conservation violations that are
            calibration artifacts rather than real overruns.
        fixed_overhead: Per-scout tokens for calls outside the selection loop,
            such as the team arm's two negotiation rounds.
        multiple: Provisioning margin over the per-call figure, applied
            uniformly to every role. A single stated rule, never tuned per
            arm: tuning until every arm certifies would make H-C vacuous.
        scout_c95s: One c95 per scout, for arms with other than two scouts
            (the three-agent ablation). When given it defines the scout
            count and ``c95``/``c95_b`` are ignored; ``(c95, c95_b)`` is
            the legacy two-scout call, byte-identical in its allocations.

    Returns:
        A sealed graph with nodes ``scout_a``, ``scout_b``, ...,
        ``aggregator``.
    """
    # Each scout forwards 0.75*a95, so the aggregator holds 1.5*a95 -- a 50%
    # margin over the 95th-percentile aggregation call -- while NEITHER scout
    # alone can fund it. That inequality is the point of the arm, not an
    # accident of budgeting: it puts the reconciliation call inside P2's
    # incompleteness window, `max_i a_i < c <= sum_i a_i`, where the DAG law
    # admits the call and no tree encoding does.
    #
    # An earlier draft forwarded 1.5*a95 EACH. That funds the aggregator to
    # 3*a95 and leaves every single fragment (1.5*a95) already larger than the
    # call it has to make -- a tree encoding would have succeeded, the arm
    # would have demonstrated nothing about P2, and the aggregator would have
    # been over-provisioned 2x besides.
    if scout_c95s is None:
        scout_c95s = (c95, c95 if c95_b is None else c95_b)
    n = len(scout_c95s)
    names = scout_names(n)
    budgets = split_budget(k, n)
    # The aggregator holds 1.5*a95 however many scouts feed it; each forwards
    # an equal share, so at n=2 this is the ceil(0.75*a95) of the original
    # design and at any n every single fragment stays below the call.
    forward = math.ceil(1.5 * a95 / n)
    tokens = tuple(
        math.ceil(multiple * c * b) + forward + fixed_overhead
        for c, b in zip(scout_c95s, budgets, strict=True)
    )
    # The root must fund whichever scout is dearer, n times over, or sealing
    # fails before a single cell runs.
    scout_tokens = max(tokens)
    root = Contract(
        id=f"m6-root-k{k}",
        name="M6 root",
        resources=ResourceConstraints(
            tokens=n * scout_tokens, per_tool_limits={"intervene": k, "observe": 0}
        ),
    )
    graph = DelegationGraph(root)
    for name in (*names, "aggregator"):
        graph.add_node(name)

    # `tool_invocations` MUST be explicit on every edge. `allocate()` defaults
    # every unspecified dimension to 0 -- not None -- and `can_use_tool`
    # checks the aggregate branch before the per-tool one, where
    # `tool_invocations is not None and usage >= tool_invocations` is already
    # True at zero usage. An omitted grant therefore blocks the node's very
    # first tool call, before any per-tool budget is consulted. Verified by
    # execution: without this, a freshly sealed graph reports
    # `monitor_for("scout_a").can_use_tool("intervene") is False`, every cell
    # returns empty, and H-C inverts from 100% compliance to 100% failure.
    for name, t, b in zip(names, tokens, budgets, strict=True):
        graph.allocate(
            DelegationGraph.ROOT,
            name,
            tokens=t,
            tool_invocations=b,
            per_tool={"intervene": b, "observe": 0},
        )
    for scout in names:
        graph.allocate(
            scout,
            "aggregator",
            tokens=forward,
            tool_invocations=0,  # the aggregator makes no chamber tool calls
            # Both keys, both zero. An omitted per-tool key means
            # *unconstrained*, and `_require_per_tool_propagation`
            # short-circuits on `granted == 0`, so a zero on an unknown key
            # raises nothing while the real key stays unbounded.
            per_tool={"intervene": 0, "observe": 0},
        )
    graph.seal()
    return graph
