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


def build_fan_in_graph(
    k: int,
    c95: int,
    a95: int,
    c95_b: int | None = None,
    fixed_overhead: int = 0,
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

    Returns:
        A sealed graph with nodes ``scout_a``, ``scout_b``, ``aggregator``.
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
    forward = math.ceil(0.75 * a95)
    c95_b = c95 if c95_b is None else c95_b
    tokens_a = math.ceil(2 * c95 * math.ceil(k / 2)) + forward + fixed_overhead
    tokens_b = math.ceil(2 * c95_b * (k // 2)) + forward + fixed_overhead
    # The root must fund whichever scout is dearer, twice over, or sealing
    # fails before a single cell runs.
    scout_tokens = max(tokens_a, tokens_b)
    root = Contract(
        id=f"m6-root-k{k}",
        name="M6 root",
        resources=ResourceConstraints(
            tokens=2 * scout_tokens, per_tool_limits={"intervene": k, "observe": 0}
        ),
    )
    graph = DelegationGraph(root)
    for name in ("scout_a", "scout_b", "aggregator"):
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
    graph.allocate(
        DelegationGraph.ROOT,
        "scout_a",
        tokens=tokens_a,
        tool_invocations=math.ceil(k / 2),
        per_tool={"intervene": math.ceil(k / 2), "observe": 0},
    )
    graph.allocate(
        DelegationGraph.ROOT,
        "scout_b",
        tokens=tokens_b,
        tool_invocations=k // 2,
        per_tool={"intervene": k // 2, "observe": 0},
    )
    for scout in ("scout_a", "scout_b"):
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
