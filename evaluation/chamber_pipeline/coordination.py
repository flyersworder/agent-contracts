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


def build_fan_in_graph(k: int, c95: int, a95: int) -> DelegationGraph:
    """Budget graph for the fan-in rungs: two scouts feeding one aggregator.

    This is the topology a tree cannot express (whitepaper §4.6 P2). The
    aggregator is funded by both scouts, and its reconciliation call is a
    single indivisible request larger than either scout's forward -- exactly
    the ``max_i a_i < c <= sum_i a_i`` regime that no tree encoding admits.

    Args:
        k: Total intervention budget, split between the scouts. An odd budget
            gives the remainder to ``scout_a``.
        c95: 95th-percentile tokens for one selection call.
        a95: 95th-percentile tokens for one aggregation call.

    Returns:
        A sealed graph with nodes ``scout_a``, ``scout_b``, ``aggregator``.
    """
    forward = math.ceil(1.5 * a95)
    scout_tokens = math.ceil(2 * c95 * math.ceil(k / 2)) + forward
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
        tokens=scout_tokens,
        tool_invocations=math.ceil(k / 2),
        per_tool={"intervene": math.ceil(k / 2), "observe": 0},
    )
    graph.allocate(
        DelegationGraph.ROOT,
        "scout_b",
        tokens=scout_tokens,
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
