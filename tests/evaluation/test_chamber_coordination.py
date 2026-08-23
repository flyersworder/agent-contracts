"""Coordination metrics for the M6 ladder.

See docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md
"""

from __future__ import annotations

from evaluation.chamber_pipeline.coordination import overlap_fraction


def test_overlap_fraction_disjoint_is_zero():
    assert overlap_fraction(["a", "b"], ["c", "d"]) == 0.0


def test_overlap_fraction_identical_is_one():
    assert overlap_fraction(["a", "b"], ["a", "b"]) == 1.0


def test_overlap_fraction_uses_min_denominator():
    """A subset relationship is total overlap for the smaller selection."""
    assert overlap_fraction(["a", "b", "c"], ["a"]) == 1.0


def test_overlap_fraction_empty_is_none_not_zero():
    """Undefined, not zero.

    Zero is the H-B success case -- perfectly disjoint scouts. A cell where a
    scout got no picks at all must not be indistinguishable from it, or the
    analyzer averages the artifact into the headline.
    """
    assert overlap_fraction([], ["a"]) is None
    assert overlap_fraction(["a"], []) is None
    assert overlap_fraction([], []) is None


def test_overlap_fraction_ignores_within_scout_duplicates():
    """The metric is over sets: a scout repeating itself is not agreement."""
    assert overlap_fraction(["a", "a", "b"], ["a"]) == 1.0
    assert overlap_fraction(["a", "a"], ["b", "b"]) == 0.0


def test_overlap_fraction_is_symmetric():
    for a, b in ((["a", "b", "c"], ["b", "d"]), (["x"], ["x", "y", "z"])):
        assert overlap_fraction(a, b) == overlap_fraction(b, a)


def test_overlap_fraction_is_bounded():
    assert 0.0 <= overlap_fraction(["a", "b", "c"], ["b", "c", "d"]) <= 1.0


# --------------------------------------------------------------------------
# Task 4: the fan-in delegation graph
# --------------------------------------------------------------------------

import math  # noqa: E402

from evaluation.chamber_pipeline.coordination import build_fan_in_graph  # noqa: E402


def test_fan_in_graph_seals_and_funds_the_aggregator():
    graph = build_fan_in_graph(k=30, c95=2303, a95=38752)
    assert graph.is_sealed
    forward = math.ceil(0.75 * 38752)
    assert graph.in_flow("aggregator").tokens == 2 * forward
    assert graph.in_flow("scout_a").per_tool["intervene"] == 15
    assert graph.in_flow("scout_b").per_tool["intervene"] == 15


def test_aggregator_is_zeroed_on_both_chamber_tools():
    """`observe` needs the explicit zero as much as `intervene`.

    An omitted per-tool key means *unconstrained*, so without this the
    aggregator could acquire data outside the certified budget and the
    matched-budget control would be unenforceable.
    """
    per_tool = build_fan_in_graph(k=30, c95=2303, a95=38752).in_flow("aggregator").per_tool
    assert per_tool["intervene"] == 0
    assert per_tool["observe"] == 0


def test_scout_monitors_permit_their_first_intervention():
    """Regression: allocate() defaults tool_invocations to 0, not None.

    `can_use_tool` checks the aggregate branch first, and `usage >= 0` holds at
    zero usage, so an omitted grant blocks the node's very first tool call
    before any per-tool budget is consulted -- every cell would come back empty
    and H-C would invert from 100% compliance to 100% failure.
    """
    graph = build_fan_in_graph(k=6, c95=1350, a95=21163)
    for scout in ("scout_a", "scout_b"):
        assert graph.monitor_for(scout).can_use_tool("intervene") is True
    assert graph.monitor_for("aggregator").can_use_tool("intervene") is False


def test_odd_budget_gives_the_remainder_to_scout_a():
    graph = build_fan_in_graph(k=45, c95=2778, a95=39191)
    assert graph.in_flow("scout_a").per_tool["intervene"] == 23
    assert graph.in_flow("scout_b").per_tool["intervene"] == 22


def test_fan_in_graph_conserves_the_intervention_budget():
    """The matched-budget guarantee: scouts together hold exactly k."""
    for k in (6, 30, 45):
        graph = build_fan_in_graph(k=k, c95=2303, a95=38752)
        total = sum(graph.in_flow(s).per_tool["intervene"] for s in ("scout_a", "scout_b"))
        assert total == k, (k, total)


def test_fan_in_graph_verifies_before_any_consumption():
    build_fan_in_graph(k=30, c95=2303, a95=38752).verify()


def test_the_aggregation_call_lands_in_p2s_incompleteness_window():
    """The arm must actually exercise the proposition it exists to test.

    P2's window is `max_i a_i < c <= sum_i a_i`: the aggregator can afford its
    reconciliation call only by pooling both forwards. If either bound fails
    the arm demonstrates nothing -- below the window a single tree fragment
    suffices, above it not even the DAG can fund the call.
    """
    a95 = 38752
    graph = build_fan_in_graph(k=30, c95=2303, a95=a95)
    incoming = [e.amount.tokens for e in graph.edges() if e.target == "aggregator"]
    assert len(incoming) == 2  # genuinely multi-parent
    assert max(incoming) < a95, "a single scout could fund it; no tree failure"
    assert a95 <= sum(incoming), "not even the DAG could fund it"


def test_aggregator_funding_keeps_a_margin_over_the_p95_call():
    a95 = 38752
    graph = build_fan_in_graph(k=30, c95=2303, a95=a95)
    assert graph.in_flow("aggregator").tokens >= 1.5 * a95


def test_per_role_budgets_track_the_dearer_scout():
    """The roles are not interchangeable and must not share one figure.

    Measured through the production provider order, the targeted role costs
    4.7x the plain one. A shared `c95` under-funds whichever scout reasons
    harder, and the resulting conservation violations are calibration
    artifacts rather than real overruns.
    """
    graph = build_fan_in_graph(k=30, c95=5969, a95=8557, c95_b=18136)
    assert graph.in_flow("scout_b").tokens > graph.in_flow("scout_a").tokens
    graph.verify()


def test_fixed_overhead_funds_calls_outside_the_selection_loop():
    """The team arm's two negotiation rounds are per scout, not per pick."""
    plain = build_fan_in_graph(k=30, c95=2809, a95=8557)
    team = build_fan_in_graph(k=30, c95=2809, a95=8557, fixed_overhead=4 * 4138)
    for node in ("scout_a", "scout_b"):
        delta = team.in_flow(node).tokens - plain.in_flow(node).tokens
        assert delta == 4 * 4138


def test_symmetric_call_keeps_the_old_behaviour():
    """`c95_b=None` means both scouts are budgeted identically."""
    graph = build_fan_in_graph(k=30, c95=2809, a95=8557)
    assert graph.in_flow("scout_a").tokens == graph.in_flow("scout_b").tokens
