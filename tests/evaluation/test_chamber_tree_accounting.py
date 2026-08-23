"""Turning whitepaper §4.6 P2 into a per-cell measurement.

P2 as originally drafted claimed tree accounting is *unsound* on a DAG. That
was withdrawn: checked against this framework's own `ContractingCapability`,
the natural (split) encoding is sound. The surviving result is
**incompleteness** -- for a node with m >= 2 in-edges, an indivisible
consumption `c` with `max_i a_i < c <= sum_i a_i` is admitted by the DAG law
and by no tree encoding.

So the measurable quantity is not an over-certification gap but a refusal:
how often does the aggregator's actual reconciliation call land in a window
where every tree encoding would have blocked it?
"""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline.coordination import build_fan_in_graph
from evaluation.chamber_pipeline.tree_accounting import (
    dag_capacity,
    fragmentation_factor,
    max_tree_fragment,
    tree_would_refuse,
)

A95 = 38752
C95 = 2303


def _graph(k: int = 30):
    return build_fan_in_graph(k=k, c95=C95, a95=A95)


def test_dag_capacity_pools_every_in_edge():
    graph = _graph()
    assert dag_capacity(graph, "aggregator") == graph.in_flow("aggregator").tokens


def test_max_tree_fragment_is_the_largest_single_parent_grant():
    graph = _graph()
    incoming = [e.amount.tokens for e in graph.edges() if e.target == "aggregator"]
    assert max_tree_fragment(graph, "aggregator") == max(incoming)
    assert max_tree_fragment(graph, "aggregator") < dag_capacity(graph, "aggregator")


def test_fragmentation_factor_is_the_penalty_p2_quantifies():
    """Two equal parents cost a factor of two in largest fundable call."""
    graph = _graph()
    assert fragmentation_factor(graph, "aggregator") == pytest.approx(2.0, abs=0.01)


def test_a_call_inside_the_window_is_refused_by_every_tree_encoding():
    graph = _graph()
    frag = max_tree_fragment(graph, "aggregator")
    cap = dag_capacity(graph, "aggregator")
    assert tree_would_refuse(graph, "aggregator", frag + 1) is True
    assert tree_would_refuse(graph, "aggregator", cap) is True


def test_a_call_a_single_parent_could_fund_is_not_evidence():
    graph = _graph()
    frag = max_tree_fragment(graph, "aggregator")
    assert tree_would_refuse(graph, "aggregator", frag) is False
    assert tree_would_refuse(graph, "aggregator", 1) is False


def test_a_call_the_dag_cannot_fund_either_is_not_evidence():
    """Above `sum_i a_i` both encodings fail, so the comparison says nothing."""
    graph = _graph()
    cap = dag_capacity(graph, "aggregator")
    assert tree_would_refuse(graph, "aggregator", cap + 1) is None


def test_single_parent_nodes_yield_none_not_false():
    """Undefined, not negative: a scout has one parent, so P2 does not apply."""
    graph = _graph()
    assert tree_would_refuse(graph, "scout_a", 10**9) is None
    assert fragmentation_factor(graph, "scout_a") == pytest.approx(1.0)


def test_the_real_aggregation_call_is_expected_to_land_in_the_window():
    """End-to-end: the arm as budgeted exercises P2 at a typical call size."""
    graph = _graph()
    assert tree_would_refuse(graph, "aggregator", A95) is True


def test_run_record_carries_the_p2_measurement_columns():
    """The columns must survive a Parquet round-trip alongside legacy rows."""
    import tempfile
    from pathlib import Path

    import pandas as pd

    from evaluation.chamber_pipeline.results import RunRecord, write_records_parquet

    graph = _graph()
    tokens = A95
    rec = RunRecord(
        chamber="lt",
        configuration="standard",
        agent_name="fan_in_spec",
        budget_k=30,
        budget_fraction=0.51,
        seed=0,
        status="ok",
        started_at="2026-08-23T00:00:00Z",
        finished_at="2026-08-23T00:01:00Z",
        overlap_frac=0.25,
        n_experiments_distinct=28,
        conservation_certified=True,
        aggregator_tokens=tokens,
        max_tree_fragment=max_tree_fragment(graph, "aggregator"),
        tree_would_refuse=tree_would_refuse(graph, "aggregator", tokens),
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "p2.parquet"
        write_records_parquet([rec], out)
        df = pd.read_parquet(out)
    assert bool(df.tree_would_refuse.iloc[0]) is True
    assert df.max_tree_fragment.iloc[0] == max_tree_fragment(graph, "aggregator")
    assert df.overlap_frac.iloc[0] == pytest.approx(0.25)
    assert bool(df.conservation_certified.iloc[0]) is True


def test_the_graph_refuses_to_create_an_unbounded_in_edge():
    """Why the `None` handling below can never bite in practice.

    `allocate()` rejects `tokens=None` outright, so no real graph reaches the
    unbounded case -- the same shape as P3's cycle refusal, where the hazard
    is made unconstructible rather than detected.
    """
    from agent_contracts.core.contract import Contract, ResourceConstraints
    from agent_contracts.core.delegation_graph import DelegationGraph

    root = Contract(id="u", name="u", resources=ResourceConstraints(tokens=None))
    graph = DelegationGraph(root)
    graph.add_node("p")
    with pytest.raises(ValueError, match="finite"):
        graph.allocate(DelegationGraph.ROOT, "p", tokens=None)


def test_helpers_treat_none_as_unbounded_not_zero():
    """`None` is ResourceVector's *unbounded* sentinel, never zero.

    Unreachable through `allocate` today, but these functions are exported for
    general graphs and `EdgeAllocation.amount` is typed to permit it.
    Collapsing `None` to 0 would report an unbounded parent as contributing
    nothing: one unbounded edge beside a 6,418-token edge would give
    `max_tree_fragment == 6418`, and two unbounded edges would make every
    positive call read as refused by every tree encoding.
    """
    from types import SimpleNamespace

    from evaluation.chamber_pipeline.tree_accounting import (
        dag_capacity,
        fragmentation_factor,
    )

    def edge(target, tokens):
        return SimpleNamespace(target=target, amount=SimpleNamespace(tokens=tokens))

    fake = SimpleNamespace(edges=lambda: [edge("agg", None), edge("agg", 6418)])
    assert max_tree_fragment(fake, "agg") is None
    assert dag_capacity(fake, "agg") is None
    assert fragmentation_factor(fake, "agg") == 1.0
    assert tree_would_refuse(fake, "agg", 7000) is None

    bounded = SimpleNamespace(edges=lambda: [edge("agg", 6418), edge("agg", 6418)])
    assert max_tree_fragment(bounded, "agg") == 6418
    assert tree_would_refuse(bounded, "agg", 7000) is True


def test_a_call_that_never_happened_is_not_evidence():
    """Zero tokens means the aggregator never ran, not that a tree coped."""
    graph = _graph()
    assert tree_would_refuse(graph, "aggregator", 0) is None
    assert tree_would_refuse(graph, "aggregator", -1) is None
