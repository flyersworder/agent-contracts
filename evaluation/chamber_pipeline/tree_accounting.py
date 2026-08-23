"""Measuring whitepaper §4.6 P2 per cell: where a tree encoding would refuse.

P2 as first drafted claimed tree accounting is *unsound* on a DAG. That claim
was withdrawn after it was checked against this framework's own tree law:
`ContractingCapability` resolves a multi-parent node by splitting it, and the
split encoding is sound -- it refuses the over-commitment the claim relied on.
Only a *drop*-policy accountant, which no real implementation uses, is unsound.

The surviving result is **incompleteness**. For a node ``v`` with ``m >= 2``
in-edges and an indivisible consumption ``c``:

    max_i a(u_i -> v)  <  c  <=  sum_i a(u_i -> v)

is admitted by the DAG's local invariant and by no tree encoding of the same
grants -- by merge (no parent holds the sum), by split (``c`` exceeds every
fragment), or soundly by drop. This module turns that inequality into a
per-cell measurement.

Nothing here re-runs the counterfactual: `DelegationGraph.allocate` refuses
over-commitment by construction, so the tree encoding is modelled explicitly
from the edge amounts rather than simulated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_contracts.core.delegation_graph import DelegationGraph


def _in_edge_tokens(graph: DelegationGraph, node: str) -> list[int | None]:
    """In-edge token grants, preserving `None`.

    `None` is `ResourceVector`'s *unbounded* sentinel, never zero. Collapsing
    it to 0 would report an unbounded parent as contributing nothing: a node
    with one unbounded edge and one 6,418-token edge would get
    `max_tree_fragment == 6418`, and with two unbounded edges every positive
    call would read as refused by every tree encoding.
    """
    return [e.amount.tokens for e in graph.edges() if e.target == node]


def _is_unbounded(values: list[int | None]) -> bool:
    return any(v is None for v in values)


def dag_capacity(graph: DelegationGraph, node: str) -> int | None:
    """``sum_i a_i`` -- the largest indivisible call the DAG law admits.

    The node's contract is materialized from its summed in-flow, so a single
    request is charged once against the pooled total. ``None`` if any in-edge
    is unbounded.
    """
    values = _in_edge_tokens(graph, node)
    if _is_unbounded(values):
        return None
    return sum(v for v in values if v is not None)


def max_tree_fragment(graph: DelegationGraph, node: str) -> int | None:
    """``max_i a_i`` -- the largest indivisible call any tree encoding admits.

    Under the split encoding the node is represented by one fragment per
    parent, and an indivisible call must be charged to exactly one of them.
    The best a tree can do is therefore its single largest in-edge.
    """
    incoming = _in_edge_tokens(graph, node)
    if _is_unbounded(incoming):
        return None
    return max((v for v in incoming if v is not None), default=0)


def fragmentation_factor(graph: DelegationGraph, node: str) -> float:
    """How much capacity the tree encoding forfeits: ``sum_i a_i / max_i a_i``.

    Equals ``m`` for ``m`` equal parents -- the factor-of-m penalty P2
    quantifies -- and 1.0 for a single-parent node, where the encodings agree.
    """
    largest = max_tree_fragment(graph, node)
    capacity = dag_capacity(graph, node)
    if largest is None or capacity is None or largest == 0:
        # An unbounded fragment forfeits nothing; neither does a node with no
        # funding at all.
        return 1.0
    return capacity / largest


def tree_would_refuse(graph: DelegationGraph, node: str, tokens: int) -> bool | None:
    """Would every tree encoding have refused this indivisible call?

    Returns ``None`` -- undefined, not ``False`` -- in the two cases where the
    comparison carries no evidence:

    * no call was made at all (``tokens <= 0``);
    * the node has fewer than two parents, so P2 does not apply at all;
    * the call exceeds ``sum_i a_i``, where the DAG cannot fund it either, so
      a tree's refusal says nothing about the encoding.

    Collapsing either case to ``False`` would let cells that tested nothing
    dilute the reported refusal rate.
    """
    if tokens <= 0:
        # No call was made (an early-return cell, or a provider response with
        # no usage block). Reporting False would record a cell that tested
        # nothing as positive evidence that a tree encoding would have coped.
        return None
    incoming = _in_edge_tokens(graph, node)
    if len(incoming) < 2 or _is_unbounded(incoming):
        # An unbounded parent could fund any call, so no tree encoding is
        # forced to refuse and the comparison carries no evidence.
        return None
    bounded = [v for v in incoming if v is not None]
    if tokens > sum(bounded):
        return None
    return tokens > max(bounded)
