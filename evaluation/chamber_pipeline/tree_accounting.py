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


def _in_edge_tokens(graph: DelegationGraph, node: str) -> list[int]:
    return [e.amount.tokens or 0 for e in graph.edges() if e.target == node]


def dag_capacity(graph: DelegationGraph, node: str) -> int:
    """``sum_i a_i`` -- the largest indivisible call the DAG law admits.

    The node's contract is materialized from its summed in-flow, so a single
    request is charged once against the pooled total.
    """
    return sum(_in_edge_tokens(graph, node))


def max_tree_fragment(graph: DelegationGraph, node: str) -> int:
    """``max_i a_i`` -- the largest indivisible call any tree encoding admits.

    Under the split encoding the node is represented by one fragment per
    parent, and an indivisible call must be charged to exactly one of them.
    The best a tree can do is therefore its single largest in-edge.
    """
    incoming = _in_edge_tokens(graph, node)
    return max(incoming) if incoming else 0


def fragmentation_factor(graph: DelegationGraph, node: str) -> float:
    """How much capacity the tree encoding forfeits: ``sum_i a_i / max_i a_i``.

    Equals ``m`` for ``m`` equal parents -- the factor-of-m penalty P2
    quantifies -- and 1.0 for a single-parent node, where the encodings agree.
    """
    largest = max_tree_fragment(graph, node)
    if largest == 0:
        return 1.0
    return dag_capacity(graph, node) / largest


def tree_would_refuse(graph: DelegationGraph, node: str, tokens: int) -> bool | None:
    """Would every tree encoding have refused this indivisible call?

    Returns ``None`` -- undefined, not ``False`` -- in the two cases where the
    comparison carries no evidence:

    * the node has fewer than two parents, so P2 does not apply at all;
    * the call exceeds ``sum_i a_i``, where the DAG cannot fund it either, so
      a tree's refusal says nothing about the encoding.

    Collapsing either case to ``False`` would let cells that tested nothing
    dilute the reported refusal rate.
    """
    incoming = _in_edge_tokens(graph, node)
    if len(incoming) < 2:
        return None
    if tokens > sum(incoming):
        return None
    return tokens > max(incoming)
