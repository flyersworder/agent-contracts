"""Flow-conservation delegation over a DAG.

Generalizes the tree conservation law in :mod:`agent_contracts.core.delegation`
to a directed acyclic graph whose edges carry resource allocations.

A node's budget is the sum of its in-edges. The invariant checked at every
node is::

    in-flow >= own consumption + out-flow

Summing this over all nodes telescopes: every internal allocation appears once
as in-flow at its head and once as out-flow at its tail, so only the root's
exogenous budget and total system consumption survive. Local checks therefore
imply the global bound ``sum(consumption) <= root budget``.

The *control* graph may contain cycles (a node retries, a council
re-deliberates); the *budget* graph must not, or the telescoping argument
collapses. Cycle-creating edges are rejected.

This module is not thread-safe. ``run_sweep`` is serial today; if parallelism
lands, ``allocate``/``release``/``abandon`` need a lock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_contracts.core.contract import Contract
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.core.resource_vector import ResourceVector


class CycleError(Exception):
    """Raised when an edge would create a cycle in the budget graph."""


class GraphLintError(Exception):
    """Raised by ``seal()`` when the graph fails validation.

    Reports every problem found, not just the first, so one pass fixes them all.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"graph failed validation:\n  - {joined}")


class FlowConservationError(ConservationViolationError):
    """Raised when a node's flow invariant is or would be violated.

    Subclasses :class:`ConservationViolationError` so existing handlers of the
    tree conservation law also catch flow violations.

    ``contributing_edges`` is an audit trail of who funded the node, not blame
    assignment: the invariant is checked at ``node_id``, so that node is at
    fault. Parents are accountable only for their own out-flow.
    """

    def __init__(
        self,
        message: str,
        node_id: str,
        dimension: str,
        in_flow: float | None,
        consumed: float,
        out_flow: float,
        deficit: float,
        contributing_edges: list[str],
    ):
        self.node_id = node_id
        self.dimension = dimension
        self.in_flow = in_flow
        self.consumed = consumed
        self.out_flow = out_flow
        self.deficit = deficit
        self.contributing_edges = contributing_edges
        # The inherited int fields round *away* from each other: a 1.9 overrun
        # against a 1.5 budget must not truncate to 1 vs 1 and read as no
        # violation at all.
        super().__init__(
            message,
            requested=math.ceil(out_flow + consumed),
            available=math.floor(in_flow) if in_flow is not None else 0,
            parent_id=node_id,
        )


@dataclass
class EdgeAllocation:
    """A budget allocation carried along one edge.

    ``amount`` is the live allocation and shrinks when the edge is released.
    ``original_amount`` is fixed at construction and is what proportional
    refunds are computed against — using the live amount would make refunds
    depend on the order siblings are released in.

    ``released`` records that ``release()`` has already run on this edge; it
    never removes the edge from flow accounting.
    """

    source: str
    target: str
    amount: ResourceVector
    original_amount: ResourceVector | None = None
    created_at: datetime = field(default_factory=datetime.now)
    released: bool = False

    def __post_init__(self) -> None:
        if self.original_amount is None:
            self.original_amount = self.amount

    @property
    def key(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class AbandonSnapshot:
    """A node's flow state at the moment it was abandoned.

    Abandonment refunds unconsumed budget upstream, which shrinks the dead
    node's live in-flow, and a later release of one of its out-edges shrinks
    its live out-flow. Checking an abandoned node against *live* values would
    therefore let those follow-on movements quietly clear an overspend that
    really happened. ``verify()`` checks abandoned nodes against this frozen
    triple instead, so an over-spent node stays flagged for good.
    """

    in_flow: ResourceVector
    consumed: ResourceVector
    out_flow: ResourceVector


@dataclass
class GraphNode:
    """A node in the delegation graph."""

    node_id: str
    name: str
    contract_kwargs: dict[str, Any] = field(default_factory=dict)
    contract: Contract | None = None
    monitor: ResourceMonitor | None = None
    abandoned: bool = False
    snapshot: AbandonSnapshot | None = None


class DelegationGraph:
    """Delegation as a DAG with flow conservation.

    Lifecycle is build -> seal -> run. Topology is frozen at ``seal()``.
    """

    ROOT = "root"

    def __init__(self, root_contract: Contract, root_monitor: ResourceMonitor | None = None):
        self.root_contract = root_contract
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, EdgeAllocation] = {}
        self._sealed = False

        root = GraphNode(node_id=root_contract.id, name=self.ROOT, contract=root_contract)
        root.monitor = root_monitor or ResourceMonitor(root_contract.resources)
        self._nodes[self.ROOT] = root

    # ---------------------------------------------------------------- build

    def add_node(self, name: str, **contract_kwargs: Any) -> str:
        """Register a node. It carries no budget until an edge feeds it.

        ``contract_kwargs`` accepts the same contract-shaping fields as
        ``ContractingCapability.create_subcontract`` (capabilities, execution,
        temporal, metadata) — everything except resources, which arrive via
        edges.
        """
        self._require_unsealed()
        if name in self._nodes:
            raise ValueError(f"node '{name}' already exists")
        node_id = f"{self.root_contract.id}/{name}"
        self._nodes[name] = GraphNode(
            node_id=node_id, name=name, contract_kwargs=dict(contract_kwargs)
        )
        return node_id

    def allocate(self, source: str, target: str, **resources: Any) -> EdgeAllocation:
        """Allocate budget along edge ``source -> target``.

        Raises:
            KeyError: if either node is unknown.
            CycleError: if the edge would create a cycle.
            ValueError: if any allocated dimension is unbounded.
            FlowConservationError: if source's own consumption plus its
                prospective out-flow would exceed its in-flow.
        """
        self._require_unsealed()
        self._require_node(source)
        self._require_node(target)

        per_tool = resources.pop("per_tool", {}) or {}
        amount = ResourceVector(
            tokens=resources.pop("tokens", 0),
            cost_usd=resources.pop("cost_usd", 0.0),
            tool_invocations=resources.pop("tool_invocations", 0),
            iterations=resources.pop("iterations", 0),
            per_tool=dict(per_tool),
        )
        if resources:
            raise TypeError(f"unexpected resource dimensions: {sorted(resources)}")
        if not amount.is_finite():
            raise ValueError("allocations must be finite; None (unbounded) cannot be allocated")

        if source == target:
            raise CycleError(f"self-edge '{source}' -> '{target}' is not allowed")
        if self._reaches(target, source):
            raise CycleError(f"edge '{source}' -> '{target}' would create a cycle")

        key = f"{source}->{target}"
        if key in self._edges:
            raise ValueError(f"edge '{key}' already exists")

        prospective_out = self.out_flow(source) + amount
        source_in_flow = self.in_flow(source)
        self._require_per_tool_propagation(source, amount, source_in_flow)
        source_consumed = self._consumed(source)
        if not source_consumed + prospective_out <= source_in_flow:
            self._raise_flow_error(
                source,
                in_flow=source_in_flow,
                consumed=source_consumed,
                out_flow=prospective_out,
                kind="allocation",
            )

        edge = EdgeAllocation(source=source, target=target, amount=amount)
        self._edges[key] = edge
        return edge

    def _require_per_tool_propagation(
        self, source: str, amount: ResourceVector, source_in_flow: ResourceVector
    ) -> None:
        """Reject a per-tool grant the source's own in-flow does not constrain.

        ``ResourceVector.__le__`` compares only the keys the *budget* side
        names — the convention ``AllocationRecord`` already documents — so a
        tool missing from a node's in-flow is unconstrained rather than zero.
        Without this rule a node funded with no experiment budget could grant
        experiments without limit, and the M6 conserved resource is exactly
        that per-tool dimension.

        The root is exempt: its budget is exogenous, so a tool it leaves
        unconstrained is genuinely unbounded rather than absent.
        """
        if source == self.ROOT:
            return
        for tool in sorted(amount.per_tool):
            if tool in source_in_flow.per_tool:
                continue
            granted = amount.per_tool[tool]
            raise FlowConservationError(
                f"node '{source}' cannot grant tool '{tool}': its in-flow does not "
                f"constrain that tool, so the grant would be unbounded downstream",
                node_id=source,
                dimension=f"tool:{tool}",
                in_flow=0,
                consumed=0,
                out_flow=granted,
                deficit=granted,
                contributing_edges=self.contributing_edges(source),
            )

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        """Validate the whole graph and freeze its topology.

        Checks: every non-root node has at least one in-edge; no node's
        out-flow exceeds its in-flow; no node names a per-tool budget in its
        out-edges that its own in-flow does not constrain (the root excepted —
        its budget is exogenous). Acyclicity is maintained incrementally by
        ``allocate()``, which also rejects the per-tool case edge by edge; the
        lint here is the independent whole-graph check.

        Idempotent: sealing an already-sealed graph is a no-op.
        """
        if self._sealed:
            return

        problems: list[str] = []
        for name in self._nodes:
            if name == self.ROOT:
                continue
            if not self.contributing_edges(name):
                problems.append(f"node '{name}' is an orphan: no in-edge funds it")

        for name in self._nodes:
            in_flow = self.in_flow(name)
            out = self.out_flow(name)
            if not out <= in_flow:
                problems.append(f"node '{name}' out-flow exceeds in-flow")
            if name == self.ROOT:
                continue
            for tool in sorted(out.per_tool):
                if tool not in in_flow.per_tool:
                    problems.append(
                        f"node '{name}' grants tool '{tool}' that its in-flow does not constrain"
                    )

        if problems:
            raise GraphLintError(problems)
        self._sealed = True

    # ------------------------------------------------------------- queries

    def in_flow(self, name: str) -> ResourceVector:
        """Total live allocation arriving at ``name``. For the root, its own budget.

        Released edges are *not* excluded: ``release()`` shrinks an edge's
        ``amount`` by the refunded share, so the remaining amount is still real
        budget the target holds.
        """
        self._require_node(name)
        if name == self.ROOT:
            return ResourceVector.from_constraints(self.root_contract.resources)
        total = ResourceVector.ZERO
        for edge in self._edges.values():
            if edge.target == name:
                total = total + edge.amount
        return total

    def original_in_flow(self, name: str) -> ResourceVector:
        """Total allocation arriving at ``name`` as originally granted.

        Refund shares are computed against this, not ``in_flow``, so that
        sibling releases are order-independent.
        """
        self._require_node(name)
        total = ResourceVector.ZERO
        for edge in self._edges.values():
            if edge.target == name and edge.original_amount is not None:
                total = total + edge.original_amount
        return total

    def out_flow(self, name: str) -> ResourceVector:
        """Total live allocation leaving ``name``."""
        self._require_node(name)
        total = ResourceVector.ZERO
        for edge in self._edges.values():
            if edge.source == name:
                total = total + edge.amount
        return total

    def node_names(self) -> list[str]:
        return list(self._nodes)

    def edges(self) -> list[EdgeAllocation]:
        return list(self._edges.values())

    def contributing_edges(self, name: str) -> list[str]:
        return [e.key for e in self._edges.values() if e.target == name]

    # ------------------------------------------------------------------ run

    def contract_for(self, name: str) -> Contract:
        """Materialize ``name``'s Contract with its summed in-flow as budget.

        The same object is returned on every call, so node-level monitoring
        stays attached to one contract.
        """
        self._require_sealed()
        self._require_node(name)
        node = self._nodes[name]
        if node.contract is None:
            node.contract = Contract(
                id=node.node_id,
                name=name,
                resources=self.in_flow(name).to_constraints(),
                **node.contract_kwargs,
            )
        return node.contract

    def monitor_for(self, name: str) -> ResourceMonitor:
        """Monitor for ``name``, bound to its materialized contract.

        Node-level enforcement is the existing ``ResourceMonitor`` machinery:
        because the contract carries the summed in-flow as its constraints,
        strict/lenient modes, callbacks, and per-tool priority all apply
        unchanged. The graph layer owns only the edges.
        """
        self._require_sealed()
        self._require_node(name)
        node = self._nodes[name]
        if node.monitor is None:
            node.monitor = ResourceMonitor(self.contract_for(name).resources)
        return node.monitor

    def residual(self, name: str) -> ResourceVector:
        """in-flow - own consumption - out-flow."""
        self._require_node(name)
        return self.in_flow(name) - self._consumed(name) - self.out_flow(name)

    def check_node(self, name: str) -> None:
        """Raise FlowConservationError if ``name``'s invariant is violated.

        An abandoned node is checked against its :class:`AbandonSnapshot` — the
        flow state it died in — rather than against live values, which the
        refund it triggered has since moved.
        """
        self._require_node(name)
        in_flow, consumed, out_flow = self._flow_state(name)
        if not consumed + out_flow <= in_flow:
            self._raise_flow_error(name, in_flow=in_flow, consumed=consumed, out_flow=out_flow)

    def verify(self) -> None:
        """Check the invariant at every node, abandoned nodes included.

        Abandoned nodes are *not* excused. Abandonment is the timeout case, and
        a timed-out node is the likeliest of all to have overspent; excusing it
        would let ``verify()`` certify a graph whose total consumption exceeds
        the root budget.
        """
        for name in self._nodes:
            self.check_node(name)

    def abandon_snapshot(self, name: str) -> AbandonSnapshot | None:
        """The flow state ``name`` was abandoned in, or ``None`` if it is live."""
        self._require_node(name)
        return self._nodes[name].snapshot

    def _flow_state(self, name: str) -> tuple[ResourceVector, ResourceVector, ResourceVector]:
        """``(in_flow, consumed, out_flow)`` to check ``name``'s invariant against."""
        snapshot = self._nodes[name].snapshot
        if snapshot is not None:
            return snapshot.in_flow, snapshot.consumed, snapshot.out_flow
        return self.in_flow(name), self._consumed(name), self.out_flow(name)

    def _require_sealed(self) -> None:
        if not self._sealed:
            raise RuntimeError("graph must be sealed before materializing contracts")

    # ------------------------------------------------------------ internal

    def _require_unsealed(self) -> None:
        if self._sealed:
            raise RuntimeError("graph is sealed; topology is frozen")

    def _require_node(self, name: str) -> None:
        if name not in self._nodes:
            raise KeyError(f"unknown node '{name}'")

    def _reaches(self, start: str, goal: str) -> bool:
        """True if ``goal`` is reachable from ``start`` following edges."""
        stack = [start]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == goal:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(e.target for e in self._edges.values() if e.source == current)
        return False

    def _raise_flow_error(
        self,
        name: str,
        *,
        in_flow: ResourceVector,
        consumed: ResourceVector,
        out_flow: ResourceVector,
        kind: str = "runtime",
    ) -> None:
        """Identify the first violated dimension and raise.

        ``consumed`` and ``out_flow`` are reported as the separate quantities
        they are: a node that delegated nothing and overspent must not be
        described as over-allocating. ``kind`` selects the phrasing —
        ``"allocation"`` for the build-phase check in :meth:`allocate`, where
        the out-flow is prospective, ``"runtime"`` for :meth:`check_node`.
        """
        verb = "would over-allocate" if kind == "allocation" else "violates flow conservation on"

        def raise_for(
            dimension: str,
            limit: float,
            used: float,
            allocated: float,
            what: str,
        ) -> None:
            raise FlowConservationError(
                f"node '{name}' {verb} {what}: consumption {used} + "
                f"out-flow {allocated} exceeds in-flow {limit}",
                node_id=name,
                dimension=dimension,
                in_flow=limit,
                consumed=used,
                out_flow=allocated,
                deficit=used + allocated - limit,
                contributing_edges=self.contributing_edges(name),
            )

        for dimension in ("tokens", "cost_usd", "tool_invocations", "iterations"):
            limit = getattr(in_flow, dimension)
            used = getattr(consumed, dimension) or 0
            allocated = getattr(out_flow, dimension) or 0
            if limit is not None and used + allocated > limit:
                raise_for(dimension, limit, used, allocated, dimension)
        for tool, limit in in_flow.per_tool.items():
            used = consumed.per_tool.get(tool, 0)
            allocated = out_flow.per_tool.get(tool, 0)
            if used + allocated > limit:
                raise_for(f"tool:{tool}", limit, used, allocated, f"tool '{tool}'")
        raise FlowConservationError(
            f"node '{name}' violates flow conservation",
            node_id=name,
            dimension="unknown",
            in_flow=None,
            consumed=0,
            out_flow=0,
            deficit=0,
            contributing_edges=self.contributing_edges(name),
        )

    def _consumed(self, name: str) -> ResourceVector:
        """Consumption recorded at ``name``. Zero before a monitor exists."""
        node = self._nodes[name]
        if node.monitor is None:
            return ResourceVector.ZERO
        return ResourceVector.from_usage(node.monitor.usage)

    # -------------------------------------------------------- reclamation

    def release(self, source: str, target: str) -> ResourceVector:
        """Refund edge ``source -> target``'s share of the target's unused budget.

        Shares are computed against *original* allocations, so releasing
        sibling edges in any order yields the same final state. Computing
        against live in-flow instead would make each sibling's refund depend on
        how many siblings released first.
        """
        key = f"{source}->{target}"
        edge = self._edges.get(key)
        if edge is None:
            raise KeyError(f"unknown edge '{key}'")
        if edge.released:
            raise ValueError(f"edge '{key}' already released")

        share = self._refund_share(edge, target)
        edge.amount = edge.amount - share
        edge.released = True
        return share

    def abandon(self, name: str) -> ResourceVector:
        """Mark ``name`` dead; refund its unconsumed budget to its parents.

        A node that times out or crashes otherwise leaves a stranded
        allocation that silently corrupts every downstream residual for the
        rest of a run — the failure mode the M4b pilot produced 8 times.

        Three limits are deliberate and must not be read as more than they are:

        * **Only in-edges are refunded.** Budget the dead node had already
          delegated downstream stays stranded at the child, which may still be
          running. Abandoning the child too is what reclaims it.
        * **Reclaimed budget is not re-delegatable in v1.** ``allocate()`` is
          build-phase only, so a refund changes accounting and reporting — the
          parent's residual, and what ``verify()`` will certify — but nothing
          can re-spend it within the sealed graph.
        * **The node is presumed to stop consuming.** Its invariant is frozen
          into an :class:`AbandonSnapshot` at this moment; consumption recorded
          against a node after it was declared dead is outside the model.
        """
        self._require_node(name)
        if name == self.ROOT:
            raise ValueError("cannot abandon the root node")
        node = self._nodes[name]
        if node.abandoned:
            raise ValueError(f"node '{name}' already abandoned")

        node.snapshot = AbandonSnapshot(
            in_flow=self.in_flow(name),
            consumed=self._consumed(name),
            out_flow=self.out_flow(name),
        )
        reclaimed = ResourceVector.ZERO
        for edge in list(self._edges.values()):
            # Skip edges already released: `_refund_share` computes the pool
            # from ORIGINAL in-flow, so a released edge would compute the same
            # share a second time, drive its amount negative, and double-credit
            # the parent. This is not the forbidden "filter flow queries on
            # released" — in_flow/out_flow stay unfiltered; only refunds skip.
            if edge.target == name and not edge.released:
                share = self._refund_share(edge, name)
                edge.amount = edge.amount - share
                edge.released = True
                reclaimed = reclaimed + share
        node.abandoned = True
        return reclaimed

    def is_reachable(self, name: str) -> bool:
        """False if ``name`` is abandoned or every path to it passes an abandoned node."""
        self._require_node(name)
        if self._nodes[name].abandoned:
            return False
        if name == self.ROOT:
            return True
        funders = [e.source for e in self._edges.values() if e.target == name]
        return any(self.is_reachable(source) for source in funders)

    def _refund_share(self, edge: EdgeAllocation, target: str) -> ResourceVector:
        """Edge's proportional share of ``target``'s unused budget."""
        originals = self.original_in_flow(target)
        pool = originals - self._consumed(target) - self.out_flow(target)
        assert edge.original_amount is not None  # set in __post_init__
        return self._proportional_share(edge.original_amount, originals, pool)

    @staticmethod
    def _proportional_share(
        original: ResourceVector, total: ResourceVector, pool: ResourceVector
    ) -> ResourceVector:
        """``original / total * pool``, floored at zero, integers rounded down."""

        def scalar(part: float | None, whole: float | None, available: float | None) -> float:
            if part is None or not whole or available is None or available <= 0:
                return 0.0
            return available * (part / whole)

        per_tool = {}
        for tool, part in original.per_tool.items():
            per_tool[tool] = int(
                scalar(part, total.per_tool.get(tool, 0), pool.per_tool.get(tool, 0))
            )

        return ResourceVector(
            tokens=int(scalar(original.tokens, total.tokens, pool.tokens)),
            cost_usd=scalar(original.cost_usd, total.cost_usd, pool.cost_usd),
            tool_invocations=int(
                scalar(original.tool_invocations, total.tool_invocations, pool.tool_invocations)
            ),
            iterations=int(scalar(original.iterations, total.iterations, pool.iterations)),
            per_tool=per_tool,
        )
