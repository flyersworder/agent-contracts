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

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.core.resource_vector import ResourceVector

if TYPE_CHECKING:
    from agent_contracts.core.contract import Contract


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
        super().__init__(
            message,
            requested=int(out_flow + consumed),
            available=int(in_flow) if in_flow is not None else 0,
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


@dataclass
class GraphNode:
    """A node in the delegation graph."""

    node_id: str
    name: str
    contract_kwargs: dict[str, Any] = field(default_factory=dict)
    contract: Contract | None = None
    monitor: ResourceMonitor | None = None
    abandoned: bool = False


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
            FlowConservationError: if source's out-flow would exceed its in-flow.
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
        if not prospective_out <= self.in_flow(source):
            self._raise_flow_error(source, prospective_out)

        edge = EdgeAllocation(source=source, target=target, amount=amount)
        self._edges[key] = edge
        return edge

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        """Validate the whole graph and freeze its topology.

        Checks: every non-root node has at least one in-edge; no node's
        out-flow exceeds its in-flow; per-tool allocations are consistent with
        the funding side. Acyclicity is maintained incrementally by
        ``allocate()``.

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
            out = self.out_flow(name)
            if not out <= self.in_flow(name):
                problems.append(f"node '{name}' out-flow exceeds in-flow")

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

    def _raise_flow_error(self, name: str, prospective_out: ResourceVector) -> None:
        """Identify the first violated dimension and raise."""
        available = self.in_flow(name)
        consumed = self._consumed(name)
        for dimension in ("tokens", "cost_usd", "tool_invocations", "iterations"):
            limit = getattr(available, dimension)
            requested = getattr(prospective_out, dimension)
            if limit is not None and requested is not None and requested > limit:
                raise FlowConservationError(
                    f"node '{name}' would over-allocate {dimension}: "
                    f"out-flow {requested} exceeds in-flow {limit}",
                    node_id=name,
                    dimension=dimension,
                    in_flow=limit,
                    consumed=getattr(consumed, dimension) or 0,
                    out_flow=requested,
                    deficit=requested - limit,
                    contributing_edges=self.contributing_edges(name),
                )
        for tool, limit in available.per_tool.items():
            requested = prospective_out.per_tool.get(tool, 0)
            if requested > limit:
                raise FlowConservationError(
                    f"node '{name}' would over-allocate tool '{tool}': "
                    f"out-flow {requested} exceeds in-flow {limit}",
                    node_id=name,
                    dimension=f"tool:{tool}",
                    in_flow=limit,
                    consumed=consumed.per_tool.get(tool, 0),
                    out_flow=requested,
                    deficit=requested - limit,
                    contributing_edges=self.contributing_edges(name),
                )
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
