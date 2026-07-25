# Delegation Flow Conservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the tree-shaped budget conservation law in `ContractingCapability` to a DAG flow invariant, enabling fan-in delegation where a node receives budget from multiple parents.

**Architecture:** A new `DelegationGraph` in its own module models delegation as a DAG whose edges carry resource allocations. Each node's budget is the sum of its in-edges; the invariant `in-flow ≥ own consumption + out-flow` is checked locally at each node, which implies the global bound `Σ C(v) ≤ B(root)` by telescoping. `core/delegation.py` is not modified — the existing tree implementation remains the single-parent special case, and a cross-validation test proves the two agree.

**Tech Stack:** Python 3.12+, uv, pytest, ruff, mypy (strict). No new runtime or test dependencies.

## Global Constraints

- Python `>=3.12`; package manager is `uv`. Run tests with `uv run pytest`.
- **No new dependencies.** Property-style tests use a seeded pseudo-random generator, not `hypothesis`.
- **Do not modify `src/agent_contracts/core/delegation.py`.** The tree implementation stays as-is.
- All 1073 existing tests must remain green after every task.
- `None` in a resource dimension means **unbounded**, never zero.
- `FlowConservationError` must subclass the existing `ConservationViolationError` so current `except` blocks still catch it.
- Pre-commit hooks (ruff lint, ruff format, mypy strict, markdownlint) run on commit and must pass.
- v1 is **not thread-safe**; do not add locking to `DelegationGraph`.
- Spec: `docs/superpowers/specs/2026-07-25-delegation-flow-conservation-design.md`

---

### Task 1: Track and enforce `iterations`

`ResourceConstraints.iterations` exists but is never tracked. `core/prompts.py:174` carries a comment admitting this. Fix both.

**Files:**
- Modify: `src/agent_contracts/core/monitor.py` (add field, validation entry, `add_iteration()`, constraint check)
- Modify: `src/agent_contracts/core/prompts.py:173-177` (remove stale comment)
- Test: `tests/core/test_monitor.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ResourceUsage.iterations: int`, `ResourceUsage.add_iteration() -> None`, and a `ViolationInfo` with `resource == "iterations"`. Task 2 reads `usage.iterations`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_monitor.py`:

```python
def test_add_iteration_increments_usage():
    usage = ResourceUsage()
    usage.add_iteration()
    usage.add_iteration()
    assert usage.iterations == 2


def test_iterations_over_limit_reports_violation():
    monitor = ResourceMonitor(ResourceConstraints(iterations=2))
    for _ in range(3):
        monitor.usage.add_iteration()
    violations = monitor.check_constraints()
    assert any(v.resource == "iterations" for v in violations)
    violation = next(v for v in violations if v.resource == "iterations")
    assert violation.limit == 2
    assert violation.actual == 3


def test_iterations_at_limit_is_not_a_violation():
    monitor = ResourceMonitor(ResourceConstraints(iterations=2))
    for _ in range(2):
        monitor.usage.add_iteration()
    assert not any(v.resource == "iterations" for v in monitor.check_constraints())


def test_iterations_unlimited_when_none():
    monitor = ResourceMonitor(ResourceConstraints())
    for _ in range(100):
        monitor.usage.add_iteration()
    assert not any(v.resource == "iterations" for v in monitor.check_constraints())


def test_negative_iterations_rejected():
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        ResourceUsage(iterations=-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_monitor.py -k iteration -v`
Expected: FAIL with `AttributeError: 'ResourceUsage' object has no attribute 'add_iteration'`

- [ ] **Step 3: Add the field and validation entry**

In `src/agent_contracts/core/monitor.py`, add the field immediately after `cost_usd: float = 0.0` (line 52). Placing it at the end of the counter block keeps counters grouped while avoiding any risk to positional construction in existing tests:

```python
    cost_usd: float = 0.0
    iterations: int = 0
```

Add `"iterations"` to the docstring attribute list (after `cost_usd`) and to the validation list in `__post_init__` (line 61-71), after `"cost_usd"`:

```python
            "cost_usd",
            "iterations",
        ]:
```

- [ ] **Step 4: Add the `add_iteration()` method**

In `monitor.py`, immediately after `add_web_search()` (line 126-130), following the same lock pattern:

```python
    def add_iteration(self) -> None:
        """Record one agent loop iteration."""
        with self._lock:
            self.iterations += 1
            self.last_updated = datetime.now()
```

- [ ] **Step 5: Enforce the constraint**

In `ResourceMonitor.check_constraints()`, immediately after the `tool_invocations` block (around line 380-390) and before the per-tool loop:

```python
        if (
            self.constraints.iterations is not None
            and self.usage.iterations > self.constraints.iterations
        ):
            violations.append(
                ViolationInfo(
                    resource="iterations",
                    limit=self.constraints.iterations,
                    actual=self.usage.iterations,
                )
            )
```

- [ ] **Step 6: Remove the stale comment**

In `src/agent_contracts/core/prompts.py`, lines 173-177 currently read:

```python
    if resources.iterations is not None:
        # Note: We don't track iterations in ResourceUsage currently,
        # ...
        lines.append(
            f"- LLM Calls: {resources.iterations} maximum (plan your reasoning steps accordingly)"
        )
```

Delete only the two-line `# Note:` comment. Leave the `if` and the `lines.append(...)` exactly as they are.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/core/test_monitor.py -v`
Expected: PASS, including the five new tests.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: 1078 passed, 1 skipped (1073 + 5 new).

- [ ] **Step 9: Commit**

```bash
git add src/agent_contracts/core/monitor.py src/agent_contracts/core/prompts.py tests/core/test_monitor.py
git commit -m "feat(monitor): track and enforce ResourceConstraints.iterations"
```

---

### Task 2: `ResourceVector` arithmetic helper

A frozen value type supporting `+`, `-`, `<=` across the conserved dimensions, with `None` meaning unbounded.

**Files:**
- Create: `src/agent_contracts/core/resource_vector.py`
- Test: `tests/core/test_resource_vector.py`

**Interfaces:**
- Consumes: `ResourceUsage.iterations` from Task 1.
- Produces:
  - `ResourceVector(tokens: int | None, cost_usd: float | None, tool_invocations: int | None, iterations: int | None, per_tool: Mapping[str, int])`
  - `ResourceVector.ZERO` — classmethod-free module constant with all scalars `0` and empty `per_tool`
  - `ResourceVector.from_constraints(rc: ResourceConstraints) -> ResourceVector`
  - `ResourceVector.from_usage(u: ResourceUsage) -> ResourceVector`
  - `ResourceVector.to_constraints() -> ResourceConstraints`
  - `ResourceVector.is_finite() -> bool`
  - `__add__`, `__sub__`, `__le__`
  - Tasks 3-7 use all of these.

**Semantics (implement exactly):**

| Operation | Rule |
|---|---|
| `a + b` scalar | `None` if either is `None`, else `a + b` |
| `a - b` scalar | `None` if `a is None`; raise `ValueError` if `b is None` and `a` is not |
| `a <= b` scalar | `True` if `b is None`; `False` if `a is None` and `b` is not; else `a <= b` |
| `per_tool` add/sub | union of keys, missing treated as `0` |
| `a <= b` per_tool | iterate keys of `b` only; `a.per_tool.get(k, 0) <= b.per_tool[k]`. Keys in `a` but not `b` are unconstrained and skipped. |

The per-tool rule reproduces existing behavior: `AllocationRecord`'s docstring states tools the parent does not constrain "don't participate in conservation accounting."

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_resource_vector.py`:

```python
import pytest

from agent_contracts.core.contract import ResourceConstraints
from agent_contracts.core.monitor import ResourceUsage
from agent_contracts.core.resource_vector import ResourceVector


def test_add_finite_scalars():
    a = ResourceVector(tokens=10, cost_usd=1.0, tool_invocations=2, iterations=1)
    b = ResourceVector(tokens=5, cost_usd=0.5, tool_invocations=3, iterations=4)
    total = a + b
    assert total.tokens == 15
    assert total.cost_usd == 1.5
    assert total.tool_invocations == 5
    assert total.iterations == 5


def test_add_unbounded_absorbs():
    a = ResourceVector(tokens=None)
    b = ResourceVector(tokens=5)
    assert (a + b).tokens is None


def test_subtract_from_unbounded_stays_unbounded():
    a = ResourceVector(tokens=None)
    b = ResourceVector(tokens=5)
    assert (a - b).tokens is None


def test_subtract_unbounded_from_finite_raises():
    a = ResourceVector(tokens=5)
    b = ResourceVector(tokens=None)
    with pytest.raises(ValueError, match="unbounded"):
        a - b


def test_le_anything_under_unbounded():
    assert ResourceVector(tokens=10**9) <= ResourceVector(tokens=None)


def test_le_unbounded_not_under_finite():
    assert not (ResourceVector(tokens=None) <= ResourceVector(tokens=5))


def test_per_tool_add_unions_keys():
    a = ResourceVector(per_tool={"exp": 3})
    b = ResourceVector(per_tool={"exp": 2, "web": 1})
    total = a + b
    assert total.per_tool == {"exp": 5, "web": 1}


def test_per_tool_le_ignores_tools_budget_does_not_constrain():
    used = ResourceVector(per_tool={"exp": 3, "unconstrained": 99})
    budget = ResourceVector(per_tool={"exp": 5})
    assert used <= budget


def test_per_tool_le_detects_overrun():
    used = ResourceVector(per_tool={"exp": 6})
    budget = ResourceVector(per_tool={"exp": 5})
    assert not (used <= budget)


def test_from_constraints_reads_all_dimensions():
    rc = ResourceConstraints(
        tokens=100, cost_usd=2.0, tool_invocations=7, iterations=3, per_tool_limits={"exp": 4}
    )
    v = ResourceVector.from_constraints(rc)
    assert v.tokens == 100
    assert v.cost_usd == 2.0
    assert v.tool_invocations == 7
    assert v.iterations == 3
    assert v.per_tool == {"exp": 4}


def test_from_usage_reads_all_dimensions():
    usage = ResourceUsage(tokens=50, cost_usd=1.0, tool_invocations=2)
    usage.add_tool_invocation("exp")
    usage.add_iteration()
    v = ResourceVector.from_usage(usage)
    assert v.tokens == 50
    assert v.cost_usd == 1.0
    assert v.tool_invocations == 3
    assert v.iterations == 1
    assert v.per_tool == {"exp": 1}


def test_to_constraints_round_trips():
    rc = ResourceConstraints(tokens=100, cost_usd=2.0, per_tool_limits={"exp": 4})
    assert ResourceVector.from_constraints(rc).to_constraints() == rc


def test_is_finite():
    assert ResourceVector(tokens=1, cost_usd=0.0, tool_invocations=0, iterations=0).is_finite()
    assert not ResourceVector(tokens=None).is_finite()


def test_zero_is_all_zeros():
    assert ResourceVector.ZERO.tokens == 0
    assert ResourceVector.ZERO.per_tool == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_resource_vector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_contracts.core.resource_vector'`

- [ ] **Step 3: Implement the module**

Create `src/agent_contracts/core/resource_vector.py`:

```python
"""Resource arithmetic for flow-conservation delegation.

A ``ResourceVector`` is an amount across every conserved resource dimension.
``None`` in a scalar dimension means *unbounded*, never zero.

Per-tool semantics mirror ``AllocationRecord``: only tools the budget side
explicitly constrains participate in comparison. A tool absent from a budget
is unconstrained; a tool absent from an amount counts as zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from agent_contracts.core.contract import ResourceConstraints
from agent_contracts.core.monitor import ResourceUsage


def _add(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a + b


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None:
        return None
    if b is None:
        raise ValueError("cannot subtract an unbounded amount from a finite budget")
    return a - b


def _le(a: float | None, b: float | None) -> bool:
    if b is None:
        return True
    if a is None:
        return False
    return a <= b


@dataclass(frozen=True)
class ResourceVector:
    """An amount across conserved resource dimensions. ``None`` == unbounded."""

    tokens: int | None = None
    cost_usd: float | None = None
    tool_invocations: int | None = None
    iterations: int | None = None
    per_tool: Mapping[str, int] = field(default_factory=dict)

    ZERO: ClassVar[ResourceVector]

    def __add__(self, other: ResourceVector) -> ResourceVector:
        per_tool = dict(self.per_tool)
        for name, count in other.per_tool.items():
            per_tool[name] = per_tool.get(name, 0) + count
        return ResourceVector(
            tokens=_add(self.tokens, other.tokens),
            cost_usd=_add(self.cost_usd, other.cost_usd),
            tool_invocations=_add(self.tool_invocations, other.tool_invocations),
            iterations=_add(self.iterations, other.iterations),
            per_tool=per_tool,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        per_tool = dict(self.per_tool)
        for name, count in other.per_tool.items():
            per_tool[name] = per_tool.get(name, 0) - count
        return ResourceVector(
            tokens=_sub(self.tokens, other.tokens),
            cost_usd=_sub(self.cost_usd, other.cost_usd),
            tool_invocations=_sub(self.tool_invocations, other.tool_invocations),
            iterations=_sub(self.iterations, other.iterations),
            per_tool=per_tool,
        )

    def __le__(self, other: ResourceVector) -> bool:
        scalars_ok = (
            _le(self.tokens, other.tokens)
            and _le(self.cost_usd, other.cost_usd)
            and _le(self.tool_invocations, other.tool_invocations)
            and _le(self.iterations, other.iterations)
        )
        if not scalars_ok:
            return False
        # Only tools the budget side constrains participate.
        return all(self.per_tool.get(name, 0) <= limit for name, limit in other.per_tool.items())

    def is_finite(self) -> bool:
        """True when no scalar dimension is unbounded."""
        return all(
            value is not None
            for value in (self.tokens, self.cost_usd, self.tool_invocations, self.iterations)
        )

    @classmethod
    def from_constraints(cls, rc: ResourceConstraints) -> ResourceVector:
        return cls(
            tokens=rc.tokens,
            cost_usd=rc.cost_usd,
            tool_invocations=rc.tool_invocations,
            iterations=rc.iterations,
            per_tool=dict(rc.per_tool_limits),
        )

    @classmethod
    def from_usage(cls, usage: ResourceUsage) -> ResourceVector:
        return cls(
            tokens=usage.tokens,
            cost_usd=usage.cost_usd,
            tool_invocations=usage.tool_invocations,
            iterations=usage.iterations,
            per_tool=dict(usage.tool_usage_by_name),
        )

    def to_constraints(self) -> ResourceConstraints:
        return ResourceConstraints(
            tokens=self.tokens,
            cost_usd=self.cost_usd,
            tool_invocations=self.tool_invocations,
            iterations=self.iterations,
            per_tool_limits=dict(self.per_tool),
        )


ResourceVector.ZERO = ResourceVector(
    tokens=0, cost_usd=0.0, tool_invocations=0, iterations=0, per_tool={}
)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/core/test_resource_vector.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/agent_contracts/core/resource_vector.py`
Expected: `Success: no issues found`

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/resource_vector.py tests/core/test_resource_vector.py
git commit -m "feat(core): add ResourceVector arithmetic with unbounded semantics"
```

---

### Task 3: `DelegationGraph` — nodes, edges, allocation

Build-phase graph construction with satisfiability and cycle checks.

**Files:**
- Create: `src/agent_contracts/core/delegation_graph.py`
- Test: `tests/core/test_delegation_graph.py`

**Interfaces:**
- Consumes: `ResourceVector` and all its operations from Task 2.
- Produces:
  - `CycleError(Exception)`
  - `FlowConservationError(ConservationViolationError)` with attributes `node_id, dimension, in_flow, consumed, out_flow, deficit, contributing_edges`
  - `EdgeAllocation(source, target, amount: ResourceVector, original_amount: ResourceVector | None, created_at: datetime, released: bool)` — `original_amount` defaults to `amount` in `__post_init__` and never changes; Task 6 computes refunds from it
  - `GraphNode(node_id: str, name: str, contract_kwargs: dict, contract: Contract | None, monitor: ResourceMonitor | None, abandoned: bool)`
  - `DelegationGraph(root_contract: Contract, root_monitor: ResourceMonitor | None = None)`
  - `.ROOT: str = "root"`, `.add_node(name, **contract_kwargs) -> str`, `.allocate(source, target, **resources) -> EdgeAllocation`, `.in_flow(name) -> ResourceVector`, `.original_in_flow(name) -> ResourceVector`, `.out_flow(name) -> ResourceVector`, `.contributing_edges(name) -> list[str]`, `._consumed(name) -> ResourceVector`
  - Tasks 4-7 build on all of these. Task 6 specifically needs `original_in_flow` and `_consumed`.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_delegation_graph.py`:

```python
import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import (
    CycleError,
    DelegationGraph,
    FlowConservationError,
)


def make_root(**kwargs) -> Contract:
    defaults = {"tokens": 100_000, "per_tool_limits": {"exp": 59}}
    defaults.update(kwargs)
    return Contract(
        id="root-contract",
        name="Root",
        resources=ResourceConstraints(**defaults),
    )


def test_root_node_registered_on_construction():
    graph = DelegationGraph(make_root())
    assert graph.in_flow(DelegationGraph.ROOT).tokens == 100_000


def test_add_node_then_allocate_sets_in_flow():
    graph = DelegationGraph(make_root())
    graph.add_node("scout_a")
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    assert graph.in_flow("scout_a").tokens == 40_000
    assert graph.out_flow(DelegationGraph.ROOT).tokens == 40_000


def test_fan_in_accumulates_budget():
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    assert graph.in_flow("aggregator").tokens == 30_000


def test_over_allocation_raises_flow_conservation_error():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("child")
    with pytest.raises(FlowConservationError) as excinfo:
        graph.allocate(DelegationGraph.ROOT, "child", tokens=101)
    assert excinfo.value.dimension == "tokens"
    assert excinfo.value.node_id == DelegationGraph.ROOT


def test_flow_conservation_error_is_catchable_as_conservation_violation():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("child")
    with pytest.raises(ConservationViolationError):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=101)


def test_cumulative_over_allocation_across_two_edges_raises():
    graph = DelegationGraph(make_root(tokens=100))
    graph.add_node("a")
    graph.add_node("b")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=60)
    with pytest.raises(FlowConservationError):
        graph.allocate(DelegationGraph.ROOT, "b", tokens=60)


def test_per_tool_over_allocation_raises():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    with pytest.raises(FlowConservationError):
        graph.allocate(DelegationGraph.ROOT, "child", per_tool={"exp": 60})


def test_unbounded_parent_allows_any_finite_allocation():
    root = Contract(id="r", name="R", resources=ResourceConstraints())
    graph = DelegationGraph(root)
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10**9)
    assert graph.in_flow("child").tokens == 10**9


def test_unbounded_allocation_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    with pytest.raises(ValueError, match="finite"):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=None)


def test_cycle_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    graph.add_node("b")
    graph.allocate(DelegationGraph.ROOT, "a", tokens=100)
    graph.allocate("a", "b", tokens=50)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=10)


def test_self_edge_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    with pytest.raises(CycleError):
        graph.allocate("a", "a", tokens=10)


def test_duplicate_node_name_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("a")
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node("a")


def test_allocate_to_unknown_node_rejected():
    graph = DelegationGraph(make_root())
    with pytest.raises(KeyError):
        graph.allocate(DelegationGraph.ROOT, "nope", tokens=10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_delegation_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_contracts.core.delegation_graph'`

- [ ] **Step 3: Implement the module**

Create `src/agent_contracts/core/delegation_graph.py`:

```python
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
from typing import Any

from agent_contracts.core.contract import Contract
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.core.resource_vector import ResourceVector


class CycleError(Exception):
    """Raised when an edge would create a cycle in the budget graph."""


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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/core/test_delegation_graph.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Type-check and run the full suite**

Run: `uv run mypy src/agent_contracts/core/delegation_graph.py && uv run pytest -q`
Expected: mypy clean; 1105 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/delegation_graph.py tests/core/test_delegation_graph.py
git commit -m "feat(core): add DelegationGraph with edge allocation and cycle rejection"
```

---

### Task 4: `seal()` — graph linting

**Files:**
- Modify: `src/agent_contracts/core/delegation_graph.py`
- Test: `tests/core/test_delegation_graph.py`

**Interfaces:**
- Consumes: `DelegationGraph` internals from Task 3.
- Produces: `GraphLintError(Exception)` with `.problems: list[str]`, and `DelegationGraph.seal() -> None`, `.is_sealed` property. Task 5 requires `seal()` before materialization.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_delegation_graph.py`:

```python
from agent_contracts.core.delegation_graph import GraphLintError


def test_seal_accepts_valid_graph():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    assert graph.is_sealed


def test_seal_rejects_orphan_node():
    graph = DelegationGraph(make_root())
    graph.add_node("orphan")
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert any("orphan" in problem for problem in excinfo.value.problems)


def test_seal_reports_all_problems_not_just_first():
    graph = DelegationGraph(make_root())
    graph.add_node("orphan_a")
    graph.add_node("orphan_b")
    with pytest.raises(GraphLintError) as excinfo:
        graph.seal()
    assert len(excinfo.value.problems) == 2


def test_allocate_after_seal_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        graph.add_node("late")
    with pytest.raises(RuntimeError, match="sealed"):
        graph.allocate(DelegationGraph.ROOT, "child", tokens=1)


def test_double_seal_is_idempotent():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    graph.seal()
    assert graph.is_sealed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_delegation_graph.py -k seal -v`
Expected: FAIL with `ImportError: cannot import name 'GraphLintError'`

- [ ] **Step 3: Add `GraphLintError`**

In `delegation_graph.py`, after `CycleError`:

```python
class GraphLintError(Exception):
    """Raised by ``seal()`` when the graph fails validation.

    Reports every problem found, not just the first, so one pass fixes them all.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"graph failed validation:\n  - {joined}")
```

- [ ] **Step 4: Add `seal()` and `is_sealed`**

In `DelegationGraph`, after `allocate()`:

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/core/test_delegation_graph.py -v`
Expected: PASS, 18 tests.

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/delegation_graph.py tests/core/test_delegation_graph.py
git commit -m "feat(core): add seal() graph linting to DelegationGraph"
```

---

### Task 5: Materialization, residual, and the runtime invariant

**Files:**
- Modify: `src/agent_contracts/core/delegation_graph.py`
- Test: `tests/core/test_delegation_graph.py`

**Interfaces:**
- Consumes: `seal()` from Task 4.
- Produces: `.contract_for(name) -> Contract`, `.monitor_for(name) -> ResourceMonitor`, `.residual(name) -> ResourceVector`, `.check_node(name) -> None`, `.verify() -> None`. Tasks 6-7 use `residual` and `verify`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_delegation_graph.py`:

```python
def test_contract_for_sums_in_flow():
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    graph.seal()
    contract = graph.contract_for("aggregator")
    assert contract.resources.tokens == 30_000
    assert contract.id == "root-contract/aggregator"


def test_contract_for_before_seal_rejected():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    with pytest.raises(RuntimeError, match="seal"):
        graph.contract_for("child")


def test_contract_for_is_stable_across_calls():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=10)
    graph.seal()
    assert graph.contract_for("child") is graph.contract_for("child")


def test_residual_is_in_flow_minus_consumption_minus_out_flow():
    graph = DelegationGraph(make_root())
    graph.add_node("mid")
    graph.add_node("leaf")
    graph.allocate(DelegationGraph.ROOT, "mid", tokens=1000)
    graph.allocate("mid", "leaf", tokens=400)
    graph.seal()
    graph.monitor_for("mid").usage.add_tokens(100)
    assert graph.residual("mid").tokens == 500


def test_check_node_raises_when_consumption_breaks_invariant():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    graph.monitor_for("child").usage.add_tokens(101)
    with pytest.raises(FlowConservationError) as excinfo:
        graph.check_node("child")
    assert excinfo.value.node_id == "child"


def test_verify_checks_every_node():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    graph.verify()  # clean
    graph.monitor_for("child").usage.add_tokens(101)
    with pytest.raises(FlowConservationError):
        graph.verify()


def test_node_monitor_enforces_summed_budget_via_existing_machinery():
    graph = DelegationGraph(make_root())
    graph.add_node("child")
    graph.allocate(DelegationGraph.ROOT, "child", tokens=100)
    graph.seal()
    monitor = graph.monitor_for("child")
    monitor.usage.add_tokens(101)
    assert any(v.resource == "tokens" for v in monitor.check_constraints())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_delegation_graph.py -k "contract_for or residual or check_node or verify or monitor" -v`
Expected: FAIL with `AttributeError: 'DelegationGraph' object has no attribute 'contract_for'`

- [ ] **Step 3: Implement materialization and checks**

Append to `DelegationGraph`:

```python
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
        """Raise FlowConservationError if ``name``'s invariant is violated."""
        self._require_node(name)
        commitment = self._consumed(name) + self.out_flow(name)
        if not commitment <= self.in_flow(name):
            self._raise_flow_error(name, commitment)

    def verify(self) -> None:
        """Check the invariant at every non-abandoned node."""
        for name, node in self._nodes.items():
            if not node.abandoned:
                self.check_node(name)

    def _require_sealed(self) -> None:
        if not self._sealed:
            raise RuntimeError("graph must be sealed before materializing contracts")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/core/test_delegation_graph.py -v`
Expected: PASS, 25 tests.

- [ ] **Step 5: Type-check and run the full suite**

Run: `uv run mypy src/agent_contracts/core/delegation_graph.py && uv run pytest -q`
Expected: mypy clean; 1117 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/delegation_graph.py tests/core/test_delegation_graph.py
git commit -m "feat(core): materialize node contracts and check the flow invariant"
```

---

### Task 6: Proportional refunds and node abandonment

**Files:**
- Modify: `src/agent_contracts/core/delegation_graph.py`
- Test: `tests/core/test_delegation_graph.py`

**Interfaces:**
- Consumes: `residual` from Task 5.
- Produces: `.release(source, target) -> ResourceVector`, `.abandon(name) -> ResourceVector`, `.is_reachable(name) -> bool`.

**Refund rule.** Let `orig(u→v)` be the edge's original allocation and
`ORIG(v) = Σ orig(·→v)`. Then:

```
pool(v)      = ORIG(v) − consumed(v) − out_flow(v)        floored at 0
refund(u→v)  = orig(u→v) / ORIG(v) × pool(v)              integers rounded down
```

**Shares are computed against the *original* totals, not the current ones.** This is
what makes releases order-independent: `pool(v)` and `ORIG(v)` are unchanged by a
sibling's release, so each edge's share is fixed the moment consumption is known.
Computing against live in-flow would make `scout_a` reclaim 5000 and `scout_b` 3000
when released in that order, and the reverse when swapped — the exact order-dependence
the spec's §6.3 rules out. LIFO and first-come fail the same way.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_delegation_graph.py`:

```python
def _diamond() -> DelegationGraph:
    graph = DelegationGraph(make_root())
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=40_000)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=40_000)
    graph.allocate("scout_a", "aggregator", tokens=15_000)
    graph.allocate("scout_b", "aggregator", tokens=15_000)
    graph.seal()
    return graph


def test_release_returns_proportional_share():
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(20_000)  # residual 10_000
    refund = graph.release("scout_a", "aggregator")
    assert refund.tokens == 5_000


def test_release_is_order_independent():
    first = _diamond()
    first.monitor_for("aggregator").usage.add_tokens(20_000)
    a_then_b = (
        first.release("scout_a", "aggregator").tokens,
        first.release("scout_b", "aggregator").tokens,
    )

    second = _diamond()
    second.monitor_for("aggregator").usage.add_tokens(20_000)
    b_then_a = (
        second.release("scout_b", "aggregator").tokens,
        second.release("scout_a", "aggregator").tokens,
    )

    # Both siblings funded the aggregator equally, so both reclaim half the
    # 10_000 unused tokens regardless of which released first.
    assert a_then_b == (5_000, 5_000)
    assert b_then_a == (5_000, 5_000)
    assert first.residual("scout_a").tokens == second.residual("scout_a").tokens
    assert first.residual("aggregator").tokens == second.residual("aggregator").tokens


def test_release_restores_parent_residual():
    graph = _diamond()
    before = graph.residual("scout_a").tokens
    graph.monitor_for("aggregator").usage.add_tokens(20_000)
    graph.release("scout_a", "aggregator")
    assert graph.residual("scout_a").tokens == before + 5_000


def test_release_of_fully_consumed_edge_refunds_nothing():
    graph = _diamond()
    graph.monitor_for("aggregator").usage.add_tokens(30_000)
    assert graph.release("scout_a", "aggregator").tokens == 0


def test_double_release_rejected():
    graph = _diamond()
    graph.release("scout_a", "aggregator")
    with pytest.raises(ValueError, match="already released"):
        graph.release("scout_a", "aggregator")


def test_abandon_refunds_unconsumed_budget_to_parents():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(1_000)
    graph.abandon("scout_a")
    # scout_a held 40_000, spent 1_000, passed 15_000 downstream; root reclaims 24_000
    assert graph.residual(DelegationGraph.ROOT).tokens == 100_000 - 40_000 - 40_000 + 24_000


def test_abandon_marks_downstream_unreachable():
    graph = _diamond()
    graph.abandon("scout_a")
    assert not graph.is_reachable("scout_a")
    assert graph.is_reachable("scout_b")
    assert graph.is_reachable("aggregator")  # scout_b still funds it


def test_abandoned_node_excluded_from_verify():
    graph = _diamond()
    graph.monitor_for("scout_a").usage.add_tokens(40_000)
    graph.abandon("scout_a")
    graph.verify()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_delegation_graph.py -k "release or abandon or reachable" -v`
Expected: FAIL with `AttributeError: 'DelegationGraph' object has no attribute 'release'`

- [ ] **Step 3: Implement refunds and abandonment**

Append to `DelegationGraph`:

```python
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
        """
        self._require_node(name)
        if name == self.ROOT:
            raise ValueError("cannot abandon the root node")
        node = self._nodes[name]
        if node.abandoned:
            raise ValueError(f"node '{name}' already abandoned")

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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/core/test_delegation_graph.py -v`
Expected: PASS, 33 tests.

- [ ] **Step 5: Type-check and run the full suite**

Run: `uv run mypy src/agent_contracts/core/delegation_graph.py && uv run pytest -q`
Expected: mypy clean; 1125 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/delegation_graph.py tests/core/test_delegation_graph.py
git commit -m "feat(core): add proportional refunds and node abandonment"
```

---

### Task 7: Cross-validation and telescoping soundness

The two tests that make the paper's claims checkable rather than merely asserted.

**Files:**
- Create: `tests/core/test_delegation_graph_properties.py`

**Interfaces:**
- Consumes: everything from Tasks 2-6, plus `ContractingCapability` from the untouched `core/delegation.py`.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the cross-validation test**

Create `tests/core/test_delegation_graph_properties.py`:

```python
"""Property-style tests using a seeded generator (no hypothesis dependency)."""

import random

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ContractingCapability
from agent_contracts.core.delegation_graph import DelegationGraph
from agent_contracts.core.resource_vector import ResourceVector

SEEDS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def _root(tokens: int = 1_000_000) -> Contract:
    return Contract(id="root", name="Root", resources=ResourceConstraints(tokens=tokens))


@pytest.mark.parametrize("seed", SEEDS)
def test_tree_topology_matches_contracting_capability(seed):
    """On a tree, DelegationGraph and ContractingCapability must agree exactly."""
    rng = random.Random(seed)
    child_budgets = {f"child_{i}": rng.randint(1, 50_000) for i in range(rng.randint(1, 6))}

    tree = ContractingCapability(_root())
    for name, tokens in child_budgets.items():
        tree.create_subcontract(name, tokens=tokens)

    graph = DelegationGraph(_root())
    for name, tokens in child_budgets.items():
        graph.add_node(name)
        graph.allocate(DelegationGraph.ROOT, name, tokens=tokens)

    assert graph.residual(DelegationGraph.ROOT).tokens == tree.remaining_tokens
    for name, tokens in child_budgets.items():
        assert graph.in_flow(name).tokens == tokens
        assert tree.get_allocation(name).tokens_allocated == tokens
```

- [ ] **Step 2: Run it to verify it passes against real code**

Run: `uv run pytest tests/core/test_delegation_graph_properties.py -v`
Expected: PASS, 10 parametrized cases. A failure here means the generalization claim is false — stop and reconcile before continuing.

- [ ] **Step 3: Write the telescoping soundness test**

Append to the same file:

```python
def _random_dag(rng: random.Random, root_tokens: int) -> tuple[DelegationGraph, list[str]]:
    """Build a random DAG in which every node is funded.

    Node ``i`` only ever receives edges from nodes ``< i``, so the result is
    acyclic by construction. Each target is offered candidates in random order
    and takes the first one or two with headroom; the root is always a
    candidate and each allocation takes at most half the funder's headroom, so
    with at most 7 nodes the root retains at least ``root_tokens / 2**7``
    tokens and can always fund. Every node therefore gets at least one in-edge,
    the graph always seals, and the test never skips.
    """
    graph = DelegationGraph(_root(root_tokens))
    names = [DelegationGraph.ROOT]
    for i in range(rng.randint(2, 7)):
        name = f"n{i}"
        graph.add_node(name)
        names.append(name)

    for target_index in range(1, len(names)):
        target = names[target_index]
        candidates = names[:target_index]
        rng.shuffle(candidates)
        wanted = rng.randint(1, min(2, len(candidates)))
        funded = 0
        for source in candidates:
            headroom = graph.residual(source).tokens
            if headroom is None or headroom < 2:
                continue
            graph.allocate(source, target, tokens=rng.randint(1, headroom // 2))
            funded += 1
            if funded == wanted:
                break
        assert funded > 0, f"generator left '{target}' unfunded; root should always have headroom"
    return graph, names


@pytest.mark.parametrize("seed", SEEDS)
def test_local_invariants_imply_global_bound(seed):
    """If every node satisfies its local invariant, total consumption <= root budget."""
    rng = random.Random(seed)
    root_tokens = 100_000
    graph, names = _random_dag(rng, root_tokens)

    for name in names:
        if name != DelegationGraph.ROOT:
            assert graph.contributing_edges(name), f"'{name}' must be funded"
    graph.seal()

    # Saturate: every node consumes its ENTIRE residual. Consuming a random
    # fraction instead leaves roughly 2x slack under the bound, which lets a
    # diamond that double-counts its in-flow pass undetected on all 10 seeds
    # (verified by mutation testing). Saturation makes the telescoping identity
    # tight, so the assertion below is an equality, not a loose inequality.
    for name in names:
        headroom = graph.residual(name).tokens
        if headroom and headroom > 0:
            graph.monitor_for(name).usage.add_tokens(headroom)

    graph.verify()  # every local invariant holds

    total_consumed = sum(
        ResourceVector.from_usage(graph.monitor_for(name).usage).tokens for name in names
    )
    assert total_consumed == root_tokens
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/core/test_delegation_graph_properties.py -v`
Expected: PASS, 20 parametrized cases, **zero skips**. A skip here is a bug in the generator, not an acceptable outcome — every seed must reach `graph.verify()` and assert the global bound.

- [ ] **Step 5: Run the full suite with coverage**

Run: `uv run pytest -q --cov=agent_contracts --cov-report=term:skip-covered`
Expected: 1146 passed, 1 skipped (Task 7 adds 2 tests × 10 seeds on top of Task 6's 1126); coverage at or above 90%.

- [ ] **Step 6: Commit**

```bash
git add tests/core/test_delegation_graph_properties.py
git commit -m "test(core): cross-validate tree equivalence and telescoping soundness"
```

---

### Task 8: Export the public API

**Files:**
- Modify: `src/agent_contracts/core/__init__.py`
- Test: `tests/core/test_delegation_graph.py`

**Interfaces:**
- Consumes: all public names from Tasks 2-6.
- Produces: importable `from agent_contracts.core import DelegationGraph, ResourceVector, FlowConservationError, CycleError, GraphLintError`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_delegation_graph.py`:

```python
def test_public_api_exports():
    from agent_contracts.core import (
        CycleError,
        DelegationGraph,
        FlowConservationError,
        GraphLintError,
        ResourceVector,
    )

    assert DelegationGraph.ROOT == "root"
    assert issubclass(FlowConservationError, ConservationViolationError)
    assert ResourceVector.ZERO.tokens == 0
    assert CycleError is not None
    assert GraphLintError is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/core/test_delegation_graph.py::test_public_api_exports -v`
Expected: FAIL with `ImportError: cannot import name 'DelegationGraph'`

- [ ] **Step 3: Add the exports**

In `src/agent_contracts/core/__init__.py`, follow the file's existing import and `__all__` style:

```python
from agent_contracts.core.delegation_graph import (
    CycleError,
    DelegationGraph,
    EdgeAllocation,
    FlowConservationError,
    GraphLintError,
    GraphNode,
)
from agent_contracts.core.resource_vector import ResourceVector
```

Add each name to `__all__`, keeping the list's existing ordering convention.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: 1147 passed, 1 skipped.

- [ ] **Step 5: Verify the docs example runs**

Run:

```bash
uv run python -c "
from agent_contracts.core import DelegationGraph
from agent_contracts.core.contract import Contract, ResourceConstraints
g = DelegationGraph(Contract(id='r', name='R', resources=ResourceConstraints(tokens=100_000)))
for n in ('scout_a','scout_b','aggregator'): g.add_node(n)
g.allocate('root','scout_a',tokens=40_000); g.allocate('root','scout_b',tokens=40_000)
g.allocate('scout_a','aggregator',tokens=15_000); g.allocate('scout_b','aggregator',tokens=15_000)
g.seal()
print('aggregator budget:', g.contract_for('aggregator').resources.tokens)
"
```

Expected output: `aggregator budget: 30000`

- [ ] **Step 6: Commit**

```bash
git add src/agent_contracts/core/__init__.py tests/core/test_delegation_graph.py
git commit -m "feat(core): export DelegationGraph and ResourceVector from core"
```

---

## Out of scope for this plan

The **M6 topology experiment** (spec §8) is a separate plan, written closer to its
2026-08-24 window so it can incorporate M5's WT results. It depends on this plan
being complete and adds: a `fanin_scouts` variant in the chamber AgentSpec registry,
a 90-cell sweep, the reuse-validity guard, and the topology-tax figures.

## Verification checklist

Before declaring this plan complete:

- [ ] `uv run pytest -q` — all tests pass, none skipped beyond the pre-existing 1
- [ ] `uv run mypy src/agent_contracts/core/` — strict mode clean
- [ ] `uv run ruff check . && uv run ruff format --check .` — clean
- [ ] `git diff main --stat -- src/agent_contracts/core/delegation.py` — **empty**, proving the tree implementation was not modified
- [ ] Coverage at or above 90%
