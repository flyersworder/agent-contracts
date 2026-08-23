# M6 Theory Track: Propositions P1–P6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `docs/whitepaper.md` §4.6's prose commentary into six numbered propositions, each with a proof and an executable counterexample or property test, and reach the 2026-08-29 go/no-go decision.

**Architecture:** Each proposition gets an executable artifact in `tests/core/test_delegation_graph_propositions.py` first (a counterexample construction, a property test, or an equivalence check), then a prose proof in §4.6. Executing the claim before writing the proof is deliberate: two of the six are suspected false as currently stated, and a test tells you that in minutes where a proof attempt can absorb a day.

**Tech Stack:** Python 3.12+, pytest, `agent_contracts.core.delegation_graph`, `agent_contracts.core.delegation`, `agent_contracts.core.resource_vector`. Run with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md` §2

## Global Constraints

- Every proposition must be **falsifiable by an executable artifact** before its proof is written. A proposition whose test cannot be constructed is a proposition that does not survive the gate.
- Property tests must be **mutation-checked**: inject the defect the test exists to catch and confirm the test fails. An assertion with slack that a fan-in double-count can hide in is decorative (whitepaper §4.6, "Empirical validation").
- Assert **identities, not inequalities**, wherever saturation makes an identity available. `Σ C(v) == B(root)` catches a double-count that `Σ C(v) <= B(root)` does not.
- Existing tests must stay green: `uv run pytest tests/core/ -q`.
- Do **not** modify `src/agent_contracts/core/delegation_graph.py` behaviour in this track. If a proposition requires a code change, stop and record it as a finding for the implementation track.
- `mypy 2.3.1 --strict` clean on anything added under `src/`.
- **Gate:** if fewer than three of P2–P6 survive, Task 6 records a re-target decision rather than a §4.6 rewrite.

---

### Task 1: P2 — tree insufficiency, stated over allocations

The spec states P2 over *executions* ("tree accounting certifies an execution exceeding `B(root)`"). That framing needs a consumption dict and invites a reference model that checks only some nodes. State it over **allocations** instead: a tree accountant approves a set of grants under which the agents can collectively spend more than `B(root)`, whatever they then choose to spend.

**Files:**
- Create: `tests/core/test_delegation_graph_propositions.py`

**Interfaces:**
- Produces (test-local):
  - `permitted_total(edges, root_budget) -> int` — the maximum total consumption the allocation physically permits, `Σ_v max(0, in_flow(v) − out_flow(v))`. The `max(0, …)` matters: a node cannot spend a negative amount, so an over-committed node does not offset its neighbours.
  - `tree_admits(edges, root_budget) -> bool` — whether a drop-policy tree accountant approves the grants.

- [ ] **Step 1: Write the two helpers and the failing assertion**

```python
# tests/core/test_delegation_graph_propositions.py
import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.core.delegation_graph import DelegationGraph

ROOT = DelegationGraph.ROOT


def make_root(tokens: int = 100) -> Contract:
    return Contract(
        id="p-root", name="Root", resources=ResourceConstraints(tokens=tokens)
    )


def _in_flow(edges, node, root_budget):
    if node == ROOT:
        return root_budget
    return sum(a for _s, d, a in edges if d == node)


def _out_flow(edges, node):
    return sum(a for s, _d, a in edges if s == node)


def permitted_total(edges, root_budget):
    """Maximum total consumption the allocation physically permits.

    A node can spend whatever arrives minus whatever it forwards, and never
    less than zero — an over-committed node cannot offset its neighbours by
    spending negatively. That asymmetry is exactly what a tree accountant
    misses when it cannot see one of a node's out-edges.
    """
    nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
    return sum(
        max(0, _in_flow(edges, n, root_budget) - _out_flow(edges, n)) for n in nodes
    )


def tree_admits(edges, root_budget):
    """Drop-policy tree accountant: keep one in-edge per node, ignore the rest.

    The dropped edge is invisible to the accountant but real to the agents —
    the receiving node still holds that budget.
    """
    parents = {}
    for src, dst, _amt in edges:
        parents.setdefault(dst, []).append(src)
    kept = [
        (s, d, a)
        for s, d, a in edges
        if len(parents[d]) == 1 or s == parents[d][0]
    ]
    nodes = {s for s, _d, _a in kept} | {d for _s, d, _a in kept}
    return all(
        _out_flow(kept, n) <= _in_flow(kept, n, root_budget) for n in nodes
    )


def test_p2_tree_admits_an_allocation_permitting_more_than_the_root_budget():
    edges = [
        (ROOT, "a", 50),
        (ROOT, "b", 50),
        ("a", "d", 30),
        ("b", "d", 30),   # unrepresentable: d already has parent "a"
        ("b", "e", 30),   # b re-grants budget it has already committed to d
    ]
    assert tree_admits(edges, 100) is True
    assert permitted_total(edges, 100) == 110   # a:20 + b:0 + d:60 + e:30
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p2_tree_admits -v`
Expected: PASS. If `permitted_total` returns 100 rather than 110, the `max(0, …)` clamp was dropped — an over-committed `b` is silently offsetting the rest.

- [ ] **Step 3: Show the DAG law rejects the same allocation**

```python
def test_p2_dag_law_rejects_the_over_commitment():
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d", "e"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "d", tokens=30)
    graph.allocate("b", "d", tokens=30)
    with pytest.raises(ConservationViolationError):
        graph.allocate("b", "e", tokens=30)


def test_p2_dag_law_accepts_the_same_graph_without_the_over_commitment():
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "d", tokens=30)
    graph.allocate("b", "d", tokens=30)
    graph.seal()
    assert graph.in_flow("d").tokens == 60
    edges = [(ROOT, "a", 50), (ROOT, "b", 50), ("a", "d", 30), ("b", "d", 30)]
    assert permitted_total(edges, 100) == 100   # exactly the root budget
```

`FlowConservationError` subclasses `ConservationViolationError`, so the `pytest.raises` above catches either.

- [ ] **Step 4: Run both**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p2 -v`
Expected: 3 PASS.

- [ ] **Step 5: Mutation-check**

Change `tree_admits` to keep every edge (delete the `kept` filter). `test_p2_tree_admits_an_allocation_permitting_more_than_the_root_budget` must then FAIL, because a complete accountant sees `b`'s 60 against its 50 and refuses. Restore afterwards.

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p2_tree_admits -v`
Expected: FAIL while mutated, PASS after restoring. If it passes while mutated, `tree_admits` is not checking every node.

- [ ] **Step 6: Commit**

```bash
git add tests/core/test_delegation_graph_propositions.py
git commit -m "test: P2 tree accounting admits an over-committed allocation"
```

---

### Task 2: P3 — what acyclicity is actually necessary for

**P3 as written in the spec is suspected false.** The telescoping never uses acyclicity: summing `in_v >= C(v) + out_v` cancels every internal edge whether or not there are cycles. Determine that empirically, then restate P3 around whatever acyclicity *is* load-bearing for — the whitepaper points at reclamation ("a node could refund its own ancestor").

**Files:**
- Modify: `tests/core/test_delegation_graph_propositions.py`

**Interfaces:**
- Consumes: `permitted_total` from Task 1.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Test whether the static bound survives cycles**

```python
def test_p3_static_bound_survives_budget_cycles():
    """If this PASSES, P3 as specified is false and must be restated."""
    edges = [(ROOT, "a", 100), ("a", "b", 50), ("b", "a", 50)]
    assert permitted_total(edges, 100) == 100
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p3_static -v`
Expected: PASS — cycles telescope like any other internal edge. If it FAILS, P3 stands as written; record that and skip to Step 5.

- [ ] **Step 3: Search for a cyclic counterexample, and assert the search was not vacuous**

```python
import random


def _is_valid_allocation(edges, root_budget):
    """Every node forwards no more than it received."""
    nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
    return all(
        _out_flow(edges, n) <= _in_flow(edges, n, root_budget) for n in nodes
    )


def _generate(rng, force_cycle):
    names = ["a", "b", "c"]
    edges = [(ROOT, rng.choice(names), rng.randint(1, 60))]
    for _ in range(rng.randint(1, 4)):
        src, dst = rng.sample(names, 2)
        edges.append((src, dst, rng.randint(1, 30)))
    pairs = {(x, y) for x, y, _a in edges}
    has_cycle = any((d, s) in pairs for s, d, _a in edges)
    return edges if has_cycle == force_cycle else None


def test_p3_cyclic_and_acyclic_allocations_saturate_identically():
    """Valid allocations saturate B(root) exactly, cycles or not.

    Assert the IDENTITY, not `<= root_budget`. The inequality is implied by
    `_is_valid_allocation` alone — the filter asserts the very invariant whose
    consequence is under test — so an inequality here can never fail and proves
    nothing. The identity plus the acyclic control is what carries the claim:
    the two populations behave the same, so acyclicity is nowhere used.
    """
    rng = random.Random(20260823)
    counts = {True: 0, False: 0}
    for force_cycle in (True, False):
        for _ in range(4000):
            edges = _generate(rng, force_cycle)
            if edges is None or not _is_valid_allocation(edges, 100):
                continue
            counts[force_cycle] += 1
            assert permitted_total(edges, 100) == 100, edges
    assert counts[True] >= 50, f"only {counts[True]} cyclic allocations"
    assert counts[False] >= 50, f"only {counts[False]} acyclic allocations"


def test_p3_mutation_dropping_the_clamp_breaks_the_identity():
    """Mutation check: without max(0, ...) an over-committed node offsets its
    neighbours, so the identity stops detecting over-commitment."""
    def unclamped(edges, rb):
        nodes = {s for s, _d, _a in edges} | {d for _s, d, _a in edges}
        return sum(_in_flow(edges, n, rb) - _out_flow(edges, n) for n in nodes)

    over = [(ROOT, "a", 50), (ROOT, "b", 50), ("a", "d", 30), ("b", "d", 30), ("b", "e", 30)]
    assert permitted_total(over, 100) == 110      # clamped: over-commitment visible
    assert unclamped(over, 100) == 100            # unclamped: hidden
```

Three guards. `_is_valid_allocation` rejects edge sets that are not allocations at all — an unfiltered generator breaches the budget because a node forwards what it never received, which says nothing about cycles. The two `>= 50` counts keep both populations non-empty. And the identity plus the mutation test is what makes the pair non-vacuous: measured empirically, every accepted allocation returns exactly 100, so `<= 100` could never have failed.

- [ ] **Step 4: Run both tests**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p3 -v`
Expected: PASS, with both counts above 50. The claim carried is *equality across both populations*: cyclic allocations saturate B(root) exactly as acyclic ones do, so the telescoping argument never uses acyclicity.

- [ ] **Step 5: Assert the property that is actually enforced**

The cyclic-refund scenario cannot be built and then tested: `allocate()` calls
`_reaches()` before inspecting any amount, so the graph is unconstructible. The
enforceable property is precisely that refusal, and that is what the test must
assert — not an arithmetic identity over two literal dicts, which could never
fail and would violate this plan's own Global Constraints.

```python
def test_p3_budget_cycles_are_refused_at_allocation_time():
    """The graph cannot be built, so cyclic reclamation can never arise.

    This is what acyclicity actually buys: not the static bound (see
    test_p3_static_bound_survives_budget_cycles), but a guarantee that
    refund propagation is well-founded, enforced structurally.
    """
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "b", tokens=10)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=10)     # would close the budget cycle


def test_p3_zero_amount_does_not_exempt_a_cycle():
    """The cycle check precedes the amount check, so zero is refused too."""
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=50)
    graph.allocate(ROOT, "b", tokens=50)
    graph.allocate("a", "b", tokens=10)
    with pytest.raises(CycleError):
        graph.allocate("b", "a", tokens=0)
```

Add `CycleError` to the imports at the top of the module.

- [ ] **Step 6: Run and record the restatement**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p3 -v`
Expected: all PASS.

Record in the task notes: **P3 restated** — acyclicity is necessary for reclamation to be well-founded, not for the static bound, which is cycle-robust. Update §2 of the spec accordingly in Task 6.

- [ ] **Step 7: Commit**

```bash
git add tests/core/test_delegation_graph_propositions.py
git commit -m "test: P3 - static bound survives cycles; acyclicity is a reclamation property"
```

---

### Task 3: P4 — tightness under abandonment

**Files:**
- Modify: `tests/core/test_delegation_graph_propositions.py`

**Interfaces:**
- Consumes: `DelegationGraph.abandon`, `.verify`, `.monitor_for`, `.abandon_snapshot`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: State the bound that is actually achievable in v1**

`Σ C(v) == B(root) + Σ refunds` **cannot be saturated**: a refund returns to the
parent and reclaimed budget is not re-delegatable in v1, so total spend maxes out
at `B(root)`. The achievable and interesting tightness is *per-node*: an
abandoned node may consume up to its frozen pre-refund in-flow before detection,
and one token more is caught. That is the whitepaper's own claim — "an abandoned
node can consume up to its refunded amount before detection" — and it is what the
test must pin.

```python
def test_p4_abandonment_bound_is_tight():
    graph = DelegationGraph(make_root(100))
    for name in ("live", "doomed"):
        graph.add_node(name)
    graph.allocate(ROOT, "live", tokens=40)
    graph.allocate(ROOT, "doomed", tokens=60)
    graph.seal()

    graph.monitor_for("live").usage.add_tokens(40)      # spends all of its share
    graph.monitor_for("doomed").usage.add_tokens(10)
    refund = graph.abandon("doomed")
    assert refund.tokens == 50                          # 60 granted - 10 spent

    # The abandoned node keeps spending, up to exactly the refunded amount.
    graph.monitor_for("doomed").usage.add_tokens(50)
    total = sum(graph.monitor_for(n).usage.tokens for n in ("live", "doomed"))
    assert total == 100                                 # == B(root); refunds unusable in v1
    graph.verify()                                      # doomed sits exactly on its frozen in-flow

    # One token past the frozen in-flow breaks it.
    graph.monitor_for("doomed").usage.add_tokens(1)
    with pytest.raises(ConservationViolationError):
        graph.verify()
```

- [ ] **Step 2: Run and correct to the real API**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p4 -v`
Expected: PASS. Two API facts, both verified: `add_tokens` lives on
`ResourceUsage` (`monitor.py:79`), and the accumulated total is
**`usage.tokens`** — there is no `total_tokens`, and summing
`reasoning_tokens + text_tokens` gives 0 because a plain `add_tokens(n)` splits
nothing. Check what `abandon()` returns before asserting on `.tokens`.

- [ ] **Step 3: Find the exact tightness point**

If `verify()` raises earlier than the construction predicts, reduce the
post-abandon spend one token at a time until it passes, then assert on that
exact value and on the value one greater. The test must pin the boundary, not
straddle it.

- [ ] **Step 4: Run to confirm both halves**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p4 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/core/test_delegation_graph_propositions.py
git commit -m "test: P4 abandonment bound is tight at B(root) + refunds"
```

---

### Task 4: P5 — confluence of reclamation

**Files:**
- Modify: `tests/core/test_delegation_graph_propositions.py`

**Interfaces:**
- Consumes: `DelegationGraph.release`, `_proportional_share` semantics (refunds computed against **original** allocations).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the property test over release orders**

```python
import itertools


def _two_parent_graph(a=40, b=40, ad=10, bd=10):
    graph = DelegationGraph(make_root(100))
    for name in ("a", "b", "d"):
        graph.add_node(name)
    graph.allocate(ROOT, "a", tokens=a)
    graph.allocate(ROOT, "b", tokens=b)
    graph.allocate("a", "d", tokens=ad)
    graph.allocate("b", "d", tokens=bd)
    graph.seal()
    return graph


def test_p5_release_order_does_not_change_residuals():
    orders = list(itertools.permutations([("a", "d"), ("b", "d")]))
    results = []
    for order in orders:
        graph = _two_parent_graph()
        for src, dst in order:
            graph.release(src, dst)
        results.append(
            {n: graph.residual(n).tokens for n in graph.node_names()}
        )
    assert all(r == results[0] for r in results), results
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p5_release_order -v`
Expected: PASS.

- [ ] **Step 3: Mutation-check by simulating live-value refunds**

Add a test-local reimplementation that computes each edge's refund against the node's *live* residual rather than its original allocation, and assert that this version **is** order-dependent. This is the counterexample half of P5.

```python
def test_p5_live_value_refunds_are_order_dependent():
    def simulate(order):
        original = {("a", "d"): 10, ("b", "d"): 10}
        live = dict(original)
        refunds = {}
        for edge in order:
            total_live = sum(live.values())
            refunds[edge] = live[edge] / total_live * 20 if total_live else 0
            del live[edge]
        return refunds

    forward = simulate([("a", "d"), ("b", "d")])
    reverse = simulate([("b", "d"), ("a", "d")])
    assert forward != reverse
```

- [ ] **Step 4: Run both halves**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p5 -v`
Expected: 2 PASS. If Step 3 fails (the live version turns out order-independent too), pick a more asymmetric allocation — unequal edge amounts and partial consumption at `d` — until the dependence shows.

- [ ] **Step 5: Extend to genuinely random graphs**

The randomised version must build a *different* graph each iteration and
compare release orders **within** that graph. Building a random graph and then
testing a fixed one is the same vacuity the P3 search had to guard against.

```python
def test_p5_confluence_holds_on_random_two_parent_graphs():
    rng = random.Random(20260824)
    for trial in range(200):
        # root is 100, so a + b <= 100; and a scout cannot forward more than
        # it received, so ad <= a and bd <= b. Unconstrained draws raise
        # FlowConservationError on the first trial.
        a = rng.randint(10, 60)
        b = rng.randint(10, 100 - a)
        params = dict(a=a, b=b, ad=rng.randint(1, a), bd=rng.randint(1, b))
        base = None
        for order in itertools.permutations([("a", "d"), ("b", "d")]):
            g = _two_parent_graph(**params)      # SAME params, different order
            for src, dst in order:
                g.release(src, dst)
            snapshot = {n: g.residual(n).tokens for n in g.node_names()}
            if base is None:
                base = snapshot
            assert snapshot == base, (trial, params, order, snapshot, base)
```

- [ ] **Step 6: Run and commit**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p5 -v`
Expected: 3 PASS.

```bash
git add tests/core/test_delegation_graph_propositions.py
git commit -m "test: P5 reclamation confluence, with live-value counterexample"
```

---

### Task 5: P6 — locality separation (highest risk)

P6 claims `consumption <= in-flow` is node-locally decidable while the `+ out-flow` term is not. Without a precise definition of "node-local" this is not a proposition. **Define it first; if the definition cannot be made precise in one sitting, P6 fails the gate and that is an acceptable outcome.**

**Files:**
- Modify: `tests/core/test_delegation_graph_propositions.py`

**Interfaces:**
- Consumes: `DelegationGraph.check_node`, `.monitor_for`, `.verify`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the definition down as a docstring before any code**

```python
def test_p6_locality_separation():
    """A node-local check may read only:
      - its own contract (materialized from its summed in-flow), and
      - its own recorded usage.
    It may NOT enumerate its out-edges, read another node's state, or
    consult the graph. Claim: under this definition, `consumption <=
    in-flow` is decidable and `consumption + out-flow <= in-flow` is not.
    """
```

- [ ] **Step 2: Write the test showing the monitor decides the in-flow half alone**

```python
    graph = DelegationGraph(make_root(100))
    graph.add_node("w")
    graph.allocate(DelegationGraph.ROOT, "w", tokens=40)
    graph.seal()
    monitor = graph.monitor_for("w")
    monitor.usage.add_tokens(41)
    # check_constraints() -> list[ViolationInfo] (monitor.py:311). It NEVER
    # returns None, so assert on emptiness, not identity.
    assert monitor.check_constraints() != []        # local knowledge suffices
```

- [ ] **Step 3: Write the test showing out-flow is invisible to that same monitor**

```python
    graph2 = DelegationGraph(make_root(100))
    for name in ("w", "child"):
        graph2.add_node(name)
    graph2.allocate(DelegationGraph.ROOT, "w", tokens=40)
    graph2.allocate("w", "child", tokens=35)
    graph2.seal()
    m = graph2.monitor_for("w")
    m.usage.add_tokens(30)          # 30 consumed + 35 delegated = 65 > 40
    assert m.check_constraints() == []        # the monitor sees no violation
    with pytest.raises(ConservationViolationError):
        graph2.check_node("w")                # only the graph can see it
```

- [ ] **Step 4: Run both halves**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -k p6 -v`
Expected: PASS.

**Distinguish two failure modes before recording anything.** If the second
assertion fails because the monitor *does* report a violation, P6 is genuinely
false — contract materialization already folds out-flow in. If it fails on a
type or attribute error, that is a test bug, not a falsified proposition. Only
the first outcome goes in the gate tally.

- [ ] **Step 5: Commit**

```bash
git add tests/core/test_delegation_graph_propositions.py
git commit -m "test: P6 locality separation under an explicit node-local definition"
```

---

### Task 6: Gate decision and §4.6 rewrite

**Files:**
- Modify: `docs/whitepaper.md` (§4.6)
- Modify: `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md` (§2 status column)

**Interfaces:**
- Consumes: the pass/fail record from Tasks 1–5.
- Produces: the gate decision that the implementation track's sweep launch depends on.

- [ ] **Step 1: Tally the surviving propositions**

Run: `uv run pytest tests/core/test_delegation_graph_propositions.py -v`

Count how many of P2–P6 have a passing executable artifact **and** a statement that survived contact with it unchanged. A proposition that had to be restated (P3 is expected to) still counts as surviving — restated, not abandoned.

- [ ] **Step 2: Apply the gate**

If **three or more** of P2–P6 survive, continue to Step 3.

If **fewer than three**, stop and write the re-target decision into the spec's §2: target AAMAS Findings deliberately, or defer to a later venue with M5 data. Do not proceed to Step 3, and notify before the implementation track launches its sweep.

- [ ] **Step 3: Rewrite §4.6 as numbered propositions**

Replace each prose paragraph with a numbered proposition, its assumptions, its proof, and a pointer to the test that exercises it. Keep the existing Kirchhoff framing for P1 but state plainly that it is the easy half; the paper's theoretical weight rests on P2–P6.

- [ ] **Step 4: Update the spec's §2 status column**

Change each row's Status from "asserted only" to "proved, `test_pN_*`" or "restated (see §4.6)" or "failed the gate".

- [ ] **Step 5: Verify nothing else regressed**

Run: `uv run pytest tests/core/ -q`
Expected: all pass, no change to pre-existing test counts.

- [ ] **Step 6: Commit**

```bash
git add docs/whitepaper.md docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md
git commit -m "docs: §4.6 as numbered propositions P1-P6; record Aug 29 gate outcome"
```
