# Delegation Flow Conservation and the M6 Topology Benchmark

**Date**: 2026-07-25
**Status**: Design approved, ready for implementation planning
**Supersedes**: `causal_chamber_validation_plan.md` §7 (cross-pillar transfer study) as the M6 milestone

## 1. Motivation

Two terms entered circulation in mid-2026:

- **Loop engineering** (named 2026-06-07, Addy Osmani): designing the system that
  prompts the agent rather than prompting it yourself. Five components —
  automations, worktrees, skills, plugins/connectors, subagents — plus persistent
  external state. Its central concern is *the trigger, the topology, the verifier,
  and the stop rules*.
- **Graph engineering** (named ~2026-07-18): designing multi-agent organizations as
  programmable structures. Nodes are agents running their own loops; edges are data
  flow and dependencies. Splits into a stable *org graph* and an ephemeral *work
  graph*.

Both name a control problem and then decline to solve it:

- Loop engineering offers no formalism for resource bounds. Osmani's essay says only
  "you absolutely have to be careful about token costs."
- Graph engineering says nothing about **how budget flows along an edge**. Its own
  field guide names *graph linting* as one of two things nobody has shipped, and
  critical reviews note that as of 2026-07-20 there was no standard definition, no
  canonical reference architecture, no peer-reviewed evidence, and **no comparative
  benchmark**.

Agent Contracts already answers much of the loop side. `TerminationCondition`
(`core/contract.py:537`) formalizes stop rules as Ψ; `SuccessCriterion` formalizes
the verifiable goal as Φ; `SkillSpec` implements the agentskills.io `SKILL.md`
standard that loop engineering names as a component; the JSONL checkpoint sidecar is
the persistent external state.

The graph side has a real gap. `ContractingCapability` (`core/delegation.py`) models
a **tree**: child IDs are hierarchical paths (`f"{parent.id}/{name}"`, line 477) and
`_check_conservation` sums allocations against one parent's budget. Graph
engineering's work graph requires **fan-in** — a node with two upstream parents —
which the current conservation law would double-count.

This design closes that gap and uses it to run the comparative benchmark the field
lacks.

## 2. Scope

**In scope:**

1. A flow-conservation delegation graph generalizing the tree law to a DAG.
2. Tracking and enforcement of `ResourceConstraints.iterations`, currently declared
   but not tracked (see the stale comment at `core/prompts.py:174`).
3. A three-arm topology experiment (M6) on the existing chamber harness.

**Out of scope (deliberate):**

- Dynamic work-graph mutation (runtime spawn/merge/cancel). v1 is build-then-seal.
- Thread safety. `run_sweep` is serial today; see §6.4.
- New top-level types named after the trend (`LoopContract`, `GraphContract`). We add
  capability to existing primitives instead.
- Knowledge-graph / typed-edge memory, the *other* reading of "graph engineering".

## 3. Formal model

The delegation graph is a DAG `G = (V, E)`. Each node is a contract. Each edge
carries an allocation `a(u→v)`: a resource vector over tokens, cost, tool
invocations, per-tool counts, and iterations.

```
        ┌─ B ─┐              B(v)  = Σ  a(u→v)      node budget = sum of in-edges
   A ───┤     ├─── D         C(v)  = what v itself consumes
        └─ C ─┘
                             Invariant at every node v:
   A→B: 40k tokens             Σ a(u→v)  ≥  C(v) + Σ a(v→w)
   A→C: 40k                    └in-flow┘     └own┘   └out-flow┘
   B→D: 15k  ⎫ D's budget
   C→D: 15k  ⎭ = 30k
```

In words: **in-flow ≥ own consumption + out-flow.** Kirchhoff's current law with a
consumption sink at each node.

### 3.1 Relationship to the existing tree law

The current law is the special case where every non-root node has exactly one
in-edge. This is a strict generalization; no existing behavior changes. §7 test 1
verifies this empirically rather than by assertion.

Two qualifications, both verified by tests rather than left implicit:

- **Equivalence is exact at `reserve_ratio=0`.** `ContractingCapability` optionally
  withholds a coordination reserve (`remaining_tokens` subtracts `reserved_tokens`),
  and the flow graph has no analogue: a reserve is a policy about *how much of an
  in-flow to delegate*, not a conservation constraint. At `reserve_ratio > 0` the two
  laws differ by exactly the reserve, which the cross-validation test asserts.
- **`residual()` is signed; `remaining_tokens` clamps at zero.** The tree law reports
  `0` both for "exactly spent" and "overspent"; a graph residual goes negative so an
  overrun stays visible to `verify()`. This is deliberate — the flow invariant needs to
  see the deficit, not a floor.

### 3.2 Soundness

**Claim.** If the local invariant holds at every node, then `Σ_v C(v) ≤ B(root)`.

**Proof.** Sum the invariant over all `v ∈ V`. Every internal allocation `a(u→v)`
appears once positively as in-flow at `v` and once negatively as out-flow at `u`, so
internal terms cancel. Only the root's exogenous budget survives on the left and
total system consumption on the right. ∎

The claim is over the whole resource vector, per-tool counts included. That holds only
because the node invariant reads an *undeclared* tool as a budget of zero for every node
but the root (§5.2); under `≤` alone the per-tool components would cancel vacuously and
the bound would be real for tokens and cost while empty for experiments.

The corollary is the design's whole point: **each node checks only its own edges, and
global boundedness follows.** No global lock, no central accountant — which is what a
multi-agent org graph needs and what graph engineering currently has no answer for.

**Scope of what `verify()` certifies.** The proof above sums the invariant as checked
against each node's *live* in-flow, consumption, and out-flow. `verify()` establishes
that premise directly for live nodes. For an abandoned node it does not: per §6.3, the
check is against the node's frozen, pre-refund in-flow rather than its live in-flow,
because checking against live values would let the very refund abandonment triggered
launder a real overspend. What `verify()` actually certifies is therefore the looser

```
Σ_v C(v) ≤ B(root) + Σ refunds
```

not the tighter bound the theorem states. Absent abandonment the two coincide — no
refunds means frozen and live in-flow are identical, and `Σ_v C(v) ≤ B(root)` holds
exactly as proven. The bound is tight, not merely a conservative estimate: an abandoned
node can consume up to the amount it was refunded, and no more, without `verify()`
flagging it — that budget still sits inside its frozen in-flow. This is a deliberate,
bounded residual (§6.3's "second door"), accepted because the alternative — freezing
consumption too — reopens the same hole from the other side by letting a dead node
spend without limit post-mortem. It is a scope statement on what the verifier
certifies, not a defect in the theorem: the theorem is about the live graph, and holds
for it.

### 3.3 Control flow may cycle; budget flow must not

Graph engineering explicitly wants loops inside graphs — a node retries, a council
re-deliberates. That is fine: the *control* graph may contain cycles. The *budget*
graph must remain acyclic, or the telescoping argument collapses and a node can
refund its own ancestor. Cycle-creating edges are rejected.

This distinction clarifies a confusion present in both camps' writing and is worth
stating explicitly in the paper.

### 3.4 Unbounded dimensions

`ResourceConstraints` fields are `int | None`. In flow arithmetic `None` means
unbounded (∞): if a parent's dimension is unbounded, children may draw any finite
amount and the invariant on that dimension is vacuously satisfied. This must be
explicit in the implementation, not incidental — the failure mode otherwise is
`TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'` surfacing hours
into a sweep.

## 4. Module boundary

New module `src/agent_contracts/core/delegation_graph.py`. `core/delegation.py` is
untouched.

Rationale: `delegation.py` is ~730 lines with one clear responsibility (tree
delegation). Folding DAG flow into it would push it past 1000 lines and tangle two
responsibilities. Keeping them separate also means the existing 1073 tests stay green
by construction.

## 5. Components and API

Lifecycle: **build → seal → run.**

```python
graph = DelegationGraph(root_contract)        # root node auto-registered

# --- build phase: nodes carry no budget until an edge feeds them
graph.add_node("scout_a", capabilities=scout_caps)
graph.add_node("scout_b", capabilities=scout_caps)
graph.add_node("aggregator", capabilities=agg_caps)

# --- allocation flows along edges
graph.allocate("root", "scout_a", tokens=40_000, tool_invocations=10)
graph.allocate("root", "scout_b", tokens=40_000, tool_invocations=10)
graph.allocate("scout_a", "aggregator", tokens=15_000)
graph.allocate("scout_b", "aggregator", tokens=15_000)   # fan-in: budget accumulates

graph.seal()                                  # validate whole DAG, freeze topology

# --- run phase
contract = graph.contract_for("aggregator")   # materialized: 30k tokens
monitor  = graph.monitor_for("aggregator")
graph.residual("scout_a")                     # in-flow − consumed − out-flow
graph.release("scout_a", "aggregator")        # refund unused along one edge
graph.abandon("scout_b")                      # node died: refund, mark downstream
```

| Component | Purpose |
|---|---|
| `DelegationGraph` | Topology, allocation, invariant checking, materialization |
| `GraphNode` | node_id, materialized `Contract`, `ResourceMonitor`, capabilities |
| `EdgeAllocation` | `(source, target)` + resource vector + released flag — DAG analogue of the existing `AllocationRecord` |
| `ResourceVector` | Internal helper: `+`, `−`, `≤` across all resource dimensions |
| `FlowConservationError(ConservationViolationError)` | Subclasses the existing error so current `except` blocks still catch it |
| `CycleError` | Raised when an edge would create a budget cycle |

`add_node(name, **kwargs)` accepts the same contract-shaping fields
`ContractingCapability.create_subcontract` accepts today (capabilities, execution
config, temporal constraints, metadata) — everything except the resource budget, which
arrives via edges rather than at construction. The node's `Contract` is materialized on
first `contract_for()` call after `seal()`, with `ResourceConstraints` built from the
summed in-flow.

### 5.1 `ResourceVector` is a simplification

`delegation.py` currently repeats structurally identical arithmetic in
`remaining_tokens`, `remaining_cost`, and `remaining_per_tool` (lines 271–351) — the
same computation across three types. Expressing the flow invariant without a vector
type would repeat it six more times. One helper with `+`/`−`/`≤` makes the invariant a
single readable line.

### 5.2 `seal()` is graph linting

`seal()` validates:

- acyclicity of the budget graph;
- every non-root node has at least one in-edge, and that in-edge funds it with
  something (an all-zero edge exists but starves the node just the same);
- no node's out-flow already exceeds its in-flow;
- no node's out-edges name a per-tool budget its own in-flow does not constrain.

The per-tool rule needs stating precisely, because `ResourceVector.__le__` compares
only the keys the *budget* side names — the convention `AllocationRecord` already
documents. A tool absent from a node's in-flow is therefore *unconstrained* under `≤`,
not zero, so without an explicit rule a node funded with no experiment budget could both
grant and spend experiments without limit. The graph layer closes this on **both** paths:

- **Grant path.** `allocate()` rejects a per-tool key the source's in-flow does not
  constrain, with `FlowConservationError` naming the tool; `seal()` re-checks the same
  property across the whole graph.
- **Consumption path.** The node invariant treats an undeclared tool as a budget of
  zero, so consuming a tool nobody funded is a violation rather than vacuous compliance.
  Without this clause §3.2's bound would hold on the scalar dimensions but not on the
  per-tool ones — which are precisely the M6 conserved resource.

The **root is exempt on both paths** — its budget is exogenous, so a tool it leaves
unconstrained is genuinely unbounded rather than absent. A grant of **zero** is always
allowed, even for an undeclared tool: it strictly tightens the child, and for a node with
no budget for that tool it is the only way to constrain the child at all. `__le__` itself
is unchanged; the rules live at the graph layer, where the notion of "in-flow" exists.

`FlowConservationError` reports `in_flow=None` for an undeclared dimension. Undeclared is
not a budget of zero, and an audit artifact that conflates the two misstates the rule it
is evidence for.

This matters directly for §8: arm 3 gives the aggregator "0 experiments", which must
materialize as `per_tool_limits == {"exp": 0}` and never as an omitted key.

It reports **all** problems found, not just the first. The graph engineering field
guide names graph linting as an unshipped open problem; this is a budget-flavored
version of it, cheap because we already own the invariant.

### 5.3 Iterations tracking

`ResourceUsage` gains an `iterations` counter; `check_constraints()` enforces
`ResourceConstraints.iterations`. The stale comment at `core/prompts.py:174` is
removed. Rationale: you cannot measure the cost of a loop without counting its turns,
and the M6 arms need comparable per-arm iteration counts.

## 6. Error handling and invariant semantics

### 6.1 When checks fire

| Point | Check | Failure |
|---|---|---|
| `allocate()` | source's consumption + prospective out-flow ≤ source's in-flow | `FlowConservationError`, fail fast in build phase |
| `allocate()` | source's in-flow constrains every per-tool key granted (root exempt) | `FlowConservationError` naming the tool |
| `allocate()` | edge would not create a cycle (DFS from target for source) | `CycleError` |
| `seal()` | orphans, unfunded nodes, out-flow ≤ in-flow, per-tool propagation | aggregated report of all problems |
| runtime | node `v`'s invariant on every monitor update | existing enforcement path |

The runtime check requires **no new enforcement code**, but only for part of the
invariant. Because each node's `Contract` is materialized from its in-flow unmodified
(`contract_for()` passes `in_flow(name)` straight through), the existing
`ResourceMonitor.check_constraints()` enforces `consumed ≤ in_flow` at that node,
node-locally, including strict/lenient modes, callbacks, and per-tool priority ordering.
It does **not** enforce the invariant's `+ out_flow` term, because the materialized
contract never sees a node's out-flow — an internal node with in-flow 40k and out-flow
15k gets a 40k contract from the monitor's point of view, so consuming 30k passes the
monitor (`check_constraints()` returns `[]`) even though `consumed + out_flow = 45k`
exceeds in-flow and `verify()` correctly raises. The `+ out_flow` term is enforced by
`verify()` / `check_node()` alone; the monitor only ever sees `consumed ≤ in_flow`. The
graph layer owns the edges and this half of the check.

**A second exception, the same class of distinction:** `ResourceConstraints.per_tool_limits`
has no deny-by-default notion, so a monitor cannot flag a tool its constraints never
mention. Use of an *undeclared* tool is therefore **post-hoc detection at `verify()` /
`check_node()`, not prevention** — a node can physically make the calls before
`verify()` ever runs, since nothing at call time stops it. Contracting a node with an
explicit `0` for the tools it must not use (M6 arm 3's aggregator) is what turns
detection into prevention: an explicit zero restores node-local enforcement, which is
one more reason a zero grant has to be legal. Every dimension covered by a declared
limit, and the `consumed ≤ in_flow` term of every dimension, is enforced node-locally
as described above; the `+ out_flow` term and undeclared-tool use are not.

### 6.2 Violation payload and blame

`FlowConservationError` carries `node_id`, `dimension`, `in_flow`, `consumed`,
`out_flow`, `deficit`, and `contributing_edges`.

`consumed` and `out_flow` are reported as the separate quantities they are: a node that
delegated nothing and overspent 40k is a consumption overrun, and describing it as an
out-flow of 40k would make the audit artifact wrong exactly where it is most needed. The
message phrasing follows the call site — "would over-allocate" for the build-phase check
in `allocate()`, where the out-flow is prospective, versus a runtime violation at
`check_node()`.

`contributing_edges` is an audit trail, not blame assignment. The invariant is checked
at `v`, so `v` is at fault. Parents are only ever accountable for their own out-flow.
No cross-parent arbitration is needed.

### 6.3 Refunds are proportional

`release(u, v)` returns edge `(u→v)`'s share of `v`'s unused budget. Let `orig(u→v)` be
the edge's allocation as originally granted and `ORIG(v) = Σ orig(·→v)`:

```
pool(v)      = ORIG(v) − consumed(v) − out_flow(v)        floored at 0
refund(u→v)  = orig(u→v) / ORIG(v) × pool(v)              integers rounded down
```

**Shares are computed against original allocations, not live ones.** This is what makes
releases order-independent: a sibling's release changes neither `ORIG(v)` nor
`pool(v)`, so every edge's share is fixed the moment consumption is known, and the
shares sum exactly to the unused budget.

Computing against *live* in-flow would not have this property. With two siblings each
funding 15k of a 30k node that consumed 20k, releasing `scout_a` first reclaims 5k and
leaves `scout_b` only 3k; reversing the order swaps them. A 30-seed sweep would then
depend on the order releases happen to fire. LIFO and first-come fail the same way.
Each edge may be released at most once.

**Precondition on `release`: the target must be done consuming on that edge.** The
formula above is exact, and sibling order irrelevant, only while `consumed(v)` does not
move between releases. Two consequences, documented rather than enforced — requiring a
terminal target would break legitimate staged releases:

- `v`'s `Contract` is materialized once and cached by `contract_for()`. A release does
  **not** shrink it, so `v`'s `ResourceMonitor` keeps authorizing the pre-release
  budget; only the graph-level residual shrinks.
- `release` → consume → `abandon` over-refunds, because the release already paid out a
  share of a pool that the later consumption shrank.

`abandon(node)` refunds unconsumed allocation to parents proportionally and marks
downstream nodes unreachable. This exists because M4b produced 8 `planner_reasoner`
timeouts at k=59; in a DAG those become stranded allocations that silently corrupt
every downstream residual for the remainder of a sweep. Three limits are deliberate:

- **Only in-edges are refunded.** Budget the dead node already delegated downstream
  stays stranded at the child, which may still be running; abandoning the child too is
  what reclaims it.
- **Reclaimed budget is not re-delegatable in v1.** `allocate()` is build-phase only, so
  a refund changes accounting and reporting — the parent's residual, and what `verify()`
  will certify — but nothing can re-spend it inside the sealed graph.
- **Abandoned nodes are still verified.** Abandonment is the timeout case, and a
  timed-out node is the likeliest of all to have overspent; excusing it would let
  `verify()` certify a graph whose total consumption exceeds the root budget.
  `abandon()` freezes the node's in-flow and out-flow into a snapshot, and `verify()`
  checks abandoned nodes against those frozen budget sides — live values would let the
  refund it triggered, or a later release of one of its out-edges, quietly clear a real
  overspend. **Consumption is always read live**, including after death: freezing it too
  would reopen the same hole through the other door, since a node could then spend
  without limit post-mortem and still be certified. What post-mortem spending can hide is
  bounded by the refund, because the frozen in-flow is the pre-refund one — reclaiming
  budget is a reporting change, not a licence to spend it.

### 6.4 Concurrency boundary

v1 is **not thread-safe**, stated rather than assumed. `run_sweep` is serial today.
M4c lists `ThreadPoolExecutor` parallelism as an M5 priority; if that lands,
`allocate`/`release`/`abandon` require a lock, since each reads then writes shared
residuals.

## 7. Testing

New file `tests/core/test_delegation_graph.py`, mirroring `tests/core/test_delegation.py`.

| # | Test | Protects |
|---|---|---|
| 1 | Cross-validation vs `ContractingCapability` — same tree topology and allocation sequence, assert identical residuals and budgets | Turns "strict generalization" into a verified property |
| 2 | Telescoping soundness — generate random DAGs, consume within local invariants, assert `Σ C(v) ≤ B(root)` | §3.2, as an executable check |
| 3 | Fan-in accumulation: two in-edges → budget is their sum | The core new capability |
| 4 | Cycle rejection; orphan detection; `seal()` reports all problems | §5.2 graph linting |
| 5 | Refund order-independence — release edges in different orders, assert identical final state | §6.3 reproducibility |
| 6 | `abandon()` refunds stranded budget; downstream marked unreachable | The M4b timeout failure mode |
| 7 | `None`-as-unbounded arithmetic across every dimension | §3.4 bug class |
| 8 | `iterations` tracked and enforced | §5.3 |
| 9 | An abandoned node that overspent still fails `verify()`, including when the overspend is recorded *after* abandonment | §6.3, the laundering hole and its second door |
| 10 | A per-tool grant the source's in-flow does not constrain is rejected; a grant of zero is not | §5.2, the M6 conserved resource |
| 11 | Consuming a tool the in-flow never funded fails at any node; the root is exempt | §5.2 consumption path |

The generated DAGs carry a per-tool `exp` budget alongside tokens, cost and aggregate
tool invocations, and conserve all four. Cost is granted in exact binary fractions rather
than decimal cents: cents are not representable in binary floating point, so a generator
using them saturates a few ulps above the true residual and the test then fails on its own
rounding instead of on the law.

Tests 1 and 2 are cross-checked by mutation: excusing abandoned nodes in `verify()`, and
double-counting fan-in in `in_flow()`, must each fail on every seed. A property test that
survives such a mutation is not carrying the claim it appears to.

**No new test dependency.** Tests 1 and 2 want property-based generation, but the
project uses no `hypothesis` across its 1073 tests. A seeded pseudo-random DAG
generator with a fixed seed list gives equivalent coverage, stays reproducible, and
matches existing style.

Test 2 is unusual in that it tests a proof rather than a function. If the telescoping
argument has a hole — for example a diamond where an allocation is double-counted
along two paths — generated graphs will find it long before a sweep does.

## 8. M6 experiment: the topology benchmark

**Research question.** At matched total budget, does graph topology help or hurt?

In the chamber setup the budget `k/M` is the fraction of the 59 available experiments
an agent may run, so the conserved resource is *experiment selections* — the per-tool
budget the M4b conservation refactor already handles.

### 8.1 Arms

```
Arm 1  LOOP (llm_only)          root ──────────────► 59 experiments
       already run in M4b

Arm 2  CHAIN (planner_reasoner) root ──► planner ──► reasoner
       already run in M4b              (delegated split)

Arm 3  FAN-IN (new)             root ──┬─► scout_a (30 exp) ─┐
       needs DAG conservation           └─► scout_b (29 exp) ─┴─► aggregator (0 exp)
                                        total = 59, matched
```

Arm 3 registers as a new variant in the existing AgentSpec registry (M4a).

**Split rule.** The diagram shows `k = 59` (the `k/M = 1.00` cell). The rule
generalizes to every budget level: `scout_a` receives `⌈k/2⌉` experiments, `scout_b`
receives `⌊k/2⌋`, and the aggregator receives `0`. The scouts' experiment budgets
therefore always sum to exactly `k`, matching arms 1 and 2 at every budget level. At
`k/M = 0.10` (`k = 6`) that is 3 and 3; at `k/M = 0.50` (`k = 30`) it is 15 and 15.
The aggregator consumes tokens for the merge but no experiment selections, so the
conserved resource stays matched across arms.

### 8.2 Matrix and cost

- Chamber: LT (WT is a stretch goal, contingent on M5's WT data).
- Budgets: `k/M ∈ {0.10, 0.50, 1.00}` — the trimmed M5 set.
- Seeds: 30, matching M4b.
- **New compute: 90 cells** (fan-in arm only). Arms 1 and 2 already exist in
  `runs/m4-pilot.parquet` at the same model, seeds, budgets, and scoring.
- Estimated cost ~$1–2; hours of VPS wall-clock, not days.

**Reuse-validity guard.** Before trusting the M4b arms, re-run a 5-cell subset of
`llm_only` (1 budget × 5 seeds) and confirm it reproduces the May numbers within
noise. If the harness drifted between May and August we discover it for 5 cells rather
than in the final figure.

### 8.3 Pre-registered hypotheses

| | Hypothesis | Status |
|---|---|---|
| H-A | Chain underperforms loop at matched budget | Already supported — M4b: F1 0.397 vs 0.75 |
| H-B | Fan-in recovers delegation cost via exploration diversity | **Open — this is the experiment** |
| H-C | Conservation compliance is 100% across all arms | The governance claim |

H-B is graph engineering's best case, chosen deliberately. Two scouts select
*different* experiments, so their union may cover more causal structure than one
agent's 59 sequential picks — the ensemble effect underlying Council Deliberation and
adversarial-verify patterns. Against that: each scout sees less data, and the
aggregator may introduce merge error. Testing the strongest version of the opposing
claim is what makes a negative result credible rather than a strawman.

H-C is what keeps this a contracting result rather than a causal-discovery topology
benchmark. It is the same reasoning by which the UNCONTRACTED baselines are
non-negotiable for M5.

### 8.4 Metrics and figures

Per cell: SHD, F1 (existing), plus total tokens, experiments used, iterations,
conservation compliance, wall time, error status.

Figures: (a) SHD and F1 versus budget with one line per topology arm; (b) a
"topology tax" plot showing performance delta against the single loop at matched
budget.

Following risk R8's framing in the chamber plan, results are reported as effect sizes
with confidence intervals rather than binary effect/no-effect. Both directions are
publishable: fan-in wins identifies *when* graph structure pays, which neither camp
has established; fan-in loses is a strong negative result against the field's central
claim, backed by the full three-arm matrix of 270 cells at 30 seeds (90 of them new).
The outcome to avoid is not a bad result but an underpowered one.

## 9. Milestone impact

M6 changes from the §7 cross-pillar transfer study to this topology benchmark.

**What is given up:** cross-pillar transfer evidence. The §7 post-M4b callout already
judged deferring it to a journal extension viable (path 1 of two sanctioned options),
so this selects the timelier of two paths already approved rather than making a new
sacrifice.

**Timeline.** The capability is built 2026-07-27 → 08-23, in parallel with M5's ~900
cells occupying the VPS — M5 is wall-clock-bound, so developer time during it is
effectively free. The fan-in arm runs early in the M6 window (08-24 → 09-13), leaving
roughly two weeks for analysis and figures before M7 drafting.

## 10. References

- Addy Osmani, "Loop Engineering," 2026-06-07; republished O'Reilly Radar 2026-06-22.
- Graph engineering field guide and critical review, July 2026 (see §1).
- `docs/causal_chamber_validation_plan.md` §5.1 (variants), §6.1 (budgets), §7
  (superseded), §9 (milestones), R8 (effect-size framing).
- `docs/or_optimization_research_ideas.md` Idea 1 — stochastic budget allocation in
  `delegation.py`. Flow conservation is the structural prerequisite for it.
- `src/agent_contracts/core/delegation.py` — the tree law being generalized.
- `runs/m4-pilot.parquet` — arms 1 and 2, 450 cells, 30 seeds.
