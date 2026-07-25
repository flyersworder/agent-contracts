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

### 3.2 Soundness

**Claim.** If the local invariant holds at every node, then `Σ_v C(v) ≤ B(root)`.

**Proof.** Sum the invariant over all `v ∈ V`. Every internal allocation `a(u→v)`
appears once positively as in-flow at `v` and once negatively as out-flow at `u`, so
internal terms cancel. Only the root's exogenous budget survives on the left and
total system consumption on the right. ∎

The corollary is the design's whole point: **each node checks only its own edges, and
global boundedness follows.** No global lock, no central accountant — which is what a
multi-agent org graph needs and what graph engineering currently has no answer for.

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
- every non-root node has at least one in-edge (no node starved of budget);
- no node's out-flow already exceeds its in-flow;
- per-tool allocations are consistent with parents' `per_tool_limits`.

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
| `allocate()` | source's out-flow ≤ source's in-flow (satisfiability) | `FlowConservationError`, fail fast in build phase |
| `allocate()` | edge would not create a cycle (DFS from target for source) | `CycleError` |
| `seal()` | acyclicity, orphans, per-tool consistency | aggregated report of all problems |
| runtime | node `v`'s invariant on every monitor update | existing enforcement path |

The runtime check requires **no new enforcement code**. Because each node's `Contract`
is materialized with its summed in-flow as `ResourceConstraints`, the existing
`ResourceMonitor.check_constraints()` already enforces the invariant at that node,
including strict/lenient modes, callbacks, and per-tool priority ordering. The graph
layer owns only the edges.

### 6.2 Violation payload and blame

`FlowConservationError` carries `node_id`, `dimension`, `in_flow`, `consumed`,
`out_flow`, `deficit`, and `contributing_edges`.

`contributing_edges` is an audit trail, not blame assignment. The invariant is checked
at `v`, so `v` is at fault. Parents are only ever accountable for their own out-flow.
No cross-parent arbitration is needed.

### 6.3 Refunds are proportional

`release(u, v)` returns edge `(u→v)`'s share of `v`'s residual:

```
refund(u→v) = a(u→v) / Σ_in a(·→v) × residual(v)
```

The alternatives (LIFO, first-come) are **order-dependent**, which would make a
30-seed sweep non-reproducible depending on the order releases happen to fire.
Proportional refunds are order-independent by construction.

`abandon(node)` refunds unconsumed allocation to parents proportionally and marks
downstream nodes unreachable. This exists because M4b produced 8 `planner_reasoner`
timeouts at k=59; in a DAG those become stranded allocations that silently corrupt
every downstream residual for the remainder of a sweep.

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
