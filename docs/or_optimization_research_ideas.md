# OR / Optimization Research Ideas for Agent Contracts

**Status**: Backlog (not started) — captured for later pickup
**Created**: 2026-05-30
**Owner**: qingye
**Source**: Discussion sparked by TDS article
["Optimizing AI Agent Planning with Operations Research and Data Science"](https://towardsdatascience.com/optimizing-ai-agent-planning-with-operations-research-and-data-science/)
**Related code**: `src/agent_contracts/core/planning.py`, `src/agent_contracts/core/delegation.py`
**Related evaluators**: `evaluation/indeterminacy_evaluator.py`

---

## TL;DR

The TDS article applies four textbook Operations Research models — **set-covering,
assignment, knapsack, network-flow** — to *agent architecture decisions*, solved
with Gurobi. It is a useful tutorial but rests on a naive assumption it states
plainly: **deterministic, known, fixed** agent costs/outputs/values, with linear
and independent effects, decided **offline/statically**.

Our framework already (a) approximates several of these problems with hand-tuned
heuristics, and (b) knows something the article ignores: **the values are
stochastic and ambiguous** (this is the entire reason `IndeterminacyEvaluator`
exists, and the lesson behind the "N=3 vs N=20" finding in CLAUDE.md).

So the on-thesis upgrade is **stochastic / robust optimization**, not deterministic
LP. Three concrete directions are recorded below, ranked by value.

---

## How the article maps onto our current code

| Article OR model | Where we do this today | Current gap |
|---|---|---|
| **Knapsack** (max output s.t. budget) | `plan_resource_allocation` → `_allocate_urgent / _economical / _balanced` (`planning.py:154-277`) | Magic ratios (1.2, 0.7, 0.8, 0.9); no optimality guarantee |
| **Assignment** (agent↔task value) | `prioritize_tasks` sort key (`planning.py:317`) | Greedy lexicographic sort, not joint optimization |
| **Set-covering** (min agents to cover skills) | `SkillSpec` / `Capabilities.skills` | No selection logic exists at all |
| **Conservation budget split** | `ContractingCapability.create_subcontract` (`delegation.py:420`) | Caller picks `tokens=40_000` *by hand*; nothing optimizes the split subject to `Σ bᵢ ≤ B − used` |

**Key observation**: our heuristic allocators are an *approximation of a known
problem class*. The conservation law `Σ bᵢ ≤ B − used` (`delegation.py:11-17`) is
*literally* an OR constraint — we just choose the decision variables manually.

---

## Why NOT to copy the article directly

Naive adoption would regress findings we already established:

1. **Determinism assumption contradicts our own results.** The article assumes
   fixed point-estimate values. Our central empirical learning (CLAUDE.md
   "Scientific Process": *N=3 showed 50% variance reduction, N=20 showed the
   opposite*) and the whole `IndeterminacyAwareEvaluator` exist because output
   value is stochastic and ambiguous. A knapsack solved on point estimates would
   be confidently optimal and frequently wrong.
2. **Static vs runtime.** The article never re-plans after an overrun. Our
   differentiator is *online* governance (monitor + enforcement + conservation
   during execution). Live example: chamber pilot `planner_reasoner` timeouts at
   k=59.
3. **We already model what it assumes away** — coordination overhead
   (`reserve_ratio`), hierarchical recursion, per-tool axes. Flattening to the
   article's independent-linear world is a step down.

**Anti-goal**: do NOT add a hard Gurobi dependency, and do NOT replace the
runtime heuristics wholesale. Heuristics are O(1) inside the enforcement hot
path; an LP solve per re-plan could dominate latency, and Gurobi licensing is
hostile to an open-source framework. Prefer `scipy.optimize` / PuLP / OR-Tools
(permissive licenses) and keep any solver path optional.

---

## Idea 1 — Stochastic budget allocation in `delegation.py` (HIGHEST VALUE, on-thesis)

**What**: Replace manually-chosen child budgets `bᵢ` with an optimizer that
maximizes *expected* output subject to the existing conservation law, using the
uncertainty `IndeterminacyEvaluator` already estimates.

**Formulation sketch**:
```
maximize   E[ Σ value(child_i) · b_i ]
subject to Σ b_i ≤ B − used            (existing conservation law)
           P(overrun_i) ≤ ε            (chance constraint, replaces hard point estimate)
           b_i ≥ b_i^min for required children
```
This is **robust / chance-constrained knapsack**. The `P(overrun) ≤ ε` term is
the contribution the article lacks.

**Why it's the strongest**: exploits the exact thing the article ignores
(calibrated uncertainty), lands directly on the framework paper's "contracting
helps" thesis, and slots into an existing extension point
(`create_subcontract`) without disrupting the enforcement hot path (allocation
happens at delegation time, not per-token).

**Open questions to resolve in brainstorming**:
- Which uncertainty model feeds the chance constraint? (indeterminacy response
  sets? historical per-agent token distributions? a fitted Gaussian/lognormal?)
- Is this an optional `allocate_optimally()` helper on `ContractingCapability`,
  or a separate planner that *calls* `create_subcontract` with computed values?
  (Leaning: separate planner, so conservation enforcement stays the single
  source of truth.)
- Target venue: framework paper.

**Touch points**: `delegation.py` (new optional planner alongside, not inside,
`create_subcontract`); `indeterminacy_evaluator.py` (value/uncertainty source).

---

## Idea 2 — Optimal experiment selection baseline for the chamber pillar (MOST CONCRETE, paper-relevant)

**What**: Our `GreedyIG-lite` chamber baseline (greedy information-gain
experiment selection under budget k) *is* a greedy knapsack. Add an OR-backed
variant that selects experiments to optimize information-gain-per-token under the
budget — a provably-stronger baseline than the greedy one.

**Formulation sketch**:
```
maximize   Σ infogain(experiment_j) · x_j
subject to Σ cost_tokens(experiment_j) · x_j ≤ k
           x_j ∈ {0,1}
```
(Submodular IG ⇒ greedy is already (1−1/e)-competitive, so the *interesting*
research question is how much the optimal solver actually beats greedy on the
real chambers, and whether the gap matters next to the LLM variants.)

**Why it's valuable**: directly sharpens the §5.3 story
(`docs/causal_chamber_validation_plan.md`). M4b's headline is "DeepSeek + summary
dominates the Pareto." Giving the planner variants a tougher OR-backed competitor
to beat makes that dominance claim more defensible to AAMAS/ECAI reviewers.

**Open questions**:
- Is this a new variant in the variant registry, or a replacement for
  GreedyIG-lite? (Leaning: new variant, keep GreedyIG-lite for the
  greedy-vs-optimal comparison — that gap is itself a result.)
- LT-only (like GreedyIG-lite per plan §5.1) or both chambers?
- Cost: adds cells to the M5 sweep — quantify before committing.

**Touch points**: `evaluation/chamber_pipeline/` (variant registry, agents).

---

## Idea 3 — Solver-backed reference allocator as a validation oracle (LOWEST RISK)

**What**: Add an *optional* OR allocator purely to **measure how far the cheap
heuristics in `planning.py` are from the LP/MIP optimum**. Not used at runtime —
a measurement/validation tool.

**Why it's valuable**: turns a soft spot into a defensible claim. If
`_allocate_balanced` is within ~3% of the optimum, that's a publishable defense
of using the O(1) heuristic in the hot path. If it's 30% off, we've found a real
bug. Either outcome is useful.

**Formulation**: same knapsack/assignment as Idea 1 but *deterministic* (point
estimates are fine here — the goal is heuristic↔optimum gap, not real allocation).

**Open questions**:
- Where does it live — a `tests/` benchmark, an `evaluation/` script, or a
  `benchmarks/` demo?
- Which heuristic(s) to audit first? (`_allocate_balanced` is the default path.)

**Touch points**: `planning.py` (oracle compares against `_allocate_*`); likely
a new `evaluation/` or `benchmarks/` harness.

---

## Suggested sequencing

1. **Idea 3 first** if we want a low-risk warm-up that produces a paper-ready
   number quickly and de-risks the OR tooling choice (scipy vs PuLP vs OR-Tools).
2. **Idea 1 or 2** as the real contribution, depending on which paper is active:
   - Framework paper active → **Idea 1**.
   - Chamber paper (AAMAS 2027 / ECAI 2027) active → **Idea 2**.

Before writing code for any of these, run `superpowers:brainstorming` to pin
scope (especially the uncertainty model in Idea 1 and the cell-count cost in
Idea 2).

---

## Decisions deferred

- OR tooling choice (scipy.optimize / PuLP / OR-Tools) — pick during Idea 3.
- Which paper Idea 1 vs Idea 2 attaches to — decide when M5 status is clearer.
- Whether any of this is in scope before COINE 2026 (May 25–26) — likely no;
  this is post-COINE / M5-era work.
