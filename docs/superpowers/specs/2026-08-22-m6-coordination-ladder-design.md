# M6 — Coordination Ladder: Design Spec

**Date:** 2026-08-22 (rev. 2026-08-23, two independent review rounds — see §11)
**Status:** Approved design; theory track complete (gate cleared 2026-08-23), implementation track pending
**Target:** AAMAS 2027 main track (COINE area). Abstract 2026-10-01, paper 2026-10-08.
**Supersedes:** `docs/causal_chamber_validation_plan.md` §7.7 (see §10)

---

## 1. Framing

The paper is a **governance-mechanism paper**, not a topology benchmark.

Prior matched-budget studies control cost either by a cap that is never enforced
(Tran & Kiela, arXiv:2604.02460 — their stated limitation) or by a static
whole-graph check that assumes deterministic per-call costs and covers only
delegation *forests* (Talluri et al., arXiv:2605.05657, which cites the COINE
paper as the runtime baseline and explicitly declines multi-parent structures).
Neither assumption survives LLM agents, whose consumption is observable only
after generation.

The contribution is a conservation law for delegation **DAGs**, enforced at
runtime by **node-local** checks, sound under refunds and abandonment; plus the
comparison that law makes possible, in a regime the matched-budget literature
excludes — where a unit of budget buys an **action that creates information**
rather than reasoning over fixed context.

### Claims explicitly NOT made

- Not the first matched-budget comparison of agent topologies (Tran & Kiela; Illusion of Multi-Agent Advantage, arXiv:2606.13003)
- No novelty in measuring agent diversity (arXiv:2602.03794 owns this, with a better metric: K\*)
- No novelty in LLM causal discovery or intervention selection (MAC arXiv:2407.15073; Model Discovery Agent arXiv:2608.09696; A-CBO arXiv:2605.27567; LeGIT; CausalGame, ICML 2026)
- No novelty in using causal chambers as an agent benchmark (TemporalBench, arXiv:2602.13272)

The chamber is an **instrument for measuring coordination**, nothing more.

---

## 2. Theory: six propositions

Theory is load-bearing: the empirical section may return null on every accuracy
comparison. §4.6 of `docs/whitepaper.md` now states all six as numbered
propositions. P2–P6 each carry an executable artifact in
`tests/core/test_delegation_graph_propositions.py`; P1 is covered by the
pre-existing `test_local_invariants_imply_global_bound`.

**Gate outcome (2026-08-23, six days early): PASS — 5 of 5**, after an xhigh
code review forced three restatements. All of P2–P6 have a passing artifact
(20 tests, mutation-checked). **P2, P3 and P4 were each falsified as first
stated**; P5 and P6 survived but their artifacts did not, and both were
rebuilt. AAMAS 2027 main track remains the target.

| | Proposition | Status | Risk |
|---|---|---|---|
| **P1** | Soundness. Local invariant `in-flow ≥ consumption + out-flow` at every node ⇒ `Σ C(v) ≤ B(root)` | proved, §4.6 P1 | low — retained as the easy half; weight rests on P2–P6 |
| **P2** | ~~Tree insufficiency (unsoundness).~~ **Restated as incompleteness:** for a node with *m* ≥ 2 in-edges, an indivisible consumption *c* with `maxᵢ aᵢ < c ≤ Σᵢ aᵢ` is admitted by the DAG law and by no tree encoding (merge / split / drop exhausted) | **restated**, `test_p2_*` (6 tests) | **the unsoundness claim was a STRAWMAN** — it held only against a drop-policy accountant; the real `ContractingCapability` splits and refuses. Incompleteness is provable against the shipped tree law, and quantified: max indivisible call is B/m vs B |
| **P3** | ~~Necessity of budget acyclicity for the static bound.~~ **Restated:** the static bound is cycle-robust; acyclicity is necessary for refund propagation to be well-founded, and is enforced structurally — `allocate()` refuses a cycle-closing edge before inspecting its amount | **restated**, `test_p3_*` (5 tests) | **as stated it was FALSE** — the static bound is cycle-robust; acyclicity buys well-founded reclamation, enforced at allocation time (zero amounts included). Counts corrected to 182 cyclic / 386 acyclic after the 2-cycle-only detector was replaced by reachability |
| **P4** | Tightness under abandonment. Verification certifies `Σ C(v) ≤ B(root) + Σ refunds`, and the bound is tight **globally** (a reachable execution consumes exactly `B(root) + Σ refunds`) **and per node** | **restated**, `test_p4_*` (3 tests) | **"unsaturable in v1" was FALSE** — that conflated *re-delegatable* with *re-spendable*; the refund lands at the parent, which may spend it. Verified: total 150 passes, 151 raises. The first artifact hid it by summing over non-root nodes only |
| **P5** | Confluence of reclamation. Refunds computed against *original* allocations ⇒ final residuals are independent of edge-release order; live-value refunds admit a counterexample | **proved**, `test_p5_*` (4 tests) | statement survived; **artifact did not** — the zero-consumption fixture made both refund rules agree, so a live-value mutant passed. Rebuilt with consumption at the shared child; the mutant now dies |
| **P6** | Locality separation. `consumption ≤ in-flow` is node-locally decidable; the `+ out-flow` term is not, under a stated definition of node-local | **proved (scoped)**, `test_p6_*` (2 tests) | resolved, but **scoped to the materialization choice** — materializing from residual instead of in-flow would make the full invariant node-locally preventable; §4.6 now argues why in-flow is the right choice rather than treating the separation as unconditional |

**Go/no-go gate (2026-08-29): CLEARED 2026-08-23.** The gate required three of
P2–P6 to survive contact with a real proof; five did. The implementation track
is unblocked.

**Three** results changed the paper rather than merely confirming it, and all
three were found by executing the claim rather than by reading it. **P3 was
falsified**: the telescoping sum cancels every internal edge whether or not the
edge set contains a cycle, so acyclicity is nowhere used in the static bound —
169 random cyclic allocations saturate `B(root)` exactly as 399 acyclic ones do.
§4.6's previous claim that a budget cycle "collapses" the argument was wrong and
is now corrected. What acyclicity actually buys is structural: `allocate()`
refuses a cycle-closing edge before inspecting its amount, so cyclic reclamation
is unreachable rather than merely detectable. **P4's tightness turned out to be
per-node, not global** — reclaimed budget is not re-delegatable in v1, so
`B(root) + Σ refunds` cannot be saturated. **That was itself wrong**, caught by
a subsequent review: the refund lands at the parent, which may spend it, so a
reachable execution consumes exactly `B(root) + Σ refunds` — 150 against a root
budget of 100 — and one unit more raises. The bound is tight globally as well as
per node.

**P2's unsoundness claim was withdrawn entirely.** It held only against an
accountant that resolves multiple parents by dropping an in-edge; this
framework's own `ContractingCapability` splits the node instead and correctly
refuses the over-commitment. The surviving result is incompleteness — no tree
encoding admits an indivisible consumption larger than any single parent's grant
— which is provable against the shipped tree law and carries a quantitative
penalty of `B/m`. It also bears directly on the ladder: the fan-in aggregator's
job is one large model call, exactly the regime a tree encoding cannot express.

---

## 3. The ladder

Five coordination designs at a certified-identical **intervention** budget.
Inference is held fixed at PC for every rung, so topology is the only varying
factor. Arm names follow Tran & Kiela's taxonomy for transparent comparison.

```
rung 0  LOOP        root ──────────────────────► agent (k experiments)
rung 1  ENSEMBLE    root ──┬─► scout_a ─┐        identical prompts, blind
                           └─► scout_b ─┴─► aggregator
rung 2  PARALLEL    root ──┬─► scout_a ─┐        distinct roles, blind
        -ROLES             └─► scout_b ─┴─► aggregator
rung 3  SEQUENTIAL  root ──► planner ──► reasoner    one-way handoff
rung 4  TEAM        root ──┬─► scout_a ◄╌╌╌╌╌╌┐      bidirectional negotiation
                           └─► scout_b ╌╌╌╌╌╌►┘      (╌ = messages, no budget)
                               scout_a, scout_b ──► aggregator
```

Each *adjacent* pair changes exactly one thing: 0→1 splits the budget; 1→2 adds
role differentiation; 2→3 adds a one-way channel; 3→4 makes the channel
bidirectional.

**Stated confound:** 2→3 cannot separate "one-way channel" from "sequential
ordering," because a one-way channel requires ordering. Inherent to
communication, disclosed rather than discovered.

### The aggregator must consume budget

In a PC-inference design the aggregator would otherwise do no LLM work — PC is
not an LLM call — so the fan-in edges would carry budget nothing ever spends,
making multi-parent conservation nominal and P2's empirical demonstration
impossible.

The aggregator therefore performs one **bounded reconciliation call**: resolve
duplicate and conflicting scout selections, emit the pooled dataset. Then PC
runs. Inference stays fixed at PC across all five rungs; the reconciliation call
is part of what coordination *costs*, which is the thesis, not a confound.

### Budgets: the saturation problem

`MENU_SIZES["lt"] = 59`, so **k=59 means the budget is the entire menu**. Rung 0
takes all 59 distinct experiments; rung 3 also reaches 59 distinct because
`_llm_select_loop` receives `starting_chosen=planner_chosen`, which excludes
prior picks. Blind scouts can only reach `59 − overlap`. At k=59 the ladder
would therefore measure duplicate avoidance rather than topology, with the
handicap falling exclusively on the three new arms.

**Ladder budgets are `k ∈ {6, 30, 45}`** (k/M ∈ {0.10, 0.51, 0.76}). Rungs 0 and
3 must be run at k=45, since M4b has no data there.

k=59 is reported separately as the **saturation regime**, from existing rung 0
and rung 3 data plus the analytic observation that a blind arm cannot exceed
`59 − overlap` distinct experiments. New arms are not run at k=59.

### Arm inventory

| Rung | Variant | Inference | k=6 | k=30 | k=45 |
|---|---|---|---|---|---|
| 0 | `llm_pc` | PC | have | have | **30 new** |
| 1 | `fan_in_homog` | PC | 30 new | 30 new | 30 new |
| 2 | `fan_in_spec` | PC | 30 new | 30 new | 30 new |
| 3 | `planner_reasoner` | PC | have | have | **30 new** |
| 4 | `team` | PC | 30 new | 30 new | 30 new |

`llm_only` (SHD 26.2 / F1 0.75 at k=59) **exits the ladder** and is reported
separately as an inference finding, so the paper's largest effect is not
mistaken for a topology result.

### Guards and reserves

| Item | Cells | Purpose |
|---|---|---|
| Ladder arms | 330 | 270 new-arm cells + 60 for rungs 0/3 at k=45 |
| `c95(45)` calibration | 5 | `llm_pc` ×5 at k=45. Nothing else in the pre-flight measures `llm_pc` per-call cost at k=45: the overlap probe runs new arms only, and the reuse guard must sit at an M4b budget to test reproduction at all |
| Overlap pre-flight probe | 24 | 3 new arms × **k=30** × 8 seeds. The budget must be named: at k=6 two blind scouts draw 3 each from 59, so overlap is near zero by chance; at k=45 it is near-forced. High overlap in `fan_in_homog` is the *expected* replication of arXiv:2602.03794. Abort only if `fan_in_spec` **and** `team` also exceed `overlap_frac` > 0.8 |
| Reuse-validity guard | 20 | `llm_pc` ×15 at k=30 + `llm_only` ×5 (smoke only). **Test the mean against M4b's SEM, not the per-cell SD.** `llm_pc` per-cell SD is 0.040, so the old 5-cell / 1.5σ rule tolerated 0.059 F1 of drift — 2–8× the effect the paper measures. n=15 gives SEM 0.010, resolving ~0.02. `llm_only`'s SD of 0.274 makes any tight test on it vacuous |
| New-arm timeout reserve | 20 | Rung 3 lost 8/30 at k=59 and its survivors averaged 1381 s against the 1800 s limit. Rungs 1/2/4 have at least as many chained calls |
| Tight-budget P2 demonstration | 10 | Deliberately tight T, so total consumption approaches `B(root)`. Isolated from the main sweep — see §6 |
| `planner_reasoner` k=59 timeout re-runs | 8 | Completes the M4b record for the saturation report |
| Token-matched contingency | 90 (reserve) | Held back. Triggered only if a multi-agent rung beats rung 0 — pre-registered so the "gains bought with tokens" objection is answered with evidence |

**Total committed: ~417 cells, 65–90 h serial on the VPS, $9–14.**

Per-cell cost is strongly **k-dependent** — per-call latency itself grows with
the `already_chosen` block — so a flat rate misestimates both ends. Measured M4b
min/cell:

| | k=6 | k=30 | k=59 |
|---|---|---|---|
| `llm_pc` | 0.5 | 4.1 | 14.2 |
| `planner_reasoner` | 0.7 | 6.1 | 23.0 |

Projected for the new arms (two select loops + reconciliation; rung 4 adds four
negotiation calls): **~2 min at k=6, ~9 min at k=30, ~18 min at k=45.** That
gives ≈58 h for the ladder and ≈16 h for guards and reserves.

M4b's total spend was **$5.11** and `planner_reasoner` averaged **$0.0196/cell**;
the 4.7 min/cell implied by M4b's headline is contaminated by 180 non-LLM cells
averaging 0.15 s.

---

## 4. Graph construction

```
root  tokens=T(k), per_tool{intervene: k}
 ├─ scout_a  tokens=S(k), intervene=⌈k/2⌉ ─┐ F(k)
 └─ scout_b  tokens=S(k), intervene=⌊k/2⌋ ─┤ F(k)
                                            └─► aggregator  in-flow 2·F(k)
                                                per_tool{intervene: 0, observe: 0}
```

Budgets are derived **per role from measured per-call cost**, not as fractions
of a global pool. Two call types must be calibrated separately:

- **Selection calls** are capped at `_SELECTION_MAX_TOKENS = 200`
  (`agents.py:71`). `llm_pc`'s `n_llm_calls` equals k for every cell, so its
  whole population is capped selection calls. Let `c95(k)` be its
  95th-percentile per-call cost.
- **The reconciliation call is uncapped and reasoning-heavy**, so it cannot be
  priced from selection calls. Its analogue in M4b is `llm_only`'s adjacency
  step, isolable because `llm_only` issues exactly k+1 calls. Let `a95(k)` be the
  95th percentile of that surplus.

```
F(k) = ceil(1.5 · a95(k))                    forward to aggregator, per scout
S(k) = ceil(2 · c95(k) · ⌈k/2⌉) + F(k)       scout in-flow
T(k) = 2 · S(k)                              root token budget
```

| k | c95 | a95 | F(k) forward | S(k) scout | aggregator in-flow | T(k) root |
|---|---|---|---|---|---|---|
| 6 | 1,350 | 21,163 | 31,744 | 39,844 | 63,488 | 79,688 |
| 30 | 2,303 | 38,752 | 58,129 | 127,219 | 116,258 | 254,438 |
| 45 | ~2,778 * | ~39,191 * | 58,786 | 186,574 | 117,572 | 373,148 |

\* interpolated between k=30 and k=59; **`c95(45)` is measured by the 5-cell
calibration run in §3, not left interpolated.**

An earlier revision priced the aggregator at `2·c95(k)` — four capped-call
equivalents for one uncapped call, under-budgeting it by 7–10× and guaranteeing
H-2 failure on every fan-in cell. A revision before that used flat fractions of
a global T, leaving scouts 2.4–3.7 % headroom against their own p95 spend. Both
would have produced conservation violations that were calibration artifacts
rather than governance failures.

### Output caps

A third named constant is required. The codebase has only
`_SELECTION_MAX_TOKENS = 200` and `_ADJACENCY_MAX_TOKENS = 32768`
(`agents.py:71,91`). Reusing 200 for the reconciliation or negotiation calls
reproduces the M4b root-cause bug verbatim: DeepSeek v4 Flash spends the cap on
reasoning tokens, returns empty `content`, and every call silently degrades to
the fallback path. Size `_RECONCILE_MAX_TOKENS` and `_NEGOTIATE_MAX_TOKENS` at
4–8× expected content, per the M4b post-mortem.

**Token budgets are non-binding.** Node monitors record tokens for certification
arithmetic but do not halt execution on the token dimension. This is required:
a binding cap would truncate new-arm cells while the reused rungs 0 and 3 ran
uncapped, confounding the accuracy comparison. Interventions remain live-gated.

Other constraints:

- Splits: k=6 → 3/3; k=30 → 15/15; k=45 → 23/22.
- **Fan-in edges must carry nonzero tokens.** `seal()` (`delegation_graph.py:311`)
  rejects any non-root node whose every in-flow dimension is zero
  (`GraphLintError: funded with nothing`).
- The root token cap is supplied via the factory's existing
  `extra_resources=ResourceConstraints(tokens=T)`, **opt-in per variant**.
- Rung 4's scout-to-scout channel is **plain message passing in the agent code,
  not a `DelegationGraph` edge.** `allocate()` runs its cycle check
  (`delegation_graph.py:231-233`) *before* inspecting the amount, so a
  bidirectional scout_a↔scout_b pair raises `CycleError` even at zero. That is
  the point, not an obstacle: the interaction graph cycles while the budget DAG
  stays acyclic, which is exactly P3's claim.
- Rung 3 stays on `ContractingCapability` unchanged, protecting M4b reuse.
- **Scouts receive decorrelated seeds** (`2·seed`, `2·seed + 1`).
  `_llm_select_loop` does `rng = _random.Random(seed)` (`agents.py:418`); with
  identical seeds *and* identical prompts, rung 1's fallback would pick the same
  experiment in both scouts, collapsing the arm into rung 0 at half budget.
  `seed + 1` is not sufficient: M4b seeds are contiguous `0..29`, so scout_b of
  cell *s* would draw the same stream as scout_a of cell *s+1* for 29 of 30
  cells, correlating the replicates and deflating the between-seed variance that
  every comparison depends on.

### Zero-grant keys

The aggregator is granted `per_tool={"intervene": 0, "observe": 0}`.

- **Not `"exp"`.** `_require_per_tool_propagation` short-circuits on
  `granted == 0`, so a zero-grant on an unknown key raises nothing while
  `"intervene"` stays unconstrained.
- **`"observe"` must be named explicitly.** `create_contracted_chamber_agent`
  inserts the key only `if observation_budget > 0` (`causalchamber.py:496-499`),
  and `can_use_tool` treats an absent key as unlimited (`monitor.py:606-610`).
  Without the explicit zero, the aggregator could call `query_observation`
  without bound and acquire data outside the certified budget.

Both are pinned by regression tests (§7).

---

## 5. Enforcement

| Resource | Enforcement | Claim |
|---|---|---|
| Interventions (`per_tool["intervene"]`) | **Live, per node.** `query_intervention` routes through `graph.monitor_for(<node>)`; an aggregator experiment call is *blocked*, not merely noticed | "certified-identical intervention budget" |
| Observations (`per_tool["observe"]`) | **Live, per node**, explicit zero for the aggregator | closes the side channel |
| Tokens | **Post-hoc certification, non-binding.** Measured spend from `_CountingLLM` is written into each node's monitor via `add_tokens(...)`, then `graph.verify()` | reported as cost; not matched, and said so plainly |

**Per-node live gating is a requirement, not an upgrade.** Today all variants
spend through the adapter's single aggregate monitor, with sub-budgets honored
only by the `spend=` argument to `_llm_select_loop`.

**Hard constraint:** per-node routing must be opt-in (default `None` preserves
today's aggregate behavior exactly), so rungs 0 and 3 remain byte-identical in
behavior and M4b reuse stays valid.

---

## 6. Hypotheses

H-1 and H-2 pull in opposite directions on the same constant and must not share
a run. Tree insufficiency needs total consumption near `B(root)`; certification
needs headroom. They are therefore separated by design:

- **P2 proves insufficiency analytically** (§2).
- **A dedicated 10-cell run at deliberately tight T demonstrates it** — total
  consumption approaches `B(root)`, so tree accounting's double-counted bound
  admits executions the DAG law rejects.
- **The main sweep runs at the loose T(k) of §4** and supports H-2.

| | Claim | Where | Can it be null? |
|---|---|---|---|
| **H-1** | Tree accounting certifies a bound exceeding `B(root)` by the fan-in double-count, and executions fall in that gap | tight-budget demonstration (10 cells) | No — the gap is analytic (P2); the demonstration shows executions reaching it |
| **H-2** | DAG conservation certified on **rungs 1, 2, 4** (the only rungs with a `DelegationGraph`), including abandonment cases, against `B(root) + Σ refunds`. Rung 3 reports *tree* conservation via `ContractingCapability` separately; rung 0 has no delegation structure | main sweep | No — reported as a certification claim, per P4. **Scope stated explicitly**: reused M4b cells predate `DelegationGraph`, so `conservation_certified` is null for them. **Report the mechanism and the forecast separately** (added 2026-08-24) — see below |
| **H-3** | **Equivalence bound**: no rung differs from rung 0 by more than the minimum detectable effect at that budget | main sweep | Must be stated as a bound, not a null. See the power table below — the M4b rung-0-vs-rung-3 differences (0.028 at k=6, 0.007 at k=30) sit *below* what n=30 can resolve |
| **H-4** | If any rung beats rung 0, expect it at the tightest budget (k=6) | main sweep | Pre-registered from Tran & Kiela's finding that SAS was best "for all budgets except the lowest one". **Caveat: k=6 is a floor regime.** All five M4b variants lie within 0.035 F1 of each other there (0.183–0.218) against an MDE of ~0.034, so a null at k=6 is indistinguishable from "nothing works at 6 experiments" |
| **H-5** | Failure rate rises with coordination machinery | main sweep | M4b: rung 3 failed 8/30 at k=59, rung 0 failed 0/30 |
| **H-6** | Among rungs 1, 2 and 4, overlap falls as coordination increases | main sweep | Secondary. **Within-{1,2,4} only** — rung 3's overlap is 0 by construction (`starting_chosen` excludes prior picks) and rung 0's is undefined, so a full-ladder monotonicity claim is malformed. Framed as replication of arXiv:2602.03794 in the acquisition regime, not novelty. Report K\* alongside `overlap_frac` |

#### H-2 measures two things; report them apart

A `conservation_certified=False` cell means the node consumed more than it was
allocated. That is a true finding about **our budget forecast**, and a
*success* for the mechanism: `verify()` detected the overrun, every time.
Reporting a bare compliance rate conflates:

- **the mechanism** — does the framework enforce the flow-conservation
  invariant and detect violations? Measured at 100% across every cell run so
  far, including all overruns.
- **the forecast** — did `_A95_RECONCILE_BY_K` predict actual cost well enough
  that no node overran? At k=45 it did not until recalibration; at k=6 it
  cannot, because aggregator cost spreads 48.8x (500-24,415 tokens) while the
  provisioning basis is a median.

A reader shown "H-2 compliance = 56%" will conclude the framework failed. It
did not; our cost model did. State the mechanism result, then report
per-budget forecast adequacy as a separate, calibration-dependent number.

Note the naming skew: this document numbers hypotheses H-1..H-6 while
`CLAUDE.md` uses H-A/H-B/H-C. H-2 here is H-C there. Reconcile before drafting.

### Statistical power

Two-sample MDE at n=30 per arm, using `llm_pc`'s within-variant per-cell SD
(α=0.05, 80 % power ⇒ ≈ 2.8·SD·√(2/n)):

| k | `llm_pc` SD | MDE (F1) | Observed rung 0 − rung 3 |
|---|---|---|---|
| 6 | 0.047 | ~0.034 | 0.028 — below detection |
| 30 | 0.040 | ~0.029 | 0.007 — below detection |
| 45 | ~0.040 * | ~0.029 | unknown |

\* assumed; the new arms may be noisier, which would raise the MDE further.

The consequence must be stated in the paper: **the ladder can bound the topology
effect, not measure it.** "No rung differs from a single agent by more than 0.03
F1" is the honest and still-useful form of the deflationary claim. Reporting it
as "no effect" would be a power artifact dressed as a finding.

---

## 7. Implementation

### Files

| File | Change |
|---|---|
| `evaluation/chamber_pipeline/agents.py` | `fan_in_agents(..., differentiate: bool)` for rungs 1–2; `team_agents(...)` for rung 4. Both call `_llm_select_loop` with `starting_chosen=None` for both scouts (blind) and distinct seeds |
| `evaluation/chamber_pipeline/llm_planner.py` | **Two new blind role prompts** for rung 2. `build_reasoner_select_prompt` frames the task as refining "the Planner's picks (which appear in the `already_chosen` block)" (`llm_planner.py:167-175`); with `starting_chosen=None` that block is empty and the system message references nothing. Reusing it is not an option |
| `evaluation/chamber_pipeline/orchestrator.py` | Three `AgentSpec` entries, `extra_kwargs=("scout_a_budget","scout_b_budget")`; `_build_agent_kwargs` branch mirroring the planner/reasoner split; opt-in `extra_resources` token cap |
| `src/agent_contracts/integrations/causalchamber.py` | Optional per-node monitor routing for `query_intervention` and `query_observation` (default `None` = today's behavior) |
| `evaluation/chamber_pipeline/tree_accounting.py` | **New.** Replays a recorded execution under tree accounting; returns the certified bound and whether the execution falls in the double-count gap. Scores H-1 |
| `evaluation/chamber_pipeline/results.py` | New optional columns: `overlap_frac`, `n_experiments_distinct`, `conservation_certified`, `tree_accounting_bound` |
| `evaluation/chamber_pipeline/analyze_results.py` | Ladder figures: accuracy vs rung, cost vs rung, failure rate vs rung |

### Negotiation protocol (rung 4)

One upfront round, deliberately O(1) rather than O(k): each scout proposes its
intended selections, sees the other's proposal, revises once, then executes
blind. **Four extra LLM calls per cell** — propose and revise, per scout — each
capped at `_NEGOTIATE_MAX_TOKENS`, never at `_SELECTION_MAX_TOKENS = 200`.
The channel is plain message passing, not a `DelegationGraph` edge (§4).

Rationale for O(1): rung 3 already times out 8/30 at k=59 with three chained
calls; a per-step protocol would be k rounds and worse.

Conflict resolution: scouts resolve overlaps during the revision round.
Deterministic backstop if they fail — lower-index scout keeps the contested
experiment, the other re-picks. `overlap_frac` then measures whether negotiation
actually works.

**This is allocation negotiation, not answer adjudication** — the distinction
from Tran & Kiela's Debate arm, and the Contract Net connection.

### Duplicate handling

Each `query_intervention` is metered, so an overlapping cell spends k calls and
sees fewer than k distinct experiments — the honest matched-cost reading.
Identical frames are **deduplicated before pooling**, because feeding PC
duplicated rows inflates effective *n* and makes its independence tests
overconfident. Both `n_experiments_spent` (= k) and `n_experiments_distinct` are
recorded.

### Tests

- `seal()` / `verify()` on each rung's graph
- Per-node gating: the aggregator's `query_intervention` **and**
  `query_observation` are both refused
- **Regression test pinning the `"exp"` trap**: granting `per_tool={"exp": 0}`
  must leave `"intervene"` unconstrained
- **Regression test pinning the `"observe"` hole**: an aggregator granted only
  `{"intervene": 0}` can still observe; granting both zeros blocks it
- Distinct-seed test: rung 1's two scouts do not produce identical fallback picks
- `overlap_frac` and K\* unit tests
- `tree_accounting` replay: a known fan-in double-count is admitted by tree
  accounting and rejected by the DAG law
- FakeLLM end-to-end on a 2-node chamber for all three new arms
- Behavioral no-op check: rungs 0 and 3 produce identical results with per-node
  routing disabled

---

## 8. Sequencing

| Window | Work | Gate |
|---|---|---|
| Aug 23–29 | **Theory week**: prove P1–P6. In parallel: implement rungs 1/2/4, the two blind role prompts, per-node routing, and the tree-accounting scorer | — |
| **Aug 29** | **Go/no-go** | ≥3 of P2–P6 survive, else re-target |
| Aug 30–Sep 2 | Overlap probe (24, k=30) + reuse guard (20) + `c95(45)` calibration (5) | Abort if `fan_in_spec` **and** `team` both exceed 0.8 overlap, or `llm_pc`'s 15-cell mean drifts >0.02 F1 from M4b |
| Sep 2–9 | Main sweep, 330 ladder cells + reserves, checkpointed | — |
| Sep 9–14 | Tight-budget P2 demonstration (10), tree-accounting scoring, figures, contingency decision | Token-matched reserve fires only if a rung beats rung 0 |
| Sep 14–30 | Writing | — |
| **Oct 1** | Abstract registration (mandatory, 100–300 words) | — |
| **Oct 8** | Submission | — |

M5 (WT + UNCONTRACTED + Pro, ~900 cells) does not fit the same VPS in this
window. M6 takes precedence; M5 resumes October.

---

## 9. Submission hygiene

- Cite the COINE paper **in third person** — "Prior work [n] establishes
  conservation for delegation trees." Never "our earlier work"
- **No links** to `github.com/flyersworder/agent-contracts` or PyPI
  `ai-agent-contracts`. Use an anonymized mirror for supplementary code, swap at
  camera-ready
- Scrub `\author`, `pdfauthor`, ORCID (`orcidlink.sty`), and repo URLs from
  captions and footnotes
- The framework name is effectively a signature; keeping it is a deliberate,
  permitted choice
- arXiv preprints are **not** archival under AAMAS rules — `arXiv:2601.08815v3`
  poses no eligibility problem. Only the Springer LNAI COINE proceedings does,
  and the DAG contribution is disjoint from it
- Do not push further M6 design detail to the public repo before submission

---

## 10. Corrections to `docs/causal_chamber_validation_plan.md` §7.7

1. `per_tool={"exp": 0}` → `per_tool={"intervene": 0, "observe": 0}`. The wrong
   key fails silently; the missing key leaves a side channel
2. H-A ("chain underperforms loop") is **unresolved, not refuted** (sharpened
   2026-08-23). M4b at matched inference gives chain−loop deltas of −0.028 /
   +0.007 / −0.027 at k = 6 / 30 / 59 against a pooled MDE of ~0.036 — all
   below it, so the data licenses neither direction. Restate as "no accuracy
   effect this design can resolve (equivalence bound ±0.036 F1);
   coordination's measurable cost is reliability." Note n≈55 per arm would be
   needed to resolve the observed ~0.03 gap
3. Arm 1 was `llm_only`, which confounds topology with inference. Rung 0 is now
   `llm_pc`
4. §7.7's claim of "no comparative benchmark" is **false as of April 2026**
   (Tran & Kiela; Illusion). Remove it
5. The fan-in as specified (zero-only grant) **does not seal**. Fan-in edges must
   carry tokens
6. Budgets `{0.10, 0.50, 1.00}` → `{6, 30, 45}`. At k=59 the budget equals the
   menu, so blind arms structurally cannot match rungs 0 and 3 on distinct
   coverage
7. `~$2` for a 450-cell sweep is wrong; M4b cost **$5.11**. Also correct in
   `CLAUDE.md`
8. **P2's measurement window is k-independent and centred on the calibration
   constant** (added 2026-08-23, from `tree_would_refuse` probes). The window
   is `(0.75*a95, 1.5*a95]` = (6418, 12836] tokens at every budget, because
   `build_fan_in_graph`'s forward flow depends only on `a95`, never on `k`.
   Actual aggregator spend is strongly k-dependent, so the measured verdicts
   are:

   | k | reconcile spend | verdict |
   |---|---|---|
   | 6 | ~2,826 median (max 6,375) | `False` — below the window; a tree would cope |
   | 30 | ~8,557 median | `True` — 33% up the window |
   | 45 | pre-flight probe pending | could exceed 12,836, which returns `None` |

   **State this honestly in the paper.** `a95` is the k=30 reconcile median
   AND sets the threshold, so a typical k=30 cell landing inside the window is
   substantially guaranteed by construction — the arm is *built* to sit in
   P2's incompleteness regime, as `build_fan_in_graph`'s docstring says
   outright. The measurement therefore confirms that real token spend stays
   inside the constructed window; it does not discover that arbitrary
   workloads happen to fall there. Report it as a demonstration of P2, not as
   an estimate of how often tree encodings fail in the wild.

   Corollary risk at k=45: `_A95_RECONCILE` is fixed at the k=30 median for
   every budget, so if the real k=45 reconcile is much larger the aggregator
   can overrun a grant sized for k=30 — an H-C conservation failure that is a
   calibration artifact, exactly the failure mode the `_ROLE_C95` comment
   warns about for the scouts. This is what pre-flight probe 3 measures.


## 11. Known issue: rung 4's negotiation parser reads restatement as claim

Found 2026-08-23 while fixing the selection-loop equivalent (which IS fixed --
spent experiments now leave the menu, and parsing happens against the offered
list). The negotiation path has the same shape and is **not** fixed.

`build_negotiate_revise_prompt` renders "You proposed: <own>" and "The other
designer proposed: <other>" above the full menu. `_parse_name_list(revised,
menu)` then scans the **entire response text** for every menu name. A scout
that reasons out loud -- "the other designer wants X and Y, so I will take
Z" -- has X and Y counted as its OWN claim.

Consequences, all on rung 4's headline metric:

- `contested = set(source_a) & set(source_b)` inflates, so
  `n_contested` reports conflicts the scouts never had.
- `claim_a = list(source_a)[:scout_a_budget]` truncates in menu order, so a
  phantom claim can displace a real one.

Why it is NOT fixed the way selection was: filtering `other` out of the parse
menu would be wrong. A scout claiming what the other proposed is a *genuine*
contest, and that is exactly the signal rung 4 exists to measure. The fix has
to separate the scout's ANSWER from its restatement of the prompt -- a
delimited answer block, or parsing only lines that are bare menu names, which
is what the prompt already asks for ("one per line, and no other
commentary"). That is a prompt-and-parser change needing its own validation.

**Measure before fixing.** The §3 overlap pre-flight probe (3 new arms x k=30
x 8 seeds) records `n_contested` and `overlap_frac` and will show whether
compliant responses make this rare. Do not stack a speculative parser change
on top of the selection fix without that number.


## 12. P2's demonstrable window has width equal to the fan-in degree

Found 2026-08-24 while recalibrating `a95` from the k=45 gate. This is a
theory-to-design connection, not a bug, and it bounds what the fan-in arms can
show.

P2's incompleteness condition is `max_i a_i < c <= sum_i a_i`: the aggregator's
single call must exceed any ONE parent's forward but not their sum. With `n`
parents each forwarding `f`, the admissible window is `(f, n*f]` -- so its
**width ratio is exactly `n`, the fan-in degree.** Both fan-in rungs have two
scouts, so the window is 2x wide, and `build_fan_in_graph` deliberately gives
the aggregator no provisioning multiple (`forward = ceil(0.75 * a95)`) because
a margin would push every single fragment above the call and a tree encoding
would then have coped.

That creates a structural tension between the two hypotheses:

- **H-C (conservation)** wants generous provisioning, so no cell overruns.
- **P2** wants tight provisioning, so the call lands above a single forward.

They can both hold only while the aggregator's observed cost spread fits
inside the window. Measured:

| budget | aggregator spend | spread | window (n=2) | both hold? |
|---|---|---|---|---|
| k=45 | 9,783 - 25,168 | **2.6x** | 2x | marginal |
| k=6 | 500 - 6,064 | **12x** | 2x | **no** |

At k=6 the reconcile prompt lists only 3+3 names, so cost is dominated by
erratic reasoning length rather than prompt size -- 12x variance within a
single arm across seeds. No single `a95` can both conserve the 6,064 cell and
keep the 500 cell inside the window.

**Consequences, all reportable rather than fixable:**

1. **P2 is demonstrable at k=45 and not at k=6.** State it as a scope limit on
   the measurement: the DAG-vs-tree distinction is empirically visible only
   where the aggregator's cost is predictable within a factor of `n`.
2. **Do not tune `a95` until H-C reads 100%.** With a 2x window that trades
   directly against P2, and `_PROVISION_MULTIPLE`'s comment already forbids it.
   Report observed rates per budget.
3. **Higher fan-in degree widens the window.** Three scouts would give a 3x
   window and tolerate more variance. Not a change to make for M6 -- it alters
   the rung definitions -- but it is the principled lever if a future design
   needs P2 demonstrable at low budgets, and it is worth one sentence in the
   paper: the empirical demonstrability of P2 improves with fan-in degree,
   which is a statement about the theory, not about DeepSeek.

---

## 11. Review record

Independent review at medium effort, 2026-08-23, over the 2026-08-22 revision.
Eleven findings, all judged valid and all applied.

| Finding | Applied as |
|---|---|
| T(k) leaves scouts ~3 % headroom → conservation violations as calibration artifacts | §4 per-role budgets from `c95(k)` |
| k=59 equals the full menu → blind arms structurally handicapped | §3 budgets `{6, 30, 45}`; k=59 demoted to saturation regime |
| Opt-in token cap confounds new arms against reused arms | §4 token budgets made non-binding |
| Both scouts share a seed → identical fallback picks | §4 distinct seeds `seed`, `seed + 1` |
| H-6 non-monotone by construction (rung 3 overlap = 0) | §6 restated within-{1,2,4} |
| No timeout reserve for the new arms | §3 20-cell reserve; k=45 lowers exposure |
| Wall-clock and cost ~2× low | §3 65–95 h, $9–13, from verified per-cell figures |
| Aggregator's `"observe"` left unconstrained | §4 explicit `{"intervene": 0, "observe": 0}` + regression test |
| H-1 and H-2 coupled through one constant | §6 separated: P2 analytic + 10-cell tight-budget demonstration vs. main sweep |
| Rung 4 undercounted at two extra LLM calls | §7 four calls |
| Rung 2's role prompts unusable blind | §7 two new blind role prompts, budgeted |

Two further defects were found by executing against `runs/m4-pilot.parquet`
rather than re-reading the spec: the token split above, and the fact that a
PC-inference aggregator consumes no tokens at all, which would have made the
fan-in nominal and P2 undemonstrable (§3, *The aggregator must consume budget*).

### Round 2 — 2026-08-23

Second independent review over the round-1 revision. Ten findings, all judged
valid, all applied. Every quantitative claim in the spec was re-derived against
the parquet and held exactly.

| Finding | Applied as |
|---|---|
| Reconciliation and negotiation calls have no `max_tokens`; reusing 200 reproduces the M4b empty-content bug | §4 `_RECONCILE_MAX_TOKENS`, `_NEGOTIATE_MAX_TOKENS` |
| `F(k)` priced an uncapped reasoning call from 200-token-capped selection calls | §4 `F(k) = ceil(1.5·a95(k))` from `llm_only`'s isolable adjacency call — **7–10× larger** |
| Reuse guard (5 cells, 1.5σ of per-cell SD) tolerates drift 2–8× the measured effect | §3 n=15 at k=30, tested against SEM |
| `seed`/`seed+1` collides with the next cell's scout_a for 29 of 30 seeds | §4 `2·seed` / `2·seed+1` |
| Nothing in the pre-flight can measure `c95(45)` | §3 5-cell `llm_pc` k=45 calibration run |
| H-2 overstated: reused cells have no `DelegationGraph`, rung 3 is a tree, rung 0 has no delegation | §6 scoped to rungs 1/2/4; rung 3 reports tree conservation separately |
| A zero-amount scout↔scout edge still raises `CycleError` — the check precedes the amount | §4 channel is plain message passing, outside the graph |
| Flat 10–15 min/cell ignores measured k-dependence | §3 per-budget projections |
| k=6 is a floor regime where H-4's positive is least resolvable | §6 caveat + power table |
| Overlap gate: n=4, budget unnamed, overlap strongly budget-dependent | §3 k=30, n=8 |

The most consequential consequence is not in the table: the power analysis added
in §6 shows the ladder **cannot resolve** the effect sizes M4b suggests
(0.007–0.028 F1 against an MDE of ~0.03). H-3 was therefore restated from a null
to an equivalence bound. This is a limit of the design, not of the analysis, and
n=30 was fixed by the reuse of M4b cells.
