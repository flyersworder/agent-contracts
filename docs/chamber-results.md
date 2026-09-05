# Chamber Pillar: Results

The canonical record of every chamber-pillar experiment and what it showed.
Results live here rather than in `claude.md`, which is project memory loaded
into every session and should stay instructions plus status.

**Companions.** `docs/chamber-harness-validity-register.md` records the twenty-nine
harness defects that each changed or could have changed a result — read it
before trusting any number here. `docs/causal_chamber_validation_plan.md` is
the experiment plan; `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md`
is the ladder's design spec.

**Corpus as of 2026-09-01**: 18,063 cells, **$108.39**, **zero errored cells**,
across two chambers and two models. (The 2026-08-30 line read "2,221 / $94.05";
it predated the seven M7 files, which add 1,220 cells and $14.34, and the two
LLM-free variance probes and re-scorings, which add 14,622 cells at no cost. The table below is
the arithmetic of record.)

| dataset | cells | cost | what it establishes |
|---|---|---|---|
| `runs/m6-ladder.parquet` | 450 | $54.53 | LT coordination ladder, 5 rungs x 3 budgets x 30 seeds |
| `runs/m6-wt-ladder.parquet` | 750 | $11.64 | WT ladder, same 5 rungs, n=50 |
| `runs/m6-lt-loop-curve.parquet` | 420 | $7.26 | loop vs random across 7 budgets |
| `runs/wt-random-vps.parquet` | 150 | $0.00 | WT random baseline, platform-matched |
| `runs/agg-ablation.parquet` | 60 | $1.74 | aggregator honored vs discarded |
| `runs/uncontracted.parquet` | 60 | $1.09 | the ungoverned control, both chambers |
| `runs/pro-lt.parquet` | 60 | $7.39 | v4-pro robustness, LT |
| `runs/pro-wt.parquet` | 100 | $7.16 | v4-pro robustness, WT |
| `runs/team-preflight.parquet` + `-lowk` | 9 | $0.42 | claim-cap incidence, LT k=45 and k=6 |
| `runs/wt-team-probe.parquet` | 3 | $0.04 | substring-conflict incidence, WT k=21 |
| `runs/m6-wt-team-rerun.parquet` | 150 | $2.78 | WT `team` re-run under the fixed parser; spliced into `m6-wt-ladder-final.parquet` |
| `runs/_provprobe.parquet` | 9 | $0.00 | VPS BLAS-stability probe; reproduces `wt-random-vps` 9/9 exactly |
| `runs/m7-phase1.parquet` | 20 | $0.53 | mechanism probe: variable- vs experiment-level coverage |
| `runs/m7-coverage.parquet` + `-ms` | 150 | $0.00 | LLM-free coverage manipulation, the +0.0073/variable price |
| `runs/m7-varsplit.parquet` | 90 | $2.25 | `team_varsplit`: partition variables, not experiments |
| `runs/m7-p2-lt.parquet` | 270 | $3.72 | Phase 2 LT: `one_shot`, `critique`, `shared_blackboard` |
| `runs/m7-p2-ref.parquet` | 90 | $2.29 | same-era LT `llm_pc` reference for the Phase 2 panel |
| `runs/m7-p2-wt.parquet` | 600 | $5.56 | Phase 2 WT, all four arms including the loop, in one sweep |
| `runs/variance-probe.parquet` | 3,150 | $0.00 | selection vs measurement variance, 7 LT budgets, no LLM |
| `runs/variance-probe-1500.parquet` | 150 | $0.00 | max-rows control; refutes the subsample-thinning mechanism |
| `runs/variance-probe-wt.parquet` | 3,150 | $0.00 | the same decomposition on WT, 7 budgets, no LLM |
| `runs/rescored.parquet` (+`-bykey`) | 8,172 | $0.00 | every M7 design re-scored at 9 PC seeds; validated 191/191 against production |
| `runs/m7-coverage-lt-ends.parquet` | 180 | $0.00 | the coverage rule at LT k=6 and k=45, LLM-free |
| `runs/m7-coverage-wt.parquet` | 300 | $0.00 | WT random at 3 budgets; the LT-only coverage arm correctly skipped |
| `runs/m7-coverage-wt2.parquet` | 300 | $0.00 | the WT coverage arms, breadth and depth, 3 budgets x 50 seeds |

**Never pool rows whose `blas_backend` differs** — see register §10. Every
sweep above ran on Linux / `scipy-openblas` except `runs/m4-pilot.parquet`
and the `curve-*` files, which are macOS / Accelerate. `m6-controls`'s
backend was attributed from a run log until 2026-08-29, when it was
positively verified against a platform-stamped file (register §13).

**Never pool across the collinear-fix boundary either** (register §13). The
fix of 2026-08-25 changes 20% of LT `random` cells — with *perfect* separation
against `n_collinear_dropped`, so it is a version difference, not noise. The
marker is whether a file carries that column. Pre-fix: `m4-pilot`,
**`m6-ladder`**, `m6-controls`, `curve-lt-random`, `curve-wt-random`.
Post-fix: everything else above. **Audited 2026-08-29: no published contrast
crosses the boundary** — the LT ladder's five arms ran pre-fix together,
`agg-ablation` and `pro-*` each carry their own control, and the uncontracted
contrast draws both baselines from post-fix files. For scale: the
cross-backend gap is ΔF1 = 0.055, larger than most effects reported below.

Chambers: light tunnel (LT) 38 nodes / 57 edges / 59-experiment menu; wind
tunnel (WT) 32 / 42 / 28. PC with Fisher-Z at alpha=0.05, 300-row subsample,
collinearity threshold 0.999. MDE = 2.8 * sd * sqrt(2/n) throughout.

---

## THE COVERAGE ORACLE (2026-09-01): an LLM-free rule matches every LLM arm

> **CORRECTED 2026-09-02 — the tables below crossed BLAS backends, and two
> verdicts move.** The coverage arms ran locally on Accelerate; every LLM arm
> they are compared against came off the VPS on OpenBLAS (register §31). Only
> LT k=30 was clean. Re-scored on one backend at 9 PC seeds and clustered by
> selection, the headline "no LLM arm resolves above the rule at any budget"
> is **false at both ends**:
>
> | | best LLM − rule | MDE | verdict |
> |---|---|---|---|
> | LT k=6 | **+0.036** | 0.030 | **the LLM wins** |
> | LT k=30 | +0.001 | 0.029 | ties |
> | LT k=45 | −0.001 | 0.013 | ties |
> | WT k=7 | −0.015 | 0.020 | ties |
> | WT k=14 | +0.002 | 0.022 | ties |
> | WT k=21 | **−0.030** | 0.024 | **the rule wins** |
>
> **This is a better result than the one it replaces.** A flat row of ties
> reads as a task that cannot discriminate; a crossing is a finding — the
> model beats the heuristic when budget is scarce and loses to it when budget
> is ample, on two chambers. The "nuance that saves the story" below was
> therefore under-stated rather than over-stated: the tight-budget escape is
> not a near-miss, it resolves.
>
> **But it does not survive core-20 scoring, and that matters more.** Register
> §28 records that 18 of our 38 nodes are pure apparatus sources carrying 18
> of 57 edges. Scoring only the 20 variables the chambers' own case study uses:
>
> | LT | best LLM − rule (core-20) | MDE | verdict |
> |---|---|---|---|
> | k=6 | +0.029 | 0.030 | ties |
> | k=30 | +0.005 | 0.014 | ties |
> | k=45 | −0.000 | 0.016 | ties |
>
> `llm_pc`'s k=6 margin falls from +0.036 to **+0.014**. So the LLM's one
> resolved advantage over a ten-line rule lives substantially in the
> apparatus edges — "did you buy the experiment that makes this setting vary"
> — and **on the non-trivial subgraph no LLM arm we built beats round-robin
> coverage at any budget**. State it that way; a reviewer who checks will
> find it otherwise.
>
> Absolute values in the corrected tables are Accelerate re-scorings and are
> not comparable to the numbers below; the contrasts are, because every arm in
> them is read from `f1_rescored` on one machine.

Prompted by §29's finding that the contemporaneous ground truth is a bipartite
source→sink assignment with 32-40% trivially-structured edges: if the task is
coverage-shaped, a coverage rule should be hard to beat. It is.

`coverage_max_ms` is **no LLM at all** — round-robin over distinct variables,
weak intervention strengths excluded, seeded shuffle for tie-breaks. Same
platform, same BLAS, same post-collinear-fix era as every arm below.

| LT | coverage rule | loop | `one_shot` | best LLM − rule | MDE | verdict |
|---|---|---|---|---|---|---|
| k=6 | 0.1845 | **0.2188** | 0.1596 | +0.0343 | 0.0375 | ties |
| k=30 | **0.4336** | 0.4276 | 0.4392 | +0.0055 | 0.0293 | ties |
| k=45 | 0.4276 | 0.4359 | 0.4179 | +0.0083 | 0.0290 | ties |

**No LLM arm resolves above the rule at any budget.** At LT k=30 the full
ordering, all n=30, same era:

| arm | F1 | vs rule |
|---|---|---|
| `one_shot` | 0.4392 | ties |
| **`coverage_max_ms`** | **0.4336** | — |
| `llm_pc` (loop) | 0.4276 | ties |
| `shared_blackboard` | 0.4226 | ties |
| `critique` | 0.3957 | **worse** |
| `random` | 0.3604 | worse |

### The nuance that saves the story

The rule is **not** a universal oracle — it is near-optimal only where coverage
is the binding constraint:

| k | rule | random | rule − random |
|---|---|---|---|
| 6 | 0.1845 | 0.1771 | **+0.007** |
| 45 | 0.4276 | 0.4062 | +0.021 |
| 30 | 0.4336 | 0.3604 | **+0.073** |

**At k=6 the coverage rule is barely better than random** (+0.007), while the
loop beats random by **+0.056** and the rule by +0.034 (MDE 0.0375 — the
closest any LLM arm comes to resolving above it). So:

> **The LLM's contribution is confined to the tight-budget regime where a
> coverage heuristic does not help. Once the budget is large enough for
> coverage to bind, a ten-line rule matches every LLM arm we built.**

This dovetails with the variance decomposition — room to differ is largest at
small k, skill at exploiting it peaks mid-range — and it explains why every
Phase 2 arm converges above k/M ≈ 0.5: they are all converging on the coverage
optimum.

### Why this is an asset, not a refutation

Agent benchmarks almost never have a computable near-optimal reference policy.
This one does, and it turns every result into a distance-from-optimum:

- **loop ≈ rule** at k≥30 — the LLM reaches the coverage optimum and no further.
- **fan-in < loop** — coordination overhead, now measurable *against a known
  ceiling* rather than only against each other.
- **`team_varsplit` ≈ loop** — partitioning by variable restores the arm to the
  optimum; partitioning by task list does not.
- **`one_shot` ≈ loop** — no running record is needed to reach the optimum.

The +0.0073 F1 per distinct variable slope is the exchange rate that makes all
of this quantitative, and the rule is what makes it a *ceiling* rather than a
trend.

### WT replicates it, on a structurally opposite menu (2026-09-01)

`wt_coverage_max` / `wt_coverage_min` built the same day
(`wt_menu_taxonomy.py`: strip `validate_`, take the longest node-name prefix;
28 entries over **21 variables**). n=50 per cell, same platform and era:

| WT k | rule (max) | rule (min) | random | loop | best LLM | best LLM − rule | MDE | verdict |
|---|---|---|---|---|---|---|---|---|
| 7 | **0.1881** | 0.1242 | 0.1860 | 0.1703 | 0.1748 | −0.0132 | 0.0258 | ties |
| 14 | 0.2319 | 0.1647 | 0.2220 | **0.2469** | 0.2469 | +0.0150 | 0.0347 | ties |
| 21 | **0.2817** | 0.2292 | 0.2370 | 0.2538 | 0.2608 | −0.0209 | 0.0355 | ties |

**No LLM arm resolves above the rule on either chamber, at any budget** — six
budgets, two chambers, every arm. At WT k=21 the rule is 0.021 *above* the best
LLM, still within MDE but pointing the rule's way.

The same small-budget escape holds: at WT k=7 the rule beats random by only
+0.002 (LT k=6: +0.007), and the gap widens with budget to +0.045 at k=21
(LT k=30: +0.073). **Coverage is the binding constraint at middling and large
budgets on both chambers, and at neither small one.**

**A failed pre-registration worth reporting.** `wt_coverage_min` was predicted
in its own docstring, before the run, to do *well* — WT's only multi-entry
variables are `hatch`, `load_in` and `load_out`, precisely the highest
out-degree drivers (6, 8, 8), so concentrating there should have beaten
spreading across out-degree-1 settings. It lost badly: 0.124 / 0.165 / 0.229
against breadth's 0.188 / 0.232 / 0.282.

Buying a driver's several menu entries makes that *one* variable vary several
times — redundant in exactly the sense the M7 mechanism result measures — while
breadth activates a new source each time. **Out-degree is not what the budget
buys; a distinct varying variable is.** That this survives a menu whose fat
entries are the real drivers, rather than LT's intervention strengths, is the
stronger form of the coverage finding: the two chambers' menus are structurally
opposite and the conclusion is identical.

---

## M7 PHASE 2 COMPLETE (2026-08-31): the running record is not load-bearing

960 cells across both chambers, **960 ok / 0 errors**, $11.57. Phase 2 tests
the ladder's central axis directly — how much of the loop's running record
survives — by adding three arms at the ends of it:

| arm | record | cost shape |
|---|---|---|
| `one_shot` | **none**: one call picks all k experiments | 1 LLM call |
| `shared_blackboard` | complete, but written by two role-framed voices | k calls |
| `critique` | complete, plus a reviewer pass over the selection | k + 3 calls |

Datasets: `runs/m7-p2-lt.parquet` (270) + `runs/m7-p2-ref.parquet` (90, the
same-era `llm_pc` reference), `runs/m7-p2-wt.parquet` (600, all four arms
in one sweep). All post-collinear-fix, `scipy-openblas`, flash-0731.

### The panels

LT, n=30, against the same-day loop reference:

| k | loop | `one_shot` | `critique` | `shared_blackboard` |
|---|---|---|---|---|
| 6 | 0.219 | 0.160 (**−0.059 worse**) | 0.191 (−0.027) | 0.140 (**−0.079 worse**) |
| 30 | 0.428 | 0.439 (+0.012) | 0.396 (**−0.032 worse**) | 0.423 (−0.005) |
| 45 | 0.436 | 0.418 (−0.018) | 0.398 (**−0.038 worse**) | 0.425 (−0.011) |

WT, n=50, all four arms **within one sweep** — no cross-run splice:

| k | loop | `one_shot` | `critique` | `shared_blackboard` |
|---|---|---|---|---|
| 7 | 0.170 | 0.172 (+0.002) | 0.175 (+0.005) | 0.169 (−0.002) |
| 14 | 0.247 | 0.241 (−0.006) | 0.216 (−0.030) | 0.240 (−0.007) |
| 21 | 0.254 | 0.252 (−0.002) | 0.260 (+0.006) | 0.261 (+0.007) |

MDEs 0.028–0.037 throughout; bold marks the four contrasts that resolve.
**On WT nothing resolves at all** — nine ties.

### What it establishes

**1. One call matches k calls above the smallest budget.** `one_shot` carries
no record whatsoever, and it ties the loop at LT k=30 and k=45 and at all
three WT budgets. Five of six chances to beat it, and the loop takes none. The
record only pays at LT k=6, where it is worth +0.059.

This is the reverse of the ladder's premise. M6 ordered the rungs by how much
of the record survives a partition; Phase 2 removes the record entirely and
loses nothing. The M6 ordering is therefore **not explained by the record
axis**, and any account of why fan-in underperforms has to work without it.

**2. `critique` fails its pre-registration on both chambers.** Predicted
≈ loop; measured resolved *worse* at LT k=30 and k=45, with WT's only sizable
delta (−0.030 at k=14) pointing the same way. Nine contrasts, not one above
zero by more than noise. A reviewer pass over the selection costs three flat
calls and never helps. Reported as a negative result.

**3. Structure matters only in the middle of the budget range.** Read down the
columns: at the small budget the arms scatter (LT spread 0.079) but not along
the record axis; at the large budget everything converges (LT 0.398–0.436, WT
0.252–0.261, both narrower than a single arm's sd). The mid-budget cell is the
only place a coordination difference both exists and is detectable — consistent
with the loop saturating at F1 ≈ 0.42 by k=30 (see the LT loop curve section).

### Design-level re-scoring (2026-09-01): tighter bounds, and one verdict withdrawn

The panel above scores each cell once, with PC's subsample seeded by the cell
seed. That single draw carries the inference noise §"WHY THE MIDDLE BUDGET"
measured, and it is the larger half of the spread. Since
`chosen_experiments` is recorded, both problems are fixable **offline with no
LLM calls**: rebuild each purchased design, score it under 9 subsample seeds,
average, and cluster by distinct design so a repeated buy counts once
(register §24).

`evaluation/chamber_pipeline/rescore.py`; 908 distinct designs from 1,050
cells, 8,172 (design x seed) scores, ~15 min of CPU, **$0**.

**Validated before use.** Production scored each cell at `pc_seed = cell seed`,
so for every cell with seed < 9 the re-scoring computed that exact pair.
**191 of 191 match to the bit** (max abs diff 0.00e+00) — the rebuilt pooling
and inference are the production ones, not an approximation.

| | k | loop | `one_shot` | `critique` | `shared_blackboard` |
|---|---|---|---|---|---|
| **LT** | 6 | 0.206 | **−0.047 worse** | −0.010 | **−0.057 worse** |
| | 30 | 0.421 | +0.004 | −0.013 | −0.018 |
| | 45 | 0.420 | −0.007 | −0.015 | −0.002 |
| **WT** | 7 | 0.176 | −0.000 | −0.005 | −0.004 |
| | 14 | 0.244 | −0.004 | −0.022 | −0.005 |
| | 21 | 0.253 | −0.002 | +0.001 | +0.007 |

**MDEs fall about 35%** — LT k=30 from 0.031 to **0.019**, WT k=21 from 0.036
to **0.028** — because the averaged-away component was inference noise, which
is most of the per-cell spread and none of the arm.

**What improves.** The record claim's bound, which register §24 had widened to
±0.051 at LT k=30, is now **±0.021**. Against a loop-vs-random gap of +0.055
that bound *does* exclude "the record is worth nearly as much as selecting at
all" — the objection the cell-level analysis could not answer. The equivalence
is now a result rather than a shrug.

**What is withdrawn.** `critique` was reported as **resolved worse** at LT k=30
(−0.032) and k=45 (−0.038). Averaged over 9 subsample draws those deficits are
**−0.013 and −0.015, both inside a tighter MDE**. The original verdicts rested
on a favourable single PC draw — a ~1.8 standard-error shift, entirely
ordinary. **`critique`'s pre-registration ("≈ loop on accuracy") is therefore
SUPPORTED, not refuted**, and the "clean pre-registered negative" claim made on
2026-08-31 is retracted.

The honest statement is an equivalence with a bound: a reviewer pass over the
selection costs three extra flat calls and changes accuracy by less than 0.02
on either chamber at any budget. It does not help; it also does not hurt. That
is a weaker but more defensible negative than "it hurts".

**Corrected Phase 2 scorecard:**

| pre-registration | verdict |
|---|---|
| `one_shot` < loop | **FALSE at 5 of 6 budgets** (holds only at LT k=6, −0.047) |
| `critique` ≈ loop | **TRUE** — |Δ| < 0.022 everywhere (was reported false) |
| `shared_blackboard` ≈ loop | **TRUE except LT k=6** (−0.057 there) |

**Scope of the method.** Only M7-era files record `chosen_experiments`, so the
M6 ladders cannot be re-scored. **The axis test and every topology contrast
stay at cell-level MDEs**, and must not be quoted alongside these tighter ones
as though they were measured the same way. Re-running M6 to obtain the column
would cost ~$12 at current provider prices and would tighten those contrasts
by roughly the same 35%.

### Robustness across three metrics, and what the node set is doing (2026-09-01)

Prompted by reading the chambers' own causal-discovery case study
(`causal-chamber-paper/case_studies/causal_discovery_iid.ipynb`), which differs
from our setup on two axes worth testing rather than defending.

**Axis 1 — orientation.** They compute precision/recall for *every DAG in the
estimated CPDAG*, because orientation inside a Markov equivalence class is not
identifiable. We score one directed graph, and
`cpdag_to_directed_adjacency` encodes an undirected CPDAG edge as **both**
directions — so a correctly-found but unoriented edge scores one true positive
AND one false positive. The cheap equivalent of their protocol is to score the
skeleton, where the whole equivalence class agrees.

**Axis 2 — the node set.** Their case study uses **20** light-tunnel variables;
we use **38**. The 18 extra are not a superset chosen for coverage — measured
against the ground truth, **every one is a pure source: out-degree 1,
in-degree 0**. They are apparatus settings (`t_*` exposure time, `osr_*`
oversampling rate, `v_*` reference voltage, `diode_*` select), each driving
exactly one sensor, and they carry **18 of the 57 true edges (32%)**.

Both were re-scored offline over the same 908 designs x 9 subsample seeds, $0.

| | | directed F1 | skeleton F1 | core-20 F1 |
|---|---|---|---|---|
| **LT k=6** | `one_shot` | **−0.047 worse** | −0.022 ties | **−0.028 worse** |
| | `critique` | −0.010 ties | −0.009 ties | +0.015 ties |
| | `shared_blackboard` | **−0.057 worse** | **−0.037 worse** | −0.023 ties |
| **LT k=30** | all three | ties | ties | ties |
| **LT k=45** | all three | ties | ties | ties |
| **WT, all budgets** | all three | ties | ties | (LT only) |

**The headline survives all three.** "The running record is not load-bearing
above the smallest budget" rests on the ties at LT k=30/45 and all of WT, and
**every one of those holds under every metric**. Nothing that was a tie becomes
a difference.

**The one resolved finding is metric-sensitive.** At LT k=6 both arms are
negative under all three metrics — `one_shot` −0.047 / −0.022 / −0.028,
`shared_blackboard` −0.057 / −0.037 / −0.023 — but neither clears MDE under all
three. **The sign is robust; the resolution is marginal.** Report the k=6 result
that way rather than as a clean effect.

### What the node set is doing to the headline numbers

The core-20 subgraph is the more uncomfortable finding, and it is about
absolute values rather than comparisons:

| LT loop | k=6 | k=30 | k=45 |
|---|---|---|---|
| full 38-node F1 | 0.206 | **0.421** | 0.420 |
| core 20-node F1 | 0.176 | **0.223** | 0.226 |

**78% of the loop's budget response sits on the 18 setting→sensor edges**
(k=6→k=30: full +0.215, core +0.047). Those edges are real, but they are
trivially structured — a pure source with one child — and they are recoverable
if and only if an experiment makes that setting vary. So a third of the
recoverable structure, and most of the measured improvement with budget, is
**"did you buy the experiment that activates this setting"** rather than "did
you infer non-obvious structure".

Three consequences, stated rather than fixed:

1. **Comparative claims are unaffected.** Every arm faces the identical node
   set and menu, so the topology, record and coverage contrasts are unchanged —
   as the table above confirms across all three metrics.
2. **Absolute F1 values must not be read as "recovered the light tunnel".**
   Quote the core-20 figures beside the headline ones, or a reader will compare
   0.42 against numbers from the case study's 20-variable setting.
3. **It reframes the coverage result rather than voiding it.** The
   +0.0073 F1 per distinct variable is partly the price of activating settings.
   That is still a real and actionable mechanism for an experiment-selection
   agent — but it is a statement about *coverage of manipulable variables*, not
   about discovering physics, and §"M7 PHASE 1"'s wording should say so.

**WT is worse, and it was never checked** (register §29). Same analysis on the
wind tunnel: **17 trivial sources carrying 40% of its 42 edges**, leaving a core
of 15 nodes and 25 edges — and 9 further in-edges sit on the three barometers
the collinearity fix drops, 6 of them from real drivers. And **both chambers'
ground truth is bipartite with maximum path length 1**: 0 mediators on
lt/standard, lt/camera and wt/standard alike. There are no causal chains to
discover; the task is an assignment of manipulable sources to observed sinks.
`wt/pressure-control` is the one configuration with actual depth (3 mediators),
and it needs its own dataset release wired before it can be run.

**Why not simply switch to 20 nodes.** The menu is built from the dataset's
experiments, and it *contains* interventions on those settings
(`uniform_t_ir_1_strong`, `uniform_v_c_strong`, …). Dropping the nodes while
keeping the buys would score an informative purchase as wasted budget. Changing
both is a different experiment, and it forks all 18,000 recorded cells. The
honest move is to report both scorings, which now costs nothing.

### Robustness: re-analysed at the selection level (2026-08-31)

A single LLM call re-picks nearly the same design each seed, so `one_shot`'s
cells are not 30 independent draws — at LT k=30 they are **6 distinct
selections, one covering 17 cells**. Re-running every Phase 2 contrast with one
row per distinct buy (register §24):

**Every verdict above is unchanged**, on both chambers, at every budget. One
bound moves, and it is the one the headline rests on: the LT k=30 equivalence
is **±0.051, not ±0.029**. Since the loop-vs-random gap there is +0.055, that
bound cannot exclude "the record is worth nearly as much as selecting at all",
so **the LT half of the record claim rests on k=45** (24 distinct selections,
±0.033) **and on WT** (30–34 of 50 distinct, bounds widening by ≤0.005). Every
other arm draws 29–50 distinct designs and is unaffected.

More seeds cannot tighten it — they buy more scorings of the same six designs.
The fix is selection diversity (menu-order shuffling per seed, or a pinned
non-zero temperature) and it requires re-running the arm.

### The axis test: sharing a record beats splitting it, at one budget

`shared_blackboard` versus `fan_in_spec` isolates the axis properly. Both run
the same two role prompts; the only difference is whether the two voices write
into one record or two.

| chamber | small k | middle k | large k |
|---|---|---|---|
| LT | −0.016 (MDE 0.027) | **+0.053 (MDE 0.039) RESOLVED** | −0.006 (MDE 0.026) |
| WT | +0.007 (MDE 0.032) | **+0.046 (MDE 0.039) RESOLVED** | +0.025 (MDE 0.038) |

Different graph, different menu, different budget fractions — and both chambers
resolve **only at the middle budget**, within 0.007 of each other. This is the
sharpest replication in the corpus.

Three caveats travel with it, all of them live:

- **Cross-run.** `fan_in_spec` comes from the M6 ladders, `shared_blackboard`
  from Phase 2. Era drift measured on `llm_pc` is +0.017/+0.008/+0.019 (LT) and
  +0.025/+0.008/−0.032 (WT). At the middle budget drift is +0.008 on both, so
  adjusting leaves LT at ≈+0.045 (clear) and WT at ≈+0.038 against an MDE of
  0.039 — **WT lands exactly on the boundary**. State it as such.
- **LT crosses the collinear-fix boundary** (`m6-ladder` is pre-fix). Measured
  rather than assumed: on post-fix LT the drop fires for 0% of cells at k=45,
  3% at k=30, and only at k=6 is it common. The budget where the LT axis test
  resolves is 3%-vs-0% affected. WT is post-fix on both sides and clean.
- **`shared_blackboard` vs the *loop* varies two things** (shared record AND
  two role-framed voices), which is why `fan_in_spec` and not `llm_pc` is its
  comparator here. The same caution the spec already carries for the parked
  rationale-passing arm.

### A confound checked and killed

At LT k=6 the collinear drop fires for **90% of `one_shot` cells** against 20%
for the loop — a rate correlated with the arm, which is exactly the shape of a
harness moderator. It is not one: the drop costs ~nothing. Pooled across arms,
drop-fired 0.172 vs not-fired 0.182; within `critique` −0.006, within
`shared_blackboard` −0.002. (`llm_pc` shows +0.036 on n=6, the wrong sign for
the confound and too small to weigh.) `one_shot`'s −0.059 at k=6 is a real
record effect, not a collinearity artifact.

---

## WHY THE MIDDLE BUDGET (2026-08-31): room falls, skill rises, the payoff peaks where they cross

Phase 2 left an interpretive gap: three separate results — the loop-vs-random
gap, the axis test, and `one_shot`'s collapse at LT k=6 — all pointed at the
middle of the budget range without a shared account of why. This section
supplies one, and it is measured rather than argued.

**Dataset**: `runs/variance-probe.parquet`, 3,150 PC runs, **no LLM**, ~5 min.
`runs/variance-probe-1500.parquet` (150) is the max-rows control.

### The probe: untie the two things the seed controls

Every cell's `seed` sets both WHICH experiments a selection-free agent buys and
WHICH 300 rows `run_pc` subsamples. So the spread of `random` at any budget is
`selection variance + measurement noise` with no way to separate them, and the
corpus's noise-floor claim ("at k=M selection freedom is zero, so the spread is
pure PC noise") was an argument from construction, not a measurement.

`evaluation/chamber_pipeline/variance_probe.py` crosses the two seeds instead
of tying them: 30 independent random buys per budget, each scored under 15
different subsample seeds. That is a one-way ANOVA layout — between-group
variance is what the CHOICE is worth, within-group is what the MEASUREMENT
costs. Group means over m draws still carry `sigma_within^2 / m`, so the
between-estimate is bias-corrected; without that, pure noise reads as a
selection effect.

**The method validates itself on the bottom row.** At k=M=59 every "selection"
buys the whole menu, so all 30 are literally identical and the true selection
variance is zero by construction. The decomposition recovers **sd 0.005**
without being told.

### What the choice is worth, by budget

| k | k/M | mean F1 | sd total | sd **PC noise** | sd **selection** | loop−random | gap in selection-sd |
|---|---|---|---|---|---|---|---|
| 6 | 0.10 | 0.170 | 0.048 | 0.032 | **0.036** | +0.047 | 1.3 |
| 12 | 0.20 | 0.256 | 0.055 | 0.037 | **0.042** | +0.014 | 0.3 |
| 20 | 0.34 | 0.326 | 0.052 | 0.036 | **0.038** | +0.042 | 1.1 |
| 30 | 0.51 | 0.368 | 0.050 | 0.043 | 0.026 | +0.055 | **2.1** |
| 40 | 0.68 | 0.393 | 0.045 | 0.041 | 0.018 | +0.036 | **2.0** |
| 50 | 0.85 | 0.412 | 0.046 | 0.043 | 0.015 | +0.000 | 0.0 |
| 59 | 1.00 | 0.416 | 0.042 | 0.041 | 0.005 | +0.008 | — |

`loop−random` is from `runs/m6-lt-loop-curve.parquet` (same chamber, same
backend, post-fix). The last column expresses it in units of the room that
actually exists at that budget.

### Three findings

**1. Total spread is flat because two opposing trends cancel.** `sd_total`
sits at 0.042–0.055 at every budget, which invites the reading that nothing
changes with k. The decomposition shows the opposite: **selection variance
falls by 8x (0.042 → 0.005) while measurement noise rises slightly (0.032 →
0.041)**, and the sum happens to stay put. Any inference from the flat total —
including one made in this project on 2026-08-31 and corrected here — is
reading a coincidence.

**2. Room to differ collapses with budget; this is the honest form of "fewer
choices matter more."** Which experiments you buy is worth sd 0.036–0.042 at
k/M <= 0.34, sd 0.015–0.026 above it, and sd 0.005 at k=M. Note the shape:
below k/M = 0.34 it is a **plateau, not a rise**. Selection does not become
progressively more decisive as the budget shrinks; it hits a ceiling and stays
there. The decisive change is the collapse above k/M = 0.5.

**3. Skill at exploiting the room moves the other way.** The loop captures
**2.1 and 2.0 selection-sd** at k=30 and k=40 against **1.3** at k=6 — it
lands near the top of the achievable distribution at middle budgets and only
partway up at the smallest. So:

> **The room to differ falls with budget while the ability to exploit it
> rises. The absolute payoff to good selection peaks where the two curves
> cross — the middle of the range.**

That single sentence accounts for all three Phase 2 observations: the
inverted-U in loop-vs-random on both chambers (LT peak +0.055 at k/M=0.51, WT
peak +0.039 at k/M=0.50); the axis test resolving only at the middle budget on
both; and every arm converging at the top, where there is nothing left to
exploit.

It also explains **`one_shot`'s split personality**. At LT k=6 it captures
**0.0** of the 0.036 available — it sits exactly on random (0.160 vs 0.163) —
while at k=30 it captures +0.079 over random, matching the loop. The room at
k=6 is real; a single call simply cannot find any of it, and sequential
deliberation can find some. Picking a *reasonable half* of a menu needs little
discrimination; picking the best six of 59 needs a lot.

### A mechanism proposed, tested, and refuted

Measurement noise rises with k, and the obvious explanation is that a
fixed 300-row subsample thins per-experiment coverage linearly: at k=6 those
300 rows cover 6 experiments (~50 rows each), at k=59 they cover 59 (~5 each).
Tested at 5x the rows on **identical selections**:

| | within-sd @ 300 | @ 1500 | mean F1 @ 300 | @ 1500 |
|---|---|---|---|---|
| k=6 | 0.0345 | 0.0346 | 0.212 | 0.236 |
| k=59 | 0.0390 | 0.0369 | 0.418 | 0.442 |

**Noise does not move** (−5% at k=59, 0% at k=6). The subsample is not what
makes PC noisy; the noise is intrinsic to the accept/reject cascade at alpha,
consistent with register §10's account of why a 1e-10 numerical perturbation
forks the conditioning-set search. The proposed mechanism is refuted and
recorded as such.

**Side finding with a cost attached**: 5x the rows buys **+0.025 F1 at both
budgets** — uniform, not budget-dependent. Free accuracy for runtime. It
cannot be retrofitted: `max_rows=300` is the configuration of record for all
3,441 corpus cells, and changing it would fork the pooling boundary the way
the collinear fix did. Worth stating as a known headroom, not a change.

### WT: the mechanism half-transfers, and the noise floor is the real constraint

Run on WT the same day (`runs/variance-probe-wt.parquet`, 3,150 runs, no LLM,
7 budgets x 30 selections x 15 subsample seeds). The k=M validation passes
again — at k=28 all 30 buys are identical and the decomposition returns
sd 0.007 without being told.

| k | k/M | mean F1 | sd total | sd PC noise | sd selection | loop−random | gap in selection-sd |
|---|---|---|---|---|---|---|---|
| 3 | 0.11 | 0.130 | 0.056 | 0.033 | **0.046** | — | — |
| 7 | 0.25 | 0.191 | 0.055 | 0.039 | 0.039 | −0.011 | −0.3 |
| 10 | 0.36 | 0.206 | 0.051 | 0.043 | 0.027 | — | — |
| 14 | 0.50 | 0.213 | 0.056 | 0.049 | 0.027 | +0.039 | **+1.4** |
| 19 | 0.68 | 0.227 | 0.064 | 0.054 | 0.034 | — | — |
| 21 | 0.75 | 0.233 | 0.065 | 0.053 | 0.040 | +0.020 | +0.5 |
| 28 | 1.00 | 0.248 | 0.067 | 0.067 | **0.007** | — | — |

**What replicates.** Skill peaks in the middle: the loop captures **+1.4
selection-sd at k=14** against −0.3 at k=7 and +0.5 at k=21, the same shape as
LT's 2.1 at k=30 versus 1.3 at k=6. And selection variance collapses to nothing
at k=M on both chambers, which is forced rather than discovered.

**What does not.** On LT the room to differ falls monotonically (0.036 →
0.005). On WT it does **not**: 0.046 → 0.027 at mid-range, then back up to
0.040 at k=21 before collapsing. So the LT sentence "room falls while skill
rises, and the payoff peaks where they cross" is **LT-specific**. The
chamber-general statement is weaker and should be the one the paper makes:

> **The payoff to good selection peaks in the middle of the budget range on
> both chambers, because that is where agents exploit the available room best.
> On LT the room also shrinks with budget, which sharpens the peak; on WT it
> does not.**

The rise at WT k=19–21 is ~2 sigma on the estimate's own uncertainty (±13% at
30 groups), so it is suggestive rather than established. **Collinearity is
ruled out as its cause**: the collinear drop count is flat at ~3.1 columns from
k=7 upward (between-selection sd 0.20 at k=19 and k=21), so it cannot generate
between-selection variance there. Zero-variance drops fall monotonically
(16.25 → 2.18 columns) and correlate −0.38 to −0.47 with F1.

**A WT scope note this exposes.** At k=3, **16.25 of 32 columns** are dropped
as zero-variance and padded back with zeros — over half the graph is answered
by padding rather than inference. Much of WT's large small-budget selection
variance is therefore "which half of the graph did you make measurable at all",
not "did you choose informatively". State it before reading WT's small-budget
numbers as selection quality.

### The MDE is mostly measurement noise, and that is a design constraint

The probe permits a calculation the corpus could not do before: what the MDE
would be if two arms selected *identically* and only PC noise separated them.

| | noise-only MDE | observed MDEs |
|---|---|---|
| LT k=6, n=30 | 0.023 | 0.031–0.038 |
| LT k=30, n=30 | 0.031 | 0.029–0.036 |
| WT k=14, n=50 | 0.028 | 0.035–0.037 |
| WT k=21, n=50 | 0.029 | 0.036–0.037 |

**Most of our resolving power is spent on measurement noise, not on arm
variability** — at LT k=30 the observed MDEs sit at or below the noise-only
floor. Two consequences:

1. **WT Phase 2's nine ties are partly a noise result.** WT noise doubles
   across the budget range (0.033 → 0.067), so at k=21 roughly 80% of the MDE
   is PC. An arm genuinely 0.025 better could not have been resolved there at
   n=50.
2. **The fix is seeds, not better agents.** Resolving a 0.02 difference at WT
   k=21 needs **n ≈ 110 per arm**; at LT k=30, **n ≈ 75**. No agent design
   closes a floor set by the inference procedure. Quote these when reporting
   an equivalence bound, so "below MDE" reads as a power statement rather than
   a null.

### What this does and does not license

It **does** support scoping every coordination claim in this pillar to the
middle of the budget range, with a mechanism rather than an apology: at the top
no topology can differ, and at the bottom the room exists but agents cannot
find it.

It **does not** transfer to WT whole. Measured the same day (section above):
the "skill peaks mid-range" half replicates, the "room falls with budget" half
does not. Use the chamber-general sentence, not the LT one.

---
## STATUS 2026-08-30: WT `team` re-run COMPLETE — every verdict unchanged

The parser defect that fired **only on WT and only on rung 4** is retired.
150 cells re-run, **150 ok / 0 errors / 0 PC degeneracies**. Spliced into
`runs/m6-wt-ladder-final.parquet` (750 rows; the 26 Aug
`runs/m6-wt-ladder.parquet` is left untouched for audit).

- **What was wrong.** `_parse_name_list` carried a substring guard on top of a
  word-boundary regex that already prevented the problem it was written for,
  so every time it fired it deleted a genuine claim. Three WT names can be
  deleted this way (`validate_load_in`, `validate_load_out`, `validate_osr_in`
  — the short, unqualified member of each family), each replaced by a random
  top-up. **LT's menu has no such pairs, so LT was untouched.**
- **How often — now measured at n=50 per budget**, via the
  `n_substring_conflicts` counter that records what the removed guard *would*
  have dropped: **0.02 / 0.14 / 1.02 per cell at k = 7 / 14 / 21**. The
  incidence scales with budget, as expected (more claims, more chances to
  collide), reaching ~1 affected pick in 21 at the top budget. The earlier
  3-cell probe (0, 0, 2 at k=21) was consistent with this.
- **What it moved: nothing that resolves.** `team` rose at every budget by
  **+0.0059 / +0.0075 / +0.0048**, all far below MDE, **all p ≥ 0.56**. The
  direction is as predicted — restoring deleted claims should help the arm —
  but the magnitude is not detectable.

| k | team (26 Aug) | team (re-run) | loop | delta vs loop | MDE | Welch p | verdict |
|---|---|---|---|---|---|---|---|
| 7 | 0.1791 | 0.1850 | 0.1451 | **+0.040** | 0.033 | 0.0008 | resolved (team) |
| 14 | 0.2176 | 0.2251 | 0.2388 | −0.014 | 0.032 | 0.31 | below MDE |
| 21 | 0.2402 | 0.2451 | 0.2854 | **−0.040** | 0.040 | 0.0051 | resolved (loop) |

**All three verdicts are unchanged**, so the headline tally is unchanged:
**24 contrasts, 10 resolve, 9 favour the loop, 1 favours a topology.** The WT
k=21 loop win survives at −0.040 (was −0.045); the WT k=7 inversion survives
at +0.040 (was +0.034) and is still explained by the broken denominator there
(the loop loses to random at k=7).

The cost frontier is also unchanged in every verdict: `team` moves 0.218 →
0.225 at k=14 and 0.240 → 0.245 at k=21, dominated at both, and **both blind
fan-in rungs remain strictly dominated 12/12** across the six chamber-budget
points.

### Provenance: this splice crosses a stamp boundary, and here is why it is sound

`m6-wt-ladder.parquet` (26 Aug) predates the provenance columns — 38 columns
against the re-run's 48, no `blas_backend`. So comparing new `team` rows to
old `llm_pc` rows is exactly the mixed-provenance case
`require_homogeneous_provenance` exists to refuse, and the analyzer *did*
refuse until passed `--allow-mixed-provenance`.

We did **not** backfill stamps onto the old file — writing a stamp we did not
observe defeats the guard. Instead the provenance was established by evidence:

1. **Origin.** The VPS holds `m6-wt-ladder.parquet` at a byte-identical md5
   (`b416f14f…`) to the local copy, dated 26 Aug — it was produced there.
2. **Backend today.** The VPS reports `scipy-openblas 0.3.34.0.0` on
   `Linux-x86_64`, matching the re-run's stamp.
3. **Backend stability across the window.** A 9-cell seeded, LLM-free `random`
   sweep run on the VPS on 30 Aug **reproduces `wt-random-vps.parquet`
   (26 Aug, stamped OpenBLAS) exactly — 9/9 on both F1 and SHD, max |diff| =
   0.000e+00.** Since PC amplifies a 1e-10 perturbation into structural noise
   (§"BLAS backend"), bit-exact reproduction is a sharp test: any backend
   change would have shown.

**Rule for the next such splice**: unstamped is not the same as wrong, but it
must be *established* rather than assumed, and a seeded LLM-free arm is the
cheapest instrument for doing it.

## M7 PHASE 1 (2026-08-30): `overlap_frac = 0.0` was a mirage, but redundancy is still not the cause

`runs/m7-phase1.parquet` — **20/20 cells, 20 ok, 0 errors, 0 PC degeneracies.**
LT, `llm_pc` + `team`, k=30, n=10, `deepseek-v4-flash-0731`, VPS
(`scipy-openblas`), ~35 min wall on 6 workers.

**The question**: why does `team` lose to the loop at equal experiment count?
Three hypotheses predicted the same recorded symptom, so they could only be
separated by what each arm actually *bought* — hence the `chosen_experiments`
roster instrument (shipped 2026-08-29, which is why M6's rows cannot answer
this and the cells had to be re-run).

### It replicates M6 first

| | loop | team | delta |
|---|---|---|---|
| M6, 24 Aug, n=30 | 0.420 ± 0.039 | 0.374 ± 0.046 | **−0.046** |
| M7 P1, 30 Aug, n=10 | 0.427 ± 0.042 | 0.379 ± 0.056 | **−0.048** |

Six days, a separate sweep, and the delta reproduces to 0.002. The n=10 delta
is itself *below* its own MDE (0.062) — n=10 was sized for the mechanism, not
for the contrast — but the contrast was already resolved at n=30, and the two
runs agreeing this closely is the useful part.

### What team buys, in variables rather than menu entries

The LT menu carries up to three entries per actuated variable
(`weak`/`mid`/`strong`), so 30 distinct *experiments* can touch 30 variables or
12. Counting in variables:

| | loop | team | delta | MDE | |
|---|---|---|---|---|---|
| experiments bought | 30.0 | 30.0 | +0.000 | 0.000 | matched by construction |
| **distinct variables** | **27.9** | **23.4** | **−4.500** | 2.273 | **RESOLVED** |
| variables bought at >1 strength | 1.8 | 6.3 | +4.500 | 1.951 | **RESOLVED** |
| zero-variance columns dropped | 0.9 | 3.1 | — | — | consistent |

**`overlap_frac` reads exactly 0.0 in all ten team cells, and 5.6 variables
were bought by both scouts** — 24% of team's variable coverage. The metric is
structurally incapable of seeing this: the two pools are disjoint *at the
experiment level by construction*, so zero measured overlap is guaranteed
rather than earned, while a quarter of the variable budget is spent twice.
This is H3 (blind depth duplication), and it is confirmed as a **description
of what team does**.

### Which hypothesis, precisely

- **H1 (scouts buy depth) — rejected.** Each scout individually is *more*
  breadth-seeking than the loop, not less: scout A repeats 0.8 variables over
  15 picks (0.053/pick), scout B 0.2 (0.013/pick), the loop 2.1 over 30
  (0.070/pick). Nothing is wrong with either scout's own behaviour.
- **H2 (forced allocation, lopsided) — rejected in that form.** The split is
  even: A touches 14.2 distinct variables, B 14.8. Neither pool is starved.
- **H3 (cross-scout duplication) — confirmed as the mechanism of the variable
  deficit.** 14.2 + 14.8 − 5.6 = 23.4 exactly. The entire 4.5-variable deficit
  is scouts unknowingly buying the same variable at different strengths.

### Does the duplication explain the accuracy loss? Yes, about two-thirds of it

Answered on 2026-08-30 by direct manipulation, after a first attempt got it
wrong. Both attempts are recorded because the correction is instructive.

**Attempt 1, and why it does not count.** Within the loop arm, F1 looked flat
in how many variables it happened to touch (r = +0.027, slope +0.0007 per
variable) — but over a range of only 25–30 at n=10, extrapolated to team's
23.4, which sits below the loop's observed minimum. That reading was reported
and is **withdrawn**: it was a range-and-power artifact.

**Attempt 2: manipulate coverage directly.** Two LLM-free arms at LT k=30,
`coverage_max` and `coverage_min`, spanning 11 to 30 distinct variables at
identical budget and PC settings. 90 cells, $0 in API spend, plus `random` at
n=30 on the VPS (which also closes the cross-platform gap in the loop-vs-random
contrasts).

That design was **confounded** — all 9 `weak` menu entries sit on exactly the
variables `coverage_min` exhausts first, so the arms differed in breadth *and*
intervention strength, correlated at −0.891. See register entry 20. Repeated
with the menu restricted to mid+strong (`_ms` arms), giving 15 to 30 variables
with zero weak at either end:

| arm | variables | weak | F1 | SHD |
|---|---|---|---|---|
| `coverage_min_ms` | 15 | 0 | 0.325 ± 0.055 | 59.2 |
| `coverage_max_ms` | 30 | 0 | 0.434 ± 0.043 | 52.4 |

**+0.109 F1 across 15 variables, MDE 0.030 — RESOLVED. Slope +0.0073 F1 per
distinct variable.** Deconfounding *doubled* the effect rather than shrinking
it (the confounded estimate was +0.0036/variable), because the two channels had
been partly cancelling.

### The attribution

Both the loop (27.9) and team (23.4) sit inside the manipulated 15–30 span, so
this is interpolation, not extrapolation:

| | |
|---|---|
| coverage deficit, team vs loop | 4.5 variables |
| predicted F1 cost at +0.0073/variable | **−0.033** |
| measured team − loop | **−0.048** |
| share explained by coverage | **≈68%** |
| unexplained residual | −0.015 (below the loop-vs-team MDE of 0.036) |

**So `team`'s deficit is mostly redundancy after all** — invisible at the
experiment level, where `overlap_frac` reads exactly 0.0 by construction, and
plainly visible at the variable level, where 5.6 of its ~29 variable-slots are
spent twice. What remains after coverage is accounted for is smaller than the
contrast's own detection threshold, so this analysis cannot say whether any
genuine coordination cost exists on top.

**Two assumptions this attribution rests on**, both worth a reviewer's
attention. The slope is measured on arms with *deterministic* coverage, so
applying it to LLM arms assumes variable identity does not matter beyond
variable count — the manipulated arms choose how many, the LLM arms also choose
which. And the `_ms` pair retains a mid/strong imbalance that, taken at face
value, would move the attribution from 68% to 62% (register entry 20).

### What this corrects, twice

Project memory recorded from 25 Aug: *"`team` reaches identical 30/30 coverage
and is still −0.047, so its cost is genuine coordination, not redundancy."*
Coverage is identical at the **experiment** level and not at the **variable**
level, and the variable deficit now accounts for about two-thirds of the gap.
**The conclusion is withdrawn, not merely re-founded** — an earlier edit on
30 Aug said it survived, which was based on the flat-slope reading now
withdrawn above.

`random`'s VPS baseline from the same sweep: 21.7 variables (range 17–26),
F1 0.360 ± 0.051, `scipy-openblas` — available for the loop-vs-random contrasts
that were previously cross-platform.

**Datasets**: `runs/m7-phase1.parquet` (20 cells), `runs/m7-coverage.parquet`
(90, confounded, retained for reproducibility), `runs/m7-coverage-ms.parquet`
(60, deconfounded).

## PROVIDER DRIFT AND THE DRIFT AUDIT (2026-09-02)

While running the WT k=21 confirmation, DeepSeek's reasoning per call rose
**2.4x** under an unchanged model id, unchanged code and a pinned
`n_llm_calls` of 26 — and it was still climbing (r=+0.44 with launch order
over two hours). Full detail in register §32; two things belong in the results
record.

### 1. No archived result is affected

Sweeps ran with arms blocked in time (fixed 2026-09-02), so drift during a
sweep could have landed on one arm. Audited rather than assumed
(`analyze_drift.py`, probe = tokens per LLM call, launch order, residualised
on arm x budget):

| file | n | hours | blocks | residual trend | worst block | verdict |
|---|---|---|---|---|---|---|
| m4-pilot | 262 | 35.6 | 9 | −0.006 | 0.378 | CLEAN |
| m6-ladder | 450 | 16.9 | 15 | −0.001 | 0.330 | CLEAN |
| m6-wt-ladder | 750 | 21.4 | 15 | +0.002 | 0.384 | CLEAN |
| m6-wt-ladder-final | 750 | 98.3 | 15 | +0.001 | 0.384 | CLEAN |
| m6-wt-team-rerun | 150 | 5.0 | 3 | −0.007 | 0.246 | CLEAN |
| m6-lt-loop-curve | 210 | 13.1 | 7 | +0.008 | 0.353 | CLEAN |
| m7-p2-lt | 270 | 5.3 | 9 | +0.016 | 0.525 | CLEAN |
| **m7-p2-wt** | 600 | 7.9 | 12 | −0.007 | **0.730** | **FLAG 1/12** |
| m7-p2-ref | 90 | 3.7 | 3 | −0.032 | 0.430 | CLEAN |
| m7-varsplit | 90 | 3.8 | 3 | +0.004 | 0.105 | CLEAN |
| m7-wt-varsplit | 298 | 4.3 | 6 | −0.007 | 0.287 | CLEAN |

Residual trend is within **±0.032 of zero in every file**. Scope:
`window_overlap` is 0.00-0.15 throughout, so this establishes that no drift
was detectable *within* blocks, not that between-block drift is impossible —
arm and window are confounded by construction under the old ordering, and that
gap cannot be closed retroactively.

### 2. A 1.7x swing in reasoning moved F1 by 0.004

The single flagged block is the useful one. `shared_blackboard` WT k=14:
reasoning per call **halved** across the block (5,520 -> 3,210, r=−0.730 over
50 cells in 74 minutes), and **F1 moved −0.004 against an MDE of ~0.023**. The
same arm is flat at k=7 (1.00x) and k=21 (0.98x).

So the provider genuinely moves on hour timescales and accuracy barely
notices. **This is a robustness result the paper should state**, not merely a
caveat: it bounds how much any given reasoning shift can plausibly matter, and
it is measured rather than argued.

**For the reproducibility section: a pinned model id does not pin the
computation.** Second recorded instance — 2026-08-13 was 4.35x tokens x 1.55x
throughput under unchanged weights. Recording `n_llm_calls` and `tokens_out`
per cell is what makes the question answerable at all.

---

## WT k=21 CONFIRMATION AT n=132 (2026-09-02): the prediction lands, the house bar does not clear

`runs/m7-wt-varsplit-n132.parquet` — **464 cells, 456 ok, 8 errors**, WT
`standard` k=21, `deepseek-v4-flash-0731`, VPS (`scipy-openblas`), arms
INTERLEAVED. Pre-registered before launch in
`docs/superpowers/specs/2026-09-02-wt-varsplit-confirmation.md` (commit
`c673121`), which fixed the predicted value, the sample size and the decision
rule while no new data existed.

**Predicted +0.0149. Measured +0.0139.**

| analysis | Δ | 95% CI | p | σ | n |
|---|---|---|---|---|---|
| **pooled (primary)** | **+0.0139** | **[+0.0032, +0.0246]** | 0.0117 | 2.53 | 121 / 131 |
| bootstrap, 100k resamples | +0.0139 | [+0.0032, +0.0245] | 0.0103 | — | — |
| new seeds only | +0.0118 | [−0.0020, +0.0256] | 0.096 | 1.68 | 73 / 81 |

Scored as pre-registered: `f1_rescored` at 9 PC seeds, clustered by distinct
design, Welch on cluster means. Zero is **outside** the interval; the predicted
value is **inside** it. The two-factor model's error is **−0.0010**.

### It does not clear the pillar's own 2.8σ bar, and that is the honest headline

2.53σ against a 2.8σ convention (≈ p<0.005). Resolving it needs **n≈154**; we
ran 132 and lost 8 to the feasibility guard. So:

> **The effect is significant at conventional levels and not at ours.** State
> both. The 2.8σ bar exists to control false positives while scanning dozens
> of exploratory contrasts; this is a single confirmatory test of a point
> prediction registered in advance, where that multiplicity argument does not
> apply the same way. We report the number, the bar, and let the reader choose
> — we do not quietly switch bars to the one that pays.

**No analytic technique closes the gap, and both were checked rather than
assumed:**

* **More PC seeds cannot.** At m=9, inference noise is only **18%** of the
  remaining variance; the rest is design variance (which experiments the LLM
  bought), which never averages away. m→∞ moves σ from 2.53 to **exactly
  2.80**. A verdict that turns on raising `m` after seeing the shortfall is
  decided by an analytic choice, not by evidence.
* **Bootstrapping cannot.** It estimates the same sampling distribution; it
  does not manufacture precision. Agreement was near-exact (SE 0.0055 both),
  which is expected since the cluster means are near-normal (Shapiro p=0.077 /
  0.220, skew 0.25 / 0.27). Its value here is corroboration by an assumption-
  free route, and it was committed to being reported whatever it said.

The clean route to the house bar is an **independent replication at
pre-specified n≈180**, never an extension of this sample — that would be
optional stopping on a result already seen.

### The decision rule was mis-specified, and the correction is ours to own

The pre-registration said: *"still below MDE at n=132 → MODEL FALSIFIED where
it predicts hardest."* Applied literally that is the verdict, and the analysis
script printed it.

**The label was wrong.** It conflated two questions — *did the effect clear our
threshold* and *was the prediction accurate*. Falsification requires the point
estimate to disagree with the prediction; it agrees to 0.001. What happened is
the test was under-powered: n=132 came from the n=50 MDE, the realised spread
was larger, and 8 cells went to the guard.

Recorded plainly because the failure mode is instructive: **a decision rule
keyed on a significance threshold cannot evaluate a point prediction.** The
rule should have keyed on whether the interval contains the prediction versus
contains zero. Had the estimate come back at +0.002 the original rule would
have been right and we would have reported falsification — the rule caught the
wrong thing, not the wrong answer.

### The model's record, stated with its miss

| chamber | k | predicted | measured | agreement |
|---|---|---|---|---|
| LT | 30 | +0.033 | +0.043 (resolved) | 1.30x |
| WT | 14 | +0.010 | −0.000 (below MDE) | **miss** |
| WT | 21 | +0.0149 | **+0.0139** (p=0.012) | **0.93x** |

Two close, one miss. WT k=14 predicted a small positive and measured zero;
both sit inside their bounds, so the two are not inconsistent, but the point
estimate does not match and saying "3/3" would be counting verdicts rather
than predictions.

### Free robustness result: the arm means ignored a 2.4x reasoning shift

The pre-registered pooling check compared seeds 0-49 (2026-09-01) against
50-131 (2026-09-02), across a provider regime change that took cells from
415 s / 67k output tokens to ~1,500 s / ~165k under an unchanged model id
(register §32):

| arm | old (n=50) | new (n=81/73) | diff | bound |
|---|---|---|---|---|
| `team` | 0.2431 | 0.2480 | +0.0049 | 0.0242 |
| `team_varsplit` | 0.2598 | 0.2598 | **+0.0001** | 0.0242 |

Both agree, so pooling is permitted by the rule set in advance. **This is the
second independent measurement that accuracy is nearly insensitive to large
reasoning shifts** — the first being `shared_blackboard` WT k=14, where
reasoning per call halved and F1 moved 0.004. Together they are a reportable
robustness claim, not merely a caveat.

### Feasibility guard

`team_varsplit` raised on **8 of 132 (6.1%)**; `team` on 0 of 132. Higher than
the 4% seen at n=50 — same code, so the earlier figure was a small sample.
The guard fires before any experiment is bought or scored, so the surviving
cells are selected on partition structure, not on outcome. Quote 6.1%.

---

## WT `team_varsplit` (2026-09-02): the non-replication is PREDICTED, not a failure

`runs/m7-wt-varsplit.parquet` — **300 cells, 298 ok, 2 errors**, WT
`standard`, k ∈ {14, 21}, n=50 per arm, `deepseek-v4-flash-0731`, VPS
(`scipy-openblas`), 3 h 32 m on 6 workers. All three arms in ONE run.
Scored at **9 PC seeds, clustered by distinct design** (`rescored-vps`).

The LT result below rests on 90 cells at one budget on one chamber, so this
was the replication that mattered. **It does not replicate.**

| contrast | LT k=30 | WT k=14 | WT k=21 |
|---|---|---|---|
| `team_varsplit` − `team` | **+0.0428** | **−0.0000** | +0.0172 |
| MDE | 0.0189 | 0.0232 | 0.0242 |
| verdict | **RESOLVED** | **flat** | below MDE |

Arm means (re-scored, design-clustered):

| arm | WT k=14 | WT k=21 |
|---|---|---|
| `llm_pc` (loop) | 0.2445 | 0.2727 |
| `team_varsplit` | 0.2240 | 0.2603 |
| `team` | 0.2241 | 0.2431 |

`team` − `llm_pc` is **−0.0296 RESOLVED** at k=21 and −0.0205 (below MDE) at
k=14, so the *deficit* the mechanism is supposed to close is present on WT.
What is absent is the closing of it.

### The mechanism fires; the accuracy does not follow

The manipulation works exactly as designed — this is not an implementation
failure:

| distinct variables bought | k=14 | k=21 |
|---|---|---|
| `team` | 11.52 | 16.06 |
| `team_varsplit` | **12.40** | **17.40** |
| loop | 11.64 | 17.14 |

+0.88 and +1.34 variables, `overlap_frac` 0.000 in both arms by construction.
At k=21 `team_varsplit` buys **more** distinct variables than the loop (17.40
vs 17.14) and still scores 0.012 lower. On LT the same +5.5 variables bought
+0.036 F1; on WT +1.34 buys nothing that resolves.

### A two-factor model predicts all three verdicts (added 2026-09-02)

An earlier version of this section stopped at "WT's menu gives the
manipulation almost no room" — true, but a post-hoc rationalisation with no
prediction in it. It is now a quantity, and it predicts the outcome:

> predicted gain  =  **coverage exchange rate**  ×  **variables recovered**

Both factors are measured **without any LLM**. The exchange rate is F1 per
additional distinct variable, regressed on the LLM-free arms alone
(`coverage_max` / `coverage_min` / `random`) with budget as a fixed effect —
a property of the *chamber's inference problem*. Variables recovered is what
partitioning actually buys — a property of the *menu*.
(`evaluation/chamber_pipeline/analyze_headroom.py`.)

| chamber | exchange rate | n cells |
|---|---|---|
| LT | **0.0061 ± 0.0005** | 240 |
| WT | **0.0111 ± 0.0006** | 450 |

**The two factors move in opposite directions, which is why neither alone
explains the non-replication.** WT's exchange rate is nearly **twice** LT's —
a distinct variable is worth *more* there, not less. What WT lacks is
headroom: 28 entries over 21 variables (1.33 each) against LT's 59 over 30
(1.97), and 18 of WT's entries are singletons.

| chamber | k | rate | variables recovered | **predicted** | **measured** | MDE | predicted verdict | actual |
|---|---|---|---|---|---|---|---|---|
| LT | 30 | 0.0061 | 5.47 | **+0.033** | **+0.043** | 0.019 | resolves | **RESOLVED** ✓ |
| WT | 14 | 0.0111 | 0.88 | **+0.010** | −0.000 | 0.023 | below MDE | below MDE ✓ |
| WT | 21 | 0.0111 | 1.34 | **+0.015** | +0.017 | 0.024 | below MDE | below MDE ✓ |

**Three for three on the verdict, and WT k=21 is nearly exact** (+0.015
predicted, +0.017 measured). Using instead the LT exchange rate from the
direct 15-vs-30-variable manipulation (0.0073, `M7 PHASE 1`) rather than this
regression puts LT at +0.040 against +0.043 measured — a 7% error from an
independent measurement of the same quantity.

**So the mechanism does not fail on WT; it is predicted to be undetectable
there.** That is a different and much stronger claim than non-replication:
the model says *in advance* which action spaces reward partitioning by
variable, and it correctly called the one where our own pre-registered
prediction would not resolve at n=50.

### The moderator is the ACTION SPACE, not the chamber

The user-facing form of this — the thing that transfers off our benchmark —
is that partitioning by role pays exactly to the extent that the action space
affords duplication. That quantity is computable **before running any agent**:
split the menu into two disjoint halves, have each side buy k/2 uniformly at
random, count the variables both touch (`a_priori_headroom`).

| chamber | menu | entries/variable | k | a-priori shared | observed `team` shared |
|---|---|---|---|---|---|
| LT | 59/30 | 1.97 | 30 | 4.14 | 6.50 |
| WT | 28/21 | 1.33 | 14 | 1.06 | 1.90 |
| WT | 28/21 | 1.33 | 21 | 1.85 | 2.42 |

The uniform model **under-predicts by ~1.5×** on both chambers, and for a
reason worth stating rather than tuning away: real scouts concentrate on the
variables that look informative, so they collide more often than random draws
do. Treat it as a **lower bound that ranks action spaces correctly** — the
LT:WT ratio is 3.9 predicted against 3.4 observed — not as a point estimate.

This is the sentence that reaches the loop-vs-graph discourse: two sub-agents
handed disjoint *task lists* still work the same modules; partition the
*module space* and the duplication goes away. How much that is worth is
`exchange rate × headroom`, and both terms are measurable in any benchmark
without running a model.

### The confirmatory test this makes available

The model's WT prediction is a real number, not a null: **+0.015 at k=21**.
At the observed spread that needs **n ≈ 132** per arm to clear its own MDE
(`50 × (0.0242/0.0149)²`), against the n=50 we ran. So the pre-registerable
follow-up is: *run WT k=21 varsplit vs team at n≈132 and the effect should
resolve at +0.015*. It is ~250 additional WT cells. **If it resolves there,
the non-replication converts into a confirmed quantitative law across two
chambers; if it does not, the model is falsified on the chamber where it
predicts hardest.** Either way it is a result, which is more than the current
"below MDE" delivers.

**Caveats, so this is not over-sold.** The exchange rate is measured on
LLM-free arms and applied to LLM arms, which assumes the coverage curve is
arm-independent — supported by the coverage oracle tying every LLM arm, but
an assumption. "Variables recovered" is measured from the runs, not predicted
from the menu; the fully a-priori version is the weaker lower bound above.
And three points is a model that fits, not a law that has been tested.

### The arm is INFEASIBLE at k/M = 0.75, by construction

Both errors are `partition_pools_by_variable` raising, at k=21 seeds 7 and 49
(**2 of 50, 4%**): a variable partition left scout B a pool of 10 entries
against a budget of 10, where the selection loop is inert because every name
gets queried.

This is a real limit of partition-granularity interventions, worth stating in
the paper rather than hiding: **the trick needs menu slack, and slack vanishes
as the budget approaches the menu size.** At WT k=21 the scout budgets are 11
and 10, so a feasible split needs 23 of 28 entries — five to spare — and claims
are assigned before balancing, so a scout claiming all three fat variables
(`load_out` 4, `load_in` 3, `hatch` 3) plus eight singletons takes 18 and
starves its peer. LT k=30 had 59 entries over 30 variables and could not hit it.

At 4% the surviving cells are close to an unbiased sample, but the selection
is on claim structure, which is the mechanism variable — so quote k=21 with
the exclusion stated.

### What this does to the paper's positive result

> The partition-granularity manipulation is demonstrated on LT k=30
> (+0.043, resolved, and a pre-registered slope predicting +0.040), and
> **does not reproduce on WT at either budget with n=50**. The mechanism
> variable moves on both chambers; the accuracy gain follows only where the
> menu leaves enough duplication to remove.

Resolving WT k=21's +0.017 would need **n ≈ 250 per arm**. That is not a null —
it is an equivalence bound at ±0.024 — but it is not the two-chamber
manipulation the earlier draft assumed.

**A correction to what was reported mid-run.** At cell level the two WT
budgets gave +0.0155 and +0.0160, and the stability across independent budgets
looked like a weak but real replication. Re-scored at 9 PC seeds, k=14 is
**−0.0000**. The apparent stability was single-draw PC noise — the exact
failure mode register §27 documents, arriving a second time in the same
quantity it was written about.

## M7 `team_varsplit` (2026-08-30): the deficit was redundancy, and a one-line change to WHAT is partitioned recovers it

`runs/m7-varsplit.parquet` — **90/90 cells, 90 ok, 0 errors, 0 PC
degeneracies.** LT, k=30, n=30 per arm, `deepseek-v4-flash-0731`, VPS
(`scipy-openblas`), 2 h 55 m on 6 workers. All three arms in ONE run, so the
contrast carries no cross-era confound.

**The strongest result in the chamber pillar, because it is a manipulation
confirming a mechanism rather than another observational contrast.**

| arm | distinct variables | shared | F1 | sd | SHD |
|---|---|---|---|---|---|
| `llm_pc` (loop) | 27.5 | — | 0.411 | 0.044 | 54.3 |
| `team` | 22.7 | **6.50** | 0.388 | 0.050 | 55.5 |
| **`team_varsplit`** | **28.2** | **0.00** | **0.424** | 0.045 | **53.4** |

### The pre-registered test

The prediction was fixed before the run, from a slope measured on unrelated
LLM-free arms (`coverage_*_ms`, +0.0073 F1 per distinct variable):

| | |
|---|---|
| predicted gain, +0.0073 x 5.5 variables | **+0.0399** |
| **observed gain** | **+0.0360** |

| contrast | delta | MDE | verdict |
|---|---|---|---|
| `team_varsplit` − `team` | **+0.0360** | 0.0344 | **RESOLVED** |
| `team_varsplit` − `llm_pc` | +0.0127 | 0.0322 | below MDE |
| `team` − `llm_pc` | −0.0233 | 0.0342 | below MDE |

Changing **only what gets partitioned** — identical topology, budgets, four
negotiation calls and A-wins-ties rule — recovers the deficit and brings the
two-agent arm level with the single sequential loop. Shared variables went
6.50 → **exactly 0.00**; distinct variables 22.7 → 28.2, slightly above the
loop's own 27.5.

**For the paper**: the cost measured across the whole ladder is not the cost of
having several agents, nor even of partitioning their information. It is the
cost of partitioning it **on the wrong object**. Drawn where the information
actually lives, a two-agent split is free.

### The caveat that must travel with it

`team` − `llm_pc` came in at **−0.023** here, against M6's −0.046 (n=30) and
M7 Phase 1's −0.048 (n=10). Pooling this run with Phase 1 — provenance verified
identical (OpenBLAS, flash-0731, LT k=30, same day) — gives **n=40 per arm:
−0.0296 against an MDE of 0.0298, i.e. just below threshold.**

So the honest statement is:

> `team_varsplit` beats `team` by a resolved **+0.036** and matches the loop.
> The `team`–loop deficit it closes is itself only marginally resolved at n=40
> (−0.030, MDE 0.030) and varies run to run from −0.023 to −0.048.

The recovery is **121% of the pooled deficit** — the arm fully closes a gap
whose size we know less precisely than we would like.

**The mechanism variable carries no such doubt.** Distinct variables are
near-deterministic per arm across independent runs (`team` 22.7 / 23.4; loop
27.5 / 27.9), and `shared` is 6.5 / 5.6 against a structural 0.00. Whatever is
adding variance to F1 is not touching what the arms buy.

### Why the F1 variance, and what follows

`llm_pc` and `team` run with **temperature unpinned**, so the seed governs only
the fallback RNG and PC; every cell is an independent draw from the provider's
default sampling (project memory records the same seed giving F1 0.330 and
0.482). This run is the clearest evidence yet that it inflates variance in
**arm means**, not only in cells — three estimates of one contrast spanning
−0.023 to −0.048.

**Pin temperature on the LLM arms before the next comparative sweep**, and
record the value per cell. Recorded as a scope limit on every contrast measured
before that lands.

## M7: `team`'s coordination is at CHANCE on the axis that costs it (2026-08-30)

The mechanism result above says team duplicates variables. This asks whether
its negotiation provides *any* protection against that, and the answer is no.

### What the negotiation actually decides

Four LLM calls (propose x2, revise x2), then a deterministic assembly. Recorded
diagnostics at LT k=30, n=10:

| | |
|---|---|
| conflicts to resolve (`n_contested`) | **1.2** of 30 claims |
| negotiation parse failures | 0 |
| claims needing truncation | 0 |
| claim's share of the resulting pool (`claim_pool_share`) | **0.51** |

Each scout claims its full 15 and they barely collide, because the menu is 59
entries wide. So the negotiation arbitrates almost nothing; it sets about
**half of each scout's option set**, and the other half is a seeded shuffle of
the ~29 unclaimed entries. Each scout then runs its own selection loop and
picks 15 from a pool of ~30.

### The null model

Split the menu into disjoint pools **at random**, pick 15 **at random** within
each pool — team's structure with the intelligence removed. 8,000 draws:

| | random null | actual `team` |
|---|---|---|
| scout A distinct variables (of 15) | 12.78 | **14.20** |
| scout B distinct variables | 12.76 | **14.80** |
| **variables bought by BOTH** | **4.11 ± 1.51** | **5.60** |
| distinct overall | 21.44 | 23.40 |

Two readings, and they are not in tension — `distinct = |A| + |B| − shared`:

- **Inside its own pool each scout clearly beats chance** (14.2 and 14.8
  against 12.8). The selection loops do real work.
- **Across scouts the coordination buys nothing.** 5.6 duplicated variables
  against 4.11 expected at random, z = +0.99 — if anything slightly worse, and
  within noise at n=10. **On the axis that costs accuracy, four LLM calls of
  negotiation perform at chance.**

### Why

Every stage that builds the pools is blind to variables. Conflict detection is
a set intersection on experiment NAMES; the leftover split is a parity slice of
a shuffled NAME list. Nothing in the pipeline knows `uniform_blue_mid` and
`uniform_blue_strong` are one variable.

**The scouts do not coordinate badly. They coordinate competently over the
wrong object.** For the paper: when work is partitioned between agents, the
partition has to be drawn where the *information* lives, not where the *task
list* lives.

### The one-change control: `team_varsplit`

Built 2026-08-30. Identical topology, budgets, four negotiation calls and
A-wins-ties rule; the only change is that pools are partitioned by VARIABLE, so
every entry of a variable travels to one scout and cross-scout duplication is
structurally impossible.

**Not a free win, and the outcome is open.** Concentrating a variable's entries
in one pool converts cross-scout duplication into within-scout duplication: a
~29-entry pool now spans only ~15 variables, so a scout must pick almost
exactly one entry per variable. Under `--mock-llm`, where selection degrades to
seeded random, the two effects cancel:

| | shared vars | per-scout distinct | total distinct |
|---|---|---|---|
| `team` | 3.83 | 12.5 / 12.3 | 21.0 |
| `team_varsplit` | **0.00** | 9.7 / 10.8 | 20.5 |

So the arm pays off only if scouts avoid SELF-repetition, which the real ones
do (0.8 and 0.2 repeats over 15 picks). **Pre-registered prediction**: if that
behaviour survives the narrower pools, distinct variables should reach ~29
against `team`'s 23.4, worth about **+0.041 F1** at the measured
+0.0073/variable — enough to close most of the −0.048 gap. If it does not, the
redundancy account is incomplete and the cost is coordination itself.

## M6 WT LADDER COMPLETE (2026-08-26): the topology result replicates

`runs/m6-wt-ladder.parquet` — **750/750 cells, 750 ok, 0 errors, 0 PC
degeneracies**, 21.3h wall on the VPS, **$11.64**, 127.7h active compute
across 6 workers. Grid: 5 rungs x k in {7,14,21} (k/M = 0.25/0.50/0.75 on
WT's 28-experiment menu) x 50 seeds. n=50 rather than 30 because WT compresses
effect sizes ~2.2x versus LT.

**Headline, scoped to what resolves (revised 2026-08-29): of the 24
topology-vs-loop contrasts — 4 multi-agent rungs x 3 budgets x 2 chambers —
**10 resolve at n=30/50, and 9 of those 10 favour the sequential loop.** The
tenth runs the other way. No fan-in topology ever resolves as *better* than
the loop on either chamber.

An earlier version of this line read "no multi-agent topology beats the single
sequential loop on EITHER chamber, at any budget where selection has signal",
and the table below it showed only the three budgets where the loop wins.
That is the claim the full grid does not support, for two reasons that must
travel with it.

**Exception 1 — at the lowest WT budget the ordering inverts, because the
baseline is broken there.** `team` beats the loop by **+0.040, resolved** at
WT k=7, and all four topologies are nominally above it (fan-in +0.029, roles
+0.016, chain +0.031, all below MDE). The reason is visible one table down:
**at WT k=7 the loop is itself significantly WORSE than random** — 0.145 vs
0.181, delta −0.036 against MDE 0.031, Welch p=0.0015. LLM selection actively
hurts at that budget, so the reference arm is below chance-level selection and
"beating the loop" there is not evidence for a topology. Report k=7 as a
regime where the comparison's denominator fails, not as a counterexample to
the topology result — and note the loop only overtakes random from k=14
(+0.031, p=0.028) and k=21 (+0.052, p=0.0004).

**Exception 2 — at LT's top budget nothing resolves at all.** All four deltas
at k=45 fall inside the MDE (−0.009, +0.014, +0.023, −0.003). That is an
equivalence bound, not a null: the design cannot separate the rungs there.
Note WT's top budget (k=21, the same k/M ≈ 0.75) *does* resolve three, so this
is specific to LT rather than a general property of high budgets.

**The chain is never resolved in either direction, anywhere.** All six of its
contrasts are below MDE (−0.038 to +0.031). "Delegation has measurable cost"
is not supported by this grid, and neither is its converse.

Every contrast, so the claim can be checked rather than taken:

| rung | LT k=6 | LT k=30 | LT k=45 | WT k=7 | WT k=14 | WT k=21 |
|---|---|---|---|---|---|---|
| ensemble `fan_in_homog` | −0.013 ns | −0.079 **R** | −0.009 ns | +0.029 ns | −0.048 **R** | −0.052 **R** |
| roles `fan_in_spec` | −0.046 **R** | −0.051 **R** | +0.014 ns | +0.016 ns | −0.044 **R** | −0.049 **R** |
| chain `planner_reasoner` | −0.038 ns | +0.011 ns | +0.023 ns | +0.031 ns | +0.022 ns | −0.024 ns |
| team (negotiation) | +0.006 ns | −0.047 **R** | −0.003 ns | **+0.040 R** | −0.014 ns | −0.040 **R** |

**R** = resolved against that cell's MDE; ns = below it. Negative favours the
loop. Reproduce with `analyze_results --input <ladder>.parquet --ladder`.

**The defensible sentence** is therefore: *where the comparison resolves, the
sequential loop is matched or beaten by no fan-in topology, and where it
resolves in the loop's favour the margin is 0.040-0.079 F1, on both chambers;
the one resolved exception is negotiation at the smallest WT budget.* The replication across two
chambers, different graphs (38 nodes/57 edges vs 32/42), different menus (59
vs 28) and different sample regimes (1,000 vs ~840 rows/experiment) is real
and is **the external-validity answer to "everything rests on LT's single
graph"** — it is the middle-budget result that replicates, not a claim about
every budget.

**Loop vs random** (recomputed 2026-08-26 on a matched platform; the figures
that stood here, +0.019 and +0.037, paired the VPS ladder against a *local*
random curve at n=30 and are superseded -- see §2b). Both arms VPS/OpenBLAS,
n=50, `runs/wt-random-vps.parquet`:

| k | loop | random | delta | MDE | Welch p | verdict |
|---|---|---|---|---|---|---|
| 7 | 0.1451 | 0.1813 | **−0.036** | 0.031 | 0.0015 | **RESOLVED** |
| 14 | 0.2388 | 0.2081 | +0.031 | 0.038 | 0.028 | below MDE |
| 21 | 0.2854 | 0.2339 | **+0.052** | 0.040 | 0.0004 | **RESOLVED** |

The matched contrast is **larger** at k=21 than the cross-platform one it
replaces. Unlike LT, WT does NOT converge at its top budget -- because
k/M=0.75 there is only 21 experiments absolute, against LT's 45. So **LT's
convergence is about absolute menu coverage, not budget fraction**; a useful
disambiguation for §5.

**k=7 is a resolved negative, not an uninformative band.** The earlier
reading -- "all five arms underperform random, report as below some budget
the LLM's selection is worse than chance" -- understated it: against a
matched-platform baseline the loop is worse than random by 0.036 at
**p=0.0015**, comfortably past the MDE. The claim is now positive and
directional: *on WT, below k≈10 the LLM's selection is measurably worse than
chance.* An earlier interim reading of k=7 as "splitting helps at low budget"
remains withdrawn.

**Conservation**: WT 64.3% (193/300 fan-in cells) against LT's 95.9%,
degrading 91% → 61% → 41% across budgets. `verify()` caught every overrun, so
this is provisioning, not mechanism — the WT c95/a95 figures were calibrated
from only 27 gate cells. **Report the two separately.** WT `team` carries no
conservation result at all (`_C95_NEGOTIATE` unmeasured for WT → forced None).

**Harness**: fallbacks 37/11,550 = 0.32%; 0 degeneracies; suspend **231s**
(the macOS gate recorded 70,029s — see register §9).

**Fixed after the sweep**: `allow_fallbacks` was True, so OpenRouter served 22
cells (2.9%) from OpenInference/Relace/DigitalOcean — unpinned, unknown
quantization. Impact nil (residualised +0.0095 vs −0.0003, p=0.55) but the
guarantee was void; now False. See register §8.

## COST–ACCURACY FRONTIER (2026-08-29): both blind fan-in rungs are dominated 12/12

Free result from the existing ladders — no new compute. `analyze_results
--cost-frontier` prices each arm in **LLM calls**, not dollars: call count is a
property of the topology, while price is a property of the provider that week
(the same model has billed 4.7x more on one endpoint than another). Every arm
buys exactly *k* interventions by construction, so the difference is
**coordination overhead alone**.

| arm | calls | overhead | what the overhead buys |
|---|---|---|---|
| loop (`llm_pc`) | k | — | reference |
| relay (`planner_reasoner`) | k | **+0** | a second system prompt at the seam |
| ensemble (`fan_in_homog`) | k+1 | +1 | the aggregator call |
| roles (`fan_in_spec`) | k+1 | +1 | the aggregator call |
| team | k+5 | +5 | 2 proposals, 2 revisions, 1 reconciliation |

At LT k=6 team's flat overhead is **+83%**; by k=45 it has amortised to +11%.

**Frontier (★ = Pareto-optimal; everything else costs more calls AND scores lower):**

| arm | LT k=30 | LT k=45 | WT k=14 | WT k=21 |
|---|---|---|---|---|
| loop | 0.420 / 30 | 0.417 / 45 | 0.239 / 14 | **0.285 / 21 ★** |
| relay | **0.431 / 30 ★** | **0.440 / 45 ★** | **0.260 / 14 ★** | 0.262 / 21 |
| ensemble | 0.341 / 31 | 0.409 / 46 | 0.191 / 15 | 0.234 / 22 |
| roles | 0.369 / 31 | 0.431 / 46 | 0.194 / 15 | 0.236 / 22 |
| team | 0.374 / 35 | 0.414 / 50 | 0.225 / 19 | 0.245 / 26 |

**Across all six chamber x budget points, both blind fan-in rungs are strictly
dominated — 12 of 12. Not one is worth its overhead at any budget on either
chamber.**

Two qualifications that must travel with this:

1. **The relay's stars are not "the relay is better".** Its accuracy edge over
   the loop is below MDE at every point (§"The chain resolves in neither
   direction"). What the frontier shows is that it is the only arm adding
   structure at *zero* call overhead. The honest sentence is **"structure is
   free when it costs no extra calls, and not worth paying for when it does"**.
2. **Pareto ranking ignores statistical resolution.** At LT k=6 and WT k=7
   `team` also lands on the frontier — as the most expensive endpoint with the
   nominally highest F1, and at both points that edge is below MDE. Read those
   stars as "not dominated", never as "best". A reviewer will check this.


## LT LOOP CURVE COMPLETE (2026-08-27): the gap closes because random catches up

`runs/m6-lt-loop-curve.parquet` — **420/420 cells, 420 ok, 0 errors, 0 PC
degeneracies**, 13.1 h wall on the VPS, **$7.26**, 76.9 h active compute
across 6 workers. Grid: `llm_pc` + `random` × k ∈ {6,12,20,30,40,50,59} × 30
seeds. Both arms fresh, one machine, one provider set, one model.

| k | k/M | loop | random | delta | MDE | Welch p | Holm | verdict |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.10 | 0.2094 | 0.1627 | **+0.047** | 0.035 | 0.0004 | 0.0026 | **RESOLVED** |
| 12 | 0.20 | 0.2604 | 0.2466 | +0.014 | 0.034 | 0.26 | 0.78 | below MDE |
| 20 | 0.34 | 0.3553 | 0.3137 | **+0.042** | 0.033 | 0.0008 | 0.0038 | **RESOLVED** |
| 30 | 0.51 | 0.4156 | 0.3604 | **+0.055** | 0.035 | 0.00004 | 0.0003 | **RESOLVED** |
| 40 | 0.68 | 0.4208 | 0.3849 | **+0.036** | 0.031 | 0.0022 | 0.0089 | **RESOLVED** |
| 50 | 0.85 | 0.4106 | 0.4104 | +0.000 | 0.035 | 0.98 | 0.98 | below MDE |
| 59 | 1.00 | 0.4225 | 0.4146 | +0.008 | 0.030 | 0.46 | 0.93 | below MDE |

All four resolved points survive Holm correction across the seven contrasts.

**The mechanism of convergence, which we previously had wrong.** The loop's
own F1 saturates at ≈0.42 by k=30 and never improves (0.4156 → 0.4208 →
0.4106 → 0.4225). Random climbs the whole way (0.3604 → 0.3849 → 0.4104 →
0.4146). So the advantage does not vanish because the loop degrades — **it
vanishes because random catches up.** Above k/M ≈ 0.85 there is almost no
selection left to perform: both arms buy nearly the same set.

**Retracted: "LT's convergence is about absolute menu coverage, not budget
fraction."** That was inferred from the ladder's loop F1 plateauing between
k=30 and k=45, which is the *loop saturating*, not the *gap closing*. The
fresh curve separates the two and shows LT still holds a resolved +0.036 at
k=40. WT was never run above k/M=0.75 on either scaling, so this data does
not discriminate absolute coverage from budget fraction. **Open scope limit,
not a finding.**

**The low-budget reversal is WT-specific.** WT k=7 gives −0.036 (loop worse
than chance, p=0.0015); LT k=6 gives +0.047 (loop better, p=0.0004). Opposite
under *both* matched-k and matched-k/M readings, so it is a property of that
chamber — plausibly its 28-experiment menu — and not of small budgets as
such. Do not state "below some budget the LLM's selection is worse than
chance" without naming the chamber.

**k=12 is an observed non-monotonicity, reported not smoothed.** +0.014,
below MDE, sitting between two resolved positives, driven by random gaining
more than the loop over 6→12 (+0.084 vs +0.051). Checked against the
collinearity moderator (register §11) and it survives: restricted to cells
with no dropped columns the delta gets *smaller*, +0.0037. One of seven
points landing below MDE by chance is unremarkable; it is recorded as
observed.

**Harness**: fallbacks 9/6510 = **0.138%** and flat in k (0.00/0.00/0.17/
0.00/0.25/0.13/0.17%) — no §1 moderator. 0 degeneracies. Every cell
`scipy-openblas` / `Linux-x86_64`, `pc_alpha` 0.05, `max_rows` 300,
`collinearity_threshold` 0.999. Providers confined to the pinned fp8 set
(Parasail/SiliconFlow/Baidu), **no off-pin routing** — `allow_fallbacks:
False` holding. One live caveat: collinear-column drops are budget- and
arm-dependent on LT, biasing the low-budget contrast *conservatively*; see
register §11.

**Preflight (12 cells, k ∈ {15,59})** was clean on all four stated checks and
caught the BLAS confound that would otherwise have contaminated this curve
(§2b). Cost/wall scale as ≈`k^1.19`; the $14 projection overshot the $7.26
actual roughly 2×, so future LT estimates should use this curve's per-k costs.

## AGGREGATOR ABLATION + UNCONTRACTED CONTROL (2026-08-27)

Two experiments closing the two largest gaps in the chamber pillar's story.
Both VPS / `scipy-openblas`, `deepseek-v4-flash-0731`, pinned fp8 providers,
0 errors. Combined cost **$2.83**.

### 1. The aggregator is inert BY MEASUREMENT (`runs/agg-ablation.parquet`)

60 cells, LT k=30 (the budget where the fan-in negative lives), n=30 per arm,
both arms run fresh in one sweep.

| | F1 | sd |
|---|---|---|
| aggregator **honored** (`fan_in_agg`) | 0.3290 | 0.0395 |
| aggregator **discarded** (`fan_in_homog`) | 0.3259 | 0.0446 |

delta **+0.0031**, MDE 0.0305, Welch **p=0.78** — equivalence, well inside
the bound. The diagnostics matter more than the delta: across **30/30 cells**
the aggregator **dropped nothing, hallucinated nothing, never returned an
empty answer** (`agg_dropped`, `agg_hallucinated`, `agg_fallback` all 0;
`agg_named` mean 20.4 against 20.4 distinct experiments pooled).

**Given authority over the pooled set, the aggregator reproduces the union
verbatim.** So the Python dedup is a *faithful* implementation of what the
LLM aggregator does when asked, not a strawman — and the ladder's negative
fan-in result is not an artifact of discarding it.

The architectural reason, which the measurement confirms: by the time the
aggregator runs the scouts have **already bought** their experiments. Its
only levers are reordering (which reaches PC solely via `run_pc`'s row
subsample) and dropping (strictly less data). It cannot un-buy and holds no
information the scouts lack. That is a property of fan-in-*after-purchase*,
and it is the honest scope of the negative result: a fan-in where the
aggregator allocates budget *before* purchase is a different topology that
this ladder does not test.

Free consistency check: fresh `fan_in_homog` at k=30 scores 0.3259, against
0.337 predicted from the M6 ladder's −0.079 offset and the fresh loop curve's
0.4156. The arm reproduces across the Novita → pinned-fp8 provider change.

### 2. Contracts are a floor on effort, not only a ceiling on spend

`runs/uncontracted.parquet` — 60 cells, both chambers, n=30 each. The
UNCONTRACTED arm is `llm_pc` with the contract removed: no budget in the
prompt, the agent may answer `DONE`, and the adapter is capped at the menu
size (a physical limit, not a governance bound). **The cap never bound —
0/60 cells hit it, so every stop was voluntary.**

| chamber | bought | range | F1 | cost CV |
|---|---|---|---|---|
| LT (menu 59) | 28.9 (sd 4.9) | **6–33, 5.5×** | 0.4224 | 21% |
| WT (menu 28) | 12.8 (sd 3.6) | **7–19, 2.7×** | 0.2272 | 34% |

| comparison | delta | MDE | p | verdict |
|---|---|---|---|---|
| LT uncontracted vs contracted k=30 | +0.007 | 0.035 | 0.59 | below MDE |
| LT uncontracted vs contracted k=59 | −0.000 | 0.033 | 0.99 | below MDE |
| WT uncontracted vs contracted k=14 | −0.012 | 0.048 | 0.49 | below MDE |
| **WT uncontracted vs contracted k=21** | **−0.058** | 0.045 | **0.0007** | **RESOLVED** |

Three findings, in ascending order of interest:

1. **At matched spend the contract costs nothing in accuracy.** Every
   matched-budget contrast is below MDE. Governance is not paid for in
   quality here.
2. **Spend variance is outcome variance.** The stopping point predicts the
   result: r = **+0.51** on LT (p=0.004) and **+0.50** on WT (p=0.005). LT
   uncontracted F1 ranges 0.292–0.533. So an ungoverned agent is not merely
   unpredictable in cost — you cannot tell in advance which *graph quality*
   you will get. Contracted arms buy exactly k, sd = 0, by construction.
3. **On WT the agent stops in the wrong place, and the contract fixes it.**
   It quits at 12.8 of 28 available experiments; a contract mandating k=21
   beats it by **+0.058 at p=0.0007**. **The contract is a floor on effort,
   not only a ceiling on spend** — `k` is a commitment to do the work, not
   merely a cap on it. On LT the same agent stops at 28.9, right at the knee
   where the loop curve saturates, and loses nothing. Whether self-regulation
   suffices is therefore **chamber-dependent and not predictable in advance**,
   which is the governance argument in one sentence.

This independently reproduces the framework's Nov-2025 positioning —
"governance, not optimization" — on a pillar built five months later with a
different task, model and metric.

**Scope limit, stated not glossed:** removing the budget necessarily changes
the prompt (no budget line, plus a `DONE` option), so the contrast is
contract-plus-prompt, not contract alone. `build_uncontracted_select_prompt`
holds menu rendering, history block and answer format identical to
`build_select_prompt` to keep that difference as small as it can be, but it
cannot be zero.

## v4-PRO ROBUSTNESS CHECK (2026-08-28): the ordering survives; scale does not help

`runs/pro-lt.parquet` + `runs/pro-wt.parquet` — **160/160 cells, 0 errors**,
7.0 h wall on the VPS, **$14.55**. `deepseek-v4-pro`, LT k=30 n=30 and
WT k=21 n=50, contrast restricted to the decisive one (loop `llm_pc` vs
ensemble `fan_in_homog`). n=50 on WT because WT compresses effects ~2.2x —
at n=30 the MDE (0.051) exceeds the flash effect (0.052), so that run would
have been guaranteed inconclusive before it started.

### 1. The topology ordering is model-robust

Ensemble minus loop, provider- and platform-matched **within each model**:

| config | n | loop | ensemble | delta | MDE | p | verdict |
|---|---|---|---|---|---|---|---|
| LT k=30 flash | 30 | 0.4156 | 0.3259 | **−0.090** | 0.032 | <1e-6 | **RESOLVED** |
| LT k=30 **pro** | 30 | 0.3520 | 0.2765 | **−0.075** | 0.031 | <1e-6 | **RESOLVED** |
| WT k=21 flash | 50 | 0.2854 | 0.2336 | **−0.052** | 0.039 | 0.0004 | **RESOLVED** |
| WT k=21 **pro** | 50 | 0.2656 | 0.2453 | −0.020 | 0.036 | 0.11 | below MDE |

On LT the effect reproduces at nearly the same magnitude under a model
costing 3.9x more per cell. On WT it keeps its sign but attenuates below the
MDE. **The single sequential loop is not an artifact of one model snapshot**,
which was the largest open threat to the ladder's headline.

Report the WT pro cell as an attenuation, not a contradiction: the sign is
unchanged and the flash effect there was itself the smallest in the study.

### 2. The larger model is WORSE at this task, and it is resolved

Same arm, budget, seeds, platform and provider class; only the model differs:

| contrast | pro | flash | delta | MDE | p | verdict |
|---|---|---|---|---|---|---|
| LT k=30 loop | 0.3520 | 0.4156 | **−0.064** | 0.030 | <1e-5 | **RESOLVED** |
| LT k=30 ensemble | 0.2765 | 0.3259 | **−0.049** | 0.033 | 0.0001 | **RESOLVED** |
| WT k=21 loop | 0.2656 | 0.2854 | −0.020 | 0.039 | 0.16 | below MDE |
| WT k=21 ensemble | 0.2453 | 0.2336 | +0.012 | 0.036 | 0.36 | below MDE |

At **3.9x the cost per cell** ($0.123 vs $0.032), v4-pro selects measurably
worse experiments on LT — on both arms — and is indistinguishable on WT.
Model scale does not buy accuracy at experiment selection here.

**The comparison worth putting in the paper:** on LT, changing the
coordination topology costs **−0.075 to −0.090** F1, while changing to a
4x more expensive model costs **−0.064**. *Topology is at least as large a
lever as model choice, and it is the cheaper one.* That is the strongest
single sentence the coordination pillar can make.

No mechanism is claimed for why pro is worse. Do not speculate in the paper —
the harness, prompts, budget, seeds and platform are identical, and that is
all the design licenses.

**Harness**: 0 errors; fallbacks 3/900 (LT) and 1/1050 (WT), and **all of
them fall on the loop**, so the bias runs against the finding rather than
for it. Providers 97-99% Baidu, remainder StreamLake — the pinned fp8
per-model order, no strays. Single BLAS, single platform, single model tag.

---

## Paper readiness (assessed 2026-08-28; amended 2026-08-31, 2026-09-02)

> **2026-09-02 amendment — the two lead sentences below both need restating,
> and the corpus now supports a better pair.**
>
> **1. "Topology is at least as large a lever as model choice" oversells.**
> It is true that fan-in *costs* 0.075–0.090, but the coverage oracle shows
> there is no topology *gain* available above small k — nothing we built beats
> a ten-line rule, and at WT k=21 the rule beats everything. The honest form
> is **topology can only cost you here**: you can lose 0.09 by choosing a
> fan-in and you cannot buy anything back by choosing a cleverer one.
>
> **2. "The contract is a floor on effort" is unchanged and is now the
> strongest positive in the corpus.** It should be co-headline, not §5.
>
> **The new lead, which the data does carry:**
>
> > On a task with a computable near-optimum, we measure agent topologies as
> > distance-from-optimum under contract-enforced matched budgets. LLM
> > selection beats a ten-line coverage rule only where the budget is too
> > tight for coverage to bind — and on the non-trivial subgraph, not even
> > there. Above that, no topology we built beats the rule and several lose to
> > it. No fan-in ever beats a single sequential loop.
>
> **What this buys:** a computable near-optimal reference policy is rare in
> agent benchmarks, and it converts three negatives into one measured claim
> with a ceiling. **What it costs:** it hands a reviewer the objection "your
> task is coverage-shaped." That objection is correct — §29's bipartite,
> depth-1 ground truth says so — and the answer is to scope it in the title,
> not to argue it. See threat 5 below.
>
> **The `team_varsplit` claim is no longer a single-chamber positive.** The
> non-replication on WT is *predicted* by an LLM-free two-factor model
> (`exchange rate × variables recovered`, 3/3 on the verdict) whose moderator
> is the ACTION SPACE, not the chamber. That is a stronger contribution than
> the original result, and it comes with a pre-registerable confirmatory test
> at WT k=21, n≈132. See the varsplit section.


> **Phase 2 amendment.** The two lead sentences below both survive Phase 2 —
> neither depends on the record axis. What Phase 2 changes is the *explanation*
> the paper may offer for the M6 ordering. `one_shot` carries no record and
> ties the loop at five of six budgets, so **the ordering is not explained by
> how much of the record survives**, and a draft that argues it that way is now
> contradicted by our own data. Two consequences: threat 1 below gains a
> sibling (the ladder's organising axis is unsupported at 5/6 budgets, which is
> a finding to report, not a defect to fix), and `critique` enters as a clean
> pre-registered negative. The one place the axis does hold — sharing a record
> beats splitting it at the middle budget, +0.053 LT / +0.046 WT — replicates
> across chambers and is the strongest single result in the corpus, but it is a
> mid-budget-only claim and must be stated as one.



**The empirical work now carries an AAMAS main-track submission.** Two weeks
earlier it did not — not because the results were weak, but because the title
promised a contracting framework while the experiments validated a
causal-discovery benchmark. Every arm in the registry was contracted, so
nothing measured what governance costs.

### The three ranked threats, all closed by measurement

| # | threat | status | evidence |
|---|---|---|---|
| 1 | The aggregator's output was discarded, so the fan-in negative could be a null-aggregator artifact | **CLOSED** | +0.003 against MDE 0.031, p=0.78; 30/30 cells dropped and hallucinated nothing |
| 2 | Everything rested on one model | **CLOSED** | a 3.9x pricier model reproduces the ordering on LT (−0.075 vs −0.090, both p<1e-6) and keeps its sign on WT |
| 3 | No contracted/uncontracted contrast existed | **CLOSED** | the control now exists on both chambers and yields the pillar's strongest framework claim |

### The two sentences to lead with

1. **Topology is at least as large a lever as model choice, and the cheaper
   one.** On LT, changing topology costs 0.075–0.090 F1; changing to a 4x
   pricier model costs 0.064. This converts a negative result about
   multi-agent systems into a positive claim about where to spend attention.
2. **The contract is a floor on effort, not only a ceiling on spend.** On WT
   the ungoverned agent quit at 12.8 of 28 experiments and a mandated k=21
   beat it by +0.058 (p=0.0007); on LT the same agent stopped at the knee and
   lost nothing. Whether self-regulation suffices is chamber-dependent and
   not knowable in advance.

### Still open, ranked

0. **The ladder's organising axis is unsupported at 5 of 6 budgets**
   (new 2026-08-31). Removing the record entirely (`one_shot`) costs nothing
   outside LT k=6. Report it: the M6 ordering is real and replicated, but the
   record-survival story we built it on is not what produces it. What remains
   is narrower and better evidenced — sharing a record beats splitting one at
   the middle budget, on both chambers.

1. ~~**The negative is carried by middle budgets.**~~ **CLOSED
   2026-08-29.** The headline above now states the scope it can support: of
   24 topology-vs-loop contrasts, 10 resolve and 9 favour the loop, with all
   24 tabulated so the sentence can be checked against the table rather than
   contradicted by it. Both exceptions travel with the claim — LT k=45
   resolves nothing (an equivalence bound, and note WT's equivalent k/M does
   resolve three, so it is specific to LT), and WT k=7 inverts because the
   loop is itself below random there (−0.036, p=0.0015), which makes it a
   broken denominator rather than a counterexample. The chain resolves in
   neither direction in any of its six contrasts.
2. **Conservation conflates mechanism with forecast.** LT 92.2% (249/270),
   WT 64.3% (193/300). A failure means `verify()` correctly CAUGHT an
   overrun: the mechanism worked every time, the cost prediction did not.
   Report as two numbers or a reader concludes the framework failed.
3. **Chain vs loop cannot resolve at n=30.** At the observed spread,
   separating a ~0.03 gap needs n≈55. Reportable as an equivalence bound —
   the analyzer prints the MDE beside every delta so it cannot be read as a
   null — but it is not a finding.
4. **Fan-in is tested only after purchase.** By the time the aggregator runs
   the scouts have already bought their experiments, so it cannot un-buy and
   holds no information they lack. A topology allocating budget BEFORE
   purchase is a different design this ladder does not test. State it.
5. **Smaller disclosures.** Temperature unpinned, so the seed does not
   control the model (variance, not bias). The negotiation parser reads
   restatement as claim, inflating rung 4's headline metric. `overlap_frac`
   is structurally 0.0 for rung 4. Removing the budget necessarily changes
   the prompt, so the governance contrast is contract-plus-prompt.
6. **The task is coverage-shaped, and we now have the evidence for it**
   (new 2026-09-02, and the top-ranked threat). A ten-line LLM-free rule
   ties every LLM arm at 5 of 6 budgets and beats them all at WT k=21; on
   core-20 scoring it ties at every LT budget. §29 explains why — the
   contemporaneous ground truth is bipartite, depth 1, zero mediators — and
   §28 adds that 18 of 38 nodes are pure apparatus sources. The compressed
   objection is **"you flattened the structure that would make coordination
   pay, then reported that coordination doesn't pay."** It cannot be argued
   away and must be scoped in the title and abstract. The one closable
   version: the chamber's depth is TEMPORAL, and the authors' own WT case
   study answers the same autocorrelation problem with PCMCI+ rather than a
   different dataset — a lagged-estimator variant is the experiment that
   would answer it, at engineering cost and no API cost.

   **The scoping sentence now has a named foil** (added 2026-09-05, from
   `docs/related-work/2026-08-27-google-antigravity-teamwork.md`). Google's
   Teamwork post reports seven solved open problems, a cycle-accurate CPU
   simulator and two merged library optimisations — and **every one of them
   sits on a cheap automatic per-candidate verifier**: Lean checks the proof,
   an air-gapped Spike simulator checks the cycle counts in "continuous
   lockstep co-simulation", a benchmark checks the hash table. Their Long
   Proof pattern is built on exactly that — "Many candidate strategies are
   generated in parallel, each paired with a falsifier whose sole job is to
   break it" — and they give the reason: "the flaw stays invisible until deep
   into the attempt."

   So the boundary is statable in one sentence instead of conceded:
   **partitioning pays when the action space has headroom — which we measure
   and predict — or when a cheap verifier makes parallel generate-and-falsify
   affordable, which is their regime and not ours.** That converts the
   objection from a hole into a scope condition the paper states in its own
   voice, and it costs nothing to add. It does not remove the threat; it
   stops the threat from being the only thing a reader can say about the
   result.

### Method as a contribution

Twelve harness defects are recorded in the register, several worth reporting
rather than quietly fixing because they generalize to anyone benchmarking LLM
agents: a scaffold failure rate that varies with the experiment's x-axis
(§1); the linear-algebra backend determining a seeded result (§10); and five
separate tests that certified our REQUEST rather than what ran (§3-4, §7, §8,
§10, §12).
