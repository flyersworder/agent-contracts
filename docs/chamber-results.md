# Chamber Pillar: Results

The canonical record of every chamber-pillar experiment and what it showed.
Results live here rather than in `claude.md`, which is project memory loaded
into every session and should stay instructions plus status.

**Companions.** `docs/chamber-harness-validity-register.md` records the thirty
harness defects that each changed or could have changed a result — read it
before trusting any number here. `docs/causal_chamber_validation_plan.md` is
the experiment plan; `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md`
is the ladder's design spec.

**Corpus as of 2026-09-12 afternoon**: 18,714 cells, **$132.94**, **zero errored cells**, (2026-09-11 read 18,303 / $115.87; the WT adaptive-feedback replication adds 200 cells / $8.26, its two fix attempts 211 cells / $8.81)
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
| `runs/m7-adaptive-lt.parquet` | 60 | $6.15 | `adaptive_feedback` vs same-sweep `llm_pc`, LT k=30, n=30 each (pre-registered, spec §8.7 row 7) |
| `runs/m7-adaptive-wt.parquet` | 200 | $8.26 | `adaptive_feedback` vs same-sweep `llm_pc`, WT k=14/21, n=50 each (pre-registered 2026-09-11; P1/P3 refuted, P2 held); re-scored at 300/1500 rows in `runs/rescored-adaptive-wt-rows{300,1500}*.parquet` |
| `runs/m7-adaptive-wt2.parquet` (+ `wt3`, killed at 11 cells) | 211 | $8.81 | the two summariser fixes (register §36): disclosure changed the `osr_ambient` buy rate by 0 points; exclusion-on-drops cannot fire at feedback time |
| `runs/m7-oneshot-shuffle-lt.parquet` | 90 | $0.57 | `one_shot_shuffle`, LT k=6/30/45: menu order per seed does not diversify k=30 |
| `runs/m7-oneshot-k6-control.parquet` | 60 | $0.17 | same-day `one_shot` + `one_shot_shuffle` at LT k=6, interleaved |
| `runs/m7-loop-k6-control.parquet` | 30 | $0.59 | same-day `llm_pc` at LT k=6; the 30 Aug single-call loss does not replicate |
| `runs/oracle-probe-{lt,wt}-*.parquet` | — | $0.00 | ground-truth oracle sets, static ranking, policies, arm purchase gains (`oracle_probe.py`) |
| `runs/rescored-vps-{rows300,rows1500,jci-rows1500}.parquet` (+`-bykey`) | 2,206 designs | $0.00 | the corpus under PC at two caps and JCI-PC at 1500, 9 seeds |
| `runs/rescored-vps-utigsp-lt.parquet` (+`-bykey`) | 769 designs | $0.00 | the LT corpus under UT-IGSP, never pooled, core-20 |
| `runs/rescored-vps-ges-rows1500-subset.parquet` (+`-bykey`) | 1,190 designs | $0.00 | LT k=30/45 + WT k=21 under GES at 1500 rows, 3 seeds — the cross-family check |

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

## PRE-REGISTERED (2026-09-11, before launch): the adaptive-feedback arm on the wind tunnel

`adaptive_feedback` vs a same-sweep `llm_pc` control, WT `standard`, k=14 and
k=21, n=50 per arm per budget, interleaved (200 cells), flash-0731, VPS. The
feedback summary's menu→variable mapping was verified against
`wt_menu_taxonomy` before launch (28 of 28 entries). Analysis as for the LT
run: re-scored at 9 PC seeds on one machine at 300 and 1500 rows, clustered
by distinct design, unequal-n bound; purchases scored on the WT oracle
marginal-gain scale (`oracle-probe-wt-context.parquet`).

Predictions, written before any cell ran:

- **P1 (oracle scale).** The feedback arm's purchases sit above the
  same-sweep loop's at both budgets. On WT the loop already beats random on
  this scale (27–28% of the random→oracle range at k=14/21), so the
  prediction is that feedback STACKS on it, not merely that it clears random.
- **P2 (vs the rule).** The arm does NOT resolve above `wt_coverage_max` at
  either budget; at k=21, where the rule beats the loop (−0.026 R−), the arm
  sits at or below the rule. Same outcome as LT.
- **P3 (vs the same-sweep loop, the interval prediction).** Point prediction
  **+0.012 at k=14 and +0.010 at k=21** on directed F1 at 300 rows — LT's
  +0.027 scaled by WT's ~2.2× effect compression. Decision rule keyed on the
  interval, not a threshold (register §-lesson from the WT k=21
  confirmation): supported if the 95% CI contains the prediction and
  excludes zero; consistent-but-underpowered if it contains both; refuted if
  it excludes the prediction. At n=50 the bound is ~0.022–0.028, so a
  RESOLVED result is not expected and its absence is not a failure.
- **Also reported, not predicted:** the fallback rate (LT: 1.8% of picks),
  tokens per call vs the loop (LT: cheaper), and the 1500-row reading.

**SCORED 2026-09-12 — next section. P1 refuted at both budgets (reversed),
P2 held, P3 refuted at k=14 and undecided at k=21.**

---

## TELLING THE MODEL WAS NOT ENOUGH (2026-09-12, VPS/OpenBLAS, `runs/m7-adaptive-wt2.parquet`): the second WT feedback run, summariser naming the removed sensors

Pre-registered (section "…re-run with the summariser told what PC
dropped"). 200/200 ok, $8.44, same design as 11 Sep; re-scored at 9 seeds
at 300 and 1500 rows (`runs/rescored-adaptive-wt2-rows{300,1500}*`),
design-clustered (50 distinct designs per arm per budget again).

**M (mechanism, primary) FAILS.** With the line "Variables REMOVED from the
estimate as near-duplicates of another sensor (no edge can reach them, and
a setting that only affects one of them cannot be connected either):
pressure_downwind, pressure_ambient, pressure_intake" in every prompt
after the first feedback round, the arm bought `osr_ambient` in **90% /
98%** of cells (loop 26% / 74%), `osr_downwind` 72% / 100% (16% / 84%),
`osr_intake` 80% / 100% (12% / 42%). Unchanged from 11 Sep (78% / 100%).
The model does not make the `osr_ambient → pressure_ambient` inference
from the line, or makes it and buys anyway; `osr_ambient` still sat in
the "no edge has reached yet" list, and that list is what it acts on.

**The arm is a replicate of itself.** Arm run 2 − arm run 1: +0.002 /
+0.003 at 300 rows, +0.003 / −0.005 at 1500 — the tightest cross-day
agreement of any arm in the corpus, because the feedback makes its buy
nearly deterministic (the same poison pill and the same coverage-shaped
set every time). The loop moved −0.020 / −0.015 between days (ties).

**Predictions scored, 300 rows:**

| | k=14 | k=21 |
|---|---|---|
| P1′ purchases vs loop on the oracle scale (×10⁻³) | 0.1 vs 3.0, −2.9 [0.8] **refuted** (and below random, −0.9 [0.7]) | 0.9 vs 1.8, −0.9 [0.4] **refuted**; random tie |
| P2′ not above the rule | −0.028 [0.018] R− **held** | −0.020 [0.013] R− **held** |
| P3′ vs same-sweep loop (pred. 0.000 / +0.010) | 0.214 vs 0.241, **−0.027** [0.020], CI [−0.042, −0.013] **refuted** | 0.269 vs 0.249, **+0.021** [0.019] R+, CI [+0.007, +0.034] **supported** |

The k=21 "support" is the loop's day, not the arm's: the arm scored 0.267
on 11 Sep and 0.269 today; the loop scored 0.264 and 0.249. Skeleton:
tie at both budgets. At 1500 rows the pattern of 12 Sep repeats — tie at
k=14 (+0.017 [0.020]), +0.041 [0.020] R+ at k=21, tie with the rule at
both — for the reason already established (the loop's entries lose value
with rows; the arm reaches the rule and no further).

**What it adds to §36.** A practitioner cannot fix this by disclosure.
Naming the removed sensors, stating the implication in one sentence, and
leaving the entry in the candidate list changes the buy rate by zero
points. And the exclusion cannot be keyed on the estimator's drops either
(run 3, killed: nothing is dropped at feedback time). What is left is the
design itself: coverage-shaped feedback nominates every unbought variable
and overrides the model's prior, which was right about `osr_ambient` on
WT and wrong about the apparatus settings on LT. A feedback that could
tell the two apart has to report what each experiment CHANGED, not what
the estimate has not connected — the "what varied" arm, still unbuilt.

---

## PRE-REGISTERED (2026-09-12, before launch): third WT feedback run — settings of removed sensors EXCLUDED from the feedback, not merely named

**Why a third run.** The second run (section below, `m7-adaptive-wt2`)
told the model which sensors PC removed and what that implies; at 38 cells
the fixed arm was still buying `osr_ambient` in 17 of 18 cells (loop 4 of
20). Its pre-registered mechanism check M fails; the pre-registration said
the next step is exclusion, not more text.

**The fix** (same branch): `sensor_configured_by(setting, nodes)` maps an
`osr_X`/`v_X` setting to the sensor it configures by the chamber manual's
naming (suffix match, with the manual's table for `c`→`current`,
`1`/`2`→`signal_N`, `in`/`out`→`current_in/out`); the test checks every
such setting against the ground truth on both chambers (each has exactly
one child and the rule names it). `summarize_estimate` now treats an entry
as dead if its target is a removed variable OR a setting that configures
one, moves it out of "unreached", and lists it on the removed line marked
"do not buy". On WT that excludes `osr_ambient`, `osr_downwind`,
`osr_intake` (settings of the three dropped barometers) whenever those are
dropped — and `pot_2`'s entry when it is. Nothing changes when nothing is
dropped (LT).

**Design:** identical to the two previous runs; output
`runs/m7-adaptive-wt3.parquet`.

**KILLED at 11 cells ($0.37).** The first three arm cells bought
`osr_ambient` at positions 7, 13 and 14 — post-feedback. Rebuilding their
feedback text offline: at every round `collinear=[]` (or `['pot_2']`),
`pressure_ambient` was neither collinear- nor zero-variance-dropped, and
`osr_ambient` sat in "unreached" as before. The barometers are NOT
collinear at feedback time (min |r| 0.79 after five load experiments); the
drop happens only at final scoring on some designs. The exclusion cannot
fire, so the run would have replicated run 2. Register §36 correction.

**Predictions, written before any cell ran:**

- **M″ (mechanism).** The arm buys `osr_ambient` in ≤ 25% of cells at
  k=14 and ≤ 65% at k=21 — i.e. at or below the loop's own rate (18%/58%
  on 11 Sep, 20% on 12 Sep at k=14) — and the same for `osr_downwind` and
  `osr_intake`. Residual buys can only come from the first five purchases
  (no feedback yet) and from the model buying an entry the line says not
  to. Fails if any of the three is more than 15 points above the loop's
  rate at either budget.
- **P3″ (vs the same-sweep loop, 300 rows, directed).** Point prediction
  **+0.010 at k=14 and +0.010 at k=21**; interval rule (contains prediction
  / excludes zero). No resolved win predicted. If M″ holds and P3″ still
  comes out below zero, the LT gain does not transfer for a reason beyond
  the poison pill, and that is the answer.
- **P2″ (vs the rule).** Not above `wt_coverage_max` at either budget.
- **P1″ (oracle scale, 300).** Tie with the loop; not above random.
- Also reported: tokens per call, fallbacks, drift, and the three loops'
  (11 Sep, wt2, wt3) agreement before any pooling.

---

## PRE-REGISTERED (2026-09-12, before launch): the WT feedback arm re-run with the summariser told what PC dropped (register §36 fix)

**The fix** (`feat/feedback-dropped-columns`): `run_pc` now reports the
columns it removed (`dropped_out`), and `summarize_estimate` receives the
collinear-dropped set. Entries that perturb a removed variable leave the
"no edge has reached yet" list, and a fourth line names the removed
variables with what that means for the buyer ("no edge can reach them, and
a setting that only affects one of them cannot be connected either"). When
nothing was dropped the text is byte-identical to the pre-fix summary
(tested), so the LT result needs no re-run: 58 of its 60 cells never had a
collinear drop and the other two had one drop in one PC run.

**Not changed, deliberately:** `osr_ambient` stays on the menu (removing it
would change the task for every arm and hide the lesson); the collinearity
policy stays (without it PC returns all-zeros on WT); the fix does NOT
hard-code the `osr_ambient → pressure_ambient` link — the model is told
which sensors were removed and has to make that inference itself. So this
tests whether *telling* the agent about the estimator's preprocessing is
enough, not whether excluding the entry helps (that we already know:
−0.086/−0.050 per buy).

**Design:** identical to the 11 Sep run — `adaptive_feedback` vs a
same-sweep `llm_pc` control, WT `standard`, k=14/21, n=50 per arm per
budget, seeds 0–49, `flash-0731`, VPS, four workers; output
`runs/m7-adaptive-wt2.parquet`. Same analysis (9-seed re-score at 300 and
1500 rows, design-clustered, unequal-n bound, oracle scale at 300).

**Predictions, written before any cell ran:**

- **M (mechanism, the primary check).** The fixed arm buys `osr_ambient`
  at no more than the same-sweep loop's rate at each budget (11 Sep: arm
  78% / 100% vs loop 18% / 58%). Fails if the arm's rate is still above the
  loop's by more than 15 points at either budget — that would mean the
  model reads the line and buys anyway, and the next step is exclusion, not
  more text.
- **P3′ (vs the same-sweep loop, 300 rows, directed).** The k=14 loss
  closes: point prediction **0.000 at k=14 and +0.010 at k=21**, interval
  rule as before (contains prediction / excludes zero). A resolved win over
  the loop is NOT predicted; the LT prior is "small or nothing" and WT
  compresses effects ~2×.
- **P2′ (vs the rule).** Still not above `wt_coverage_max` at either
  budget. At 1500 rows, ties the rule as on 12 Sep.
- **P1′ (oracle scale, 300).** Purchases no longer resolved BELOW the
  loop's; a tie with the loop is the prediction. Above random is not
  predicted.
- Also reported: fallback rate, tokens per call, drift audit, and the 11 Sep
  arm as a third comparator (same regime? — check arm means of the two loops
  agree before pooling anything).

---

## THE ADAPTIVE-FEEDBACK ARM ON THE WIND TUNNEL (2026-09-12, VPS/OpenBLAS, `runs/m7-adaptive-wt.parquet`): at the cap of record the LT gain does NOT transfer; at 1500 rows the arm beats the loop at k=21 — and the root cause of both is one poison-pill menu entry the feedback always buys

The pre-registered replication (section above, written before launch).
`adaptive_feedback` vs a same-sweep `llm_pc` control, WT `standard`, k=14
and k=21, n=50 per arm per budget, 200/200 ok, $8.26, `flash-0731`, four
workers on the VPS. Re-scored at 9 PC seeds at 300 and 1500 rows
(`runs/rescored-adaptive-wt-rows{300,1500}*.parquet`, together with
`m7-coverage-wt`, `m7-coverage-wt2` and `m7-p2-wt` so every comparator sits
on one backend); clustered by distinct design (50 of 50 for both arms at
both budgets — the loop and the feedback arm never re-pick a design on WT);
unequal-n bound. Purchases scored on the WT oracle marginal-gain scale.
One scheduling note: arms were interleaved within a budget but **k=14 ran
to completion before k=21 started** (the sidecar order), so the two budgets
are different windows. Every contrast below is within a budget, so nothing
rests on the cross-budget comparison. Drift audit inside the sweep is
clean: tokens per call residualised on arm×budget vs launch order r=−0.01
(threshold 0.21).

**All three pre-registered predictions scored, at 300 rows:**

| prediction | k | measured | MDE | verdict |
|---|---|---|---|---|
| P1: purchases above the same-sweep loop's on the oracle scale (×10⁻³) | 14 | adaptive **0.4** vs loop **3.2** (random 1.0) | 0.8 | **REFUTED — resolved the other way**, Δ −2.9; and not above random (−0.6, MDE 0.8) |
| | 21 | adaptive **0.6** vs loop **2.0** (random 1.0) | 0.4 | **REFUTED**, Δ −1.4; not above random (−0.4, MDE 0.4) |
| P2: does not resolve above `wt_coverage_max`; at k=21 at or below the rule | 14 | 0.212 vs 0.241, Δ **−0.030** | 0.019 | as predicted, but stronger: resolved BELOW the rule |
| | 21 | 0.267 vs 0.289, Δ **−0.023** | 0.013 | as predicted: resolved BELOW the rule |
| P3: adaptive − same-sweep loop = **+0.012 / +0.010**, interval rule | 14 | 0.212 vs 0.261, Δ **−0.049**, 95% CI [−0.064, −0.035] | 0.021 | **REFUTED** — the interval excludes the prediction and zero; resolved the wrong way, 2.4× the bound |
| | 21 | 0.267 vs 0.264, Δ **+0.003**, 95% CI [−0.013, +0.018] | 0.022 | consistent-but-underpowered (contains prediction and zero) |

Scorecard at the cap of record: **P1 refuted at both budgets, P2 held (the
conservative half of the prediction), P3 refuted at k=14 and undecided at
k=21.** The LT result (+0.027 vs the loop, resolved) does not replicate on
the second chamber at 300 rows; at the smaller budget it reverses.

**Skeleton, at 300 rows.** The k=14 loss shrinks to a tie (−0.007, MDE
0.017) and k=21 stays a tie (+0.006); the arm is above random on the
skeleton at both budgets (+0.023 / +0.024, resolved) and below the rule at
k=21 (−0.035, resolved). So the resolved directed loss at k=14 is mostly
orientation — but it is a loss on the metric the pillar reports, and the
purchase-scale result, which has no orientation in it, is the cleaner
refutation.

**At 1500 rows the picture inverts — and the arm is the only one that
does not move.** Every comparator loses 0.04–0.05 F1 going from 300 to
1500 rows on WT (the pooled-regime harm that grows with rows, section "WHY
STRONG INTERVENTIONS HURT"); the feedback arm loses 0.006 / 0.010:

| 300 → 1500 rows, directed F1 | k=14 | k=21 |
|---|---|---|
| `adaptive_feedback` | 0.212 → **0.206** (−0.006) | 0.267 → **0.256** (−0.010) |
| `llm_pc` same sweep | 0.261 → 0.208 (−0.053) | 0.264 → 0.210 (−0.054) |
| `wt_coverage_max` | 0.241 → 0.205 (−0.036) | 0.289 → 0.249 (−0.040) |
| `random` | 0.218 → 0.178 (−0.040) | 0.233 → 0.190 (−0.043) |

So at 1500 rows: vs the same-sweep loop **−0.002 [0.017] tie at k=14,
+0.047 [0.020] RESOLVED at k=21** (CI [+0.032, +0.061]; skeleton +0.040
[0.018], also resolved — not orientation); vs the rule **+0.001 / +0.008,
ties** on directed (skeleton: tie at k=14, −0.050 resolved below at k=21);
vs random **+0.028 / +0.067, resolved**. Under the pre-registered interval
rule P3 is still "refuted" at 1500 rows at both budgets — the k=14 interval
excludes the prediction on the low side, the k=21 interval on the HIGH side
(+0.047 vs +0.010 predicted) — which is the §-lesson about point
predictions again: the rule scores a miss, the direction and size at k=21
are a bigger version of the LT result. This is the two-cap corpus finding
(10 of 39 verdicts flip with the cap) landing on the one arm whose selection
is cap-robust.

**Root cause, found the same morning by three offline probes ($0).**

1. *Not the buy order.* Every design of both arms re-scored with its
   sequence permuted (two permutations, 400 designs,
   `runs/order-probe-wt-rows{300,1500}*.parquet`): shuffled − original is
   a tie on every cell of the table (|Δ| ≤ 0.008, MDE ≥ 0.011) and the
   300→1500 drops are identical (0.014 vs 0.006, 0.011 vs 0.010 for the
   arm; 0.046 vs 0.053, 0.054 vs 0.054 for the loop). The invariance is a
   property of the SET.
2. *One entry carries it: `validate_osr_ambient`.* A ridge fit of
   per-entry marginal value on all 433/425 WT designs at each cap gives
   `osr_ambient` the largest cap shift on the menu (value −0.016 → +0.016
   at k=14, −0.051 → −0.003 at k=21) and the feedback arm buys it in **78%
   / 100%** of cells against the loop's 18% / 58%. The additive model
   reproduces both arms' drops (arm 0.009 / 0.014 predicted vs 0.006 /
   0.010 measured; loop 0.040 / 0.049 vs 0.053 / 0.054). Within every arm
   the split agrees: loop cells that happen to contain it score −0.041 /
   −0.075 at 300 rows (k=14 / 21) and drop 0.027 / 0.031 instead of 0.059
   / 0.086; the feedback arm's 11 k=14 cells without it drop 0.038 like a
   loop.
3. *It is a poison pill, and the observational split UNDER-states it.*
   Adding `osr_ambient` as a 22nd buy to six same-sweep loop designs
   (nothing removed, 9 seeds, both caps): **−0.086 at 300 rows and −0.050
   at 1500**, every design negative at 300 and four of six at 1500;
   adding `osr_intake` instead, the other inert `osr` entry, gives −0.022
   / +0.005. Buying it makes PC drop **`pot_2` as a fourth collinear
   column** beside the three barometers, at both caps — one of the
   highest-value entries on the menu leaves the graph — and adds a node
   whose only true child (`pressure_ambient`) is already collinear-dropped,
   so every edge PC hangs on it is false. (The swap version of this probe
   is confounded — removing `pot_2` or `v_out` costs at both caps — and is
   not the evidence; the add version is.)

So the mechanism has one more step than "breadth on inert variables":
**the feedback reports coverage of the running estimate; every unbought
variable is "unreached"; the model, told to cover, buys `osr_ambient`
against its own prior (the loop buys it 18–26% at k=14, the arm 78–90%);
and the buy is not merely inert but destructive under this harness's
collinearity policy.** (Corrected the same day: an earlier version of this
sentence said the feedback nominates it because its child was dropped;
at feedback time nothing is dropped — register §36 correction.)
That is the 300-row loss. The "cap invariance" is the same tax paid at a
lower rate at 1500 rows (−0.05 vs −0.09) while the loop's preferred
entries — `pot_1` (bought 100% vs 16%), `load_out_pressure_intake` (68%
vs 4%), `load_in_current_out` (98% vs 0%) — lose their value with rows
(fitted 0.018 → 0.002, −0.014 → −0.030, unchanged). The k=21 win at 1500
rows is the loop's selection being cap-fragile, not feedback being right:
the arm reaches only the rule's level there (tie, +0.008). Register §36.

**Why it loses at the cap of record — the feedback buys breadth on the
variables the 300-row estimator cannot see.** The summary reports, per menu entry, whether the variable it perturbs has
been connected by the running estimate. On LT that steered the model
toward under-covered *apparatus* settings, which are exactly the informative
buys. On WT the under-connected variables are the oversampling-rate
settings (`osr_*`) — and the oracle ranks them last, because varying them
moves nothing PC can see. Measured over 50 cells per arm:

| | k=14 adaptive | loop | k=21 adaptive | loop |
|---|---|---|---|---|
| distinct variables per cell | **13.3** | 11.8 | **19.0** | 17.0 |
| mean oracle rank of buys (1 = best of 28) | 15.0 | **12.7** | 14.9 | **13.6** |

The feedback arm covers MORE variables than the loop (+1.4 / +2.0 per
cell; the LLM-free exchange rate of 0.011 F1 per variable would predict
+0.016 / +0.022) and scores lower, because of *which* variables: at k=14
it over-buys `osr_intake` (+35 of 50 cells, oracle rank 21), `osr_ambient`
(+30, rank 28 of 28), `osr_downwind` (+25, rank 20), and under-buys `pot_1`
(−38, rank 5), `osr_2` (−22, rank 1), `load_out_current_in` (−27, rank 10),
`load_in_current_out` (−18, rank 8). At k=21 the same pattern:
`load_in_current_out` −49, `pot_1` −42, `load_out_current_in` −40. **The
summary tells the model which variables have no edges yet, and the model
buys them; on WT those are the variables whose interventions have no
visible effect, so the arm is uncertainty sampling — the LT learner that
bought exactly the spurious-edge makers (section "WHAT THE HEADROOM IS"),
rebuilt inside the loop.** The LT gain and the WT loss have one mechanism
with opposite signs: "buy what the estimate has not connected" pays when
the unconnected variables are informative and unbought, and costs when
they are unconnected *because* they are uninformative. And "uninformative"
here is the 300-row oracle's verdict (register §34: the oracle is an oracle
for the harness) — which is consistent with the same buys being harmless,
and the arm cap-invariant, at 1500 rows.

**Costs and hygiene.** The arm is NOT cheaper on WT: output tokens 27.1k
vs 22.4k at k=14 (1,935 vs 1,601 per call) and 36.2k vs 34.9k at k=21;
wall 130 s vs 113 s and 178 vs 179. Input 5.5k vs 3.8k and 8.3k vs 5.7k.
Selection fallbacks 0.4% of picks in 3 of 50 cells for BOTH arms at k=14
(loop 5 cells at k=21) — no parse penalty on the shorter WT prompt. Both
loops (this sweep vs the earlier WT files) agree within their bound
(+0.018 / +0.010 at k=14/21, MDE 0.024 / 0.026), so the comparator is
not the anomaly.

**What the paper says now.** Report it at both caps, as for every arm
contrast since the two-cap re-score — and name the poison pill. At 300 rows, the configuration of
record: the adaptive-feedback result is **one chamber, one budget** — LT
k=30, +0.027 vs the loop, resolved, not above the rule; on WT resolved
*below* the loop at k=14, a tie at k=21, below the rule at both, purchases
below the loop's on the (300-row) oracle scale at both. At 1500 rows the
same WT cells tie the loop at k=14, beat it by +0.047 at k=21 (resolved on
directed and skeleton), and tie the rule on directed F1 at both. "Feedback
from the data moves an LLM arm off the random line" is withdrawn as a
cap-free claim and replaced by: *coverage-of-the-estimate feedback steers
selection toward whatever the estimator has not connected; at the 300-row
cap that is informative on LT and, on WT, a menu entry that this harness's
collinearity policy turns destructive — and in neither chamber at either
cap does it resolve above the coverage rule.* The 1500-row k=21 win over
the loop is reported with its cause: the loop's concentrated buys lose
value with rows, and the arm only reaches the rule. The per-experiment
"what varied" feedback the headroom section asked for is still the unbuilt
arm; this one reported coverage, and coverage is what it bought.

---

## MENU SHUFFLE, AND THE LT k=6 SINGLE-CALL LOSS THAT DID NOT REPLICATE (2026-09-11 night, VPS/OpenBLAS, `runs/m7-oneshot-shuffle-lt.parquet`, `runs/m7-oneshot-k6-control.parquet`, `runs/m7-loop-k6-control.parquet`)

Three small sweeps, 180 cells, $1.33, zero errors, all flash-0731, all
re-scored together with `m7-p2-lt` and `m7-p2-ref` at 9 PC seeds on one
machine at 300 and 1500 rows (`runs/rescored-shuffle-rows{300,1500}.parquet`).

1. **`one_shot_shuffle`** (new arm, `da68be1`): `one_shot` with the prompt's
   menu order permuted on a seeded RNG per seed, nothing else changed. LT
   k=6/30/45, n=30. Register §24's fix for the single-call arm re-picking
   the same design.
2. **Same-day k=6 control**: `one_shot` and `one_shot_shuffle` interleaved in
   one sweep, n=30 each — because (1) came out above the 30 Aug `one_shot`
   at k=6 and the two files were twelve days and one reasoning regime apart.
3. **Same-day loop control**: `llm_pc` at k=6, n=30, twenty minutes after
   (2) — because the loop reference was also from 30 Aug.

### The shuffle does not diversify the middle budget

| k | fixed prompt (30 Aug), distinct sets / 30 | shuffled (11 Sep) | sets shared | dominant set |
|---|---|---|---|---|
| 6 | 30 | 30 | 0 | — |
| **30** | **6** | **7** | 2 | **identical**: 18 of 30 shuffled cells buy the same 30 experiments as the fixed prompt's 17-cell mode (20 strong buys) |
| 45 | 24 | 24 | 3 | — |

At k=30 the model has a canonical answer — one experiment per variable, the
strong level wherever one exists — and menu order does not move it. There is
no position bias to remove either: picks sit at the uniform mean menu index
(30.7 / 30.3 vs 29.0 uniform; top-15 share 0.26 vs 0.25) in both files at
every budget. So register §24's k=30 bound is a property of the model's
prior, not of the prompt layout, and **menu shuffling cannot tighten it**.
Pooling both files' distinct designs (11) gives single-call − loop at k=30
of −0.000 [0.017] at 300 rows and +0.022 [0.026] at 1500: the same tie,
the same width. The remaining lever is a pinned non-zero temperature, which
register §21/§26 already showed to be no lever at all on this endpoint.
Closed, not deferred: at LT k=30 the single-call arm's bound is what it is.

The shuffled arm's F1 ties the loop and the fixed arm everywhere (LT k=30:
−0.000 [0.019] vs loop at 300, +0.012 [0.031] at 1500; k=45: −0.006 / +0.007,
both ties) — no prompt-order sensitivity in the score either.

### The k=6 loss was a 30 Aug draw

Phase 2's one resolved single-call finding — "the record pays at the tight
budget", `one_shot` − loop −0.047 [0.025] at LT k=6, −0.042 at 1500 rows,
−0.028 on core-20 — **does not replicate**. Same day, same reasoning
regime (2.1–2.4k output tokens per call on every arm), a fresh loop
control, design-clustered, unequal-n bound:

| LT k=6, 11 Sep | directed @300 | directed @1500 | skeleton @300 | core-20 @300 | core-20 @1500 |
|---|---|---|---|---|---|
| fixed prompt − loop (28/30) | −0.010 [0.026] tie | −0.008 [0.030] tie | +0.007 tie | −0.009 tie | −0.008 tie |
| shuffled − loop (29/30) | −0.002 [0.028] tie | +0.007 [0.029] tie | +0.001 tie | +0.000 tie | +0.012 tie |
| **both single-call − loop (57/30)** | **−0.006 [0.024] tie** | **−0.001 [0.026] tie** | +0.004 tie | −0.004 tie | +0.002 tie |
| 30 Aug fixed − 30 Aug loop (30/30) | −0.047 [0.025] **R−** | −0.042 [0.026] **R−** | −0.023 tie | −0.028 R− | −0.034 R− |

What moved is the single-call arm, not the loop: loop today − loop 30 Aug
is −0.015 [0.025] (tie, and the same to 0.000 on core-20), while `one_shot`
at k=6 went 0.159 → 0.182 (fixed) and 0.190 (shuffled). The 30 Aug
`one_shot` k=6 cells bought more strong light-source experiments (2.6 per
cell vs 1.9–2.3 today) — the PC-penalised buy — and reasoned twice as long
(5,015 vs 2,061 tokens per call). Whether that is the reasoning regime
(register §32) or a low draw at n=30 cannot be separated after the fact;
what can be said is that under today's regime, with 57 single-call designs
against 30 fresh loop designs, the gap is −0.006 with a bound of 0.024.

**Consequence.** "A running record pays only while the budget is tight" is
withdrawn as a resolved claim. The honest sentence is: **the single-call arm
ties the loop at every LT budget and every WT budget in the best-controlled
reading we have; the one resolved loss (LT k=6, 30 Aug) did not replicate
same-regime at n=57 vs 30.** The record is not load-bearing anywhere we
have measured. That strengthens Phase 2's headline and removes its one
exception; it also removes the "pays only when tight" moderator sentence
from the brief, §08 and the abstract. Sixth retraction, and the second
(after `critique`) to fall to a same-regime re-run of a resolved n=30 cell.
This is register §32's rule biting in the other direction: the two arms of
the 30 Aug contrast WERE interleaved, so drift could not land on one of
them — but a single-day n=30 resolution at 1.9× its bound is still one
draw of the endpoint's regime, and a resolved verdict that matters should
be re-run on another day before it goes in a paper.

**One estimate corrected.** This sweep was forecast at "hours" from the
Phase 2 pace and took 8 minutes: `one_shot` is one call per cell, and the
Phase 2 pace was set by the loop's k calls. Estimate from the arm's own
call count, never from the sweep it will be compared with.

---

## GES — THE CROSS-FAMILY CHECK (2026-09-11, VPS/OpenBLAS, `runs/rescored-vps-ges-rows1500-subset.parquet`): the structural claims hold under a score-based estimator; the anti-prior REVERSES

`ges.py` + `rescore.py --estimator ges --pc-max-rows 1500`, causal-learn
`local_score_BIC`, on the headline subset only: the LT k=30 and k=45 designs
and the WT k=21 designs (1,394 cells, 1,190 distinct designs × 3 seeds,
3,570 scorings, 3 workers, 4 h 08 min, ~550 MB per worker). The full
corpus would have taken two days and was not run. GES pools regimes exactly
as PC does and shares the row cap, so it tests the TEST FAMILY (score vs
independence tests), not the pooling. Compared throughout against PC and
JCI-PC at the same 1500-row cap on the same designs; design-clustered,
unequal-n MDE.

**The dossier's prediction (4) was "GES reproduces the anti-prior". It does
not — it reverses it.**

### Arm contrasts, directed F1 (design-level, 3 seeds)

| contrast | PC@300 | PC@1500 | JCI-PC@1500 | GES@1500 |
|---|---|---|---|---|
| LT k=30 loop − coverage rule | −0.001 tie | −0.032 R− | −0.043 R− | +0.007 tie |
| LT k=30 rule − random | +0.065 R+ | +0.099 R+ | +0.107 R+ | +0.096 R+ |
| LT k=30 min-rule − rule | −0.100 R− | −0.241 R− | −0.246 R− | **−0.303 R−** |
| LT k=30 loop − random | +0.065 R+ | +0.067 R+ | +0.065 R+ | +0.103 R+ |
| LT k=30 `team_varsplit` − `team` | +0.042 R+ | +0.057 R+ | +0.078 R+ | +0.061 R+ |
| LT k=30 `one_shot` − loop (6 designs) | +0.002 tie | +0.038 R+ | +0.038 R+ | +0.016 tie |
| LT k=30 `critique` − loop | −0.015 R− | +0.022 R+ | +0.021 R+ | +0.004 tie |
| LT k=30 `shared_blackboard` − loop | −0.020 R− | −0.022 tie | −0.014 tie | **−0.074 R−** |
| LT k=45 loop − rule | −0.001 tie | −0.012 R− | +0.044 R+ | −0.026 tie |
| LT k=45 rule − random | +0.017 R+ | +0.021 R+ | −0.032 R− | +0.069 R+ |
| LT k=45 min-rule − rule | −0.009 tie | −0.042 R− | −0.084 R− | **+0.002 tie** |
| LT k=45 Phase 2 arms − loop | tie ×3 | tie ×3 | R− / tie / R− | tie ×3 |
| WT k=21 loop − rule | −0.026 R− | −0.038 R− | −0.025 R− | −0.021 R− |
| WT k=21 rule − random | +0.056 R+ | +0.059 R+ | +0.031 R+ | +0.027 R+ |
| WT k=21 min-rule − rule | −0.074 R− | −0.067 R− | −0.032 R− | −0.028 R− |
| WT k=21 loop − random | +0.030 R+ | +0.020 R+ | +0.006 tie | +0.005 tie |
| WT k=21 `team_varsplit` − `team` | +0.013 tie | +0.018 R+ | +0.009 tie | +0.001 tie |
| WT k=21 Phase 2 arms − loop | tie ×3 | tie / R− / tie | tie ×3 | tie ×3 |

9 of 23 directed verdicts differ between GES and PC at 1500 rows; 5 of 23
on the skeleton; 9 of 15 on core-20 (where GES's absolute level is far
higher, so its MDEs are wider in the same units). Skeleton and core-20
tables: `ges_verdicts.py f1_skeleton` / `f1_core` in the session scratch.

What holds under all four estimator settings, both chambers:

- **Breadth beats depth.** The min-rule loses to the max-rule everywhere it
  is tested except LT k=45 under GES, where the two tie at 0.679 / 0.677 —
  at three-quarters of the menu a score-based estimator no longer cares
  which entries were bought. On LT k=30 the margin under GES is the largest
  of any estimator: −0.303, with `coverage_min` at 0.246 against
  `coverage_max` at 0.671.
- **The coverage rule beats random**, and **no LLM arm resolves above the
  rule** (LT k=30 +0.007 tie, LT k=45 −0.026 tie, WT k=21 −0.021 R−).
- **`team_varsplit` beats `team` at LT k=30** (+0.061 under GES), the
  fourth estimator setting to resolve it. At WT k=21 it is a tie under GES
  (+0.001), as under PC@300 and JCI-PC; only PC@1500 resolves it.
- **The loop beats random at LT k=30** (+0.103, the largest of the four).

What moves:

- **`shared_blackboard` is resolved BELOW the loop at LT k=30 under GES on
  every metric** (directed −0.074, skeleton −0.063, core −0.055), where PC
  had it on the boundary (−0.020 R− at 300, −0.022 tie at 1500) and UT-IGSP
  at −0.004 R−. The Phase 2 "sharing a record beats splitting one" finding
  is `shared_blackboard` vs `fan_in_spec`, which is not in this subset;
  what this says is that sharing a record with the loop is WORSE than the
  loop under a score-based judge, not merely equal. Report it beside the
  Phase 2 claim.
- `one_shot` and `critique` return to ties with the loop (PC@1500 and
  JCI-PC had both R+). Consistent with the standing rule: those two
  contrasts are orientation noise and should not be adjudicated.
- WT loop − random shrinks to +0.005 (tie) under GES and JCI-PC from
  +0.030 / +0.020 under PC. On WT the LLM's advantage over random is a
  PC-only result; the rule's advantage over random is not.

### The anti-prior reverses

Within-budget partial slope of F1 on the number of STRONG light-source
buys, controlling for light-source buys overall (the same regression as
the UT-IGSP section), LT subset:

| k | metric | PC@1500 | JCI-PC@1500 | GES@1500 | UT-IGSP (corpus) |
|---|---|---|---|---|---|
| 30 | core-20 | −0.0092 (r −0.31) | −0.0072 (r −0.38) | **+0.0088 (r +0.43)** | +0.0003 (r +0.25) |
| 45 | core-20 | −0.0065 (r −0.30) | −0.0030 (r −0.25) | **+0.0171 (r +0.31)** | +0.0008 (r +0.23) |
| 30 | directed | −0.0083 | −0.0149 | **+0.0023** | — |
| 45 | directed | −0.0053 | −0.0121 | **+0.0148** | — |

Under GES a strong light-source experiment is the BEST thing to buy on the
core graph, by the same margin per experiment that PC charges for it. GES
pools the same rows PC pools; the difference is the test. A BIC score over
the pooled Gaussian is a global fit and tolerates a shifted regime; a
sequence of Fisher-Z accept/reject decisions is not, and one shifted block
flips the borderline tests (results doc "WHY STRONG INTERVENTIONS HURT").
So the anti-prior is now bracketed from both sides: absent under the
estimator that never pools (UT-IGSP), reversed under a pooled estimator
from a different family (GES), present only under the independence-test
family (PC, JCI-PC). **The models' prior — "strong root interventions give
the most signal" — is right under two of the three families we have run.**
"The LLM arms carry the opposite of the needed knowledge" stays withdrawn,
and the sentence to use is: the selection findings are properties of the
independence-test family on pooled data at a fixed cap.

### Levels, and a caveat

GES sits far above PC in absolute terms on LT (loop 0.651 directed at k=30
against PC's ~0.40; core-20 0.686 against PC's ~0.22, beside UT-IGSP's
0.642 ceiling) and slightly below on WT (0.271 against ~0.29). LT arm means
under GES span 0.246–0.671 at k=30 — unlike UT-IGSP, GES discriminates
designs strongly, which is why it can resolve contrasts at all. Seed noise
is comparable (within-design sd 0.045 vs PC's 0.040). The CPDAG is read through
`cpdag_to_directed_adjacency`, PC's own encoding: an undirected edge
counts in both directions (one true positive and one false positive when
the true edge exists); the same convention for both estimators, favouring
neither.

Caveat: this is the headline SUBSET, not the corpus — LT k=6, WT k=7 and
k=14 and the ladder arms are not scored under GES, and the WT rule-vs-loop
figures above are the only WT budget tested.

### The 5000-row JCI-PC pass did not land

The per-variable JCI-PC pass at 5000 rows on the LT k=30 designs (2
workers) stalled at 205 of 356 designs after 12 h and was killed: per-worker
memory grew from ~2.4 GB at launch to 5.8 and 8.2 GB, 8 GB of swap filled,
and the pace fell to ~3 designs/h. PC's skeleton phase at 5000 rows with
~30 context columns is exponential in surviving degree; the re-scorer has
no checkpoint, so the partial work is lost. The k=45 and per-regime 5000-row
passes queued behind it were withdrawn. If re-run: one worker, one PC seed,
LT k=30 only, and only after `rescore.py` writes a per-design sidecar.
Prediction (5) of the dossier — whether JCI-PC at 5000 rows still ranks the
strong light-source experiments last — stays open; the GES and UT-IGSP
results above make it less load-bearing, since two other families already
answer it in the models' favour.

---

## UT-IGSP — THE ESTIMATOR THAT NEVER POOLS (2026-09-10 evening, VPS, `runs/igsp-calibration-lt.parquet`): first look, alpha sweep in progress

`igsp.py` + `rescore.py --estimator utigsp` (`5491860`..`0e94bfe`): the
chamber authors' interventional method (Squires, Wang & Uhler 2020, via the
maintained split of `causaldag`). One observational sample — the reference
run, 10,000 rows, given to every arm whether or not it bought it — plus one
sample per bought experiment with its known target; CI tests inside the
observational sample, invariance tests between it and each interventional
one; no table is ever concatenated. **LT only** (WT has no observational
entry). **Core-20 by construction**: the 18 apparatus settings are constant
within every sample and drop out, so the numbers belong beside `f1_core`.
Runtime 3–8 s per design at all rows.

**Alpha calibration on NEUTRAL designs only** (`random` + `coverage_max_ms`,
LT k=6/30/45, 120 designs, 3 seeds, all rows; CI and invariance alpha set
equal):

| alpha | core-20 F1 | full F1 | SHD |
|---|---|---|---|
| 0.05 | 0.461 | 0.387 | 69.4 |
| 0.01 | 0.471 | 0.393 | 66.0 |
| **0.001** | **0.489** | **0.404** | **61.5** |

Still monotone at 1e-4 (0.497) and 1e-5 (0.543) on ALL rows — a symptom,
not a calibration: the test rejects too often on this data and only an
extreme threshold compensates. **Capping every sample — observational
included — at 1,000 rows changes both facts** (`runs/igsp-obscap-lt.parquet`,
`igsp-calibration-lt-cap1000.parquet`): the level rises (core 0.49 → 0.63 at
alpha 1e-3) and alpha gains an interior optimum:

| alpha, all samples capped at 1,000 rows | core-20 F1 | full F1 | SHD |
|---|---|---|---|
| 1e-3 | 0.622 | 0.484 | 42.0 |
| **1e-4** | **0.634** | **0.492** | **40.7** |
| 1e-5 | 0.609 | 0.469 | 41.7 |
| 1e-6 | 0.599 | 0.460 | 42.1 |

**Design of record: every sample capped at 1,000 rows (the observational
run sized like an experiment — also the fair comparison), alpha 1e-4 for
both test families.** Chosen on neutral designs only, before the corpus
pass. The authors' grid on the same chamber is 1e-4…1e-2.

The 10,000-row observational sample scoring BELOW its own 1,000-row subsample
is the single-regime row effect of "WHY STRONG INTERVENTIONS HURT" item 3,
seen through a second test family.

**Two things visible before calibration ends, both to be read as hypotheses
for the corpus pass, not results:**

1. **Selection barely registers.** At every k, every alpha AND every cap,
   `random` and the coverage rule are within 0.01 core-20 F1 of each other
   (cap 1,000, k=30: 0.622 vs 0.628). So the flatness is not the free
   observational rows — it survives sizing them like one experiment. Under PC at 1500 rows the same two arms differ
   by 0.099. If this survives the corpus, the entire "which experiments to
   buy" signal this benchmark measures lives in the pooled reduction, and
   under the authors' own estimator the purchasing task has almost no
   headroom — because the observational sample identifies the core skeleton
   on its own and interventions only orient.
2. **The budget response is nearly flat**: core-20 0.61 → 0.63 → 0.63 at
   k=6/30/45 under the design of record, against PC's 0.18 → 0.22 → 0.23.
   Same reason.

**Absolute level.** Core-20 F1 ≈ 0.49 on random designs against PC's 0.22
at LT k=30: the never-pooled estimator with 10,000 observational rows is a
much better instrument for the core graph. That is not a comparison of
purchases; it is a statement about the judge, and the paper reports it as
one.

**Pre-registered for the LT corpus pass (before it ran):** (a) arm means
converge — the spread across arms at k=30 falls below 0.02; (b) the
coverage rule does NOT resolve above the loop; (c) the marginal gain of a
strong light-source experiment is non-negative, i.e. the anti-prior does
not reproduce when regimes are never mixed.

### Outcome (2026-09-10 night, `runs/rescored-vps-utigsp-lt.parquet`, 769 designs × 3 seeds, 51 min): all three hold

| k | spread of arm means, core-20 (UT-IGSP) | same under PC@1500 | loop − rule (UT-IGSP) | strong-buy slope (UT-IGSP) | strong-buy slope (PC@1500) |
|---|---|---|---|---|---|
| 6 | 0.013 | 0.051 | −0.004 (tie) | −0.003 | −0.022 |
| 30 | 0.015 | 0.077 | −0.002 (tie) | +0.000 | −0.009 |
| 45 | 0.004 | 0.033 | +0.002 (tie) | +0.001 | −0.006 |

- **(a) Convergence — confirmed.** Every arm sits at core-20 F1 0.615–0.642;
  at k=30 the spread is 0.015 and at k=45 it is 0.004. Several arms have
  **zero variance across designs** (`coverage_max` 0.642 ± 0.000, `llm_pc`
  at k=45 0.642 ± 0.000): the estimator returns the same graph whatever was
  bought. 0.642 is its ceiling on this chamber, and almost any k=30 purchase
  reaches it.
- **(b) The rule does not beat the loop — confirmed**, nor does anything
  beat anything by more than 0.008. Three contrasts "resolve" only because
  the variance collapsed (MDEs of 0.003–0.006): `shared_blackboard` −0.005,
  rule vs random +0.008, `coverage_min` vs `coverage_max` −0.006. Report
  them as what they are: differences smaller than a rounding convention.
- **(c) The anti-prior does not reproduce — confirmed.** Within each budget,
  the partial slope of core-20 F1 on the number of STRONG light-source
  buys (controlling for light-source buys overall) is −0.003 / +0.000 /
  +0.001 under UT-IGSP against −0.022 / −0.009 / −0.006 under PC at 1500
  rows; the raw correlation is **+0.25 / +0.23 at k=30/45** under UT-IGSP
  against −0.31 / −0.30 under PC. **The models' "strong root interventions
  give the most signal" prior was right about the chamber and wrong about
  our estimator.** "The LLM arms carry the opposite of the needed
  knowledge" is withdrawn; what they lacked was knowledge of the JUDGE.

**What this does to the paper.** Under the chamber authors' own estimator
the budgeted-selection task has essentially no headroom on the core graph:
the observational sample plus any handful of interventions identifies what
the tests can identify, and purchases only add orientation the ceiling
already includes. Every selection-level finding in this document — the
coverage rule as near-oracle, the oracle headroom, the sensor-setting
regime, the anti-prior — is therefore a property of **PC on pooled data at
a fixed row cap**, the estimator of record, and is reported as such. The
arm-contrast findings survive in the only form they ever had: under one
judge, held fixed, with the judge named. And the contract finding (a floor
on effort) does not depend on the judge at all.

**Two caveats that travel with this.** (1) UT-IGSP scores the 20-variable
core; the 18 apparatus edges, where 78% of PC's budget response lives
(register §28), are invisible to it by construction — so "no headroom"
is a core-graph statement. (2) The result depends on giving the estimator
the observational sample at all; sized like one experiment it still holds,
but an estimator denied any observational data was not tested.

---

## JCI-PC ON THE CORPUS AT 1500 ROWS (2026-09-10 afternoon, VPS/OpenBLAS, `runs/rescored-vps-jci-rows1500.parquet`): the predictions scored, and a defect in the indicator design found and fixed

The same 2,206 designs, 9 seeds, `--estimator jci_pc` (per-VARIABLE
indicators, the design of record when this ran), against PC at the same cap.

**Verdict changes, PC → JCI-PC, both at 1500 rows: 9 of 39 on directed F1,
5 of 39 on the skeleton.** Every change:

| contrast | PC Δ (verdict) | JCI-PC Δ (verdict) | skeleton |
|---|---|---|---|
| LT k=45 `one_shot` − loop | −0.002 (tie) | −0.049 (**R−**) | tie → R− |
| LT k=45 `shared_blackboard` − loop | −0.004 (tie) | −0.019 (**R−**) | tie, tie |
| LT k=45 loop − rule | −0.012 (R−) | **+0.044 (R+)** | tie → **R+** |
| LT k=45 rule − random | +0.021 (R+) | **−0.032 (R−)** | R+ → **R−** |
| LT k=30 loop − rule | −0.032 (R−) | −0.043 (R−) | tie → R− |
| LT k=6 `one_shot` − loop | −0.042 (R−) | −0.032 (R−) | tie → R− |
| WT k=7 `critique` − loop | −0.000 (tie) | −0.035 (R−) | tie, tie |
| WT k=7 loop − rule | −0.026 (R−) | +0.015 (tie) | tie, tie |
| WT k=14 `one_shot` − loop | +0.012 (tie) | −0.035 (R−) | tie, tie |
| WT k=21 `critique` − loop | −0.021 (R−) | +0.001 (tie) | tie, tie |
| WT k=21 `team_varsplit` − `team` | +0.018 (R+) | +0.009 (tie) | R+, R+ |

**Arm means, JCI-PC minus PC, LT:** k=6 every arm +0.005 to +0.027; k=30
every arm +0.018 to +0.030 (`coverage_min` +0.157 — the depth rule, one
variable at three strengths, gains most from a regime column); **k=45 every
arm LOSES, and unequally: loop −0.020, random −0.023, `shared_blackboard`
−0.035, `one_shot` −0.067, `coverage_max` −0.076, `coverage_min` −0.118.**
That inequality is where four of the nine flips come from, including the
one that reverses a headline (the rule falling below the loop AND below
random at LT k=45).

**Prediction scorecard** (written before the file came back, "JCI-PC PROBE"):

1. *Every resolved arm contrast reproduces.* **Partly false**: 9 of 39
   directed, 5 of 39 skeleton. Six of the nine are boundary cases; three at
   LT k=45 are not.
2. *The rule loses ground in proportion to distinct variables bought.*
   **Wrong as stated.** At LT k=45 every arm bought ≈11 light-source
   variables (loop 11.0, rule 11.0, `one_shot` 10.96) and their penalties
   differ four-fold, so the indicator COUNT explains nothing there
   (r = +0.32 the wrong way). What does: **light-source variables bought
   at two strengths and merged into ONE indicator** — r −0.38 directed /
   −0.47 skeleton, OLS **−0.0105 per merged pair** with single-regime light
   buys at +0.0093. `coverage_min_ms` merges 5.0 pairs (−0.118), the rule
   3.8 (−0.076), the loop 2.5 (−0.020). At LT k=30 the penalty tracks
   light-source experiments bought (r −0.71 / −0.74), i.e. regimes, not
   variables. The confound with coverage is real but it runs through
   DEPTH (regimes per variable), not breadth.
3. *Skeleton verdicts move less than directed ones.* **True** (5 vs 9).

**The defect and the fix.** "One indicator per target variable" (jci.py,
decision recorded that morning) puts `red_mid` and `red_strong` rows in one
block that is itself a two-regime mixture — the very thing the indicator
exists to remove. JCI's context variable is the REGIME. `regime_label` now
emits `<variable>@<strength>` on LT and one label per menu entry on WT;
`rescore.py --context regime` selects it and stamps `rescore_context`
(`3b101fa`, 47 tests).

**Probe on the 84 designs at 1500 rows, 3 seeds — the fix does NOT help:**

| chamber, k | PC | JCI per-variable | JCI per-regime |
|---|---|---|---|
| LT 6 | 0.176 | 0.194 | 0.185 |
| LT 30 | 0.379 | **0.410** | 0.380 |
| LT 45 | 0.432 | 0.388 | **0.310** |
| WT 7 / 14 / 21 | 0.177 / 0.175 / 0.196 | 0.196 / 0.185 / 0.206 | 0.199 / 0.192 / 0.210 |

Splitting a merged indicator removes one within-block mixture but adds one
more sparse binary node, and at 1500 rows the second cost exceeds the first
on LT (WT is unchanged: its entries are already one regime each). The merged-
pair correlation was genuine; its counterfactual was not. **Per-variable
stays the JCI-PC design of record; per-regime is queued at 5000 rows on the
LT k=30/45 subset, the only cap where a sparse indicator might be cheap
enough for the split to pay.** Third lesson of the day in the same shape as
the first two: a correlate that explains a penalty is not a fix until the
counterfactual is run.

**What stands regardless of the indicator design**, because it holds under
PC at both caps and under JCI-PC: no LLM arm beats the rule at LT k=6/30 or
WT k=14/21; `one_shot` ≥ loop at LT k=30; `team_varsplit` > `team` at LT
k=30 (+0.078 under JCI); `coverage_min` ≪ `coverage_max` everywhere. What
is estimator-sensitive: everything at LT k=45, `critique` everywhere, the
WT k=21 varsplit confirmation (R+ under PC-1500 and on every skeleton
reading; tie under JCI directed).

---

## THE TWO-CAP CORPUS RE-SCORE (2026-09-10, VPS/OpenBLAS, `runs/rescored-vps-rows300.parquet`, `runs/rescored-vps-rows1500.parquet`): arm contrasts are NOT cap-invariant on directed F1 — they are on the skeleton

The twelve M7 source files (2,206 distinct designs, 9 PC seeds, $0) re-scored
on ONE machine at `--pc-max-rows 300` and `1500` (`rescore.py --pc-max-rows`,
`14ab67c`). Every headline contrast, design-clustered, unequal-n MDE:

| contrast | n | Δ @300 (MDE) | Δ @1500 (MDE) | directed verdict 300 → 1500 | skeleton verdict 300 → 1500 |
|---|---|---|---|---|---|
| LT k=6 `one_shot` − `llm_pc` | 30/30 | -0.047 (0.025) | -0.042 (0.026) | **R−** → **R−** | tie → tie |
| LT k=6 `critique` − `llm_pc` | 30/30 | -0.010 (0.026) | -0.006 (0.028) | tie → tie | tie → tie |
| LT k=6 `shared_blackboard` − `llm_pc` | 30/30 | -0.058 (0.023) | -0.047 (0.027) | **R−** → **R−** | **R−** → **R−** |
| LT k=6 `llm_pc` − `coverage_max_ms` | 30/30 | +0.037 (0.029) | +0.026 (0.032) | **R+** → tie ◀ | tie → tie |
| LT k=6 `coverage_max_ms` − `random` | 30/30 | -0.001 (0.030) | -0.007 (0.037) | tie → tie | tie → tie |
| LT k=6 `coverage_min_ms` − `coverage_max_ms` | 30/30 | -0.014 (0.028) | +0.005 (0.034) | tie → tie | tie → tie |
| LT k=30 `one_shot` − `llm_pc` | 6/70 | +0.002 (0.019) | +0.038 (0.028) | tie → **R+** ◀ | tie → tie |
| LT k=30 `critique` − `llm_pc` | 30/70 | -0.015 (0.014) | +0.022 (0.019) | **R−** → **R+** ◀ | tie → tie |
| LT k=30 `shared_blackboard` − `llm_pc` | 30/70 | -0.020 (0.013) | -0.022 (0.022) | **R−** → tie ◀ | **R−** → **R−** |
| LT k=30 `llm_pc` − `coverage_max_ms` | 70/30 | -0.001 (0.010) | -0.032 (0.014) | tie → **R−** ◀ | tie → tie |
| LT k=30 `coverage_max_ms` − `random` | 30/30 | +0.065 (0.020) | +0.099 (0.027) | **R+** → **R+** | **R+** → **R+** |
| LT k=30 `coverage_min_ms` − `coverage_max_ms` | 30/30 | -0.100 (0.014) | -0.241 (0.017) | **R−** → **R−** | **R−** → **R−** |
| LT k=30 `team_varsplit` − `team` | 30/40 | +0.042 (0.018) | +0.057 (0.019) | **R+** → **R+** | **R+** → **R+** |
| LT k=45 `one_shot` − `llm_pc` | 24/30 | -0.007 (0.016) | -0.002 (0.014) | tie → tie | tie → tie |
| LT k=45 `critique` − `llm_pc` | 30/30 | -0.015 (0.019) | +0.010 (0.013) | tie → tie | tie → tie |
| LT k=45 `shared_blackboard` − `llm_pc` | 30/30 | -0.002 (0.012) | -0.004 (0.010) | tie → tie | tie → tie |
| LT k=45 `llm_pc` − `coverage_max_ms` | 30/30 | -0.001 (0.013) | -0.012 (0.011) | tie → **R−** ◀ | tie → tie |
| LT k=45 `coverage_max_ms` − `random` | 30/30 | +0.017 (0.015) | +0.021 (0.016) | **R+** → **R+** | tie → **R+** ◀ |
| LT k=45 `coverage_min_ms` − `coverage_max_ms` | 30/30 | -0.009 (0.012) | -0.042 (0.012) | tie → **R−** ◀ | tie → **R−** ◀ |
| WT k=7 `one_shot` − `llm_pc` | 32/50 | +0.009 (0.023) | +0.014 (0.026) | tie → tie | tie → tie |
| WT k=7 `critique` − `llm_pc` | 49/50 | +0.001 (0.022) | -0.000 (0.024) | tie → tie | tie → tie |
| WT k=7 `shared_blackboard` − `llm_pc` | 50/50 | +0.001 (0.021) | -0.000 (0.026) | tie → tie | tie → tie |
| WT k=7 `llm_pc` − `wt_coverage_max` | 50/50 | -0.024 (0.020) | -0.026 (0.024) | **R−** → **R−** | tie → tie |
| WT k=7 `wt_coverage_max` − `random` | 50/50 | +0.003 (0.020) | +0.018 (0.023) | tie → tie | tie → tie |
| WT k=7 `wt_coverage_min` − `wt_coverage_max` | 44/50 | -0.067 (0.015) | -0.081 (0.019) | **R−** → **R−** | **R−** → **R−** |
| WT k=14 `one_shot` − `llm_pc` | 34/100 | -0.006 (0.023) | +0.012 (0.020) | tie → tie | tie → tie |
| WT k=14 `critique` − `llm_pc` | 50/100 | -0.023 (0.021) | -0.004 (0.018) | **R−** → tie ◀ | tie → tie |
| WT k=14 `shared_blackboard` − `llm_pc` | 50/100 | -0.005 (0.021) | +0.016 (0.018) | tie → tie | tie → tie |
| WT k=14 `llm_pc` − `wt_coverage_max` | 100/50 | +0.003 (0.019) | -0.014 (0.018) | tie → tie | tie → tie |
| WT k=14 `wt_coverage_max` − `random` | 50/50 | +0.023 (0.018) | +0.028 (0.022) | **R+** → **R+** | **R+** → **R+** |
| WT k=14 `wt_coverage_min` − `wt_coverage_max` | 50/50 | -0.073 (0.019) | -0.043 (0.022) | **R−** → **R−** | **R−** → **R−** |
| WT k=14 `team_varsplit` − `team` | 50/50 | -0.000 (0.023) | +0.016 (0.020) | tie → tie | tie → tie |
| WT k=21 `one_shot` − `llm_pc` | 30/100 | -0.012 (0.025) | -0.002 (0.021) | tie → tie | tie → tie |
| WT k=21 `critique` − `llm_pc` | 50/100 | -0.008 (0.023) | -0.021 (0.017) | tie → **R−** ◀ | tie → tie |
| WT k=21 `shared_blackboard` − `llm_pc` | 50/100 | -0.001 (0.020) | -0.002 (0.018) | tie → tie | tie → tie |
| WT k=21 `llm_pc` − `wt_coverage_max` | 100/50 | -0.026 (0.018) | -0.038 (0.018) | **R−** → **R−** | **R−** → **R−** |
| WT k=21 `wt_coverage_max` − `random` | 50/50 | +0.056 (0.017) | +0.059 (0.018) | **R+** → **R+** | **R+** → **R+** |
| WT k=21 `wt_coverage_min` − `wt_coverage_max` | 50/50 | -0.074 (0.019) | -0.067 (0.018) | **R−** → **R−** | **R−** → **R−** |
| WT k=21 `team_varsplit` − `team` | 124/132 | +0.013 (0.015) | +0.018 (0.013) | tie → **R+** ◀ | **R+** → **R+** |

`R+` = first arm resolved above the second; `R−` below; ◀ = the verdict
changes with the cap. **Directed F1: 10 of 39 verdicts flip. Skeleton F1:
2 of 39**, both at LT k=45 and both toward MORE separation at 1500 rows.

**What the 1500-row estimator says, claim by claim.**

- **The coverage headline gets stronger.** At 300 rows the loop beat the rule
  at LT k=6 (+0.037, resolved) and tied it elsewhere on LT. At 1500 rows the
  LT k=6 win is a tie (+0.026, MDE 0.032), and the RULE beats the loop at
  LT k=30 (−0.032, resolved) and k=45 (−0.012, on the bound), as it already
  did at WT k=7 and k=21. **No LLM arm beats the rule at either cap; at 1500
  rows the rule beats the loop at four of six points.** The rule's escape at
  the smallest budgets (tie with random at LT k=6, WT k=7) is unchanged.
- **The record claim holds and, if anything, reverses.** `one_shot` ties the
  loop at five points at both caps and loses only at LT k=6 (−0.047 / −0.042).
  At LT k=30 and 1500 rows it is resolved ABOVE the loop on directed F1
  (+0.038, MDE 0.028; six distinct designs all at 0.435–0.466 against the
  loop's 0.415) while the skeleton says tie (−0.013). Orientation, not
  selection — say "ties or beats".
- **`critique` is cap-chaotic and must be reported as a tie.** Directed F1:
  LT k=30 goes from resolved-below (−0.015) to resolved-ABOVE (+0.022); WT
  k=14 from resolved-below to tie; WT k=21 from tie to resolved-below. Its
  skeleton verdict is a tie at every point at both caps. Third flip of this
  cell; the standing instruction ("stop re-adjudicating") now has its reason:
  the directed verdict is orientation noise.
- **`shared_blackboard` below the loop at LT k=6 and k=30 on the skeleton at
  both caps** (−0.038/−0.043, −0.016/−0.035, all resolved); directed LT k=30
  sits exactly on its bound at 1500 (−0.022, MDE 0.022). The WT ties hold.
- **`team_varsplit`: LT k=30 resolved at both caps (+0.042 → +0.057).** WT
  k=14 tie at both. **WT k=21 — the n=132 confirmation — resolves at 1500
  rows (+0.018, MDE 0.013) and on the SKELETON at both caps (+0.017 at 300,
  +0.030 at 1500)**, having read +0.013 / MDE 0.015 on the pre-registered
  directed-300 analysis. Report it as: does not clear the house bar on the
  pre-registered analysis; clears it under two of three alternative readings;
  the prediction (+0.0149) is inside every interval. Do not switch bars.
- **Breadth-vs-depth (`coverage_min` − `coverage_max`) widens at 1500** (LT
  k=30 −0.100 → −0.241); the one-per-variable rule's advantage is larger,
  not smaller, when the estimator has power.

**Consequence for register §34's wording.** "Arm contrasts stand because
every arm ran under one estimator" was too strong. Every contrast is FAIR at
either cap, but which ones resolve depends on the cap for a quarter of them
on directed F1 — mostly boundary cases, and mostly in the direction of more
separation at 1500. **On skeleton F1 the verdicts are cap-invariant to two
boundary cases**, consistent with §28's finding that the skeleton has ~1.7×
the signal-to-noise of the directed score. Rule: report every arm contrast at
both caps and on both metrics; a directed flip whose skeleton verdict holds is
orientation, and orientation is where PC's noise lives.

**And the BLAS finding, at the design level, is gone.** The same 2,207
designs re-scored at 300 rows on Accelerate (`rescored-single-backend`, 5 Sep)
and on OpenBLAS (this file) agree on the 9-seed design mean to the digit on
**2,202 of 2,207**; the remaining five differ by at most 0.025; correlation
1.0000; all 39 contrasts give the same delta to three decimals. Register §31's
cell-level divergence (0.38 vs 0.29 on one seed) is real and averages out at
nine seeds. The reproducibility statement can now say: design-level, 9-seed
scores are backend-invariant; single-seed cell scores are not.

---

## WHY STRONG INTERVENTIONS HURT (2026-09-10, local/Accelerate, `runs/mixture-probe-lt.parquet`, `runs/dilution-probe-lt.parquet`): pooled regimes, not the row cap and not the mean shift

Two LLM-free ladders, 9 PC seeds each, built to say WHICH part of the harness
makes a strong light-source experiment score badly. Register §34 had shown
the row cap reorders selections; this asks what the cap is interacting with.

**Ladder 1 — the source's own edges.** For each of `red`, `green`, `blue` at
`mid` and `strong`: PC on the experiment ALONE; on `reference` + experiment
POOLED as the pipeline pools; and on the same pool with every column CENTRED
within its block first (mean shift removed, within-block variation kept).

| strength | mode | rows | recall of source's out-edges | false positives |
|---|---|---|---|---|
| mid | alone | 1500 / all | 0.286 | 14.7 |
| mid | pooled | 1500 / all | 0.280 / 0.238 | 14.9 / 20.0 |
| mid | centred | 1500 / all | 0.275 / 0.238 | 18.3 / 21.8 |
| strong | alone | 1500 / all | 0.286 | 13.3 |
| strong | pooled | 1500 / all | 0.286 / 0.286 | 17.7 / 18.8 |
| strong | centred | 1500 / all | 0.249 / 0.243 | 16.2 / 20.6 |
| any | any | 300 | 0.185–0.243 | 14.6–17.6 |

- Strength does not matter for the source's own edges: `mid` and `strong`
  recover the same fraction, alone or pooled.
- The strong block ALONE is fine (recall equal to mid, fewer false
  positives). The regime is not the problem; mixing regimes is.
- Pooling adds false positives and **centring does not remove them** — so
  in a TWO-block pool the extra edges are not the between-block mean shift.
  (This does NOT generalise to many-block pools: see the JCI-PC probe below,
  where indicator columns DO prevent the large-n collapse. An earlier draft
  of this section said an indicator "cannot absorb them either"; withdrawn
  the same day, by measurement.)
- 300 rows is a power floor in every mode, alone included.

**Ladder 2 — the OTHER inputs' edges.** Base pool `reference`, `green_mid`,
`blue_mid`, `t_ir_1_mid`, `osr_c_mid`; add nothing, `red_mid` or
`red_strong`; read the recall of `green`'s and `blue`'s true out-edges.

| rows | added | recall green | recall blue | false positives | F1 |
|---|---|---|---|---|---|
| all (~6,000) | none | 0.000 | 0.429 | 27.0 | 0.194 |
| all | red_mid | 0.286 | 0.286 | 27.0 | 0.174 |
| all | red_strong | **0.000** | **0.000** | 30.0 | **0.067** |
| 1500 | none | 0.222 | 0.254 | 20.9 | 0.172 |
| 1500 | red_mid | 0.111 | 0.270 | 18.6 | 0.166 |
| 1500 | red_strong | 0.079 | 0.254 | 19.1 | 0.166 |
| 300 | any | 0.03–0.06 | 0.14–0.21 | 16.1–16.4 | 0.133–0.144 |

**The harm grows with rows.** With every row used, the strong block wipes
out both other inputs' edges and F1 falls from 0.194 to 0.067; at 1500 rows
the damage is partial; at 300 it is invisible under the power floor. Variance
dilution under a cap would have gone the other way (recovered with rows).

**What this settles.**

1. The strong-intervention penalty is **not a row-cap artefact** — a larger
   cap makes it worse. This is the mechanism behind §34's non-monotone row
   response: more rows give Fisher-Z more power to find the misfit of one
   linear-Gaussian model to two regimes with different covariance.
2. In a two-block pool it is **not a mean shift** (centring is inert). In
   the 30–45-block pools the arms actually buy, **one context indicator per
   intervened variable removes the large-n collapse** (JCI-PC probe, next
   section): PC at 5000 rows falls to 0.273 / 0.255 at LT k=30 / 45 while
   JCI-PC holds 0.376 / 0.418 and its skeleton keeps rising with rows.
3. It is **not the strong regime in particular**: alone, at 1,000 rows, the
   strong experiment is as recoverable as the mid one. **But it is partly
   the chamber's data meeting a linear-Gaussian test, even in ONE regime**
   (measured 2026-09-10 evening): PC on the reference run ALONE scores
   core-20 F1 0.177 / 0.205 / 0.186 / **0.114** at 300 / 1,000 / 3,000 /
   10,000 rows, false positives 15 → 27. The inputs are uniform and the
   sensor response is not exactly linear, so a partial-correlation test with
   enough power reads nonlinear residual dependence as edges. Pooling
   regimes AMPLIFIES that misfit (ladder 2's collapse is far sharper than
   the single-regime decline); it does not create it alone. UT-IGSP shows
   the same signature on its observational sample (core 0.49 at 10,000 rows,
   0.63 at 1,000), so the effect is the test family, not PC's search.
4. It IS a property of **pooled-regime estimation that ignores the regime**
   — the reduction this harness applies. The models' "strong root
   interventions give signal-to-noise" prior is right about the regime and
   wrong about the estimator. Whether an estimator that KNOWS the regime
   (JCI-PC at a large cap, or UT-IGSP/GIES which never pool) ranks strong
   interventions differently is now a runnable question, not a deferred one.

**Rules.** State the anti-prior as an estimator-relative finding. Never quote
a "more rows is better" default for ANY Gaussian-test estimator on chamber
data, pooled or not; sweep it.
When a penalty for adding data appears, test alone / pooled / centred before
naming a mechanism — the first two explanations offered here (saturation,
then mean-shift mixture) were both wrong and both plausible.

---

## JCI-PC PROBE (2026-09-10, local/Accelerate, `runs/jci-probe.parquet`, `runs/jci-probe-rows5000.parquet`): a second estimator that knows the regime

`jci.py` + `rescore.py --estimator jci_pc`: the plain PC on the pooled table
widened by one 0/1 column per intervened variable, with every edge INTO an
indicator forbidden (Joint Causal Inference, Mooij, Magliacane & Claassen,
JMLR 2020, assumption 0 only — assumption 3, declaring the mutually exclusive
indicators adjacent, is unenforceable in causal-learn's skeleton phase and
was removed after its own test showed it inert). Indicators are stripped
before scoring. Apparatus indicators are dropped as collinear with the
setting they mark unless the setting was bought at two strengths, so the
surviving indicators are mostly the sampled light-source inputs.

**84 corpus designs (14 per chamber × budget), 3 seeds, JCI-PC minus PC:**

| chamber, k | 300 rows | 1500 rows | 5000 rows (LT only) |
|---|---|---|---|
| LT 6 | +0.009 | +0.019 | +0.018 |
| LT 30 | −0.021 | **+0.032** | **+0.103** |
| LT 45 | −0.128 | −0.044 | **+0.163** |
| WT 7 | +0.016 | +0.019 | — |
| WT 14 | −0.002 | +0.010 | — |
| WT 21 | −0.026 | +0.010 | — |

98% of design × seed cells differ; mean |Δ| 0.053. Three mechanisms, each
measured:

1. **Orientation, not skeleton, at 1500 rows.** LT k=30: directed +0.032,
   skeleton +0.005. A forced C→X edge lets Meek's rule 1 orient X—Y whenever
   C is not adjacent to Y, and the scorer charges an unoriented edge as
   TP+FP. Core-20 F1 moves +0.017.
2. **An indicator penalty at small caps.** Δ correlates −0.58 (300 rows) /
   −0.48 (1500) with the number of range-shifting buys: each indicator is a
   sparse binary node (≈10 ones at 300 rows) that adds tests and forks. This
   is CONFOUNDED WITH VARIABLE COVERAGE — the coverage rule buys the most
   indicators — so JCI-PC at 300 rows must never adjudicate the rule.
3. **Monotone in rows where PC is not.** LT, same 42 designs:

| k | rows | PC | JCI-PC | PC skeleton | JCI-PC skeleton |
|---|---|---|---|---|---|
| 30 | 300 / 1500 / 5000 | 0.387 / 0.379 / **0.273** | 0.366 / 0.410 / 0.376 | 0.447 / 0.481 / 0.391 | 0.397 / 0.486 / 0.469 |
| 45 | 300 / 1500 / 5000 | 0.413 / 0.432 / **0.255** | 0.285 / 0.388 / **0.418** | 0.460 / 0.529 / 0.357 | 0.315 / 0.452 / **0.516** |

   PC's collapse past 1500 rows (register §34) is the pooled-regime misfit of
   the previous section given enough power to find it; the indicators block
   it. JCI-PC's natural cap is therefore LARGE, the reverse of PC's.

**How to read the corpus run** (queued on the VPS at 1500 rows; pre-registered
2026-09-10 before any file came back): (1) every resolved arm-vs-arm verdict
reproduces, because the orientation gain is shared by all arms; (2) the
coverage rule loses ground in proportion to distinct variables bought — the
indicator penalty, a harness signature, not a finding about coverage; (3)
skeleton verdicts move less than directed ones; a directed flip whose
skeleton verdict holds is orientation. Selection-level claims (oracle
headroom, the anti-prior) are read under JCI-PC ONLY at a cap where its
indicator penalty is gone, i.e. ≥1500 rows, and reported beside PC's.

---

## THE 1500-ROW ORACLE (2026-09-10, VPS/OpenBLAS, `runs/oracle-probe-rows1500-lt-*`): headroom at the ends, a tie in the middle

Re-derivation of the LT oracle with `--pc-max-rows 1500` (register §34),
same procedure (greedy to k=45 at search seeds 0–4, one-swap gated on the
noise floor, static ranking over 20 contexts, everything re-scored at
seeds 100–108). **Compare only within this file** — it is OpenBLAS; the
300-row probe and the corpus are Accelerate.

| k | coverage rule | random | ranking top-k | greedy(+swap) set, fresh | ranking − rule [MDE] | set − rule [MDE] |
|---|---|---|---|---|---|---|
| 6 | 0.152 ± 0.041 | 0.181 | 0.248 | **0.338** | **+0.096 [0.038] R** | **+0.185 [0.090] R** |
| 30 | 0.447 ± 0.015 | 0.324 | 0.414 | 0.475 | **−0.033 [0.018] R** (ranking is WORSE) | +0.028 [0.032] below MDE |
| 45 | 0.429 ± 0.013 | 0.411 | 0.432 (pooled 0.452) | **0.497** | +0.003 / **+0.023 [0.017] R** | **+0.068 [0.029] R** |

**Verdict.** "Coverage is a plateau, not the ceiling" **survives at k=6
and k=45 and fails at k=30** — the middle budget, where every headline
contrast of the pillar lives. At k=30 the best set found sits +0.028 above
the rule with an MDE of 0.032 (one design vs ten rule draws), and the
static ranking is resolved *below* the rule. The 300-row picture
(headroom at all three budgets, +0.046 by ranking at k=30) does not
replicate at the middle budget. State it as: **at a well-set estimator the
coverage rule is within noise of the best known selection at the middle
budget, and beatable at the small and large ones.**

**What the 1500-row oracle buys** — and what it does not. The sensor-setting
depth regime is gone: the k=30 set covers **22 distinct variables** (rule:
30; 300-row oracle: 15), families t 10 / diode 6 / v 5 / osr 3 / l 2 /
pol 2 / green 1 / reference, 19 of 30 at mid strength and 6 weak. The
residual advantage at k=45 is coverage plus a preference for apparatus
entries at mid strength and an aversion to `osr_*` and `red_*` (per-family
mean gain at k=30: reference +61, diode +8, v +6, t −1, osr −6, red −11
×10⁻³). `rule_no_strong_sources` — the coverage rule minus strong
light-source entries — captures +0.025 of the +0.068 at k=45 (resolved),
and `rule_mid_only` +0.060 of +0.185 at k=6 (resolved); at k=30 neither
moves. **Also new at 1500 rows: the coverage rule at k=6 (0.152) is BELOW
random (0.181)** — round-robin over 30 variables with 6 picks buys six
singletons, and at this cap the reference run alone is worth more.

**Arm purchases on this scale.** Only informative at k=6, where marginal
gains are large: `llm_pc` 4.7 and `critique` 4.6 ×10⁻³ vs random −0.8 (sd
≈ 5, n=30, MDE ≈ 3.6) — **the loop's purchases ARE better than random at
the small budget on the 1500-row scale**, consistent with its resolved
k=6 win over the rule at 300 rows. At k=30 and k=45 the per-experiment
gains are within ±1×10⁻³ of zero for every arm (sd 0.7): the scale is
flat and says nothing.

**Winner's curse is small here** (search 0.500 → fresh 0.475 at k=30;
0.506 → 0.497 at k=45), unlike the 300-row run (0.474 → 0.432), which is
itself a sign the 300-row search was optimising noise.

**Status of the 2026-09-09 claims after this run.**

| claim | 300 rows | 1500 rows | paper |
|---|---|---|---|
| headroom above the rule at every budget | yes (5/6 contrasts) | k=6 and k=45 yes; k=30 tie | "beatable at the ends, within noise in the middle" |
| the residual is sensor-setting depth | yes (15 vars of 30) | no (22 of 30, coverage-like) | withdrawn |
| the models' prior points the wrong way | ρ −0.4 | not re-run; their "one strong per target" ≈ rule ≈ best at k=30 | withdrawn as stated |
| feedback arm beats the loop | +0.027 R | +0.045 R (local re-score) | stands, strengthened |
| no LLM arm beats the rule | yes | yes (loop −0.055 R at k=30, local re-score) | stands |

---

## WHAT THE HEADROOM IS (2026-09-09, evening): a one-line rule claims two-thirds of it, data-only learners claim none, and the model's prior points the wrong way

> **CONDITIONAL ON `pc_max_rows=300` — see register §34 (found the same
> night).** At 1500 rows the coverage rule scores 0.460 and the 300-row
> oracle ranking 0.394, the sensor-setting rule 0.386. Everything in this
> section and the oracle-probe section below is a statement about the
> estimator at its 300-row cap, not about the chamber. The arm contrasts
> stand (adaptive − loop is +0.045 at 1500 rows, resolved). A 1500-row
> oracle is being derived on the VPS; until it lands, the coverage rule is
> the best policy we hold and "plateau, not ceiling" is withdrawn as a
> task property.


Three LLM-free probes and one $0.37 LLM probe, all LT, all scored at PC
seeds 100–108 like the oracle. Files: `runs/learners-lt.parquet`,
`runs/depth-rule-lt.parquet`, `runs/llm-prior-lt.parquet` (+`-2`).
Scripts in the session scratchpad (`learners.py`, `depth_rule.py`,
`llm_prior.py`); worth promoting to the pipeline if the paper quotes them.

**1. Data-only learners do not climb past the rule.** Every learner sees
only the data it bought. LT k=30, n=10 rule seeds:

| learner | F1 | vs rule 0.437 [MDE] | purchases on oracle scale (×10⁻³) |
|---|---|---|---|
| uncertainty sampling (bootstrap edge instability, B=8) | 0.267 ± 0.021 | **−0.170 [0.023] RESOLVED** | 5.9 |
| uncertainty inside coverage | 0.434 ± 0.014 | −0.003 [0.019] | 9.7 |
| credit-assignment bandit over (family, strength) | 0.382 ± 0.044 | **−0.056 [0.041] RESOLVED** | 9.2 |
| perfect family-level prior, one entry per variable (uses ground truth) | 0.434 ± 0.019 | −0.003 [0.022] | 10.0 |

Same picture at k=6 and k=45 (all learners ≤ rule; uncertainty −0.025
resolved at k=45). Pure uncertainty sampling fails for an instructive
reason: instability rewards the experiments that CREATE spurious edges, so
it buys every strong light-source entry (5.0 of 5 possible). And a perfect
family ordering that still covers every variable ties the rule — so the
headroom is neither a better estimate nor a better family order.

**2. The headroom is a different regime.** The oracle ranking's top-30
covers only **15 distinct variables**: 27 of 30 picks are sensor-setting
experiments (`t_*` ×18, `diode_*` ×9), the same variable bought at two or
three strengths. Testing that as a rule, n=10, k=30:

| rule | F1 | vs coverage rule [MDE] | rule→oracle | purchases (×10⁻³) |
|---|---|---|---|---|
| all 27 sensor-setting entries + 3 other apparatus | 0.462 ± 0.011 | +0.024 [0.017] RESOLVED | 53% | 17.6 |
| 30 random from sensor-setting + `v_*` + reference (34) | 0.464 ± 0.010 | +0.027 [0.017] RESOLVED | 59% | 17.1 |
| **sensor-setting mid/strong first (24), fill from the 34** | **0.468 ± 0.018** | **+0.031 [0.021] RESOLVED** | **67%** | 17.5 |
| 30 random from ALL apparatus incl. `osr_*` (48) | 0.425 ± 0.022 | −0.013 [0.024] | −28% | 14.0 |

**This corrects the oracle-probe section's "a one-line rule recovers some
of it at the ends and none in the middle"** — the rules tried there were
all coverage-first. A rule that ABANDONS coverage of the light sources,
polarisers, LED currents and `osr_*` and spends the freed budget on repeat
purchases of the sensor settings is one line long and gets two-thirds of
the way to the oracle, with purchases scoring 17.5 against the oracle's
own 17.8.

**3. Why, mechanically (measured on the data, not argued).** In EVERY LT
experiment the same 20 columns vary: `red`, `green`, `blue`, `current`,
`pol_1`, `pol_2`, `angle_1`, `angle_2`, the six sensors and the six `l_*`
— whatever the experiment is named. The intervened apparatus setting does
NOT vary inside its own experiment (`t_ir_2` is the constant 1.0 in
`uniform_t_ir_2_mid` and the constant 3.0 in `uniform_reference`); it
varies only ACROSS the pool, as a two-level contrast, once its experiment
is bought. Consequences: (a) a light-source experiment adds no new
variation in its own variable — it shifts the range (`red` 171–255 in
`red_strong` vs 0–85 at reference). **The sensors do NOT saturate under
it** (measured: sensor sd and corr(red, ir_1) are 0.78 in `red_strong` vs
0.77 at reference; no readings pinned at a ceiling), so the earlier
"pushes sensors off the linear regime" was an assumption and is
RETRACTED. What is measured instead: the 27 sensor-setting entries + 2
`v_*` score **0.510** at 9 fresh seeds — above the oracle ranking's
top-30 (0.483) — and adding ANY 30th experiment lowers it (`+red_strong`
0.440, `+red_mid` 0.455, `+green_strong` 0.460, `+l_11_mid` 0.456,
`+reference` 0.467); `red_strong` hurts most by LOSING input→sensor edges
(9.8 → 6.7 found of the 11 input sources' edges) with 4 more false
positives, not by spurious sensor–sensor edges (3.0 → 4.1). Whether that
is a property of pooling shifted regimes or of PC's 300-row subsample
(`pc_max_rows`) is being tested; if the latter, part of the oracle
headroom is a harness setting and belongs in the register;
(b) the 18 apparatus settings are the only purchases that add information,
each as one two-level contrast, so one entry per setting is not enough and
depth pays; (c) "breadth beats depth" (WT, `wt_coverage_min`) is a
statement about the fat drivers there, not a law — on LT depth on the
sensor settings beats breadth. `osr_*` is the exception among apparatus
settings (family gain +2.8 vs `t` +17.3, `diode` +20.1); including it
costs 0.04.

**4. The models' prior points the wrong way — and it is not a capability
gap.** Asked with no data to rank all 59 entries by informativeness (names
only, and with a one-paragraph chamber description), reasoning effort high,
five draws per condition, `runs/llm-prior-lt-all.parquet`, $0.87 total:

| model | condition | n | Spearman vs oracle (min…max) | top-30 purchases (×10⁻³; random 9.7, oracle 17.8) | top-30 F1 (rule 0.437) |
|---|---|---|---|---|---|
| `deepseek-v4-flash-0731` | names only | 4 | **−0.37** (−0.50…−0.12) | 5.7 | **0.323** (below random 0.369) |
| | + description | 5 | −0.22 (−0.26…−0.14) | 9.1 | 0.438 |
| `glm-5.3-flash` | names only | 5 | +0.07 (−0.02…+0.15) | 9.3 | 0.421 |
| | + description | 5 | −0.15 (−0.27…+0.02) | 9.5 | 0.433 |
| `gpt-5.6-sol` (frontier) | names only | 5 | **−0.28** (−0.37…−0.24) | 9.4 | 0.440 |
| | + description | 5 | −0.26 (−0.36…−0.04) | 9.4 | 0.437 |

Every one of 29 parsed draws across three models and two vendors puts the
five strong light-source and polariser interventions in its top 30, and
all give the same reason: "strong interventions on the roots give the
largest signal-to-noise for Fisher-Z". The oracle ranks those five
55th–59th of 59. The frontier model is not better — it is more
consistently wrong (sd 0.06) — and lands its purchases at exactly random's
level because it executes coverage tidily ("the reference plus one strong
intervention per target"). GLM is the only model near zero, for the same
reason. DeepSeek's four empty-content draws (of 14) are the register §8
failure mode and are excluded. **The LLM arms do not merely lack the
relevant knowledge; they carry its opposite, at every model scale we can
buy** — which is why `team` scores resolved BELOW random on the oracle
scale and why no LLM arm's purchases beat a random draw on LT. The fix is
therefore not a better model; it is documentation or learning (item 5).

**4b. Documentation works for the frontier model and not for the flash
model.** The same ranking prompt with a paragraph quoted from the dataset
README (all manipulable inputs sampled independently in every experiment;
mid/strong shift one input's range; apparatus-setting experiments fix one
setting) — documentation, not the answer:

| model | n parsed | Spearman (min…max) | purchases (×10⁻³) | top-30 F1 |
|---|---|---|---|---|
| `deepseek-v4-flash-0731` | 2 of 5 (3 empty) | −0.15 (−0.19…−0.12) | 9.4 | 0.446 |
| `gpt-5.6-sol` | 5 of 5 | **+0.32** (+0.06…+0.68) | 11.9 (two draws at 15.4) | 0.446 ± 0.021 |

sol's rationale flips to the correct one — "fixed-setting interventions
rank highest because those variables otherwise have no variance" — in 3
of 5 draws (those three buy zero strong light-source entries); the other
two revert to coverage. DeepSeek, given the identical text, restates its
belief in the document's own words ("larger distributional shifts, more
detectable dependencies"). **So no model HAS the knowledge, and only the
frontier model can DERIVE it from the manual, and not reliably.** Files
`runs/llm-prior-lt-protocol-*.parquet`, $0.59.

**5. What this does to the learnability question.** The signal that
separates the oracle from the rule — which columns vary within an
experiment and which only across the pool — is visible in the bought data
after ONE purchase. It is learnable in principle and cheaply. The
adaptive-feedback arm never saw it: its summary was coverage-shaped (edges
found; variables unreached). And a prior that must be overridden makes the
feedback's job harder, not easier. The next feedback design should tell the
model exactly this (per experiment: what varied, what did not, whether the
range moved) and nothing about coverage — a pre-registrable prediction that
it clears the rule. Not required for the AAMAS submission.

**Also corrects** oracle-probe item 5 below ("A one-line rule built from
that recovers some of it at the ends and none in the middle") — true of
coverage-first rules only.

---

## THE ADAPTIVE-FEEDBACK ARM (2026-09-09): the headroom is partly learnable without ground truth — and the effect is small

The oracle probe (next section) left one question that decides how the
paper reads: is the headroom above the coverage plateau *learnable* by an
agent that never sees the ground truth, or is it visible only to an oracle
that does? `adaptive_feedback` is the loop (`llm_pc`) with one change: every
five purchases it runs PC on the data bought so far and puts a menu-keyed
summary of the current estimate in the selection prompt (edges found so far;
which menu entries perturb variables the estimate has not yet connected).
Same budget, same call count, same contract. Pre-registered in spec §8.7
row 7 before launch (commit `33dcc6d`): LT k=30, n=30, interleaved with a
fresh `llm_pc` control in the same sweep so provider drift cannot land on one
arm. `runs/m7-adaptive-lt.parquet`, 60/60 ok, $6.15, 1.2 h on six workers,
`deepseek-v4-flash-0731`, macOS/Accelerate (matching the re-scored corpus).

**The two pre-registered predictions split**, re-scored at 9 PC seeds and
clustered by distinct design (30 distinct designs for the adaptive arm, 29
for the loop):

| prediction | measured | MDE | verdict |
|---|---|---|---|
| P1: purchases above random on the oracle marginal-gain scale (random 9.7×10⁻³) | adaptive **10.6×10⁻³**, Δ **+0.81×10⁻³** | 0.79×10⁻³ | **RESOLVED**, on the boundary (ratio 1.02; Welch p=0.0065). Same-sweep loop: 9.5×10⁻³, Δ −0.26, ns. |
| P2: F1 above the coverage rule (0.437 at fresh seeds) | adaptive **0.445**, Δ **+0.008** | 0.017 | **below MDE — not supported** |

Against the same-sweep loop the arm resolves cleanly: **+0.027 F1
(0.445 vs 0.418), MDE 0.013, 2.1× the bound**. Core-20 moves the same way
but does not resolve (+0.013, MDE 0.015; 0.236 vs 0.223, rule 0.228).

**How to read it.** The arm did what the oracle said an agent could do
and the coverage plateau said it could not: its purchases score above
random on the oracle's own scale, the first LLM arm on LT to do so, and it
beats the loop it is built from. But it covers **10% of the random→oracle
range** on the purchase scale and **17% of the rule→oracle headroom** in F1
(0.437 → 0.483). It reaches the plateau; it does not climb past it. The
paper's sentence is therefore: *feedback from the data is the first thing
that moves an LLM arm off the random line on the oracle scale, and the
first single-loop change that resolves above the loop at the middle
budget — and it is still not enough to beat a ten-line coverage rule.* The
oracle headroom is real and partly learnable, and most of it is still
unclaimed.

**Costs and hygiene.** The arm is *cheaper* than the loop per cell
(336 s vs 395 s; 68k vs 78k output tokens) despite a 53% longer prompt
(28k vs 18k input) — the model reasons less when told what it already
knows. Two things to carry into the write-up:

- **Selection fallbacks: 16 of 900 picks (1.8%) in 10 of 30 adaptive cells
  went to `rng.choice`; the loop had 0.** The longer prompt occasionally
  yields an unparsable reply. This biases the arm *toward* random, so it
  understates rather than inflates the effect; cells with a fallback scored
  0.455 vs 0.445 without (n=10 vs 20, noise). Report it; do not correct
  for it.
- **Drift audit CLEAN** (`analyze_drift.py`): window overlap 0.99, no block
  trending, tokens per call flat across the sweep for both arms.

**Not yet done, and the order to do it in:** WT replication (the only
chamber where an existing arm, the loop, already beats random on the oracle
scale — if feedback stacks on top of that, the learnability claim gets a
second chamber) — **RUN 2026-09-12: it does not stack, it reverses; see
"THE ADAPTIVE-FEEDBACK ARM ON THE WIND TUNNEL" near the top of this doc.
This LT result is one chamber, one budget;** the small and large LT budgets (the middle budget is where
skill peaks, so k=30 is the friendliest test, and a k=6 result would say
whether feedback helps where selection variance is largest); and a
feedback-interval ablation (5 was a guess). None of these is required for
the AAMAS submission; the k=30 result is the one the abstract needs.

---

## THE ORACLE PROBE (2026-09-09, regenerated after review): the task is NOT solved by coverage — headroom exists at every budget and no arm reached it

> **WITHDRAWN AS A TASK PROPERTY THE SAME NIGHT — register §34.** These
> oracles were derived and scored at `pc_max_rows=300`; at 1500 rows the LT
> k=30 ranking falls 0.066 BELOW the coverage rule. The tables stand as
> "best selection for the estimator at 300 rows". Re-derivation at 1500
> rows is in progress (`runs/oracle-probe-rows1500-*`, VPS/OpenBLAS).


The top-ranked threat was "the task is coverage-shaped, so a coverage rule
tying every LLM arm is a benchmark artefact". It was being *scoped*, not
measured. `oracle_probe.py` measures it, LLM-free and for $0, with the ground
truth as an oracle: forward-greedy selection on true mean F1 (a lower bound
on the best reachable set; a one-swap search gated on the measured noise
floor accepted nothing that survived re-scoring), and a **static ranking** of
experiments by mean marginal gain over 20 random contexts. Search used PC
seeds 0–4; **every reported figure is re-scored at disjoint seeds 100–108**,
enforced by the tool, so the oracle numbers are unbiased for the sets they
name. Same machine and BLAS backend (Accelerate, stamped on every row) as
`rescored-single-backend.parquet`, from which the LLM arms are quoted (seeds
0–8; the coverage rule scored at both seed sets differs by ≤0.013).

`python -m evaluation.chamber_pipeline.oracle_probe --chamber {lt,wt}` emits
`runs/oracle-probe-{lt,wt}-{greedy,context,policies,arm-gain}.parquet`;
everything below reads from those four files.

| | oracle set (n=1) | oracle ranking (n=10) | coverage rule (n=10) | best LLM arm (n=30–150) | random (n=10) |
|---|---|---|---|---|---|
| LT k=6 | **0.249** | 0.239 ± 0.009 | 0.176 ± 0.041 | 0.207 ± 0.035 (loop) | 0.171 ± 0.035 |
| LT k=30 | 0.437 | **0.483 ± 0.014** | 0.437 ± 0.016 | 0.432 ± 0.009 (`one_shot`) | 0.369 ± 0.036 |
| LT k=45 | **0.450** | 0.449 ± 0.015 | 0.429 ± 0.014 | 0.420 ± 0.018 (loop) | 0.398 ± 0.019 |
| WT k=7 | **0.331** | 0.249 ± 0.012 † | 0.214 ± 0.019 | 0.177 ± 0.026 (`one_shot`) | 0.186 ± 0.035 |
| WT k=14 | **0.341** | 0.327 ± 0.008 † | 0.231 ± 0.033 | 0.245 ± 0.041 (loop) | 0.196 ± 0.036 |
| WT k=21 | **0.391** | 0.337 ± 0.014 | 0.291 ± 0.018 | 0.266 ± 0.043 (loop) | 0.219 ± 0.038 |

± is sd over rule seeds / shuffles / cells. † the ranking pooled over all
budgets; at WT k=7/14 it beats the at-budget ranking (0.201 / 0.293), the
reverse on LT. The oracle SET is a single design; its uncertainty is the
9-seed inference noise, sd ≈ 0.013, so every set-vs-arm gap below is >4 sd.

**1. Headroom exists at every budget on both chambers, and it resolves.**
Oracle *ranking* minus best LLM arm, unequal-n MDE in brackets:

| | LT k=6 | LT k=30 | LT k=45 | WT k=7 | WT k=14 | WT k=21 |
|---|---|---|---|---|---|---|
| ranking − best arm | **+0.033** [0.032] | **+0.051** [0.010] | **+0.029** [0.018] | +0.024 [0.024] | **+0.047** [0.036] | **+0.071** [0.038] |
| ranking − coverage rule | **+0.063** [0.038] | **+0.046** [0.019] | **+0.020** [0.018] | −0.012 [0.019] | **+0.061** [0.031] | **+0.046** [0.020] |
| oracle SET − best arm | +0.042 | +0.005 | +0.030 | **+0.154** | **+0.096** | **+0.125** |

Five of six ranking contrasts resolve (LT k=6 and WT k=7 sit exactly on
their MDE). The oracle set adds nothing over the ranking on LT but is far
above it on WT at k=7 and k=14. The coverage rule is not the ceiling; it is
the plateau every policy we built converged to. On WT an oracle set of **7**
experiments (0.331) beats every arm at **21** (0.291).

**2. On LT the headroom is in the core subgraph, not the apparatus edges.**
Core-20 F1, oracle ranking vs rule vs loop: k=6 **0.201** / 0.174 / 0.176;
k=30 **0.284** / 0.228 / 0.226; k=45 **0.251** / 0.233 / 0.226. The gain is as
large on the 20 case-study variables as on the full graph. This retires the
§28 reading that the LLM's only possible edge over a rule lives in "did you
buy the experiment that makes this setting vary".

**3. The headroom is mostly a RANKING on LT and partly a SET on WT.** The
within-candidate sd of marginal gain across random contexts is at the
measured noise floor on LT (0.028/0.027/0.025 vs noise 0.023/0.030/0.031)
and on WT k=14 (0.032 vs 0.032) — context dependence is not separable from
PC noise there — but above it at WT k=7 and k=21 (0.040 / 0.042 vs 0.025 /
0.032). Consistent with that, the static top-k reaches or beats the greedy
set on LT at k=30/45 and 74–96% of it on WT, with the residual at WT k=7
(0.249 vs 0.331) where sets matter. Practical consequence: on LT the missing
knowledge is *which experiments are informative*, a per-experiment property.

**4. No arm reached it; on LT none moved.** Each arm's actual purchases
scored on the oracle's marginal-gain scale (mean gain of the experiments
bought, ×10⁻³; unequal-n MDE vs random in brackets; **R** = resolved):

| | LT k=6 | LT k=30 | LT k=45 | WT k=7 | WT k=14 | WT k=21 |
|---|---|---|---|---|---|---|
| oracle top-k | 23.1 | 17.8 | 14.1 | 11.2 | 7.6 | 4.7 |
| random | 9.0 | 9.7 | 9.9 | 1.1 | 1.0 | 1.0 |
| loop | 10.9 [3.7] | 9.6 [0.6] | 9.9 [0.5] | 1.6 [1.4] | **2.8 [0.7] R** | **1.9 [0.4] R** |
| `one_shot` | 7.1 [3.1] | 9.6 [0.7] | 9.6 [0.6] | 2.4 [1.4] | **2.2 [0.8] R** | 0.9 [0.6] |
| `critique` | 10.0 [2.9] | 9.9 [0.8] | 9.8 [0.7] | 1.5 [1.6] | 1.2 [0.9] | 1.0 [0.5] |
| `shared_blackboard` | 6.0 [3.1] | 9.3 [0.9] | 9.3 [0.4] R↓ | 1.8 [1.5] | 1.8 [0.9] | 1.4 [0.5] |
| `team` | — | **8.4 [0.7] R↓** | — | — | 1.4 [0.7] | 1.1 [0.4] |
| `team_varsplit` | — | 9.4 [0.8] | — | — | 1.7 [0.8] | **1.5 [0.4] R** |
| coverage rule | 9.3 [3.4] | 9.7 [0.7] | 9.4 [0.5] R↓ | 1.8 [1.7] | 1.6 [0.8] | **1.6 [0.4] R** |

On LT **no LLM arm is above random at any budget** (`team` is resolved
below it). On WT the loop IS resolved above random at k=14 and k=21 — but it
covers **28% and 27% of the random→oracle range**, and 5% at k=7. The
multi-agent arms are never above the loop on this scale. The topologies did
not differ because none of them had much to coordinate: **coordination among
agents that all lack the relevant knowledge cannot create it**, and the one
arm that found a little of it was the single loop.

**5. Is the ranking describable?** Partly, on LT: apparatus-setting
experiments gain +0.014 at any strength; source experiments gain +0.002 at
mid and **hurt (−0.009) at strong** — ~~strong interventions on the light
sources push sensors off the linear regime Fisher-Z assumes~~ (retracted
the same evening: no saturation is measurable; see "WHAT THE HEADROOM IS"
item 3 for what the data show). A one-line
**coverage-first** rule built from that recovers some of it at the ends and none
in the middle (**superseded the same evening — a rule that abandons coverage
and repeats the sensor settings gets 67% at k=30; see "WHAT THE HEADROOM IS"**):
mid-only coverage **+0.028** over the rule at k=6 (0.205 vs 0.176), nothing
at k=30; excluding strong-source experiments **+0.015** at k=45 (0.444 vs
0.429), nothing elsewhere. The ranking's power at k=30 is the sum of many
small per-experiment differences (sd 0.010 across the menu), not one
feature. On WT the per-experiment means are within noise of each other
(sd 0.009) and the set effect dominates at small k. Whether an agent can
learn the ranking WITHOUT ground truth — from the adjacency it has recovered
so far — is exactly the adaptive-feedback arm (spec §8.7 row 7), which this
result promotes from optional to the most informative next experiment.

**What this does to the paper.** The scoping sentence for threat 6 is
withdrawn: the task is not coverage-shaped in the sense that matters.
Coverage is where uninformed selection saturates; a ground-truth ranking
sits 0.02–0.07 above it (resolved at 5 of 6 budgets) and a ground-truth set
0.10–0.15 above every arm on WT. The negative topology result is therefore
stronger, not weaker — room existed, and no topology found it — and every
arm now has a distance-from-optimum: **LT arms at 83–96% of the oracle
ranking, WT arms at 52–75% of the oracle set.** Report the oracle curve
beside every results table.

**Caveats.** (a) The oracle uses the ground truth; it bounds what *any*
policy could reach, not what an agent could learn. (b) Greedy is a lower
bound on the optimum; the true ceiling is at least this high. (c) The
ranking was learned from 20 contexts × 5 seeds; noise in learning can only
make it worse, never inflate the fresh-seed evaluation. (d) A ranking
learned at one budget transfers imperfectly (pooled is 0.015 lower on LT and
higher on WT at small k). (e) The oracle set is one design, n=1; its sd is
inference noise only. (f) One BLAS backend, as everything else here.

**Lessons that went into `claude.md`.** An oracle's own search score is a
max over noisy candidates: greedy+swap read 0.474 in-search at LT k=30 and
0.432 fresh, and a swap accepted at 1e-9 over 5-seed means was a max over
noise (the regenerated tool gates on the measured floor and then accepts
almost nothing). Greedy is a weak optimiser under PC noise: a static ranking
beat the greedy set at LT k=30 by +0.046.

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
record only pays at LT k=6, where it is worth +0.059. **(Did not replicate
2026-09-11: same-regime re-run with a fresh loop control gives −0.006
[0.024] over 57 single-call designs — see "MENU SHUFFLE, AND THE LT k=6
SINGLE-CALL LOSS THAT DID NOT REPLICATE". The record pays nowhere we have
measured.)**

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
| `one_shot` < loop | **FALSE at 5 of 6 budgets** (held only at LT k=6, −0.047 — and that did not replicate same-regime on 2026-09-11: −0.006 [0.024]) |
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
that way rather than as a clean effect. **Superseded 2026-09-11: the
`one_shot` k=6 loss did not replicate same-regime (−0.006 [0.024], n=57 vs
30); `shared_blackboard`'s was not re-run.**

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
the confound and too small to weigh.) `one_shot`'s −0.059 at k=6 is not a
collinearity artifact — but it is not a stable record effect either: it did
not replicate same-regime on 2026-09-11 (−0.006 [0.024]).

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

## SINGLE-BACKEND RE-SCORE (2026-09-05): every headline verdict holds

`runs/rescored-single-backend.parquet` — the twelve M7 source files re-scored
together on one machine at 9 PC seeds: **2,604 cells, 19,854 design x seed
scorings, `rescore_blas_backend` uniformly `accelerate`**, folding in sources
that were originally `accelerate` AND `scipy-openblas`. Cost $0.

Register §31 found five of six coverage-oracle contrasts crossing BLAS
backends, and two verdicts moved when they were re-scored on one machine. The
open question since was whether the rest of the corpus hid more of the same.
**It does not.**

- **The coverage-oracle table (six contrasts): all six verdicts reproduce.**
  LT k=6 the loop wins (+0.036, MDE 0.030); WT k=21 the rule wins (−0.026,
  MDE 0.024, previously quoted −0.030); the other four tie.
- **Phase 2 (eighteen contrasts): seventeen reproduce exactly**, including
  every `one_shot` tie that carries the "record is not load-bearing" claim.
- **`team_varsplit` reproduces**: LT k=30 +0.041 resolved (was +0.043); WT
  k=14 +0.000; WT k=21 +0.014 below MDE — consistent with the confirmation's
  2.53σ not clearing the pillar's bar.

### The one contrast that is not stably resolvable: `critique` at LT k=30

It lands on either side of its own MDE depending on an analysis choice that
has nothing to do with the backend:

| basis for `llm_pc` | n | delta | MDE | verdict |
|---|---|---|---|---|
| reference file only (`m7-p2-ref`) | 30 | −0.013 | 0.020 | below MDE |
| pooled over three files | 68 | −0.015 | 0.014 | RESOLVED |

Pooling is defensible — the three sources' means are 0.4211 / 0.4230 / 0.4259,
agreeing far inside the MDE, which is the check CLAUDE.md requires before
pooling across a possible regime change. It buys real power. But a verdict that
turns on it is not a finding.

**This is the SECOND flip for this exact cell** — it was reported resolved-worse
on 31 Aug, retracted on 1 Sep after design-level re-scoring, and now sits on
the boundary again. Report `critique` as **tying the loop with the LT k=30
contrast on its MDE boundary**, permanently, and stop re-adjudicating it.

### An MDE convention that nearly flipped a verdict on its own

The house formula `2.8 * sd * sqrt(2/n)` assumes equal arms. Clustering by
distinct design breaks that badly here — `wt_coverage_max` yields **27 distinct
designs from 50 cells** against `llm_pc`'s 84 — and substituting `min(n_a, n_b)`
inflates the MDE by `sqrt(2/27) / sqrt(1/27 + 1/84)` = **1.23x**, which alone
turned WT k=21 from resolved to below MDE. The unequal-n form
`2.8 * pooled_sd * sqrt(1/n_a + 1/n_b)` is the correct one and is what the
numbers above use. **Any contrast between arms with different distinct-design
counts must use it**, and the coverage arms always will, because a
near-deterministic rule re-picks designs.

### Core-20 is now a standard column, and it is LT-only

`f1_core_rescored` is populated on **800/800 LT cells and 0/1,804 WT** —
`LT_CASE_STUDY_NODES` names the chambers' LT case study, and no WT equivalent
has been defined. So "no LLM arm beats round-robin coverage on the non-trivial
subgraph" is an **LT-only** statement; on WT it rests on full-node scoring.
State the scope. Like-for-like on LT, loop minus rule: **+0.014 / +0.008 /
−0.000** at k=6/30/45, none resolved — the earlier finding that the loop's
tight-budget win does not survive core-20 scoring, reproduced on one backend.

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
> task is coverage-shaped." ~~That objection is correct — §29's bipartite,
> depth-1 ground truth says so — and the answer is to scope it in the title,
> not to argue it.~~ **Superseded 2026-09-09 (THE ORACLE PROBE): the
> objection is measured and refuted.** The rule is a plateau, not the ceiling
> — a ground-truth oracle sits above every arm at every budget, in the core
> subgraph on LT. Report the oracle curve beside every table; see threat 6.
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

   **Both WT figures exclude `team` entirely** — all 300 WT `team` cells
   report `conservation_certified = None`, because its negotiate constant had
   never been isolated on WT and `is_provisional_calibration` correctly
   refuses to report conservation for a cell provisioned by a borrowed
   figure. So WT's H-C is currently missing its most-coordinated rung, which
   is the rung a reader will most want to see. **Unblocked 2026-09-05**
   (register §33): the constant is measured (6,102 per call, 27 cells,
   $0.13, drift audit clean) and WT `team` cells now certify True/False.

   **RECOVERED FOR $0, no re-run (2026-09-05, `recertify.py`).** The claim in
   the previous sentence of this entry — that the archived cells "cannot be
   retro-certified" because they ran under the wrong grant — was wrong, and
   checking it took three measurements. (a) `verify()`'s verdict is a pure
   function of state the cells already record, so the graph can be rebuilt,
   charged with the recorded per-node spend, and **the real `verify()`
   called** — not reimplemented, which matters because it carries a per-tool
   clause a scalar comparison drops. (b) The replay reproduces the recorded
   verdict on **300 of 300** cells of the two fan-in arms, where `verify()`
   did run. (c) The old grant never shaped the spend: **98 of 300 cells
   overspend by up to 13,833 tokens and complete normally**, and no node lands
   exactly on its ceiling (0 of 900) — a monitor that records rather than
   truncates.

   **WT `team` conservation: 202/300 = 67.3%** (90/100 at k=7, 56/100 at
   k=14, 56/100 at k=21). The corrected constant changes **no** verdict —
   identical under 4138 and 6102, because the correction only enlarges a grant
   and no cell sits in the band it moves. So WT H-C over all three graph arms
   is **395/600 = 65.8%**, against the 64.3% previously reported over two.
   Including the missing arm barely moves it.

   Scope: this worked because the correction enlarged the grant. A correction
   that SHRANK one, or an archive with cells sitting on their ceilings, would
   need the re-run — `recertify_frame` raises rather than return a number in
   that case.
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
6. ~~**The task is coverage-shaped, and we now have the evidence for it**~~
   **REFUTED BY MEASUREMENT 2026-09-09 — see "THE ORACLE PROBE" at the top
   of this document.** A ground-truth oracle sits +0.02–0.07 (LT) and
   +0.02–0.07 (WT) above every arm at every budget by ranking, +0.10–0.15
   by set on WT, in the core-20 subgraph
   on LT, and every LLM arm's purchases score as random on the oracle's
   marginal-gain scale. The threat now reads: *room existed and no topology
   found it*, which is a result, not a scope limit. The lagged-estimator
   idea below is dead, not deferred — the chambers ship no lagged ground
   truth to score against. The Antigravity verifier sentence survives as the
   TRANSFER condition (when partitioning pays elsewhere), not as a defence.
   The original entry, kept for the record: (new 2026-09-02, then the
   top-ranked threat). A ten-line LLM-free rule
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
