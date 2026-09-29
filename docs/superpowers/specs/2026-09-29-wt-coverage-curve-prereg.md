# Pre-registration: do wind-tunnel arm gaps match random lists of equal coverage?

**Registered 2026-09-29, before any list below was generated or scored.** Code:
`evaluation/chamber_pipeline/wt_coverage_curve_probe.py` (committed with this
file; `analyze` must not change after this commit). Cost: $0, no LLM calls.

## What prompted it

A co-author's review of the light-tunnel k=30 analysis: each LLM arm's F1 gap
to the coverage rule, set against the gap that LLM-free lists of the same
coverage show (random lists covering exactly m variables). At LT k=30 the
arms' gaps sat within 0.008 of that prediction at 1500 rows for 7 of 9 arm x
vendor points, and about 0.02 above it for every arm at 300 rows. The review's
objection is correct: that analysis was chosen **after** seeing the data (the
budget, the prediction form, the pooling), so it is a post-hoc observation,
not a claim. This probe asks the same question **out of sample**, on the second
chamber, with the analysis fixed first.

Why WT k=21 and nothing else: the WT menu has 28 entries over 21 variables and
only `hatch` (3 entries), `load_in` (3) and `load_out` (4) have more than one.
A 21-entry list therefore covers 14-21 variables. At WT k=7 and k=14 no arm is
more than about 2.5 variables short of the rule, so there is nothing to test;
k=21 is the only WT budget with measurable coverage gaps (arms 3.7-4.8 short).
k=30 is impossible (28 entries).

**Already seen, so NOT blind:** each WT k=21 arm's F1 gap to the rule (paper
Figure 2) and its mean coverage, and their rough agreement with the paper's
straight-line WT rate (0.0111 F1 per variable). **Not yet seen:** the F1 of
LLM-free WT lists of exact coverage, which is the entire prediction.

## Design

- **Lists.** WT k=21, coverage m = 14..21 (every feasible level), 30 lists per
  m, 240 in all. Variables chosen uniformly at random subject to having >= 21
  entries between them (at m=14 this forces all three multi-entry variables),
  one random entry per chosen variable, the remaining slots filled with random
  further entries of the chosen variables, order shuffled. Seeds
  `wtcc:21:{m}:{s}`, s = 0..29. Variables resolved with the analysis code's WT
  resolver (`analyze_headroom.variable_resolver`, longest node-name prefix).
- **Arms (fixed now).** DeepSeek at WT k=21, every source in the paper's VPS
  re-score `rescored-vps`, pooled per arm exactly as Figure 2 pools them:
  `llm_pc`, `one_shot`, `team`, `team_varsplit`, `shared_blackboard`.
  `critique` is excluded (not defined in the paper); GLM never ran WT k=21.
  Rule: `wt_coverage_max` (the paper's WT rule), same file.
- **Scoring.** `rescore.py` on the VPS (OpenBLAS, same backend as the corpus
  re-score), PC, 9 PC seeds, alpha 0.05, caps 300 and 1500. Designs clustered:
  one F1 per distinct design, mean over the 9 seeds. The analysis asserts
  every input is OpenBLAS.

## Analysis (fixed)

- **Curve** per cap: mean F1 of the exact-m lists at each m, linearly
  interpolated, flat beyond the measured range.
- **Prediction** for arm A: mean over A's lists of curve(coverage) minus the
  rule's mean F1. **Measured:** A's mean F1 minus the rule's. **Residual:**
  measured minus predicted. The prediction never uses an arm's F1.
- **Uncertainty:** bootstrap, 2,000 replicates, one shared draw per replicate:
  designs resampled within each arm, within the rule, and within each
  calibration level m.
- **Informativeness check (first):** curve(21) - curve(16) with bootstrap CI.
  If its lower bound is <= 0, coverage does not move F1 over the arms' range
  and the probe is **uninformative** at that cap: no verdict is drawn.

## Decision rule (on the interval, per the project rule)

**Primary statistic, per cap:** the mean residual over the five arms, 95% CI.
Equivalence margin **+/-0.015 F1**, fixed now (about a third of the smallest
arm gap at WT k=21).

- CI inside +/-0.015 -> **A: coverage accounts for the WT gaps.**
- CI entirely above 0 -> **B: arms beat random lists of equal coverage.**
- CI entirely below 0 -> **C: arms lose more than coverage predicts.**
- CI contains 0 but leaves the margin -> **D: inconclusive.** Reported as a
  bound, never as support.

The rules are checked in that order: a CI inside the margin is **A** even if it
excludes 0 (a residual too small to matter is reported with its sign, not as B).

**Predictions, written before scoring (from the LT k=30 pattern):**
**A at 1500 rows; B at 300 rows**, with a mean residual near +0.02. Both caps
are reported whatever they show; neither is dropped.

**Reported, no bearing on the verdict:** each arm's residual with its CI; the
curve itself; team - loop and varsplit - team, predicted and measured. The
arms differ from each other by about one variable at WT k=21, so no
between-arm ordering test is made: it would have no power.

## What each outcome means for the paper

- A at 1500 (and B at 300): the LT pattern replicates out of sample on the
  second chamber; the paper may state it as a pre-registered result for WT
  k=21, with the LT k=30 analysis as the post-hoc observation that motivated it.
- C or D at 1500: the LT observation does not transfer; the paper reports the
  WT result and keeps the LT analysis labelled post hoc, in robustness only.
- Uninformative: WT cannot test the question at any budget; say so.

## What would embarrass the reading

- The coverage account is wrong on WT if the exact-m curve is flat over
  16-21 variables (uninformative) or if the arms sit well away from it at
  1500 rows (C or D).
- "LLM picks beat random at 300 rows" does not transfer if the 300-row
  residual is A, C or D.

## Caveats known now

- WT data carry two recording sessions 2.3 kPa apart (register section 37);
  random and arm lists draw on the same menu, so both are exposed alike.
- At m=14 every random list contains all three multi-entry variables; at
  low coverage, which variables a WT list holds is partly forced by the menu.
  This constrains the arms equally.
