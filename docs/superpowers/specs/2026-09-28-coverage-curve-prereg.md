# Pre-registration: is the coverage law's size miss a calibration gap?

**Registered 2026-09-28, before any list below was scored.** Code:
`evaluation/chamber_pipeline/coverage_curve_probe.py` (committed with this
file; `analyze` must not change after this commit). Cost: $0, no LLM calls.

## What prompted it

A co-author's review of the coverage-law check (paper Table 3 / Fig. 3) at the
main setting (DeepSeek, PC at 300 rows):

- The law predicts the **direction** of every resolved arm contrast, but not
  the **size**: pooled slope of measured on predicted gaps 1.70 [1.34, 2.03]
  on the 10 DeepSeek 300-row rows; 1.97 on the light-tunnel rows alone. At
  1500 rows it is 0.81 [0.68, 0.94]; GLM at 300 rows fits (0.63, wide).
- Controlling coverage leaves no arm effect (team vs loop at equal coverage is
  unresolved), so the miss is in the **exchange rate**, not a team penalty.
- Out-degree and strength weighting of coverage were tried and do not fix it
  (degree weighting makes it worse).
- The paper's rate is one straight line fitted on LLM-free lists at ~11, ~15,
  ~21 and 29-30 variables. At 300 rows those lists are flat from 11 to ~25
  variables (0.33-0.36) and high at 30 (0.42). **No LLM-free list covers 26-29
  variables** -- exactly where the loop and varsplit sit. The LLM lists
  themselves rise steeply from ~23 to ~28 at 300 rows (DeepSeek 0.381 -> 0.423).

Already seen, so NOT blind: the LLM arms' F1 by coverage (above). Not yet seen:
the F1 of LLM-free lists at 23-29 variables, which is what this probe measures.

## Question

Is the 300-row size miss (a) a **calibration gap** -- the F1-coverage curve is
steep in 24-28 for ANY list and the straight line misses it -- or (b) do
**LLM-chosen lists gain more F1 per variable** than chance-chosen lists at the
same coverage?

## Design

- **Lists.** LLM-free lists covering exactly m distinct variables (reference
  always included and counted, as in the paper's `lt_variable_count` and as
  every LLM list does): LT k=30 at m = 13..30, LT k=45 at m = 25..30, 30 lists
  per (k, m), 720 in all. Variables chosen uniformly at random (subject to
  having >= k entries), one random entry per chosen variable, remaining slots
  filled with random further entries of the chosen variables, order shuffled.
  Seeds `cc:{k}:{m}:{s}`, s = 0..29.
- **Comparators, same machine.** Everything is scored on this Mac
  (Accelerate): the new lists; the DeepSeek LLM lists re-scored locally from
  `m7-varsplit`, `m7-phase1`, `m7-p2-lt`, `m7-p2-ref`, `fig2-fill-lt` (LT,
  k in {30, 45}, arms `llm_pc`, `team`, `team_varsplit`, `one_shot`); GLM from
  `rescored-xv-glm-lt` (already Accelerate). No cross-backend comparison.
- **Scoring.** `rescore.py`, PC, 9 PC seeds, alpha 0.05, caps 300 and 1500.

## Analysis (fixed)

- **Calibration curve** per (k, cap): mean F1 of the exact-m lists at each m,
  linearly interpolated in coverage, flat beyond the measured range.
- **Prediction** for arms A, B: mean over A's lists of curve(coverage) minus
  the same for B. The prediction never uses an arm's F1.
- **Contrasts:** team - loop and varsplit - team at DeepSeek LT30 (the
  `m7-varsplit` day, as the paper), DeepSeek LT45 (`fig2-fill-lt`), and GLM
  LT30.
- **Statistic:** slope of measured on predicted gaps through the origin,
  weighted by 1/SE^2 (Welch SE per contrast). 95% CI by bootstrap (2,000):
  lists resampled within each arm and within each (k, m) calibration level,
  one draw shared by all contrasts in a replicate.
- The old straight-line law is recomputed on the same locally scored data and
  reported beside it, for comparison only.

## Decision rule (on the interval, per the project rule)

**Primary: DeepSeek, 300 rows (4 contrasts).**

- CI contains 1 -> **A, calibration gap.** The size miss is the straight-line
  calibration; the paper says the law's size holds once the rate is measured
  where the arms operate, and "LLM lists gain more per variable" is dropped.
- CI entirely above 1 -> **B, LLM lists gain more per variable** than
  chance-chosen lists at equal coverage; reported as a finding with its size.
- CI entirely below 1 -> **C**, the curve over-corrects; reported as such.

**Secondary (reported, same rule, no bearing on the primary verdict):** GLM
at 300 rows; DeepSeek and GLM at 1500 rows. Prediction under A: all contain 1.

**Descriptive:** the curve itself; the mean gap between LLM lists and the
curve at their own coverage, per vendor, budget and cap, with bootstrap CI.

## What would embarrass each reading

- A is wrong if the exact-m lists stay flat through 24-28 at 300 rows (then
  the curve predicts small gaps and the DeepSeek slope stays above 1).
- B is wrong if the exact-m lists rise as steeply as the LLM lists there.
