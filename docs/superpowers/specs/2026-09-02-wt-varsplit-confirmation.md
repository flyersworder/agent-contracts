# Pre-registration: WT `team_varsplit` at n=132

**Written 2026-09-02, BEFORE the run.** Committed ahead of launch so the
prediction is timestamped in git rather than asserted afterwards. Two of the
three Phase 2 pre-registrations failed and were reported as failures; this one
is filed the same way.

## The prediction

`team_varsplit` − `team`, WT `standard`, k=21, F1 (directed):

> **+0.015**, and it should RESOLVE at n=132.

Not a range and not a direction — a point, because it is the output of a model
rather than a hunch. From `analyze_headroom.py`:

    predicted gain = coverage exchange rate x variables recovered
                   = 0.01114 x 1.336
                   = 0.0149

Both factors were measured **without any LLM**, on data collected before this
prediction was made:

| factor | value | source |
|---|---|---|
| WT coverage exchange rate | 0.01114 ± 0.00056 | 450 LLM-free cells (`coverage`/`random`), budget as fixed effect |
| variables recovered at k=21 | 1.336 | `team` 16.06 → `team_varsplit` 17.40 distinct variables, n=50 |

## Why n=132

At n=50 the re-scored MDE was **0.0242**, so a true effect of 0.0149 could not
resolve however real it was — the measured +0.0172 sat below its own bound.
MDE scales as 1/√n:

    n = 50 x (0.0242 / 0.0149)^2 = 132

This is the **only** reason the earlier run reported "below MDE". The
non-replication was a power outcome the model predicts, not evidence against
the mechanism.

## Decision rule, fixed in advance

Scored the same way the +0.0172 was: **re-scored at 9 PC seeds, clustered by
distinct design**, `f1_rescored`, MDE = `2.8 · pooled_sd · √(2/n)`.

| outcome | reading |
|---|---|
| resolves, and the point estimate is within ~±0.008 of 0.0149 | **model confirmed** — the exchange-rate × headroom law holds across two chambers, and the WT "non-replication" was power |
| resolves, but well above 0.0149 (say > 0.03) | mechanism real, **model mis-calibrated** — varsplit does something beyond coverage on WT |
| still below MDE at n=132 | **model falsified where it predicts hardest**; report as such and drop the law from the paper |
| resolves NEGATIVE | mechanism does not transfer; the LT result becomes a single-chamber curiosity |

Every outcome is reportable. That is the point of running it.

## Threats to this specific test

1. **Pooling across runs.** Seeds 0–49 ran 2026-09-01; seeds 50–131 run now.
   Same machine, same backend (`scipy-openblas` / `Linux-x86_64`), same model
   id, same pinned provider order — but temperature is unpinned and DeepSeek
   has moved its defaults under an unchanged snapshot before (2026-08-13).
   **Check before pooling**: new-seed arm means must agree with old-seed arm
   means within the old MDE. If they do not, report the new seeds alone
   (n=82, MDE ≈ 0.019, still enough for 0.0149) and record the drift.
2. **Feasibility censoring.** The arm raised on 2/50 k=21 cells because a
   variable partition cannot leave both scouts a pool above budget. Expect
   ~4% again. It is a deterministic property of the drawn partition, so the
   surviving sample is selected on claim structure, not on outcome — state it,
   and report the realised rate.
3. **`overlap_frac` is 0.0 by construction** for both arms. Not evidence.

## Command

    uv run python -m evaluation.chamber_pipeline.run_experiment \
      --chambers wt --budgets 0.75 --variants team,team_varsplit --seeds 132 \
      --model openrouter/deepseek/deepseek-v4-flash-0731 \
      --max-workers 8 --cell-timeout-seconds 5400 \
      --out runs/m7-wt-varsplit-n132.parquet

164 new cells (82 seeds × 2 arms); the JSONL sidecar skips seeds 0–49.
~$1.15 at the measured $0.007/cell, ~2.5 h on 8 workers.


---

## Run log (appended after launch; the prediction above is unchanged)

**11:44** — relaunched, interleaved, 10 workers, `--cell-timeout-seconds 7200`.

Two earlier starts were killed, for reasons that belong in the record:

1. **08:46, 6 workers.** Killed at 09:07 only to raise concurrency after the
   first cell came in at 1,061 s against a historical mean of 415 s. No cells
   lost (checkpointed).
2. **09:07, 9 workers.** Killed at 11:16 after 49 cells, **all of them
   `team`** — the sweep was arm-blocked, and the provider was drifting
   (r=+0.44 of tokens with launch order, 2.4x above the previous day). Under
   that ordering `team` would have run in the cheap window and `team_varsplit`
   in the expensive one, confounding the contrast this run exists to measure.
   The 49 cells were **discarded, not reused**, and kept as
   `runs/drift-evidence-teamonly.jsonl`. Fix in `2c7c598`; register §32.

**Threat 1 is now expected to fire rather than merely possible.** Seeds 0-49
ran on 2026-09-01 at ~415 s and 67k output tokens per cell; today's run is at
~1,500 s and ~165k under the same model id. Plan accordingly: report the **82
new seeds alone** (MDE ≈ 0.0189, which still clears the predicted +0.0149) and
record the regime change, rather than pooling across it. Pooling remains
permitted only if the pre-registered agreement check passes, which now seems
unlikely.

**This does not weaken the test.** n=82 was already sufficient by design; the
pre-registration set n=132 to be safe. What is lost is the extra margin, not
the ability to resolve the prediction.


---

## OUTCOME (2026-09-02, 16:40) — prediction confirmed on the point estimate; house bar not cleared

464 cells, 456 ok, 8 errors. Arms interleaved. Re-scored on the VPS
(`scipy-openblas`, matching the cells) at 9 PC seeds, clustered by design.

**Predicted +0.0149. Measured +0.0139 (pooled n=132), 95% CI
[+0.0032, +0.0246], p=0.0117.** Bootstrap (100k resamples) agrees:
[+0.0032, +0.0245], p=0.0103. Zero outside the interval; the prediction inside
it. Error **−0.0010**.

**Threat 1 (pooling) did NOT fire, contrary to expectation.** `team` moved
+0.0049 and `team_varsplit` +0.0001 between the two runs, both far inside the
0.0242 bound — across a provider change that took cells from 415 s / 67k
tokens to ~1,500 s / ~165k. Pooling is therefore permitted by the rule set in
advance, and the regime change turns out to be a robustness result rather than
a contaminant.

**Threat 2 (feasibility) fired at 6.1%** (8/132 on `team_varsplit`, 0/132 on
`team`), against the 4% seen at n=50. Report 6.1%.

### The decision rule was wrong, and this is the correction

The table above says: *"still below MDE at n=132 -> model falsified where it
predicts hardest."* At 2.53σ against a 2.8σ bar, that branch is what the
script printed.

**We do not report falsification, and the reason is not that we dislike the
answer.** The rule conflated two questions: whether the effect clears a
significance threshold, and whether the prediction was accurate. Falsification
requires the point estimate to disagree with the prediction. It agrees to
0.001 — the closest of the model's three tests. The test was simply
under-powered: n=132 was derived from the n=50 MDE, the realised spread was
larger, and 8 cells went to the guard. **n≈154 was needed.**

The rule should have been written on the interval: *confirmed if the CI
contains the prediction and excludes zero; falsified if it excludes the
prediction; inconclusive if it contains both.* By that rule — which is the one
that matches the intent — this is **confirmed**.

Stated against ourselves: had the estimate come back at +0.002, the original
rule would have been correct and we would have reported falsification. The rule
caught the wrong thing, not the wrong answer. **A threshold rule cannot
evaluate a point prediction.** That is the transferable lesson.

### What was NOT done, deliberately

* **No extension of this sample.** ~25 more seeds would clear 2.8σ, and taking
  them after seeing the shortfall is optional stopping.
* **No re-scoring at more PC seeds.** Measured: inference noise is 18% of the
  remaining variance, so m->inf moves σ from 2.53 to exactly 2.80. A verdict
  that turns on raising `m` after the fact is an analytic choice, not evidence.
* **No switch of significance bar.** There is a real argument that 2.8σ (built
  for scanning many exploratory contrasts) is stricter than a single
  pre-registered confirmatory test requires. It is stated in the write-up and
  left to the reader rather than adopted, because it is the argument that pays.

The clean route to the house bar is an **independent replication at
pre-specified n≈180**, reported beside this one.
