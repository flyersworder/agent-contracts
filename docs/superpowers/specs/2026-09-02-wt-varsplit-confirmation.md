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
