# Pre-registration: does GLM varsplit really edge the loop?

**Registered 2026-09-29 (evening), before any cell below was run; to run on
2026-09-30.** Code: branch `low-negotiation-rerun`, with `--seed-start`
(`797c6b0`) as the only change since the first run. Output
`runs/lownego-glm-rep.parquet`.

## What prompted it

The all-low negotiation re-run (`2026-09-29-low-negotiation-prereg.md`,
Amendment 2) gave, for GLM `glm-5.3-flash` at LT k=30, 300 rows, 50 distinct
designs per arm:

| contrast | diff [95% Welch CI] | MDE | registered reading |
|---|---|---|---|
| team - loop | -0.010 [-0.020, +0.001] | 0.015 | not established |
| varsplit - team | +0.021 [+0.010, +0.031] | 0.014 | repair holds |
| varsplit - loop | +0.011 [+0.001, +0.021] | 0.014 | **P3 falsified** |

That verdict stands as registered. It is one of about 26 P3 intervals, it
excludes zero by 0.001, and it is below the MDE, so it is not yet a result
the paper can rest on. The project rule is that such a verdict gets
re-run on another day, beside its comparator, before it is written as
resolved. At 1500 rows the same contrast is a tie, +0.003 [-0.011, +0.018]
(MDE 0.020), so the exception holds only at the paper's primary cap. The
other 1500-row contrasts: team - loop -0.027 [-0.042, -0.013] (team loses),
varsplit - team +0.031 [+0.017, +0.045] (repair holds).

## Design

- `openrouter/z-ai/glm-5.3-flash`, LT k=30 (30/59), arms `llm_pc` and
  `team_varsplit`, interleaved, **n=100 per arm, seeds 100-199** (disjoint from
  the corpus and the first run, seeds 0-49). 200 cells. `--selection-effort low
  --coordination-effort low`, timeout 7200 s, VPS, pinned `_GLM_PROVIDER_ORDER`.
  **Run on 2026-09-30**, a different day from the first run. Command flags:
  `--chambers lt --budgets 0.5084745762711864 --seeds 100 --seed-start 100
  --variants llm_pc,team_varsplit --model openrouter/z-ai/glm-5.3-flash`.
- `team` is not re-run: P1 and P2 are not in question.
- Re-scored on the VPS (OpenBLAS), 9 PC seeds, caps 300 and 1500, clustered
  by distinct design; `lownego_analysis.py` (commit `7838374`).
- Before launch: re-probe GLM provider prices and precision; a 2-cell probe
  (one per arm) checked for `reasoning_effort == "low"`.
- **Cost** about $0.30 (first run $0.0017 per cell), about 2 h.

**Power.** Pooled sd from the first run is about 0.025, so at n=100 per arm
a true difference of +0.011 is detected with about 87% power (about 59% at
n=50). A null here is therefore informative about an effect of the observed
size.

## Analysis and decision rules (on the interval)

**Primary: `varsplit - loop`, 300 rows, this run alone.** 95% Welch CI over
distinct designs.

- CI entirely above 0: **replicated.** The paper states that under GLM at
  LT k=30, 300 rows, `team_varsplit` edges the same-day loop, with both
  intervals and the coverage difference beside them.
- CI contains 0: **not replicated.** The paper reports the first run's
  exception as registered, followed by this replication, and does not
  claim an advantage.
- CI entirely below 0: **reversed**, reported as such.

**Secondary (reported, no bearing on the primary):** the same contrast at
1500 rows; the pooled estimate over both runs (150 designs per arm, Welch),
stated as pooled across two days; distinct variables per arm and the
coverage-only prediction for the difference.

**Prediction, written before running:** not replicated. The point estimate
is between 0 and +0.011. Coverage alone predicts about +0.002 at 300 rows
and +0.004 at 1500 (0.36 more distinct variables for varsplit, at the
first run's own F1-per-variable slopes, 0.0066 at 300 and 0.0123 at 1500,
which match the nine-arm corpus line, +0.0066 and +0.0121).

## What this does not change

The first run's P3 verdict stands whatever this shows. The abstract's
"no multi-agent arm beats the loop" is scoped per the first registration;
this run decides only whether the exception is reported as an effect or
as one unreplicated interval.
