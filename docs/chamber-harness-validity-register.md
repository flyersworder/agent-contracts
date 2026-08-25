# Chamber harness validity register

Every defect found in the causal-chamber pipeline that **changed a scientific
result or would have**, with the measurement that established it and the check
that now prevents recurrence.

This file exists because the same defect class has now appeared five times and
was, each time, initially mistaken for a finding. It is the reference for the
paper's threats-to-validity and reproducibility sections. Add to it rather
than rediscovering.

**The recurring class:** *a harness property that silently determines the
result, where the failure looks like a finding.* What makes it dangerous is
not that the harness degrades — it always will — but that the degradation
**rate can correlate with the experiment's independent variable**. When it
does, the curve measures the harness.

**The rule that follows:** every degradation path must be counted per cell and
surfaced in `harness_validity_report`. An absent column reads as `UNMEASURED`,
never as clean.

---

## Register

| # | Defect | Looked like | Actually was | Now prevented by |
|---|---|---|---|---|
| 1 | `_SELECTION_MAX_TOKENS` too small | "LLM selection stops helping as budget grows" | 43% of k=30 selections were `rng.choice`; 0% at k=6 | caps at 32768; `n_selection_fallbacks` counted |
| 2 | `_A95_RECONCILE` single constant | conservation failures | aggregator cost grows with k; one constant cannot fit | `_A95_RECONCILE_BY_K` at p75; raises on uncalibrated k |
| 3 | Provider serves fp4 while pinned as fp8 | — (silent) | AtlasCloud is fp4; 27 of 450 M6 cells | `PROVIDER_PRECISION` + precision-homogeneity test |
| 4 | Together returns empty after 32768 reasoning tokens | a valid but odd selection | truncation degraded to `rng.choice`, provider-level | excluded from `DEFAULT_PROVIDER_ORDER` |
| 5 | `wt_walks_v1` fed to an i.i.d. test | "the wind tunnel is insensitive to selection" | random-walk autocorrelation 0.9999; ~19 effective samples | `wt_validate_v1`; `n_collinear_dropped` counted |

---

## 1. Selection truncation (2026-08-24)

`_SELECTION_MAX_TOKENS` was calibrated on the loop's *first* call, where
reasoning is 415–976 tokens. Reasoning scales with the prompt, and the prompt
grows one spent-experiment line per step. Late-loop it reached 2,175 (0731) to
11,690 (flash).

**Why it was a moderator:** failure rate tracked history length, so it was
0/36 at k=6 and ~43% at k=30 — correlated with the x-axis.

**Exposure in M6 was worse:** the ladder's IV is how budget is *split across
agents*, and splitting shortens each agent's history. Two scouts at k=15
truncate less than one loop at k=30, so the fan-in rungs would have beaten the
loop for reasons unrelated to coordination. **H-B could have come out positive
as a pure `max_tokens` artifact.**

Post-fix: 29 fallbacks / 12,780 calls = 0.23%.

## 2. Aggregator cost calibration (2026-08-24)

Aggregator cost grows with k, so one constant could not work: p75 was 7,646
(k=6), 11,427 (k=30), 18,790 (k=45), with spread 48.8× / 5.2× / 2.6×.

p75 rather than median for a design asymmetry: `_ROLE_C95` medians get
multiplied by `_PROVISION_MULTIPLE = 4`, but the aggregator gets `1.5 * a95`
and **no multiple** (a margin would destroy P2), so a median-sized budget
overruns ~50% of executions by construction.

**Consequence the paper must carry:** P2's window is `(f, n·f]` — width equals
the fan-in degree, so 2× for two scouts. k=6's 48.8× spread cannot fit inside
it. P2 is demonstrable at k=30/45 and effectively not at k=6. The lever is
more parents, not a better constant.

## 3–4. The provider is part of the configuration (2026-08-25)

One OpenRouter model id is served by many independent endpoints. They differ in
**price**, **numerical precision**, and **how much they reason**.

Identical prompt, identical model id, `max_tokens=32768`:

| provider | quant | $/M out | wall | output tokens | content |
|---|---|---|---|---|---|
| Parasail | fp8 | 0.280 | 17–34 s | 980–2,468 | valid |
| Novita | fp8 | 1.320 | 21 s | 1,899 | valid |
| SiliconFlow | fp8 | 0.280 | 110 s | 6,303 | valid |
| Together | unknown | 0.280 | 344 s | **32,768** | **EMPTY** |
| AtlasCloud | **fp4** | 1.320 | — | — | — |

Three consequences:

- **Cost.** 422 of 450 M6 cells ran on Novita: **$54.53**, against ~$11.60 for
  the same sweep on Parasail. Price is not a proxy for speed.
- **Precision.** The code *comment* asserted Novita and AtlasCloud were "both
  fp8". AtlasCloud is fp4. Impact measured: raw F1 on fp4-touched cells differs
  at p<1e-4, but that is entirely arm×budget composition — residualised on
  arm×budget it is −0.004 vs +0.000, **Welch p=0.61**. Reported as a
  limitation, not a correction.
- **Truncation.** Together burns the whole cap on reasoning and returns
  nothing, which this harness degrades to `rng.choice` — defect #1 one level
  down.

**Why the old test could not catch it:** it asserted
`order == DEFAULT_PROVIDER_ORDER`, true by construction whatever the constant
held. The fp8 claim lived only in prose.

**For the reproducibility statement:** archive the resolved *endpoint*, not
just the model string. Same weights does not mean same behaviour — reasoning
length varied 1,360 / 6,298 / 32,768 tokens for one prompt.

## 5. The WT dataset violated PC's test assumption (2026-08-25)

`wt_walks_v1` is a random-walk time series. Median lag-1 autocorrelation
**0.9999** (88% of variables above 0.9), so its 320,000 rows per experiment
carry roughly **19 independent observations**. LT's
`lt_interventions_standard_v1` is 0.007 — effectively i.i.d.

Fisher-Z assumes i.i.d. samples. Fed the walks release, the budget response
**inverts**: F1 falls as experiments are added, SHD worsens 55 → 67, and
predicted edges grow 25 → 38 — spurious density, not a property of the wind
tunnel. Same menu, 30 seeds per point:

| k/M | 0.11 | 0.25 | 0.50 | 0.75 | 1.00 | slope |
|---|---|---|---|---|---|---|
| `wt_walks_v1` | 0.166 | 0.170 | 0.171 | 0.161 | 0.149 | −0.0007 (p=0.06) |
| `wt_validate_v1` | 0.147 | 0.190 | 0.220 | 0.249 | 0.254 | **+0.0042 (p=1.4e-13)** |
| LT reference | | | | | | **+0.0041 (p=7.3e-52)** |

`wt_validate_v1` covers the same 28-experiment menu at lag-1 autocorrelation
0.14 and responds to budget at a rate statistically indistinguishable from
LT's. Dynamic range 0.022 → 0.107.

**I recorded the wrong conclusion first.** Before finding this, the flat walks
curve was written up as an external-validity finding — "the wind tunnel's
discovery problem is insensitive to selection". It is not; the pipeline was.

### The collinear barometers

WT's four pressure sensors — `pressure_upwind`, `pressure_downwind`,
`pressure_ambient`, `pressure_intake` — all read essentially ambient pressure
in the `standard` configuration. All six pairs exceed r=0.9998 and **none is a
true edge**. Four variables spanning ~one dimension leaves `cond(R)` ≈ 1e7,
and Fisher-Z's sub-matrix inversion raised — aborting the whole run and
returning all-zeros for all 32 nodes, F1=0 for the cell. **15 of 60 runs.**

`run_pc` now drops near-duplicate columns and pads them back with zeros, the
same policy already applied to zero-variance columns. Degeneracies: 0 of 150.

**The cost, stated plainly:** the four barometers are pure sinks and 13 of 42
true edges point into them. Dropping three of the four forfeits those edges.
That is strictly better than forfeiting all 42, but it is a real recall
ceiling and belongs in the WT results as a scope limit.

**It is counted**, because which columns are duplicate depends on which
experiments were bought and can move with the budget axis. On WT it does —
0.90 → 1.00 of cells — and `validity_warnings` flags it as contaminating.

---

## Standing scope limits (not defects)

- **`overlap_frac` is structurally 0.0 for rung 4** — pools are disjoint by
  construction.
- **The aggregator's reconcile output is discarded.** The merge is a Python
  dedup plus PC. The token flow is real, so the P2/conservation demonstration
  holds, but **the paper must not claim the aggregator improves the result.**
- **Rung 4's negotiation parser reads restatement as claim**, inflating
  `n_contested`. Cannot be fixed by filtering — a genuine contest is the
  signal. See spec §11.
- **Temperature is unpinned**, so the seed does not control the LLM. Same seed
  and config gave F1 0.330 and 0.482. Variance, not bias; every cell is an
  independent draw. Confirmed independently: cross-arm correlation within a
  seed is −0.03, i.e. seed pairing carries no information.
- **Library drift breaks per-cell reproducibility, not distributions.** PC is a
  threshold algorithm, so a p-value perturbed in the 12th decimal flips any
  edge near α=0.05. Flips are unbiased in direction: individual cells move a
  lot (seed 0: F1 0.204 → 0.127), the distribution does not (Welch p=0.265 on
  F1, p=0.743 on SHD). Archive the resolved environment, not just the seed.

## How to add an entry

1. Measure the effect on the result — do not assert it. Residualise on
   arm×budget before claiming a difference is real; three of the five entries
   above looked significant raw and were composition.
2. Add the counter at source and surface it in `harness_validity_report`.
3. Write the test against the *behaviour*, then mutate the implementation and
   confirm the test fails. Two tests in this repo passed against a deliberately
   broken implementation before this step was added.
