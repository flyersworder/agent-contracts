# Chamber harness validity register

Every defect found in the causal-chamber pipeline that **changed a scientific
result or would have**, with the measurement that established it and the check
that now prevents recurrence.

This file exists because the same defect class has now appeared eight times and
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
| 5 | `wt_walks_v1` fed to an i.i.d. test | "the wind tunnel is insensitive to selection" | random-walk autocorrelation 0.9999; ~19 effective samples from 320,000 rows | `wt_validate_v1`, documented at the constant |
| 6 | Collinear columns abort the whole PC run | a cell with genuinely no signal | four barometers read one quantity; all-zeros for all 32 nodes, F1=0 | local drop + pad; `n_collinear_dropped` counted |
| 7 | 30s request timeout | 3 cells "failed" at k=21 | the timeout sat at the MEDIAN call latency; retries hid it until the tail | raised to 300s, ratio pinned against measured p99 |
| 8 | `allow_fallbacks: True` | a precision-homogeneous sweep | OpenRouter routed past the pinned list; 22 WT cells on unpinned, unknown-quant endpoints | `allow_fallbacks: False`, asserted by test |

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

- **Cost.** 422 of 450 M6 cells ran on Novita, and the sweep billed **$54.53**
  in total ($49.68 of it Novita's). The endpoint price model reproduces that —
  7.4M in + 39.5M out at Novita's $0.44/$1.32 predicts $55.40 — and the same
  token counts at Parasail's $0.14/$0.28 predict **$12.10**. Price is not a
  proxy for speed.
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
**inverts**: F1 falls as experiments are added, SHD worsens 54.6 → 68.8, and
predicted edges grow 23.5 → 38.8 — spurious density, not a property of the wind
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

## 6. Collinear columns aborted the whole run (2026-08-25)

Found while fixing #5, but a **separate defect with a separate cause**: #5 is
about which dataset, #6 about how PC handles redundant sensors in any of them.

### The collinear barometers

WT's four pressure sensors — `pressure_upwind`, `pressure_downwind`,
`pressure_ambient`, `pressure_intake` — all read essentially ambient pressure
in the `standard` configuration. All six pairs exceed r=0.9998 and **none is a
true edge**. Four variables spanning ~one dimension leaves `cond(R)` ≈ 1e7,
and Fisher-Z's sub-matrix inversion raised — aborting the whole run and
returning all-zeros for all 32 nodes, F1=0 for the cell. **15 of 60 runs.**

`run_pc` now drops near-duplicate columns and pads them back with zeros, the
same policy already applied to zero-variance columns. Degeneracies: 0 of 150.

**The cost, stated plainly:** the four barometers are pure sinks, and 13 of 42
true edges point into them. `pressure_upwind` survives the filter (in-degree
4), so the three dropped sinks forfeit **9 of 42 true edges** — a recall
ceiling of 0.786 on WT. Strictly better than forfeiting all 42, but real, and
it belongs in the WT results as a scope limit.

**It is counted**, because which columns are duplicate depends on which
experiments were bought and can move with the budget axis. On WT it does —
0.90 → 1.00 of cells — and `validity_warnings` flags it as contaminating.


## 7. The request timeout sat at the median call latency (2026-08-25)

`DEFAULT_REQUEST_TIMEOUT_SECONDS` was 30.0, with a comment calling it
"generous for normal completions (~1-15s)". True in May 2026; invalidated by
the 32768 token caps and by August's provider-side reasoning increase.
Measured mean seconds per LLM call on the WT gate: **p50 30.3**, p90 53.7,
p99 78.9, max 86.1. The timeout sat at the middle of the distribution, with
21 of 42 cells above it.

**Why it was a moderator, not noise:** `num_retries=3` rescues most
over-runs, so a failure surfaces only where latency is highest -- the heaviest
arms at the largest budget. The gate lost 3 of 45 cells, ALL at k=21, all in
`team` and `fan_in_spec`. The error rate was therefore a function of both the
topology IV and the budget axis: survivorship bias aimed squarely at the arms
under comparison, the same shape as M4b's 8 `planner_reasoner` timeouts.

Raised to 300s (~3.5x measured p99). The measurement is now a constant,
`MEASURED_CALL_P99_SECONDS`, with a test pinning the ratio. Retrying the three
failed cells under the new value: all three passed.

**Note the near-miss.** LT's M6 ladder reported 0 errors in 450 cells. That
was not health -- 26.4% of its cells also averaged above 30s per call, and it
survived only because Novita ran a few seconds faster than Parasail.

## 8. `allow_fallbacks: True` defeated the precision pin (2026-08-26)

`PROVIDER_PRECISION` and `test_default_provider_order_is_precision_homogeneous`
certify that every PINNED provider is fp8. They cannot certify what actually
served a call: with `allow_fallbacks: True`, OpenRouter routes past
`provider.order` when the listed providers fail.

The 750-cell WT ladder was served in part by **OpenInference (14 cells),
Relace (9) and DigitalOcean (4)** -- 22 cells, 2.9%, none pinned, none in the
precision table, quantization unknown.

Impact on that sweep was nil and is reported rather than corrected:
residualised on arm x budget, off-pin cells sit at **+0.0095** against
**-0.0003**, Welch **p=0.55**. The defect is in the guarantee, not the data.

Now `allow_fallbacks: False`. The trade is availability for reproducibility:
if all four pinned fp8 endpoints fail the call fails, rather than silently
succeeding on an unknown stack. Rotation plus `num_retries` still give four
providers x four attempts inside the pinned set.

**This is the fp4/AtlasCloud lesson one layer down.** There, a pinned provider
had drifted precision. Here, routing left the pinned set entirely. Both make
the same shape of claim unverifiable, and in both cases the test asserted
something true about our REQUEST rather than about what ran.

## 9. Machine suspend masqueraded as a 5x cost error (2026-08-25)

Not a pipeline defect, but it consumed an investigation and produced a wrong
public claim, so it belongs here.

The WT gate ran **1.01h of active worker time inside a 6.66h wall-clock
span**. Per cell, `finished_at - started_at` was up to 5.5x
`wall_time_seconds`, and the gap scaled with budget -- the shape of a real
per-cell overhead. Three hypotheses were eliminated by direct measurement
(adapter construction 0.0s, imports 3.2s, scoring ~0s) before the cause
surfaced.

`wall_time_seconds` comes from `time.perf_counter()`, which on macOS is
`mach_absolute_time` and **does not advance while the system is asleep**;
`started_at`/`finished_at` are wall-clock and do. The laptop was idle-sleeping
on battery. Against `pmset -g log`, sleep windows account for **100.4%** of
the 70,029s gap (residual -288s over 48 cells). The machine slept 13.74h that
day.

**An interim claim that the cost model was wrong by 5x is retracted.**
`wall_time_seconds` was correct throughout and remains the figure sweep
estimates must use.

**The diagnostic tell, recorded because it should have been read sooner:**
two CONCURRENT cells lost an *identical* interval -- 1475s in one batch, 853s
in another. Per-cell work cannot be identical across processes; a shared
external interval can. That points at the clock, not at the code.

`suspend_seconds` is now reported by `harness_validity_report`, computed from
two columns that were always recorded. Deliberately NOT in `contaminating`:
sleeping between LLM calls cannot change what an agent selected. Run sweeps
under `caffeinate -is`, or better, on the Linux VPS -- the 750-cell WT ladder
there recorded **231s** of suspend against the gate's 70,029s.

---

## 10. The BLAS backend silently determined the graph (2026-08-26)

`runs/m6-controls.parquet` and `runs/curve-lt-random.parquet` disagree on
`random` LT k=6 -- a **seeded, LLM-free** arm that must be deterministic.
Seed 0: F1 0.1266 with 22 predicted edges against 0.2041 with 41. A
structurally different graph, not a boundary flip.

**Cause, measured directly:** the two files were produced on different
machines, and macOS/Accelerate and Linux/OpenBLAS do not agree bit-for-bit.
On the same LT matrix, `np.corrcoef` differs, and `inv(C)[0,1]` agrees to
~10 hex digits then diverges at relative ~1e-10. Dataset md5s match and all
four numeric packages are at identical versions on both machines.

PC amplifies this. It is a sequence of accept/reject tests at alpha, each
conditioned on the previous ones, so flipping one borderline test forks the
conditioning-set search. Numerical noise becomes structural noise.

**Two earlier explanations of this same observation were wrong** and are
retracted: "library drift breaks per-cell reproducibility" (`c77c610`,
written 14 minutes after the anomalous run began) and "an unrecoverable
uncommitted working tree" (`bd46b0d`, same day). Both were reached by
elimination. The mechanism was found by *measuring the linear algebra
directly* -- the discipline the register exists to enforce.

Excluded by direct measurement before that: committed code (HEAD reproduces
M4b exactly, per-seed, on macOS), dependency versions, arm mix (`random` is
bit-identical alone and alongside `greedy_ig_lite`), the parallel path
(`--max-workers 3` is seed-faithful), and `pc_alpha` (neither 0.01 nor 0.10
reproduces it). All five held; the sixth candidate was the machine.

**Consequence for reported results:** both M6 ladders ran on the VPS, so the
cross-chamber replication is platform-consistent and stands. But every
loop-vs-random contrast so far pairs a VPS ladder against a local random
curve, including WT's +0.019 / +0.037. Those must be recomputed against a
VPS random baseline.

**Fixed** by recording the configuration instead of assuming it. PC had
three silent determinants and the schema carried none: `pc_alpha`,
`DEFAULT_MAX_ROWS` (300) and `DEFAULT_COLLINEARITY_THRESHOLD` (0.999); the
backend was a fourth. All now stamped on every `RunRecord` -- including
skips and errors, since a re-run needs a failed cell's configuration too --
plus `blas_backend` and `platform_tag`. **Never pool rows whose
`blas_backend` differs.**

**The first test written for the fix was trivially true, and mutation caught
it.** `run_cell` stamps constants; `run_pc` binds its defaults at def time.
Editing the constant in the source changes both together, so
`assert record.pc_max_rows == DEFAULT_MAX_ROWS` held by construction. The
real hazard is a *runtime* reassignment, which moves the constant and leaves
`run_pc` alone. The stamp now reads `run_pc`'s bound signature via
`pc_call_defaults()`, and the test monkeypatches the constant and asserts the
stamp does **not** follow.

That is the fourth entry whose test certified our REQUEST rather than what
ran (§3-4 provider order, §7 the hardcoded 30.0, §8 `allow_fallbacks`). The
rule, stated outright: **assert against recorded execution, never against
the constant that configured it.**

## 11. Collinear-column drops are not WT-only, and they track the IV (2026-08-27)

§6 recorded collinearity as a WT problem -- four barometers reading ambient
pressure. The 420-cell LT loop curve shows LT has it too, and worse in the
respect that matters: **the drop rate varies with the budget AND differs
between the two arms being compared.**

| arm | k=6 | k=12 | k=20 | k=30 | k=40 | k=50 | k=59 |
|---|---|---|---|---|---|---|---|
| `llm_pc` | 43% | **70%** | 37% | 0% | 0% | 0% | 0% |
| `random` | 27% | 23% | 13% | 13% | 7% | 3% | 0% |

(percent of cells with at least one column dropped; 71/420 cells, 76 columns)

The mechanism is not a defect: with few experiments bought, fewer distinct
interventions are present, so more column pairs are indistinguishable in the
pooled data. It vanishes as k grows. But the LLM's selections are *more*
collinear than random draws at low budget, which is a behavioural difference
between the arms sitting on top of a budget-dependent harness effect -- the
§1 shape exactly.

**Measured, so it can be reported rather than feared.** Within arm and
within k, cells with a dropped column score lower: `llm_pc` k=6, 0.2244
(n=17) vs 0.1898 (n=13), **p=0.02**; other cells same sign, not significant.
So drops do cost F1.

Two consequences, both checked:

1. **It does not manufacture the k=12 anomaly.** Restricting k=12 to cells
   with no drops moves the loop-vs-random delta from +0.0138 to **+0.0037** --
   smaller, not larger. The below-MDE dip at k=12 survives; it is not a
   collinearity artifact.
2. **It biases the headline conservatively.** The loop takes drops more often
   than random at low budget, so its resolved +0.047 at k=6 is if anything
   understated (+0.050 restricted to clean cells).

Report as a scope limit with these numbers attached. Do NOT report the LT
curve's low-budget points without it: a reader who knows only §6 will assume
LT was unaffected.

## 12. The provider order is per-MODEL (2026-08-27)

`DEFAULT_PROVIDER_ORDER` was one constant applied to every model, but the
price and precision RANKING of endpoints is model-specific. Measured from
`GET /models/{id}/endpoints`:

| provider | flash-0731 | v4-pro |
|---|---|---|
| Parasail | **$0.28/M (cheapest fp8)** | **$3.48/M (most expensive fp8)** |
| Baidu | $0.28/M | **$1.58/M (cheapest fp8)** |

Caught in preflight, before spending: running the flash-calibrated order on
v4-pro would have overpaid **2.2x**, silently, during the one experiment
whose entire purpose is to vary the model. This is §3-4 recurring on a
different axis — there the order was stale, here it was *model-inappropriate*.

Fixed by `PROVIDER_ORDER_BY_MODEL`, matched on the **exact model tag**.
Substring matching is wrong because a dated snapshot is a different product:
Baidu does not serve `deepseek-v4-pro-0813` at all, and StreamLake serves
`deepseek-v4-pro` at fp8 while its `-0813` endpoint reports `unknown`.
**Quantization is a property of the (provider, model) pair, not of the
provider** — so `PROVIDER_PRECISION`, which is keyed by provider alone,
is an approximation that holds only for the models whose endpoints were
actually inspected. Unlisted models fall back to the default.

**A real test gap surfaced here.** `test_injects_default_provider_order`
drives `model="m"`, which falls back to the default — so it passed whether or
not per-model routing was wired in *at all*. Mutation caught it: removing the
resolver from the request path broke nothing. There is now a test asserting a
pro-model request carries the pro order **on the wire**. Fifth instance of
"the test certified our REQUEST, not what ran" (§3-4, §7, §8, §10).

Also stale and now corrected: the note that `deepseek-v4-pro` "truncates at
2048 with empty content" dated from the era of the 2048 cap. At 32768 it
returns clean content — 4/4 preflight cells and 160/160 sweep cells with
near-zero fallbacks.

## 13. The pipeline version is part of the configuration, like the BLAS backend (2026-08-29)

Entry 10 established that two machines produce different graphs from identical
inputs. This is the same lesson one level up: **two runs on the same machine,
same backend, same seed, same arm produce different graphs if the pipeline
changed between them** — and our files straddle exactly such a change.

`random` is deterministic given its seed, which makes it a probe. Comparing
overlapping `(budget_k, seed)` cells:

| pair | overlap | identical | mean abs ΔF1 |
|---|---|---|---|
| `curve-lt-random` vs `m6-lt-loop-curve` | 120 | **0** (0%) | **0.055** |
| `m6-controls` vs `m6-lt-loop-curve` | 60 | 48 (80%) | 0.005 |
| `m6-controls` vs `curve-lt-random` | 90 | 0 (0%) | 0.053 |

Two things follow. First, the middle row **positively verifies** that
`m6-controls` is OpenBLAS — its platform had only been attributed from a run
log, never stamped. Second, the 20% that differ are not noise. Cross-tabulating
the changed cells against `n_collinear_dropped` on the newer run:

| `n_collinear_dropped` | unchanged | changed |
|---|---|---|
| 0 | 48 | 0 |
| 1 | 0 | **12** |

**Perfect separation** — every changed cell dropped a collinear column, every
unchanged cell dropped none. The difference is entirely defect 6's fix (2026-08-25),
not drift. It also confirms defect 11: collinear drops are **not** WT-only, they
fire on 20% of LT cells at k ∈ {6, 30}.

**The version boundary, by file.** The presence of the `n_collinear_dropped`
column is the marker:

- **Pre-fix**: `m4-pilot`, **`m6-ladder`** (the LT ladder), `m6-controls`,
  `curve-lt-random`, `curve-wt-random`.
- **Post-fix**: `curve-wt-validate`, `wt-gate`, `m6-wt-ladder`,
  `m6-lt-loop-curve`, `wt-random-vps`, `agg-ablation`, `uncontracted`,
  `pro-lt`, `pro-wt`.

**Audited consequence: no published contrast crosses the boundary.** Every
headline comparison is within one file or within one side of it — the LT
ladder's five arms all ran pre-fix together; `agg-ablation` and the `pro-*`
files each carry their own control; the uncontracted contrast draws its
contracted baselines from `m6-lt-loop-curve` and `m6-wt-ladder`, both post-fix.
The rule going forward is the same as for BLAS: **never pool rows across the
boundary**, and prefer datasets that carry their own control.

Note also that ΔF1 = 0.055 across backends is **larger than most effects the
paper reports** (team − loop = −0.047). Cross-platform pooling would not have
added noise; it would have manufactured or erased findings.

## 14. `run_pc`'s subsample seed is pinned for two arms and not the others (2026-08-29)

`random_agent` and `greedy_ig_lite_agent` call `run_pc(pooled, nodes,
alpha=pc_alpha)`; the seven LLM-bearing agents call it with `seed=seed`. So the
300-row subsample is drawn with `random_state=0` in **every** cell of the two
non-LLM arms, and varies per cell in every other arm. The two arms in the
loop-vs-random headline are therefore not sampled the same way.

**How much that matters, measured rather than argued.** Holding the experiment
selection completely fixed and varying only the PC seed (LT, 8 seeds):

| k | selection seed | mean F1 | sd | range |
|---|---|---|---|---|
| 12 | 0 / 1 / 2 | 0.305 / 0.289 / 0.249 | 0.033 / 0.050 / 0.038 | ~0.10–0.15 wide |
| 30 | 0 / 1 / 2 | 0.380 / 0.440 / 0.378 | 0.038 / 0.038 / 0.039 | ~0.10 wide |

**The subsample alone moves F1 by sd ≈ 0.04 — the size of the effects we
report.** It is noise, not bias, and n=30 reduces the standard error to ≈0.007;
but it is why single-cell comparisons are meaningless here.

Reproducing `random_agent`'s selection exactly and scoring both conditions over
30 seeds:

| k | pinned (seed=0) | varying | delta | paired p |
|---|---|---|---|---|
| 12 | 0.2646 (sd 0.056) | 0.2470 (sd 0.060) | +0.018 | 0.15 |
| 30 | 0.3595 (sd 0.056) | 0.3656 (sd 0.043) | −0.006 | 0.59 |

**No detectable bias**, both below MDE, and the pinned arm's sd is not smaller.
The k=12 sign favours `random`, so pinning makes the "loop beats random at low
k" claim *conservative* rather than inflated.

Left as-is deliberately: changing it now would make the recorded `random` rows
irreproducible from the code (defect 13's own lesson), for a correction measured
at below MDE. Fix it at the next data freeze, re-running the LLM-free `random`
cells, which cost nothing.

## 15. What an adversarial code review found that the audit did not (2026-08-29)

Entry 13/14's audit asked "does this measure what it claims?". A `/code-review`
pass over the branch (60 commits, ~8.5k insertions) asked whether the code is
correct under conditions the sweeps happened not to hit. Different question,
different defects: **all 551 chamber tests passed before and after.**

Fifteen findings; each verified by execution here rather than accepted.

**Touches a published number — one, and it is NOT the LT result:**

- **`_parse_name_list` discards a genuinely-claimed experiment** whose name is
  a substring of a longer co-claimed one. The word-boundary regex above the
  guard already prevents the prefix inflation the guard was written for
  (`_1` does not match inside `_10`), so the guard has no true positives left
  and every firing is a false one. Verified: claiming `_1` and `_10` returns
  only `_10`. **Scope, measured against the real menus: LT has ZERO
  substring pairs**, so `m6-ladder` — the main topology result — is untouched.
  WT has 3 of 28 (`validate_load_in`, `validate_load_out`, `validate_osr_in`),
  so the exposure is the WT `team` arm's `n_contested` and claim split.
  `fan_in_agg` ran LT-only, so the aggregator ablation is clean.
- **`claim_a = list(source_a)[:budget]` truncates in MENU order.**
  `_parse_name_list` returns menu order, which the comment eight lines below
  explains is grouped by variable family — the exact bias the seeded shuffle
  exists to remove — and the comment above confirms the truncation fires
  ("10 + 4 against a 20 budget"). Every seed therefore truncates to the same
  head families. `team` is one of only two resolved negatives (−0.047), so
  **part of that deficit may be harness rather than coordination.** Not fixed
  here: fixing changes behaviour and requires re-running the arm.

**Real defects, measured zero effect on published data (fixed):**

- **`fallback_rate`'s denominator counted provider ATTEMPTS**, not logical
  calls, so a cell reported a *lower* degradation rate the worse the serving
  stack behaved — a harness statistic moving with conditions, the class this
  report exists to catch. Measured impact: **rotation fired 0 times across all
  450 loop cells**, inflation 0.00%, so every published rate is correct.
  `n_llm_calls` now means logical calls (what its name and every recorded
  figure already implied) and `n_llm_attempts` carries the billed count.
- **A dead worker killed the whole sweep.** `fut.result()` was unwrapped inside
  `with ProcessPoolExecutor(...)`, so one OOM-killed worker propagates
  `BrokenProcessPool` through an `__exit__` that calls `shutdown(wait=True)` —
  the same wait-on-exit shape this project already root-caused for
  `ThreadPoolExecutor` — discarding every in-flight cell with no sidecar line.
  A 20-hour sweep would die at hour 12 over one cell.
- **A mis-configured sweep looped forever instead of failing.**
  `_ladder_calibration`'s deliberate "raise rather than extrapolate" was
  swallowed by `run_cell`'s `except Exception` into per-cell error records;
  `done_cell_keys` excludes errored cells so they retry, so every resume
  re-attempted all of them while the message naming the fix sat truncated in
  `error_message`. Configuration faults now raise `SweepConfigurationError`,
  which both sweep paths re-raise.
- **An unlisted model inherited a pin verified for another model.** Harmless
  while the provider order was a preference; since `allow_fallbacks: False`
  (entry 8) it is a hard constraint, so `--model <anything>` either failed
  every cell or ran on an uncertified precision *while the homogeneity test
  passed*, because that test checks the pin and not the model. Unknown models
  now raise. The three we run are named explicitly.
- **`MENU_SIZES` drift was checked only for the uncontracted arm**, though
  `_budget_k_for` converts every sweep's budget fraction to `k` through that
  table for every arm — and the WT dataset changed release inside this branch.
  Now an unconditional equality check.
- **A NaN-variance column escaped both counters.** `> 1e-12` and `<= 1e-12`
  are both False for NaN, so the two lists were not a partition and an all-NaN
  column was dropped with every counter reading 0 — the exact untraceable path
  entry 14's counter was added to eliminate, reintroduced by its own patch.
  Latent only: zero NaNs in either chamber.
- **`validity_warnings` crashed on an empty selection** (`--ladder` on a WT
  file while `--check-chamber` still defaults to `lt`).

**Quality (fixed):** the negotiation prompts were the only builders skipping
`_MAX_MENU_LINES`, and they belong to the arm whose per-call spend
`_ladder_calibration` treats as a fixed overhead; the uncontracted prompt was a
hand-maintained copy of `build_select_prompt`, so the arm's own scope-limit
promise ("identical in every other respect") was enforced by two copies staying
in sync — one `_render_menu` now serves all four builders, verified
**byte-identical on all 12 prompt variants** before and after; a dead branch
after the collinearity filter; O(n²) redundant array conversions in
`select_noncollinear_columns` (hoisted, **not** replaced with `.corr()`, which
would change the arithmetic — entry 10 measures what a 1e-10 perturbation does
to PC — and verified bit-identical on 8 real cells including the WT ones where
collinearity fires); `__all__` sat mid-file and omitted the three ladder arms.

**Still open, deliberately:** `blas_backend` and the `pc_*` fields are stamped
on every row and **read by no analyzer**. Concatenating `m4-pilot` (Accelerate)
with `m6-ladder` (OpenBLAS) and running `--ladder` silently averages two
backends across a 0.055 gap. The rule lives in prose in three documents and
nowhere in code.

## 16. The claim cap never fires, so its bias never happened (2026-08-29)

Entry 15 flagged two team-arm defects and rated the second as possibly
contaminating a published number. **Measured, it does not.** Recorded because
the reasoning that got it wrong is more reusable than the fix.

The argument for contamination was half right. The pool arithmetic is real and
reproduces to three decimals: a full claim leaves no top-up, so it is
`claim/(claim + half the leftover)` of what a scout can see —

| LT budget | predicted | measured `claim_pool_share` | cells |
|---|---|---|---|
| k=6 | ~0.10 | **0.103** | 3/3 |
| k=45 | 23/30 = 0.767 | **0.767** | 6/6 |

So at the top budget the claim really does decide 77% of the scout's pool, and
a menu-order cut of it would have been a family-shaped hole repeated in every
seed. **But `n_claim_truncated` is 0 in all 9 cells at both budgets.** The cap
never runs, because `build_negotiate_propose_prompt` says *"List the {budget}
experiment name(s) you intend to claim, one per line, and no other
commentary"* — and the model complies exactly, returning 23 names for a 23
budget. A real bias channel that nothing ever pushes through.

The estimate that a 240-cell re-run was needed rested on a CODE COMMENT
("measured 10 + 4 against a 20 budget when `claim_a` reached 55 names"), which
describes an earlier state of the pipeline. A comment is a claim about the
past; it is not a measurement of the present. **Cost of checking: 9 cells,
$0.42, ~2h. Cost of not checking: 240 cells and a week of wall time.**

**Consequences:**

- Both entry-15 team fixes are **behaviour-neutral on LT**: the cap has nothing
  to cut, and LT's menu has no substring pairs. So there is no version boundary
  (entry 13) — the new code reproduces the published LT `team` rows, and the
  −0.047 loop-vs-team gap stands as measured.
- **WT is the one place the fixes change behaviour, and there it FIRES.**
  Measured with a dedicated counter (`n_substring_conflicts`) on a 3-cell WT
  probe at k=21: **0, 0, 2** — one cell in three, two shadowed names in that
  cell. So this is not a scope limit, it is a real perturbation of recorded
  data, and the WT `team` re-run became a requirement rather than a
  cleanliness option. (n=3, so the *rate* is barely constrained; what is
  established is that it is non-zero.)

  Exactly three WT names can ever be dropped, and only when a longer sibling
  is claimed in the same response:

  | droppable | shadowed by |
  |---|---|
  | `validate_load_in` | `validate_load_in_mic`, `validate_load_in_current_out` |
  | `validate_load_out` | `validate_load_out_pressure_intake`, `validate_load_out_current_in`, `validate_load_out_mic` |
  | `validate_osr_in` | `validate_osr_intake` |

  They are the short, unqualified member of each family, so the old guard
  biased systematically against the plain load-in / load-out / osr-in
  interventions, replacing each with a random top-up. It also propagates to
  `n_contested`, computed from the same parsed lists — the affected probe cell
  reported 5 contested against 3 for the other two. LT has zero such pairs,
  which is why the defect is structurally impossible there.

  **Re-run launched 2026-08-29 16:55 UTC** on the VPS (`scipy-openblas`,
  matching `m6-wt-ladder`): 150 cells, WT x `team` x k in {7,14,21} x 50 seeds,
  6 workers, `runs/m6-wt-team-rerun.parquet`. The other four WT rungs are
  untouched by both fixes, so replacing only `team` keeps the ladder
  within-version.
- **k=30 is un-measured**, bracketed by k=6 and k=45. The mechanism (a prompt
  that hard-codes the count) does not vary with budget, so the interpolation is
  safe, but say "measured at k=6 and k=45" rather than "at every budget".

**Also confirmed on this run:** provider rotation fired **0 extra attempts**
across 300 calls (`n_llm_attempts == n_llm_calls == 50`/cell), independently
reconfirming that entry 15's fallback-denominator defect was real but inert.
The LiteLLM errors in the log were litellm's own `num_retries`, not our
rotation. A team cell at k=45 is exactly 50 calls: 4 negotiation + 45
selection + 1 reconcile.

**Operational, and self-inflicted:** the preflight took 4h45m of wall clock for
96 minutes of compute because the machine slept — defect 9 exactly, whose own
recorded fix is "run sweeps under `caffeinate -is`" and which I did not apply
when launching. `wall_time_seconds` (`perf_counter`) stayed honest while
`etime` inflated 6x, which is how the stall was diagnosed rather than guessed.
**Launch every local sweep under `caffeinate -is`.**

## 17. A second review found eight defects in one day's fixes (2026-08-29)

Entry 15 was a review of the branch; this is a review of the *repairs*, scoped
to `3c67b32..HEAD` (~1.5k lines, all written the same day). **Eight findings,
four confirmed by execution, and the suite was green at 545 passed
throughout** — none were test-visible.

Two made a guard decorative, which is worse than not having written it:

- **`--allow-mixed-provenance` was a dead flag.** `load_records` threaded it;
  `aggregate_pareto` and `ladder_frame` called the guard bare, and `main`
  calls `aggregate_pareto` on every invocation. The escape hatch the CLI
  documents could not be reached at all.
- **`SweepConfigurationError` was swallowed on the agent-invocation path.**
  The adapter-construction `try` re-raised it; the second `try`, around
  `_invoke_with_timeout`, did not — and `_provider_order_for`'s unpinned-model
  raise fires inside `_CountingLLM.__call__`, i.e. in the second block. So
  `--model <unlisted-id>` produced N identical error records that every resume
  re-attempts: the exact loop the class docstring claims to prevent, defeated
  in the same commit that introduced the class.

Then four mediums and two lows. The ones worth remembering:

- **`n_claim_truncated`'s scout-B term was measured against a list the cap
  never saw** — `uncapped_b` excluded a menu-order slice of `source_a` while
  the cap ran on the shuffled `claim_a`. Same size, different membership.
  Reproduced at seed 1: reported 2 truncated where 1 had been. A counter lying
  by being measured against the wrong list, inside the commit series about
  counters that lie.
- **The broken-pool handler did not do what its comment claimed.** Once the
  pool dies every remaining future raises the same `BrokenProcessPool`, so a
  sweep that OOMed at cell 250 "completed" with 200 fabricated error rows and
  exit 0 — a success-shaped ending for a fraction of a run. It now aborts,
  naming how many cells actually ran; per-cell faults the pool survives still
  become one error record each.
- **`_pc_provenance_snapshot` hardcoded `pc_alpha=0.05`** while `run_cell`
  stamps `sweep.pc_alpha`. Since `pc_alpha` is a provenance column and the new
  guard runs *before* non-ok rows are dropped, one dead worker on a non-default
  alpha would have made the entire Parquet un-analysable — by the guard added
  two commits earlier.
- **The partial-counter check false-alarmed on every frame with a non-LLM
  arm**: `n_selection_fallbacks` is legitimately null for `random` and
  `greedy_ig`, which nonetheless run PC.

**The pattern across all three rounds, and the thing to carry forward.** Three
of these eight are the same defect class fixed elsewhere hours earlier: a
guard that treats **absent / partial / present** as two states instead of
three. It appeared as `dropna().unique()` in the provenance check, as
`fillna(0)` in the counter check, as an all-null column passing both guards,
and as the wrong base list for scout B. Writing the guard is not the hard
part; enumerating the states is.

**Second lesson, procedural:** `/code-review high 3c67b32` was read as "review
that commit", not "review since that commit". The first pass therefore covered
one commit and missed the four that contained the sweep-runner control flow —
the highest-risk surface. Use the range form, `3c67b32..HEAD`, and check the
scope line in the reply before trusting a clean result.

## 18. The substring guard: real defect, measured incidence, undetectable effect (2026-08-30)

**Closes the open item from entry 15.** `_parse_name_list` carried a substring
guard on top of a word-boundary regex that already prevented the collision it
was written for. Every time the guard fired it deleted a *genuine* claim, which
was then replaced by a random top-up. Only WT can trigger it — exactly three
menu names are the short, unqualified member of a family
(`validate_load_in`, `validate_load_out`, `validate_osr_in`) — and only rung 4
parses peer claims, so only WT `team` was exposed.

**What makes this entry worth keeping is the shape of the answer, not the fix.**
The defect was real, its incidence was non-trivial and *budget-dependent*, and
its effect on the reported result was still below MDE. All three had to be
measured; none could be argued.

`n_substring_conflicts` counts what the removed guard would have dropped,
recorded on all 150 re-run cells:

| k | conflicts / cell | picks affected |
|---|---|---|
| 7 | 0.02 | 0.3% |
| 14 | 0.14 | 1.0% |
| 21 | 1.02 | 4.9% |

Incidence rises with budget — more claims, more chances to collide — which is
the signature of a harness fault that **moderates the independent variable**
(entry 4). That is why it could not be dismissed on the 3-cell probe (0, 0, 2)
and had to be re-run rather than argued away.

**Effect, at n=50 per budget:** `team` rose **+0.0059 / +0.0075 / +0.0048** at
k = 7 / 14 / 21. Every one below MDE, **every p ≥ 0.56**. Direction as
predicted (restoring deleted claims should help), magnitude not detectable.
**All three verdicts against the loop are unchanged**: +0.040 R, −0.014 ns,
−0.040 R.

**The generalisable lesson.** A guard that is *redundant* is not harmless. This
one sat behind a regex that already did its job, so it never prevented
anything and only ever deleted true positives — a pure false-positive filter
with no true-positive workload. When removing a defensive check, the question
is not "is it correct?" but "what does it fire on that the layer beneath
already handled?" Where the answer is "everything", the check is subtracting
signal.

### The provenance sub-problem this created

The re-run is stamped (`blas_backend`, `platform_tag`, PC parameters); the
26 Aug `m6-wt-ladder.parquet` it must be compared against predates those
columns — 38 columns against 48. So the contrast crosses the boundary
`require_homogeneous_provenance` refuses, and the analyzer correctly refused
until passed `--allow-mixed-provenance`.

**We did not backfill the stamps.** Writing a value we did not observe is
precisely the failure the guard exists to catch, and it would have left no
trace for the next reader. Provenance was established by evidence instead:

1. **Origin** — the VPS holds `m6-wt-ladder.parquet` at a byte-identical md5
   to the local copy, dated 26 Aug.
2. **Backend now** — VPS reports `scipy-openblas 0.3.34.0.0`, `Linux-x86_64`.
3. **Backend stability across the window** — a 9-cell seeded, LLM-free
   `random` sweep run 30 Aug **reproduces `wt-random-vps.parquet` (26 Aug,
   stamped) exactly: 9/9 on F1 and SHD, max |diff| = 0.000e+00.**

Step 3 is the sharp one. Because PC amplifies a ~1e-10 numerical perturbation
into structural graph differences (entry 10), bit-exact reproduction of a
seeded arm is a strong test of backend identity — far stronger than comparing
version strings, which were identical across the macOS/Linux split that
*did* diverge.

**Method to reuse:** an unstamped legacy file is not disqualified, but its
provenance must be *established* rather than assumed, and **a seeded,
LLM-free arm is the cheapest instrument for establishing it** — nine cells,
no tokens, minutes.

## 19. Our own test suite carried the BLAS dependency it documents (2026-08-30)

Found by CI on PR #89, not by us. `test_warns_on_singular_matrix` passed on
macOS/Accelerate for weeks and **failed on both Linux CI runners** —
`assert np.int64(6) == 0`.

**The mechanism is entry 10, one level up.** The test built a matrix that is
exactly rank 4 of 8 in exact arithmetic (`p=x+y`, `q=y+z`, `r=z+a`,
`s=x+y+z+a`), `cond ≈ 3e17`, and asserted that PC hits the singular fallback
and returns an all-zeros adjacency. Under Accelerate the inversion raises and
the fallback fires. **Under OpenBLAS it does not raise** — it returns a
finite, meaningless inverse, PC proceeds normally, and emits 6 edges.

**Whether a near-singular matrix raises is a property of the linear-algebra
backend, not of our code.** It must therefore never be an assertion. The
original test had encoded a platform-specific numerical outcome as if it were
universal, in a suite whose own register documents that exact hazard.

**The fix is a split, not a tolerance.** The test conflated two properties;
each is now pinned separately and each is backend-independent:

1. `test_pairwise_filter_is_blind_to_higher_order_collinearity` — asserts on
   the correlation structure directly: rank-deficient as a set
   (`matrix_rank < 8`) while the largest pairwise |r| is 0.76, far below the
   0.999 cutoff, so `select_noncollinear_columns` drops nothing. This is a
   claim about *our filter* and is checkable in exact terms. The 0.76-vs-0.999
   margin is wide enough that backend noise at 1e-10 cannot flip it.
2. `test_singular_failure_degrades_to_zeros_and_warns` — **injects**
   `LinAlgError("Singular matrix")` via monkeypatch rather than provoking it,
   then asserts zeros plus the "fell back" marker. This is a claim about *our
   handler*, and injection is the only portable way to reach it.

Both mutation-verified, and each mutation kills exactly one test: breaking
`_LINALG_SINGULAR_PHRASE` fails only (2); dropping
`DEFAULT_COLLINEARITY_THRESHOLD` to 0.5 fails only (1).

**The generalisable rule.** In a suite that depends on floating-point linear
algebra, *never assert that a computation fails.* Failure to converge, failure
to invert, and failure to reach a tolerance are all backend-chosen. Assert the
handler's behaviour by injecting the failure, and assert the mathematical
property (rank, correlation, margin) directly. A test that provokes a
numerical error is testing LAPACK's opinion, not your code.

**Why local runs could not catch this.** Development is macOS/Accelerate; every
sweep of record is Linux/OpenBLAS. The suite was green locally for the entire
life of the branch. **CI is the only place the published platform is
exercised** — which is an argument for pushing early, not for trusting a local
green.

## 20. The coverage manipulation measured strength, not only breadth (2026-08-30)

The first defect in this register that sits in an arm built *to settle a
question raised by the register itself*, and it was caught by checking a design
rather than by a failing test.

**The design.** M7 Phase 1 left open whether distinct-variable coverage
predicts F1. Two LLM-free arms were built to manipulate it directly at LT
k=30: `coverage_max` (one entry per variable, attaining 30) and `coverage_min`
(fattest variables exhausted first, attaining 11). 90 cells, no API spend, and
it resolved cleanly: **+0.069 F1, MDE 0.036.**

**The defect.** On the LT menu all 9 `weak` entries sit on the 9 three-entry
variables — exactly the variables `coverage_min` exhausts first. So the two
arms differ in *two* things at once:

| | variables | weak picks |
|---|---|---|
| `coverage_max` | 30 | 3.2 |
| `coverage_min` | 11 | 9.0 |

Across the 90 cells `n_variables` and `n_weak` correlate at **−0.891**. A weak
intervention perturbs less and carries less signal, so the two channels were
inseparable. Pooled, F1-vs-weak (r = −0.407) fitted about as well as
F1-vs-variables (r = +0.471); and within `random`, where the two partly
decouple (r = −0.355), a multiple regression put the effect on **weak
(−0.0032) rather than variables (−0.0018, wrong sign)**. The sweep announced as
decisive was not.

**The fix and its result.** `coverage_ordered` gained `exclude_strengths`, and
`coverage_max_ms` / `coverage_min_ms` restrict the menu to mid+strong: **15 to
30 variables, zero weak at either end.** 60 further cells.

**The correction ran the opposite way to the usual one.** Removing a confound
normally shrinks an effect. Here it **doubled** it:

| design | span | weak | delta F1 | per variable |
|---|---|---|---|---|
| confounded | 11→30 | 9.0→3.2 | +0.069 | +0.0036 |
| deconfounded | 15→30 | 0→0 | **+0.109** | **+0.0073** |

The confounded design was *understating* breadth, because its narrow arm was
being handed extra weak picks that hurt it less than the missing breadth helped
the wide arm — the two channels partly cancelled.

**Residual imbalance, stated rather than waved away.** The `_ms` pair still
differs in mid/strong mix (18.7/10.3 against 15.0/15.0), so the NARROW arm
holds more `strong`. Measured within `coverage_max_ms`, where the split varies
by seed, the strong channel is not measurably different from mid (slope
−0.002/pick, r = −0.098, n=30). Taking that point estimate at face value would
move the slope from 0.0073 to ~0.0066 and the attribution below from 68% to
62%. It does not change any conclusion.

**Lesson.** A manipulation built from a menu's own structure inherits that
structure's correlations. Before trusting one, tabulate every recorded
attribute of the picks across the arms — not only the attribute being
manipulated. Here one `groupby` over strength counts was the whole diagnosis.

## 21. The seed does not control the LLM — and pinning temperature will not fix it (2026-08-30, corrected 2026-09-01)

**The original finding stands.** `llm_pc_agent` calls `_llm_select_loop(...)`
with no temperature, so the provider default applies. Same seed, same config,
two runs: **F1 0.330 and 0.482**. The seed governs only the fallback RNG and
PC's subsample. Every cell is an independent draw, so seed pairing carries no
information (cross-arm r = −0.03) and unpaired MDEs are the correct ones.

It reaches **arm means**, not only cells: three independent n>=10 estimates of
the same `team` − `llm_pc` contrast span **−0.023 to −0.048**.

**Audit of what was actually pinned (2026-09-01).** Temperature was never
pinned in any DeepSeek run, and the arms are not even mutually consistent:

| arms | temperature |
|---|---|
| `llm_pc`, `one_shot`, `critique`, `planner_reasoner`, `shared_blackboard`, `llm_only` | **unpinned** (no field sent) |
| scouts inside `fan_in_homog`, `fan_in_spec`, `team`, `team_varsplit` | **1.0** (`_SCOUT_TEMPERATURE`) |

Confirmed in the data: M6 ladders, curves, `pro-*`, `uncontracted`,
`agg-ablation` and `m4-pilot` carry **no `temperature` column**; the Phase 2
files carry it and it is **null in all 960 rows**, the encoding for "no field
sent". So the headline loop-vs-fan-in contrast differs in topology *and* in
whether a temperature was specified.

**The correction: pinning temperature would not buy determinism.** OpenRouter
declares no default temperature anywhere in its model or endpoint metadata, so
it was measured — identical prompt, production config (`effort=low`, Parasail
pinned), one selection decision per draw:

| temperature | draws | distinct outputs |
|---|---|---|
| unset | 14 | 6 |
| 1.0 | 11 | 6 |
| **0.0** | **9** | **6** |

**Temperature 0.0 is not deterministic on this endpoint** — six different
experiments chosen in nine draws. That claim needs only two differing draws and
has six distinct in nine, so it is not a power question. The nondeterminism
comes from somewhere else (MoE routing, batching, or the reasoning trace), not
from sampling temperature.

Two consequences, pointing in opposite directions:

1. **Good for the corpus.** If temperature does not drive the variance, the
   loop-vs-scout mismatch above is a difference on paper rather than in
   behaviour, and the confound it threatened is small. It was never a *bias* in
   any case — unpinned sampling inflates variance symmetrically, and the MDEs
   already absorb it.
2. **Bad for the fix.** "Pin temperature so the seed controls the model" — the
   remedy this entry originally implied — **does not work**, and planning around
   it would waste a re-run. Where design *diversity* is needed (register §24's
   `one_shot`, which produced 6 distinct designs in 30 cells), the lever must be
   the prompt — shuffling menu order per seed — not the temperature.

**Power limit, stated because the two claims differ in strength.** "temp=0 is
nondeterministic" is established. "temperature does not change the diversity
level" is NOT — n is 9–14 per condition on a single prompt, enough to rule out
determinism, not enough to bound a modest effect.

**Note the reconciliation with §24.** A single *pick* is highly variable (6
distinct in 9 draws) while a 30-pick *set* is nearly canonical (6 distinct in 30
cells). Both are true: choosing one of 59 is underdetermined, while choosing a
sensible 30 of 59 converges. Diversity at the call level does not imply
diversity at the design level, and it is the design level that governs an arm's
effective sample size.

**For the paper's reproducibility statement**, this joins §10 (BLAS): neither
the seed, nor temperature, nor a pinned provider makes a *cell* reproducible.
Reproducibility in this pillar lives at the level of **arm means over n seeds**,
and that is what should be claimed.

## 22. The collinear-drop rate correlates with the ARM — checked, inert (2026-08-31)

**Shape of the suspicion.** §1's lesson is that a scaffold failure rate which
varies with the experiment's x-axis makes the curve measure the harness. M7
Phase 2 produced a rate that varies with the *arm*, which is the same defect
one axis over. On LT k=6, the fraction of cells where PC dropped a collinear
column is:

| arm | k=6 | k=30 | k=45 |
|---|---|---|---|
| `one_shot` | **0.90** | 0.00 | 0.00 |
| `critique` | 0.37 | 0.00 | 0.00 |
| `shared_blackboard` | 0.30 | 0.03 | 0.00 |
| `llm_pc` (loop) | 0.20 | 0.00 | 0.00 |

`one_shot` is the arm that *loses* at k=6 (−0.059, resolved) and it is also the
arm that trips the degenerate path 4.5x as often as its comparator. If dropping
a column cost accuracy, the entire k=6 result would be an artifact.

**Measured, not argued: the drop costs ~nothing.** Stratifying LT k=6 cells by
whether the drop fired:

- pooled across arms: fired 0.172 vs not-fired 0.182 (n=53 / n=67)
- within `critique`: −0.006 · within `shared_blackboard`: −0.002
- within `llm_pc`: **+0.036** on n=6 — the wrong sign for the confound

A −0.010 pooled penalty cannot produce a −0.059 arm deficit at a 0.90-vs-0.20
rate differential; the implied contribution is under −0.01. `one_shot`'s loss
at LT k=6 is a record effect.

**Why the rate is arm-dependent at all** (mechanism, not defect): at k=6 the
purchased design matrix is tiny, so whether two columns exceed r > 0.999
depends on which six experiments were bought — and *what gets bought* is the
independent variable. The rate collapses to zero by k=30 on LT because more
experiments break the duplication. It stays high on WT at every budget (479 of
600 Phase 2 cells) for the unrelated barometer reason in §13.

**Standing rule this reinforces.** A degeneracy counter is not decoration —
`n_collinear_dropped` existed only because §13 added it, and it is what made
this a five-minute check instead of an unanswerable objection. Keep counting
every path where the harness silently substitutes different behaviour.

**Related:** §1 (rate varying with the x-axis), §13 (the collinear fix and the
pre/post-fix pooling boundary).

## 23. A flat total variance hid two opposing trends (2026-08-31)

**The misreading.** The spread of `random` cells is nearly constant across LT
budgets — sd 0.048 at k=6, 0.042 at k=59 — and on 2026-08-31 that flatness was
read, in-session, as evidence that *which* experiments you buy contributes
little variance at small budgets. The reasoning was: k=M has zero selection
freedom by construction, so its sd is pure PC noise; a similar sd at k=6 must
therefore be mostly noise too.

The premise is right and the inference is wrong, because it assumes PC noise is
budget-independent. `runs/variance-probe.parquet` (3,150 LLM-free PC runs)
crosses the selection seed against the subsample seed and separates them:

| k | sd total | sd PC noise | sd selection |
|---|---|---|---|
| 6 | 0.048 | 0.032 | **0.036** |
| 30 | 0.050 | 0.043 | 0.026 |
| 59 | 0.042 | 0.041 | **0.005** |

Selection variance falls **8x** while measurement noise **rises**, and the sum
happens to stay flat. The choice is worth *more* at k=6 than the flat total
suggested, not less — the error pointed away from a real effect.

**Why this is a register entry and not just a corrected number.** It is the
same failure as §1 one level up: a quantity that looks stable across the
experiment's x-axis, taken as evidence that nothing varies with it, when in
fact two components vary and cancel. A single aggregate statistic cannot
license a claim about its components. The general rule: **before reading a
flat curve as "no effect", check whether the flatness is a sum.**

**How it was caught.** By building the instrument rather than arguing. The
probe cost five minutes and no LLM spend, and it validates itself on a row
whose answer is known by construction: at k=M all 30 "selections" buy the
identical whole menu, and the decomposition returns sd 0.005 without being
told. A method that could not recover that would not be trusted on the rows
where the answer is unknown.

**A second claim killed in the same probe.** The obvious explanation for
rising noise — a fixed 300-row subsample thinning per-experiment coverage as k
grows — was tested at 5x the rows on identical selections and **refuted**:
within-sd moved −5% at k=59 and 0% at k=6. The noise is intrinsic to the
accept/reject cascade (§10), not to the row cap. Recorded because the
mechanism was plausible enough to have been asserted without testing.

**Standing headroom, deliberately not taken**: those same runs show
`max_rows=1500` buys **+0.025 F1 at both budgets**. It is real accuracy for
runtime, but `max_rows=300` is the configuration of record for all 3,441
LLM-bearing corpus cells, so adopting it would fork the pooling boundary the
way the collinear fix did (§13). Stated as known headroom, not applied.

**Related:** §1 (a harness quantity varying with the x-axis), §10 (why PC
converts numerical noise into structural noise), §13 (pooling boundaries).

## 24. `one_shot`'s cells were not independent draws (2026-08-31)

**Found while measuring something else.** A single LLM call with a fixed prompt
re-picks nearly the same design every time — the seed reaches only the fallback
RNG and PC's subsample, not the model. At **LT k=30, `one_shot` produced 6
distinct selections across its 30 cells, one covering 17 of them.** Those 30
rows are ~6 draws of the strategy, not 30, and treating them as 30 independent
observations is pseudo-replication: the cell-level sd is dominated by PC noise
on repeated *identical* buys, so it understates how uncertain the arm's mean is.

The exposure is asymmetric and lands on the claim that can least afford it.
Phase 2's headline is an EQUIVALENCE ("one call matches the loop"), and an
equivalence is only as strong as its bound.

**Re-analysed at the selection level** — one row per distinct buy, so the unit
of independent variation is the design rather than the scoring:

| | cells | selections | effect on the claim |
|---|---|---|---|
| LT k=6, `one_shot` | 30 | 30 | none |
| **LT k=30, `one_shot`** | 30 | **6** | **MDE 0.029 -> 0.051**, delta +0.012 -> −0.000 |
| LT k=45, `one_shot` | 30 | 24 | MDE 0.031 -> 0.033 |
| WT, all budgets | 50 | 30–34 | MDE +0.002 to +0.005 |
| every other arm, both chambers | 30/50 | 29–50 | negligible |

**Every verdict is unchanged.** No contrast flips in either direction, on
either chamber, at any budget. What changes is one bound, and it is the
important one: the LT k=30 equivalence must be reported as **±0.051, not
±0.029**. Against a loop-vs-random gap of +0.055 that bound cannot exclude
"the record is worth nearly as much as selecting at all", so the LT half of the
"record is not load-bearing" claim rests on k=45 and on WT, where the arm's
designs do vary.

**Why the bound cannot be tightened with seeds.** More seeds buy more
*scorings* of the same six designs, not more designs. The fix is selection
diversity — menu-order shuffling per seed, or a pinned non-zero temperature —
and it requires re-running the arm, not re-analysing it.

**A near-miss worth recording.** Averaging each cell over 9 PC subsample seeds
shrinks `one_shot`'s cell sd by **6.6x** (against 1.8x for the loop) and makes
the k=30 contrast read "RESOLVED, `one_shot` better by +0.011". That is the
pseudo-replication amplified, not a discovery: the variance being averaged away
is measurement noise on 6 repeated designs. **Multi-seed averaging is only
sound after clustering by selection**, never as a substitute for it.

**Standing rule.** Any arm that does not draw a fresh design per seed must be
analysed at the selection level, and every arm's distinct-selection count
belongs in the results table. `chosen_experiments` is recorded from M7 onward;
the M6 ladders predate it, so their diversity cannot be audited — noted as a
limitation of those files, not a defect in the contrasts.

**Related:** §21 (the seed does not control the LLM — this is that fact's
consequence for inference, not just for variance), §23 (a variance whose
components must be separated before it can be read).

## 25. Endpoint defaults diverge 130x — but we already pin past them (2026-08-31)

**Recorded as a corrected investigation, because the first version of this
entry was wrong and the correction is the useful part.**

**What was measured.** Probing `glm-5.3-flash` as the cross-vendor candidate,
one identical worst-case selection prompt across four pinned endpoints gave a
**130x spread in reasoning tokens**: Z.AI 6,889 / GMICloud 2,600 / DeepInfra 52,
all returning a valid on-menu name so nothing failed loudly. Reproduced with
raw `curl` — no litellm — and the response's `provider` field confirms the pin
was honoured, so it is neither a client bug nor a routing failure.

**The wrong conclusion, drawn first.** That "we send no `reasoning` parameter,
so each endpoint applies its own default." **False for the pipeline.** Every
production call sets it explicitly: `_SELECTION_REASONING_EFFORT = "low"` and
`_COORDINATION_REASONING_EFFORT = "high"`, and every recorded sweep carries
`reasoning_effort` of `"low"` or `"high,low"`. The 130x belongs to the *probe*,
which sent no parameter; it is not what a sweep does.

**With effort pinned, the endpoints converge** — the same prompt again:

| endpoint | no parameter | `effort=high` | `effort=low` |
|---|---|---|---|
| Z.AI | 4,134 | 2,450 | 163 |
| GMICloud | 2,600 | — | **0** |
| DeepInfra | **78** | 1,605 | 29 |
| Novita | — | — | 28 |

At `low`, where every selection call runs: 0–163 tokens across four endpoints.
At `high`: 2,450 vs 1,605, single draws. **The pipeline was already using the
API correctly on this axis.** `reasoning: {enabled: false}` is rejected
outright — reasoning is mandatory for this model.

**What the probe did legitimately establish:**

1. **Relace is fp4 on BOTH models**, and at $0.071/$0.237 (GLM) and $0.180
   (deepseek) it is among the cheapest endpoints, so price-first routing
   selects it. It was absent from `PROVIDER_PRECISION`; now declared fp4.
2. **`PROVIDER_PRECISION`'s provider-only keying is unsound in principle.**
   `Reka` is **fp4 for deepseek-v4-flash-0731 and fp8 for glm-5.3-flash** — the
   same fact `_provider_order_for`'s own docstring states ("quantization is a
   property of the (provider, model) pair") but the table cannot express. Not
   currently biting: Reka appears in no pinned order. Fix it before adding a
   third model, not after.
3. **A stray-provider audit, previously never enumerated.** 17 of 750 cells in
   `m6-wt-ladder-final` were served by endpoints in no pinned order — Relace
   (8, fp4), OpenInference (10, fp8), DigitalOcean (4, unknown) — because those
   runs predate `allow_fallbacks: False` (§8). Residualised on arm x budget:
   **+0.005 vs −0.000, Welch p = 0.76**, and 13 of the 17 sit in
   `planner_reasoner` k=14, an arm that resolves in neither direction anywhere.
   Not distorting; recorded so the count exists.
4. **One CoreWeave draw burned 30,573 of a 32,768 cap on reasoning and returned
   EMPTY content** — the Together failure mode (§"the provider and the WT
   dataset were both moderators") appearing in a *pinned* provider. That draw
   sent no effort parameter, so it is not the production path; CoreWeave's 900
   recorded cells show `n_selection_fallbacks` of 0.06, identical to every
   other endpoint. Watch it rather than act on it.

**The lesson, which is not the one the first draft drew.** A probe that omits a
parameter the pipeline sets does not measure the pipeline — it measures a
configuration nobody runs, and its most dramatic number is an artifact of the
omission. **Probe the production call path, or state loudly that you did not.**
The genuine defects here were found in the *endpoint metadata* (precision,
pricing, who actually served past cells), not in the reasoning behaviour.

**Related:** §21 (temperature IS unpinned — the real instance of this class),
§8 (`allow_fallbacks`), §"the provider and the WT dataset were both moderators".

## Standing scope limits (not defects)

- **Noise floor** — at k=M there is no selection freedom, so the spread there
  is pure PC noise, and every effect should be quoted against it. LT (k=59):
  sd 0.038, max−mean 0.069. WT `wt_validate_v1` (k=28): sd **0.076**,
  max−mean **0.134**, roughly twice LT's. The 0.065 figure measured on
  `wt_walks_v1` is stale — do not reuse it.
- **WT resolves less than LT at the same n**, which is why the ladder ran at
  n=50 there: WT's dynamic range is 0.107 against MDE 0.031–0.055 at n=30,
  where LT's is 0.238 against ~0.030. At n=50 the achieved MDEs were
  0.031–0.043 and every fan-in contrast resolved.
- **WT's k=7 band is uninformative, not a reversal.** Every arm underperforms
  random there (loop −0.045, team −0.011, all five negative) and the band
  (0.145–0.179) sits just above WT's PC noise floor of 0.134. Report it as
  "below some budget the LLM's selection is worse than chance on WT", not as a
  ladder result. An interim reading of it as "splitting helps at low budget"
  was withdrawn once k=14 and k=21 completed.
- **Conservation compliance is chamber- and budget-dependent.** LT 95.9%; WT
  64.3% overall (193/300 fan-in cells), degrading 91% → 61% → 41% across
  k=7/14/21. `verify()` caught every overrun, so this is a statement about
  provisioning — the WT c95/a95 figures, calibrated from 27 gate cells,
  under-predict cost as budget grows — and NOT about the mechanism. Report the
  two separately or a reader concludes the framework failed.
- **WT `team` carries no conservation result.** `_C95_NEGOTIATE` was never
  isolated for WT, so those 150 cells run on LT's figure with
  `conservation_certified` forced to None. They contribute accuracy only.

- **`overlap_frac` is structurally 0.0 for rung 4** — pools are disjoint by
  construction.
- **The aggregator's reconcile output is discarded — and that is now measured,
  not assumed** (2026-08-27, `runs/agg-ablation.parquet`). The `fan_in_agg`
  ablation gives the aggregator authority over the pooled set: F1 0.3290
  honored vs 0.3259 discarded, delta +0.0031 against an MDE of 0.0305,
  Welch p=0.78. The diagnostics are the finding rather than the delta —
  across 30/30 cells the aggregator **dropped nothing, hallucinated nothing
  and never returned an empty answer**. Given the power to change the pooled
  set it reproduces the union verbatim, so the Python dedup is a *faithful*
  implementation of what the LLM aggregator actually does. The negative
  fan-in result is therefore not an artifact of a null aggregator.
  The paper still must not claim the aggregator *improves* the result.
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

## Audited clean (2026-08-29)

A register of defects is survivorship-biased: it lists what went wrong and is
silent on what was checked and held. These were checked in the same pass that
produced entries 13 and 14, and each is a threat a reviewer can reasonably
raise.

- **The identifiability ceiling is not binding.** PC returns a CPDAG and we
  report undirected edges in both directions, so a reader will ask how much F1
  that convention forfeits. Computed by taking each ground-truth DAG to its
  CPDAG (`dag2cpdag`) and scoring it through our own converter and scorer: **LT
  has zero undirected edges in its CPDAG — it is fully identifiable, ceiling F1
  = 1.000, SHD = 0.** WT has exactly one, ceiling F1 = 0.988. The observed
  saturation at F1 ≈ 0.42 is therefore **not** a Markov-equivalence artifact.
  It is a property of PC on pooled interventional data, which is the fixed,
  uniform inference step every arm shares.
- **Budget matching is exact, and now verified from the data rather than from
  the code.** `build_fan_in_graph` splits as `ceil(k/2) + k//2`, which sums to
  `k` for odd budgets too (k=45 → 23+22, not 22+22). Solving
  `distinct = |A| + |B| − shared` against the recorded `overlap_frac` on all
  270 fan-in cells gives **max residual 0.00e+00 and zero non-integer shared
  counts**, so both scouts spent their full allocation in every cell.
- **No exclusion bias.** All 450 LT-ladder and 750 WT-ladder cells have
  `status == "ok"`; no cell was dropped from any mean.
- **Fallback contamination cannot explain any gap.** Mean `rng.choice`
  fallbacks per cell peak at 0.33 (`team`, k=45) — 0.7% of selections. It does
  rise with k in `team`, the entry-1 pattern, but two orders of magnitude too
  small to move a 0.047 effect.
- **The pooled-interventional design does not penalise exploration.** Dropping
  the `intervention` column makes the pooled mixture an unobserved common
  cause, so diversity could in principle *hurt* accuracy and thereby punish the
  fan-in arms for exploring. Measured within every arm × budget cell, F1
  correlates **positively** with `n_experiments_distinct` (r = +0.11 to +0.32),
  never negatively. The mechanism is not operating.
- **Menu truncation never fires.** `_MAX_MENU_LINES = 200` against a 59-entry
  LT menu — unlike the `> 30` threshold with zero margin that this repo shipped
  once before.
- **No duplicate cell keys, and ground truth is constant** (57 LT / 42 WT
  edges) across every headline file; one `model_id` per file.

**Instrumentation gap CLOSED (same day):** `run_pc` dropped zero-variance
columns silently — no logger call, so unlike PC degeneracy and collinear drops
it had no counter. `n_zero_variance_dropped` now records it, via a third
handler on the same logger; the three markers are asserted mutually exclusive
by driving all three paths, not by comparing strings.

What it shows, measured on `random` seed 0 immediately after wiring:

| chamber | | | | |
|---|---|---|---|---|
| LT (38 nodes) | k=1: **18** | k=3: 15 | k=12: **9** | k=59: 1 |
| WT (32 nodes) | k=1: **20** | k=7: 14 | k=14: **9** | k=28: 0 |

So at the low end of our budget curves **a quarter of the graph is answered by
zero-padding rather than by inference** — 9 of 38 LT nodes at k=12, 9 of 32 WT
nodes at k=14 — and that was previously invisible. This is not a defect and not
a confound, but NOT for the reason first written here. An earlier draft said
"every arm at one budget faces the same padding", which is false and checkable:
at a fixed k each arm buys a different set of experiments, so the set of
unperturbed — hence constant, hence padded — columns differs by arm. That is
precisely why the counter is recorded per arm. The correct argument is
MEDIATION: padding differences are *caused by* selection quality, and
activating more variables is what good selection does, so the path lies on the
causal route being measured rather than beside it. It is a **decomposition** the budget curve could not
otherwise support: how much of the graph PC was asked about, versus how well it
answered. Deliberately kept out of `contaminating` and out of the `rates` tuple
that drives warnings — its rate MUST fall with budget, so warning on that would
fire on every valid sweep, the trap `conservation_fail_rate` is already kept out
of. A frame lacking the column gets a `NOT RECORDED` notice worded distinctly
from `UNMEASURED`, so the two are not conflated.

Two side findings from the same measurement:

- **LT has a mild structural recall ceiling from subsampling, not from data.**
  At the full 59-experiment pool no LT column is constant; in the 300-row
  subsample `diode_vis_2` is, costing **1 of 57 edges — a recall ceiling of
  0.982 at k=59**. Negligible next to WT's barometer ceiling of 0.786, and it
  moves with the subsample seed (see entry 14).
- **The path never fires on either chamber below the handler.** An earlier
  measurement that reported zero drops everywhere was wrong for a harness
  reason worth recording: the probe imported the pipeline via
  `sys.path.insert("evaluation")`, so the module registered as
  `chamber_pipeline.inference` while `run_cell` attaches its handlers to
  `evaluation.chamber_pipeline.inference`. **The handlers were never attached
  and every counter read 0** — including the collinear one, which we already
  know fires 3 columns on WT. A counter that reads zero because nothing is
  listening looks exactly like a clean run.

## How to add an entry

1. Measure the effect on the result — do not assert it. Residualise on
   arm×budget before claiming a difference is real — the fp4 contamination
   looked significant at p<1e-4 raw and was pure arm×budget composition.
2. Add the counter at source and surface it in `harness_validity_report`.
3. Write the test against the *behaviour*, then mutate the implementation and
   confirm the test fails. Two tests in this repo passed against a deliberately
   broken implementation before this step was added.
