# Chamber Pillar: Results

The canonical record of every chamber-pillar experiment and what it showed.
Results live here rather than in `claude.md`, which is project memory loaded
into every session and should stay instructions plus status.

**Companions.** `docs/chamber-harness-validity-register.md` records the sixteen
harness defects that each changed or could have changed a result — read it
before trusting any number here. `docs/causal_chamber_validation_plan.md` is
the experiment plan; `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md`
is the ladder's design spec.

**Corpus as of 2026-08-28**: 2,050 cells, **$90.80**, **zero errored cells**,
across two chambers and two models.

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

## M6 WT LADDER COMPLETE (2026-08-26): the topology result replicates

`runs/m6-wt-ladder.parquet` — **750/750 cells, 750 ok, 0 errors, 0 PC
degeneracies**, 21.3h wall on the VPS, **$11.64**, 127.7h active compute
across 6 workers. Grid: 5 rungs x k in {7,14,21} (k/M = 0.25/0.50/0.75 on
WT's 28-experiment menu) x 50 seeds. n=50 rather than 30 because WT compresses
effect sizes ~2.2x versus LT.

**Headline: no multi-agent topology beats the single sequential loop on
EITHER chamber, at any budget where selection has signal.**

| rung | LT k=30 | WT k=14 | WT k=21 |
|---|---|---|---|
| ensemble `fan_in_homog` | −0.079 **R** | −0.048 **R** (p=0.0008) | −0.052 **R** (p=0.0004) |
| roles `fan_in_spec` | −0.051 **R** | −0.044 **R** (p=0.0025) | −0.049 **R** (p=0.0004) |
| chain `planner_reasoner` | +0.011 ns | +0.022 ns | −0.024 ns |
| team | −0.047 **R** | −0.021 ns | −0.045 **R** (p=0.0024) |

Two chambers, different graphs (38 nodes/57 edges vs 32/42), different menus
(59 vs 28), different sample regimes (1,000 vs ~840 rows/experiment) — same
ordering. **This is the external-validity answer to "everything rests on LT's
single graph".** The chain is the only multi-agent rung that never resolves as
worse, consistent across both chambers.

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

## Paper readiness (assessed 2026-08-28)

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

1. **The negative is carried by middle budgets.** Nothing resolves at LT
   k=45, and WT k=7 resolves the *other* way (team beats loop, +0.034
   RESOLVED). Both are defensible — k=45 has no selection left to perform,
   and at WT k=7 the loop loses to random so it is a degenerate baseline —
   but the ladder table is in the paper too, and as currently worded the
   table contradicts the sentence. **Answerable in prose; must be answered.**
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

### Method as a contribution

Twelve harness defects are recorded in the register, several worth reporting
rather than quietly fixing because they generalize to anyone benchmarking LLM
agents: a scaffold failure rate that varies with the experiment's x-axis
(§1); the linear-algebra backend determining a seeded result (§10); and five
separate tests that certified our REQUEST rather than what ran (§3-4, §7, §8,
§10, §12).
