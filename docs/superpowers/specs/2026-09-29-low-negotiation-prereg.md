# Pre-registration: do the team topologies still lose with every call at low effort?

**Amended 2026-09-29, before any cell ran: scope widened from LT k=30 to all six
Figure 2 budgets (the co-author's main claims rest on all of them); GLM is not
re-run and keeps its high-effort negotiation, disclosed.** **Amended again
2026-09-29 15:09 (commit e5a83eb), before any GLM cell ran: GLM LT k=30 is re-run too (see
Amendment 2).** **Amended a third time 2026-10-03, before any cell of it ran:
GLM WT k=14 is re-run too (see Amendment 3).**

**Registered 2026-09-29, before any cell below was run.** Code: branch
`low-negotiation-rerun` (`--coordination-effort`, the environment-carried
override `CHAMBER_COORDINATION_REASONING_EFFORT`). Outputs
`runs/lownego-lt.parquet`, `runs/lownego-wt.parquet`.

## What prompted it

A co-author found that every multi-agent cell in the corpus records
`reasoning_effort = "high,low"`: all selection calls ran at `low` (as the loop's
do), but the coordination calls -- negotiate, revise, reconcile -- ran at
`high`. The value was pinned on 2026-08-23 (`ac386f1`) as a drift fix, because
those calls were the only ones left tracking the provider default; nobody
checked it against the loop, which makes no coordination call. So the paper's
main contrasts compare a loop at `low` with teams that deliberate at `high` in
one stage. Every setting is pinned in code and recorded per cell; it was never
disclosed in the paper. **No team cell has ever run with coordination at `low`.**
Whether `high` helps or hurts the team is unknown: more deliberation could
split the variables better, or differently.

## Design

- **Arms, interleaved within each budget** (`iter_sweep_cells`): `llm_pc`
  (same-day loop control), `team`, `team_varsplit`. DeepSeek
  `openrouter/deepseek/deepseek-v4-flash-0731`, the paper's main vendor.
  Budgets as the corpus fractions: **LT k = 6, 30, 45** (6/59, 30/59, 45/59),
  n=30 per arm, 270 cells, `runs/lownego-lt.parquet`; **WT k = 7, 14, 21**
  (0.25, 0.50, 0.75), n=50 per arm, 450 cells, `runs/lownego-wt.parquet`.
  720 cells, seeds 0..n-1, matching the corpus n. `team_varsplit` at LT k=45
  uses the existing infeasible-split repair, reported with its repair rate.
- **GLM is not re-run.** Its team and varsplit cells (LT k=30 only) keep
  high-effort coordination and are labelled so wherever they appear.
- **Effort:** `--selection-effort low --coordination-effort low`. Every LLM call
  in every arm at `low`. Checked per cell: `reasoning_effort == "low"`.
- **Run** on the VPS with the current pinned provider order (CoreWeave,
  DeepInfra, Parasail, Baidu; list prices re-checked 2026-09-29, unchanged
  since 2026-09-21), cell timeout 7200 s, LT and WT as two parallel sweeps.
  Re-scored with `rescore.py` on the VPS (OpenBLAS, the corpus backend) at 9 PC seeds, caps 300 and 1500; clustered by distinct
  design.

**Before launch, in this order:** (1) re-probe the DeepSeek provider order and
price for `flash-0731`; (2) cost probe, one cell per arm per budget (18
cells) on the VPS, and re-estimate from those cells; confirm each records
`reasoning_effort="low"` and report conservation certification. Commit this file before (1).

**Estimated cost**, extrapolated from the Figure 2 fill on the same routing
($8.60 for 330 cells, LT k=45 the dearest): about **$20** ($15-30) and
12-15 h. The probe decides.

## Analysis and decision rules (on the interval, per the project rule)

Contrasts are same-day, from this run only. 95% Welch CI of the difference in
mean directed-edge F1 over distinct designs; MDE in the unequal-n form.
Reported at **both caps**; **300 rows is primary** (the paper's main setting).

- **P1 (primary): `team - loop` at LT k=30, 300 rows.**
  CI entirely below 0 -> the team loses with every call at `low`: the paper's
  main claim holds under matched effort. CI contains 0 -> not established under
  matched effort; the paper scopes the team's loss to high-effort negotiation
  and says so. CI above 0 -> falsified; the headline changes and is reported.
- **P2 (the repair): `team_varsplit - team` at LT k=30:** CI entirely above 0
  -> the repair holds at matched effort.
- **P3 (the headline, every budget): no multi-agent arm resolves above the
  same-day loop.** Falsified if the CI of `team - loop` or `team_varsplit -
  loop` excludes 0 on the positive side at any of the six budgets, at either
  cap. If it fails anywhere, the abstract's sentence is scoped to the budgets
  where it holds and names the exception.
- **Secondary (same rules as P1/P2, reported, no bearing on P1):** LT k=45 and
  WT k=21, the other budgets where the team falls measurably short of the loop
  in coverage.
- **P4 (the denominator):** loop - random at WT k=7, both caps, against the
  existing random designs on the same backend; a multi-agent win over a loop
  that is itself below random is reported as such, not as a team advantage.

**Predictions, written before running** (the corpus pattern, 300 rows):
`team - loop` below 0 at LT 30 (corpus -0.051), LT 45 (-0.016) and WT 21
(-0.029); ties at LT 6, WT 7 and WT 14; `team_varsplit - team` above 0 at LT 30
(+0.044); P3 holds everywhere.

**Descriptive, no verdict:** distinct variables per arm and the coverage gaps
(team - loop, varsplit - team); `overlap_frac`; distinct designs per arm;
selection fallbacks; certification. The effect of negotiation effort itself --
this run's team against the corpus team at `high` -- is a **cross-day**
comparison (different day, routing and price) and is reported as such, for
coverage and F1, never as a verdict.

## Amendment 2 (2026-09-29 15:09 (commit e5a83eb), before any GLM cell ran): GLM re-run

The DeepSeek sweep is in progress (LT 144/270, WT 193/450 at 15:06); no GLM
cell under this registration has run. **Scope widened to GLM**, so that no
team result in the paper keeps high-effort negotiation. This supersedes "GLM is
not re-run" above; nothing else in the DeepSeek design changes.

- **Design:** `openrouter/z-ai/glm-5.3-flash`, LT k=30 (30/59) only -- the one
  budget the corpus has GLM team data for. Arms `llm_pc`, `team`,
  `team_varsplit`, interleaved, n=50 per arm (seeds 0-49, the corpus n), 150
  cells, `runs/lownego-glm-lt.parquet`. `--selection-effort low
  --coordination-effort low`, timeout 7200 s, 4 workers, on the VPS beside the
  DeepSeek sweeps. `one_shot` makes no coordination call and is not re-run.
- **Routing:** the pinned `_GLM_PROVIDER_ORDER` (GMICloud, Novita, Z.AI),
  unchanged since the corpus run of 2026-09-12. Re-probed 2026-09-29: all three
  up and fp8; output prices $0.30 / $0.28 / $0.50 per M (were $0.375 / $0.44 /
  $0.50). Different providers from DeepSeek's CoreWeave, so the two runs do not
  share endpoint throughput.
- **Before launch:** a 3-cell probe (one per arm), checked for
  `reasoning_effort == "low"` and certification.
- **Cost:** about $0.50 (corpus GLM cells $0.0025-0.0031); about 2 h.
- **Analysis:** as P1 and P2, same-day, on the interval, both caps, re-scored
  on the VPS, clustered by distinct design. **P1-GLM:** `team - loop`, 300
  rows. **P2-GLM:** `team_varsplit - team`, 300 rows. Both 1500-row contrasts
  reported beside them. Secondary to the DeepSeek P1, which stays primary.
- **Predictions, from the GLM corpus at `high` negotiation** (VPS re-score,
  Welch 95% CI, 50 designs per arm): at 300 rows `team - loop` -0.009
  [-0.021, +0.004] and `varsplit - team` +0.007 [-0.004, +0.018], both ties;
  at 1500 rows -0.035 [-0.049, -0.020] and +0.029 [+0.015, +0.043], both
  resolved. Distinct variables: loop 25.2, team 22.3, varsplit 24.8. We
  predict the same pattern at `low`: ties at 300 rows, a team loss and a
  varsplit repair at 1500 rows. No GLM multi-agent arm resolves above the
  same-day loop at either cap (P3 extended to GLM).

## Pre-committed use in the paper

Whatever P1-P3 show, this run becomes the paper's evidence for the topology
contrasts at all six budgets (Figure 2's team and varsplit points and every
table row that uses them), with every call at `low`, and the setup states
the effort of every call kind. The corpus's high-negotiation cells become a
robustness check, labelled with their effort.

## Amendment 3 (2026-10-03, before any cell of it ran): GLM WT k=14 re-run

**The omission.** Amendment 2 limited the GLM re-run to LT k=30, calling it
"the one budget the corpus has GLM team data for". That was wrong: the corpus
also holds 50 GLM `team` cells at WT k=14 against 50 GLM loop cells
(`xv-glm-wt`, 2026-09-12, coordination at `high`), and the paper's
cross-vendor figure used them. Found during the paper swap, 2026-10-03. It is
the only team result in the main paper that still rests on high-effort
negotiation; this amendment closes it, as Amendment 2 intended.

- **Design:** `openrouter/z-ai/glm-5.3-flash`, WT k=14 (budget fraction 0.50),
  arms `llm_pc` and `team`, interleaved, n=50 per arm (seeds 0-49, the corpus
  n), 100 cells, `runs/lownego-glm-wt.parquet`. `--selection-effort low
  --coordination-effort low`, timeout 7200 s, 4 workers, on the VPS. Code
  unchanged since Amendment 2 (the sweep path; `rescore.py` gained only the
  separate `pc_exo` estimator).
- **Routing:** the pinned `_GLM_PROVIDER_ORDER` (GMICloud, Novita, Z.AI),
  unchanged. Amendment 2's replication found GMICloud's negotiation calls
  reasoning 4-6x longer one day later at the same pinned `low`; the loop
  control runs the same day, interleaved, and negotiation tokens are reported
  against the 2026-09-29 GLM run.
- **Before launch:** a 2-cell probe (one per arm), checked for
  `reasoning_effort == "low"` and certification.
- **Cost:** about $0.10 (corpus GLM WT cells: $0.023 for 50 loop, $0.041 for
  50 team); about 1 h.
- **Analysis:** same-day, on the interval, both caps, re-scored on the VPS,
  clustered by distinct design, Welch 95% CI, unequal-n MDE. **P1-GLM-WT:**
  `team - loop` at 300 rows; the 1500-row contrast reported beside it.
- **Prediction, from the GLM corpus at `high` negotiation** (re-score of
  `xv-glm-wt`, Welch 95% CI, 50 designs per arm): `team - loop` +0.006
  [-0.008, +0.019] at 300 rows and +0.003 [-0.012, +0.018] at 1500, both ties
  (MDE 0.019 / 0.021). Distinct variables: loop 12.4, team 12.2. We predict a
  tie at both caps at `low`: the interval contains zero, and `team` does not
  resolve above the same-day loop (P3 extended).
- **Use in the paper, pre-committed:** whatever it shows, this run replaces
  the GLM WT k=14 `team - loop` row of the cross-vendor figure and the
  supplement's GLM wind-tunnel table. The `xv-glm-wt` high-effort cells are
  then no longer used in the paper.
