# Pre-registration: do the team topologies still lose with every call at low effort?

**Amended 2026-09-29, before any cell ran: scope widened from LT k=30 to all six
Figure 2 budgets (the co-author's main claims rest on all of them); GLM is not
re-run and keeps its high-effort negotiation, disclosed.**

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

## Pre-committed use in the paper

Whatever P1-P3 show, this run becomes the paper's evidence for the topology
contrasts at all six budgets (Figure 2's team and varsplit points and every
table row that uses them), with every call at `low`, and the setup states
the effort of every call kind. The corpus's high-negotiation cells become a
robustness check, labelled with their effort.
