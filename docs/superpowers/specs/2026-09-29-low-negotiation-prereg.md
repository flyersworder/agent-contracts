# Pre-registration: do the team topologies still lose with every call at low effort?

**Registered 2026-09-29, before any cell below was run.** Code: branch
`low-negotiation-rerun` (`--coordination-effort`, the environment-carried
override `CHAMBER_COORDINATION_REASONING_EFFORT`). Output
`runs/lownego-lt30.parquet`.

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

- **Arms, interleaved** (`iter_sweep_cells`): `llm_pc` (same-day loop control),
  `team`, `team_varsplit`. The paper's main setting: LT, k=30 (budget fraction
  30/59), DeepSeek `openrouter/deepseek/deepseek-v4-flash-0731`, n=30 per arm
  (seeds 0-29), 90 cells, matching `m7-varsplit`.
- **Effort:** `--selection-effort low --coordination-effort low`. Every LLM call
  in every arm at `low`. Checked per cell: `reasoning_effort == "low"`.
- **Run** on the VPS with the current pinned provider order, cell timeout
  7200 s, 8 workers. Re-scored with `rescore.py` on the VPS (OpenBLAS, the
  corpus backend) at 9 PC seeds, caps 300 and 1500; clustered by distinct
  design.

**Before launch, in this order:** (1) re-probe the DeepSeek provider order and
price for `flash-0731`; (2) cost probe, one cell per arm on the VPS, and
re-estimate from those cells; confirm each records `reasoning_effort="low"`
and report conservation certification. Commit this file before (1).

**Estimated cost** from `m7-varsplit` (Baidu-first routing, since repriced):
$0.021 / $0.027 / $0.027 per cell, about $2.25; current routing may differ, so
the probe decides.

## Analysis and decision rules (on the interval, per the project rule)

Contrasts are same-day, from this run only. 95% Welch CI of the difference in
mean directed-edge F1 over distinct designs; MDE in the unequal-n form.
Reported at **both caps**; **300 rows is primary** (the paper's main setting).

- **P1 (the headline).** `team - loop`.
  CI entirely below 0 -> the team loses with every call at `low`: the paper's
  main claim holds under matched effort. CI contains 0 -> not established under
  matched effort; the paper scopes the team's loss to high-effort negotiation
  and says so. CI above 0 -> falsified; the headline changes and is reported.
- **P2 (the repair).** `team_varsplit - team`: CI entirely above 0 -> the
  repair holds at matched effort.
- **P3 (no win).** `team_varsplit - loop`: falsified if the CI excludes 0 on the
  positive side.

**Predictions, written before running:** P1 below 0, P2 above 0, P3 holds (no
win) -- i.e. the corpus pattern (team - loop -0.051, varsplit - team +0.044 at
300 rows on the `m7-varsplit` day) survives matched effort.

**Descriptive, no verdict:** distinct variables per arm and the coverage gaps
(team - loop, varsplit - team); `overlap_frac`; distinct designs per arm;
selection fallbacks; certification. The effect of negotiation effort itself --
this run's team against the corpus team at `high` -- is a **cross-day**
comparison (different day, routing and price) and is reported as such, for
coverage and F1, never as a verdict.

## Pre-committed use in the paper

Whatever P1-P3 show, this run becomes the paper's main-setting evidence for the
topology contrasts at LT k=30, with every call at `low`, and the setup states
the effort of every call kind. The corpus's high-negotiation cells become a
robustness check, labelled with their effort. The other budgets in Figure 2
keep their high-negotiation team points, disclosed as such, unless re-run.
