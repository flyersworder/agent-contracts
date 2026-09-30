# Pre-registration: is negotiation effort inert?

**Registered 2026-09-30, before any cell below was run.** Code: branch
`low-negotiation-rerun` as deployed (`--coordination-effort`, `--seed-start`),
no change. Outputs `runs/negoeffort-high.parquet`, `runs/negoeffort-low.parquet`.

## What prompted it

The all-low re-run (`2026-09-29-low-negotiation-prereg.md`) and the corpus
(team negotiation at `high`) differ by at most about 0.015 F1 and 0.8 distinct
variables per arm, the same size as the drift of the loop, whose settings did
not change (up to 0.016 F1, 0.5 variables). At LT k=30 the corpus team scored
0.377 / 0.372 (300 / 1500 rows, 22.9 variables) and the all-low team 0.384 /
0.386 (23.7 variables). That comparison is cross-day and cannot decide
anything. This run makes it a same-day verdict.

## Design

- **One arm, two settings:** `team`, DeepSeek
  `openrouter/deepseek/deepseek-v4-flash-0731`, LT k=30 (30/59),
  `--selection-effort low` in both; `--coordination-effort high` in one sweep
  and `low` in the other. **n=50 per setting, seeds 100-149 in both.**
  100 cells.
- **Same day, same window:** effort is set per process, so the two settings
  run as two sweeps launched together on the VPS, 6 workers each, same
  routing (the pinned `_FLASH_PROVIDER_ORDER`: CoreWeave, DeepInfra, Parasail,
  Baidu; re-checked 2026-09-30: the first three fp8 and up, Baidu degraded).
  Neither setting gets its own block of time.
- **No loop arm:** the contrast is between the two team settings.
- **No 1-cell probe:** both code paths ran in the last 24 h (`high` in the
  corpus, `low` in the all-low sweep). Instead the first finished cells of
  each sweep are checked for status and `reasoning_effort` (`"high,low"` and
  `"low"`).
- Re-scored on the VPS (OpenBLAS) at 9 PC seeds, caps 300 and 1500, clustered
  by distinct design.
- **Cost** about $5, about 5-6 h.

**Power.** Team F1 sd at LT k=30 is 0.0275 (300 rows, all-low run), so at
n=50 per setting the 95% CI half-width is about 0.011 and, if the true effect
is zero, the CI falls inside ±0.02 with about 90% probability.

## Analysis and decision rules (on the interval)

**Primary: `team@high - team@low`, mean directed F1 over distinct designs,
300 rows, 95% Welch CI** (`lownego_analysis.welch`). Equivalence margin
**±0.02** (about the MDE at n=30, and half the team's registered loss to the
loop, 0.040).

- CI entirely inside [-0.02, +0.02]: **inert.** Negotiation effort does not
  change the team's F1 by more than 0.02. If the CI also excludes 0, the
  sign is reported as an effect smaller than the margin.
- CI entirely outside the margin on one side (lower bound above +0.02 or
  upper bound below -0.02): **effort matters**, direction reported.
- Otherwise: **not established.**

**Secondary (reported, no bearing on the primary):** the same at 1500 rows;
distinct variables `high - low` (Welch); negotiation tokens by call kind per
setting. **Manipulation check:** the `high` sweep must show more negotiation
tokens than the `low` one. If it does not, the run did not manipulate
anything and no verdict is drawn.

**Prediction, written before running:** inert at both caps; point estimate
between -0.015 and +0.005 (the cross-day difference was high - low = -0.007
at 300 rows and -0.014 at 1500); distinct variables within 1 of each other.

## Use in the paper

If inert: the setup states that coordination calls ran at `high` in the
corpus and `low` in the re-run, and that the setting is inert by a same-day
measurement. If effort matters: the all-low re-run stays the evidence of
record and the effect of negotiation effort is reported as a result.
