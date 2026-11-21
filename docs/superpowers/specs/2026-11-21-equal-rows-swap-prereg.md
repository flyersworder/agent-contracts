# Pre-registration: the swap intervention behind the equal-rows result

Written 2026-11-21, before any cell of `rebuttal_swap.sh` ran. AAMAS 2027
rebuttal, reviewer xQHG, point C3.

## What is being tested

The equal-rows re-score (`rebuttal_equal_rows.sh`) found that on the wind
tunnel at k=21 with 1500 rows, the coverage rule's lead over random falls from
+0.059 under the uniform cap to +0.004 under an equal-rows cap. A regression on
the 230 non-rule lists (`rebuttal_equal_rows_diagnose.py`) attributes two
thirds of that to two menu entries: `validate_hatch_mic` (4000 rows) and
`validate_load_out_pressure_intake` (3000 rows), repeat entries on two hub
variables. The reading: under the uniform cap they swamp the pool, so skipping
them, as the rule mostly does, paid; under equal rows they are useful second
looks at hub variables, so skipping them costs.

That reading is correlational. This is the direct test.

## Design

- The 50 rule lists at WT k=21 (`runs/rebuttal-eqrows-src.parquet`).
- `big`: each of the two big entries a list lacks replaces, in position, a
  one-edge sensor-setting entry (one of the eight OSR entries other than
  `osr_ambient`) drawn with seed 20261121. Budget stays 21.
- `small` (size control): the same positions receive a SMALL repeat on the
  same hub variable (`hatch_rpms`, else `hatch_pressures`; `load_out_mic`,
  else `load_out_current_in`, else `load_out`).
- `original`, `big`, `small` scored with plain PC, 9 seeds, under the uniform
  and equal-rows caps, at 1500 and 300 rows, all on the VPS in this run.
- Primary sample: the 27 lists that lack both big entries. Secondary: the 47
  that lack at least one.
- Contrast: paired by list, mean difference with a 95% t interval.

## Predictions (1500 rows, primary sample)

Point predictions are the regression's coefficients applied to the swap.

| contrast | uniform | equal rows |
|---|---|---|
| P1 big − original | −0.046 | +0.056 |
| P2 big − small | −0.045 | +0.045 |
| P3 small − original | 0.00 | +0.01 |

Decision rule, on the interval and not on a threshold:

- **Confirmed** if P1's interval lies below zero under uniform AND above zero
  under equal rows. The rebuttal sentence "Cause: ..." stands.
- **Size confirmed** if P2's interval lies below zero under uniform: the harm
  under the uniform cap belongs to the entries' row counts, not to buying a
  repeat on those variables.
- **Refuted** if P1's uniform interval contains or lies above zero. The
  swamping story is wrong and the rebuttal sentence is cut back to a
  description of what changed.
- If P1 holds under uniform but not under equal rows, the rebuttal keeps
  "skipping them paid under uniform sampling" and drops any claim that they
  help under equal rows.

At 300 rows (secondary, reported whatever it shows): big − original −0.039
under uniform and about zero (+0.002) under equal rows.

## Limits stated in advance

- One chamber, one budget, one estimator. It tests why the rule loses its
  lead there and says nothing about other arms.
- The swap changes two things at once relative to `original`: it adds a hub
  repeat and removes a one-edge variable. `small` separates entry size from
  that; it does not separate adding from removing.
