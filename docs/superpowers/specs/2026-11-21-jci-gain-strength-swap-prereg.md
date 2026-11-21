# Pre-registration: does intervention strength on the light sources drive the JCI-PC gain?

Written 2026-11-21, before any cell of `rebuttal_jci_swap.sh` ran. AAMAS 2027
rebuttal, reviewer xQHG, point C4: the coverage rule gains least from JCI-PC
although it covers the most variables.

## What is known before the run

Table 5 of the paper, light tunnel, 5000 rows, gain = F1(JCI-PC, one indicator
per regime) − F1(PC). At k=30 the rule gains +0.045, resolved below every
other arm (loop +0.080, varsplit team +0.078, team +0.093, random +0.071). At
k=45 it is not the lowest (team +0.036). So the question is a k=30 question.

`rebuttal_jci_gain_diagnose.py`, on the lists behind Table 5:

- Not coverage, not repeats: among the 120 non-rule lists at k=30 the gain's
  correlation with distinct variables is −0.05 and with repeated variables
  +0.04.
- Not a ceiling: gain on PC F1 predicts +0.071 for the rule.
- Not the pooled-variance quantity of the paper's Section 3.2: the
  between-experiment share of pooled variance correlates +0.11 with the gain
  and predicts +0.080 for the rule.
- Not a count of strong or non-mid experiments: one-variable fits predict
  +0.062 to +0.073.
- A per-experiment ridge fit on the non-rule lists does predict it (+0.049
  against +0.045 measured) and reproduces rule − loop (−0.031 against −0.035).
  Three quarters of that difference sits on the light-source variables: the
  rule buys them at `mid` where the loop buys them at `strong`.

That last reading is correlational and rests on a 59-coefficient fit to 120
lists. This is the direct test.

## Design

- The rule's 30 light-tunnel k=30 lists (those in Table 5).
- `strong`: every `mid` experiment on a light-source variable is replaced, in
  position, by the `strong` experiment on the same variable, where the menu has
  one and the list lacks it. Variables covered, budget and order unchanged.
  29 lists have at least one such entry (mean 2.3 swaps, range 1 to 4); the
  thirtieth is dropped.
- `original` and `strong` scored with PC and with JCI-PC (regime context) at
  5000 rows, 9 seeds, on the Mac (the backend of Table 5), both in this run.
- Contrast: paired by list, mean difference with a 95% t interval.

## Predictions

From the ridge coefficients applied to the exact swap:

| strong − original | prediction |
|---|---|
| P1 gain | +0.017 |
| P2 F1 under JCI-PC | +0.022 |
| P3 F1 under PC | +0.005 (about zero) |

Decision rule, on the interval:

- **Supported** if P1's interval lies above zero. The rebuttal may then say
  that part of the rule's smaller gain comes from its choosing milder
  interventions on the light sources, which leave less for JCI-PC to repair.
- **Not supported** if P1's interval contains zero or lies below it. C4 then
  keeps "we have no tested explanation", with the list of what was ruled out.

Even if supported, the swap is predicted to close about half of the rule's gap
to the loop (+0.017 of 0.035), so "part of" is the most the result can carry.

## Limits stated in advance

- One chamber, one budget, one row cap, the backend of Table 5.
- The swap changes strength on the swappable light-source variables only
  (those with both a `mid` and a `strong` entry on the menu).
- At k=45 the ridge puts the rule − loop difference elsewhere (on `mid`
  experiments on other variables), so nothing here transfers to k=45.
