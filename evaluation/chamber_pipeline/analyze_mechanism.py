"""M7 Phase 1: why does `team` lose to `loop` at equal experiment count?

Three hypotheses predict the same symptom (equal `n_experiments_distinct`,
lower F1), so they are separated by what the arms actually BOUGHT, not by
what they scored:

* **H1 experiments != variables.** The LT menu carries up to three entries per
  actuated variable (`weak`/`mid`/`strong`), so 30 distinct experiments can
  touch 30 variables or 12. If `team` buys fewer distinct variables at equal
  experiment count it bought depth where the loop bought breadth.
* **H2 forced allocation.** Scout pools are disjoint by construction, so each
  scout must spend its half inside its own half of the menu whether or not
  that half deserves it. The signature is a LOPSIDED per-scout split -- one
  pool contributing most of the distinct variables -- with total variable
  coverage otherwise comparable to the loop.
* **H3 blind depth duplication.** Zero overlap at the experiment level is
  compatible with redundancy at the VARIABLE level: two scouts buying
  `uniform_v_weak` and `uniform_v_strong` overlap 0.0 by the recorded metric
  and still learn about one variable twice.

The instrument is `chosen_experiments` (the roster in spending order, recorded
at the adapter so no agent can misreport it) plus `n_zero_variance_dropped`
(variables that never moved, i.e. H1/H3 in PC's own terms).

Scout attribution is POSITIONAL and only valid for `team`: `team_agents` runs
scout_a's selection loop to completion before scout_b's, and the two pools are
disjoint, so the first `ceil(k/2)` purchases are scout_a's. Asserted, not
assumed -- a roster with a repeat would break the attribution silently.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

STRENGTHS = ("weak", "mid", "strong")
_PREFIX = "uniform_"


def experiment_variable(name: str) -> str:
    """`uniform_osr_c_strong` -> `osr_c`; `uniform_reference` -> `reference`.

    Variable names themselves contain underscores, so the split is on the
    KNOWN strength suffix, never on a token count.
    """
    stem = name[len(_PREFIX) :] if name.startswith(_PREFIX) else name
    for s in STRENGTHS:
        if stem.endswith("_" + s):
            return stem[: -(len(s) + 1)]
    return stem


def experiment_strength(name: str) -> str:
    """The intervention strength, or `none` for the observational entry."""
    for s in STRENGTHS:
        if name.endswith("_" + s):
            return s
    return "none"


def parse_roster(cell: str | float | None) -> list[str]:
    if not isinstance(cell, str) or not cell:
        return []
    return [n for n in cell.split(",") if n]


@dataclass(frozen=True)
class CellMechanism:
    """Per-cell breakdown of what one arm bought."""

    agent_name: str
    seed: int
    budget_k: int
    f1: float
    n_experiments: int
    n_variables: int
    n_repeat_variables: int  # variables bought at >1 strength
    max_depth: int  # most strengths bought on any one variable
    n_zero_variance_dropped: float | None
    n_selection_fallbacks: float | None
    # team only; None elsewhere
    a_variables: int | None = None
    b_variables: int | None = None
    shared_variables: int | None = None


def _cell(row: pd.Series) -> CellMechanism:
    roster = parse_roster(row.get("chosen_experiments"))
    if len(set(roster)) != len(roster):
        raise ValueError(
            f"roster for {row['agent_name']} seed {row['seed']} repeats a name; "
            "positional scout attribution is not valid"
        )
    per_var = Counter(experiment_variable(n) for n in roster)
    a_vars = b_vars = shared = None
    if row["agent_name"] == "team" and roster:
        # The split point comes from the CONTRACT, not from the roster length.
        # `run_cell` sets scout_a_budget = k - k//2 and scout_b_budget = k//2,
        # and scout_a's loop runs to completion first, so purchase `ceil(k/2)`
        # is the seam. Deriving it from `len(roster)` instead would silently
        # slide the seam whenever a scout under-spends -- which is exactly the
        # failure `team_agents` had before the pools were made to exceed the
        # budgets, and exactly the case where the attribution must not lie.
        budget_k = int(row["budget_k"])
        if len(roster) != budget_k:
            raise ValueError(
                f"team seed {row['seed']} bought {len(roster)} of {budget_k} "
                "experiments; a shortfall moves the scout seam, so this is a "
                "harness defect to investigate, not a cell to average"
            )
        split = budget_k - budget_k // 2
        va = {experiment_variable(n) for n in roster[:split]}
        vb = {experiment_variable(n) for n in roster[split:]}
        a_vars, b_vars, shared = len(va), len(vb), len(va & vb)
    return CellMechanism(
        agent_name=str(row["agent_name"]),
        seed=int(row["seed"]),
        budget_k=int(row["budget_k"]),
        f1=float(row["f1"]),
        n_experiments=len(roster),
        n_variables=len(per_var),
        n_repeat_variables=sum(1 for c in per_var.values() if c > 1),
        max_depth=max(per_var.values(), default=0),
        n_zero_variance_dropped=row.get("n_zero_variance_dropped"),
        n_selection_fallbacks=row.get("n_selection_fallbacks"),
        a_variables=a_vars,
        b_variables=b_vars,
        shared_variables=shared,
    )


def mechanism_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    missing = ok["chosen_experiments"].isna().sum() if "chosen_experiments" in ok else len(ok)
    if missing:
        raise ValueError(
            f"{missing} ok-cells carry no `chosen_experiments`. Rows recorded "
            "before 2026-08-29 predate the instrument and cannot answer this "
            "question -- re-run rather than analyse them."
        )
    return pd.DataFrame([_cell(r).__dict__ for _, r in ok.iterrows()])


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "f1",
        "n_experiments",
        "n_variables",
        "n_repeat_variables",
        "max_depth",
        "n_zero_variance_dropped",
        "n_selection_fallbacks",
        "a_variables",
        "b_variables",
        "shared_variables",
    ]
    present = [c for c in cols if c in cells.columns]
    g = cells.groupby("agent_name")[present]
    out = g.mean().round(3)
    out.insert(0, "n", g.size())
    return out


def mde(cells: pd.DataFrame, column: str) -> float:
    """Same convention as `analyze_results`: 2.8 * pooled sd * sqrt(2/n)."""
    groups = [g[column].dropna() for _, g in cells.groupby("agent_name")]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    n = min(len(g) for g in groups)
    pooled = math.sqrt(sum(g.var(ddof=1) for g in groups) / len(groups))
    return 2.8 * pooled * math.sqrt(2 / n)


def verdict(summary: pd.DataFrame, cells: pd.DataFrame) -> list[str]:
    """The spec's decision rule, applied mechanically.

    Deliberately reports "below MDE" rather than "no difference": at n=10 the
    bound is wide, and an equivalence bound is not a null.
    """
    if not {"llm_pc", "team"}.issubset(summary.index):
        return ["both `llm_pc` and `team` are required to apply the decision rule"]
    lines = []
    for col in ("n_experiments", "n_variables", "n_repeat_variables", "f1"):
        d = summary.loc["team", col] - summary.loc["llm_pc", col]
        m = mde(cells, col)
        mark = "RESOLVED" if abs(d) > m else "below MDE"
        lines.append(f"{col:<22} team-loop = {d:+.3f}   MDE {m:.3f}   {mark}")
    dv = summary.loc["team", "n_variables"] - summary.loc["llm_pc", "n_variables"]
    if abs(dv) > mde(cells, "n_variables"):
        lines.append("=> H1/H3: team buys measurably fewer distinct variables.")
    else:
        lines.append(
            "=> variable coverage matches within MDE; H1/H3 not supported. "
            "Read the per-scout split below for H2."
        )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Parquet written by run_experiment")
    args = p.parse_args()
    df = pd.read_parquet(args.path)
    cells = mechanism_table(df)
    summary = summarize(cells)
    print(summary.to_string())
    print()
    for line in verdict(summary, cells):
        print(line)


if __name__ == "__main__":
    main()
