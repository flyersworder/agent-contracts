"""Why `team_varsplit` pays on one chamber and not the other.

`team_varsplit` partitions the VARIABLE space between two scouts instead of
the experiment list. On LT k=30 it recovers +0.043 F1 over `team` (resolved);
on WT it recovers nothing detectable at either budget. The tempting reading is
that the mechanism is chamber-specific. It is not — the mechanism fires on
both, and a two-factor model built from LLM-FREE measurements predicts the
size of the effect, including predicting that WT cannot show it at n=50:

    predicted gain  =  coverage exchange rate  x  variables recovered

* **exchange rate** — F1 per additional distinct variable, regressed on
  LLM-free arms only (`coverage_max`/`coverage_min`/`random`) with budget as
  a fixed effect. This is a property of the CHAMBER's inference problem.
* **variables recovered** — how many duplicate variables partitioning removes.
  This is a property of the MENU: a menu whose entries map near-1:1 onto
  variables gives two blind scouts almost nothing to duplicate, so there is
  nothing for a partition to recover.

The two factors move in OPPOSITE directions across our chambers, which is why
neither alone explains the non-replication. WT's exchange rate is nearly twice
LT's (0.0111 vs 0.0061) — a distinct variable is worth MORE there. But WT's
menu carries 28 entries over 21 variables (1.33 each) against LT's 59 over 30
(1.97), so there is far less duplication to remove. Headroom wins.

**The moderator is the action space, not the chamber.** That is what makes
the claim transferable: it predicts that partitioning by role pays exactly to
the extent that the action space affords duplication, which is a quantity any
benchmark can compute for itself before running an agent.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from agent_contracts.integrations.causalchamber import (
    create_contracted_chamber_agent,
)

from .menu_taxonomy import experiment_variable as _lt_variable
from .wt_menu_taxonomy import experiment_variable as _wt_variable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Arms that select without an LLM. The exchange rate must be regressed on
#: these alone: an LLM arm's variable count is chosen jointly with everything
#: else it conditions on, so its slope would confound coverage with selection
#: skill. These arms vary coverage by construction and nothing else.
LLM_FREE_ARMS = frozenset(
    {
        "random",
        "coverage_max",
        "coverage_min",
        "coverage_max_ms",
        "coverage_min_ms",
        "wt_coverage_max",
        "wt_coverage_min",
    }
)


def variable_resolver(chamber: str, node_names: Sequence[str]) -> Callable[[str], str]:
    """One call shape for two taxonomies with different signatures.

    LT parses a strength suffix and needs nothing else; WT needs the node
    names to find the longest matching prefix. Callers should not have to
    know which, so the asymmetry is absorbed here.
    """
    if chamber == "lt":
        return _lt_variable
    if chamber == "wt":
        names = list(node_names)
        return lambda name: _wt_variable(name, names)
    raise ValueError(f"unknown chamber {chamber!r}")


def default_resolver(chamber: str) -> Callable[[str], str]:
    """Resolver for a real chamber, loading node names only when WT needs them.

    LT's taxonomy is a pure string rule, so building an adapter for it would
    make every LT-only analysis depend on a dataset download for nothing.
    """
    if chamber == "lt":
        return _lt_variable
    return variable_resolver(chamber, list(_adapter(chamber).ground_truth().index))


def parse_roster(cell: object) -> list[str]:
    """`chosen_experiments` -> the purchased names, in spending order."""
    if not isinstance(cell, str) or not cell:
        return []
    return [name for name in cell.split(",") if name]


def distinct_variables(roster: Sequence[str], resolve: Callable[[str], str]) -> int:
    return len(Counter(resolve(name) for name in roster))


def _adapter(chamber: str) -> Any:
    if chamber not in ("lt", "wt"):
        raise ValueError(f"unknown chamber {chamber!r}")
    return create_contracted_chamber_agent(
        chamber=cast("Literal['lt', 'wt']", chamber),
        configuration="standard",
        intervention_budget=1,
    )


def coverage_slope(
    frame: pd.DataFrame,
    *,
    column: str = "f1_rescored",
    resolver_for: Callable[[str], Callable[[str], str]] = default_resolver,
) -> pd.DataFrame:
    """F1 per additional distinct variable, per chamber, LLM-free arms only.

    Budget enters as a fixed effect (both terms are centred within
    chamber x budget) because F1 rises steeply with k for reasons that have
    nothing to do with coverage; pooling raw across budgets would report that
    trend as the exchange rate.
    """
    if column not in frame.columns:
        raise ValueError(
            f"no `{column}` column. The slope must be read off re-scored "
            "values: a single PC draw carries enough inference noise to move "
            "it, and the arms being pooled here must share one BLAS backend."
        )
    ok = frame[(frame["status"] == "ok") & frame["agent_name"].isin(LLM_FREE_ARMS)]
    rows = []
    for chamber, block in ok.groupby("chamber"):
        resolve = resolver_for(chamber)
        for _, row in block.iterrows():
            roster = parse_roster(row.get("chosen_experiments"))
            value = row[column]
            if not roster or pd.isna(value):
                continue
            rows.append(
                {
                    "chamber": chamber,
                    "budget_k": int(row["budget_k"]),
                    "n_variables": distinct_variables(roster, resolve),
                    "value": float(value),
                }
            )
    cells = pd.DataFrame(rows)
    out = []
    for chamber, block in cells.groupby("chamber"):
        y = block["value"] - block.groupby("budget_k")["value"].transform("mean")
        x = block["n_variables"] - block.groupby("budget_k")["n_variables"].transform("mean")
        denominator = float((x**2).sum())
        if denominator == 0:
            continue
        slope = float((x * y).sum() / denominator)
        residual = y - slope * x
        dof = len(block) - block["budget_k"].nunique() - 1
        se = math.sqrt(float((residual**2).sum()) / dof / denominator)
        out.append(
            {
                "chamber": chamber,
                "n_cells": len(block),
                "slope": slope,
                "se": se,
                "budgets": sorted(block["budget_k"].unique()),
            }
        )
    return pd.DataFrame(out)


def variables_recovered(
    frame: pd.DataFrame,
    *,
    baseline: str = "team",
    treatment: str = "team_varsplit",
    resolver_for: Callable[[str], Callable[[str], str]] = default_resolver,
) -> pd.DataFrame:
    """Distinct variables the partition buys, per chamber x budget."""
    ok = frame[(frame["status"] == "ok") & frame["agent_name"].isin({baseline, treatment})]
    rows = []
    for chamber, block in ok.groupby("chamber"):
        resolve = resolver_for(chamber)
        for _, row in block.iterrows():
            roster = parse_roster(row.get("chosen_experiments"))
            if not roster:
                continue
            rows.append(
                {
                    "chamber": chamber,
                    "budget_k": int(row["budget_k"]),
                    "agent_name": row["agent_name"],
                    "n_variables": distinct_variables(roster, resolve),
                }
            )
    cells = pd.DataFrame(rows)
    means = cells.groupby(["chamber", "budget_k", "agent_name"])["n_variables"].mean().unstack()
    if baseline not in means.columns or treatment not in means.columns:
        raise ValueError(f"both {baseline!r} and {treatment!r} must be present to difference them")
    return pd.DataFrame(
        {
            "n_variables_baseline": means[baseline],
            "n_variables_treatment": means[treatment],
            "variables_recovered": means[treatment] - means[baseline],
        }
    ).reset_index()


def a_priori_headroom(chamber: str, budget_k: int, *, trials: int = 20000, seed: int = 0) -> float:
    """Variables two BLIND scouts would duplicate, from the menu alone.

    Splits the menu into two disjoint halves and has each scout buy its share
    uniformly at random, which is what a scout with no view of its peer
    reduces to. Deliberately a LOWER BOUND: real scouts concentrate on the
    variables that look informative, so they collide more often than uniform
    draws do (measured: 4.14 predicted against 6.50 observed at LT k=30, 1.06
    against 1.90 at WT k=14). Use it to rank action spaces and to bound the
    effect from below before running anything, not as a point estimate.
    """
    adapter = _adapter(chamber)
    menu = list(adapter.available_experiments())
    resolve = variable_resolver(chamber, list(adapter.ground_truth().index))
    variables = [resolve(name) for name in menu]
    budget_a, budget_b = budget_k - budget_k // 2, budget_k // 2
    half = len(menu) // 2
    if budget_a > half or budget_b > len(menu) - half:
        raise ValueError(
            f"k={budget_k} does not fit two disjoint pools of a {len(menu)}-entry menu"
        )
    rng = np.random.default_rng(seed)
    shared = []
    for _ in range(trials):
        order = rng.permutation(len(menu))
        pool_a, pool_b = order[:half], order[half:]
        vars_a = {variables[i] for i in rng.choice(pool_a, budget_a, replace=False)}
        vars_b = {variables[i] for i in rng.choice(pool_b, budget_b, replace=False)}
        shared.append(len(vars_a & vars_b))
    return float(np.mean(shared))


def predict(slopes: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    """Join the two factors into a predicted F1 gain per chamber x budget."""
    merged = recovered.merge(slopes[["chamber", "slope"]], on="chamber", how="left")
    merged["predicted_gain"] = merged["slope"] * merged["variables_recovered"]
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", nargs="+", required=True, help="re-scored LLM-free files")
    parser.add_argument("--varsplit", nargs="+", required=True, help="team / team_varsplit files")
    parser.add_argument("--column", default="f1_rescored")
    args = parser.parse_args(list(argv) if argv is not None else None)

    coverage = pd.concat([pd.read_parquet(p) for p in args.coverage], ignore_index=True)
    varsplit = pd.concat([pd.read_parquet(p) for p in args.varsplit], ignore_index=True)

    slopes = coverage_slope(coverage, column=args.column)
    recovered = variables_recovered(varsplit)
    print("\n=== exchange rate (F1 per distinct variable, LLM-free arms) ===")
    print(slopes.to_string(index=False))
    print("\n=== variables recovered by partitioning the variable space ===")
    print(recovered.round(3).to_string(index=False))
    print("\n=== predicted gain = exchange rate x variables recovered ===")
    print(predict(slopes, recovered).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
