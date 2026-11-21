"""Why does the equal-rows cap erase the wind-tunnel rule's lead at 1500 rows?

Describes every WT purchase list by what each sampling scheme actually feeds
PC (rows per entry, per recording session, effective number of entries) and
sets that beside the uniform and equal-rows F1 of the same list. No PC run.

    uv run python -m evaluation.chamber_pipeline.rebuttal_equal_rows_diagnose
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

from .inference import equal_row_quotas
from .jci import session_ids
from .rebuttal_equal_rows_analysis import RUNS, frames
from .rescore import parse_selection
from .wt_menu_taxonomy import experiment_variable


def menu_table() -> pd.DataFrame:
    adapter = create_contracted_chamber_agent(
        chamber="wt", configuration="standard", intervention_budget=64
    )
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    dfs = [adapter.query_intervention(m) for m in menu]
    return pd.DataFrame(
        {
            "rows": [len(d) for d in dfs],
            "variable": [experiment_variable(m, nodes) for m in menu],
            "session": session_ids(dfs),
        },
        index=menu,
    )


def describe(names: list[str], menu: pd.DataFrame, cap: int) -> dict[str, float]:
    sizes = menu.loc[names, "rows"].to_numpy()
    n_raw = int(sizes.sum())
    uni = sizes / n_raw  # expected share under the uniform cap
    eq = np.array(equal_row_quotas(list(sizes), cap, np.random.default_rng(0)), dtype=float)
    eq = eq / eq.sum()
    minority = menu["session"].value_counts().idxmin()
    in_minor = (menu.loc[names, "session"] == minority).to_numpy()
    return {
        "n_raw": n_raw,
        "n_used": min(n_raw, cap),
        "distinct_vars": menu.loc[names, "variable"].nunique(),
        "n_big": int((sizes >= 1000).sum()),
        "top_share_u": float(uni.max()),
        "top_share_e": float(eq.max()),
        "eff_entries_u": float(1 / (uni**2).sum()),
        "eff_entries_e": float(1 / (eq**2).sum()),
        "minor_sess_entries": int(in_minor.sum()),
        "minor_sess_share_u": float(uni[in_minor].sum()),
        "minor_sess_share_e": float(eq[in_minor].sum()),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    menu = menu_table()
    print("== WT menu ==")
    print(menu.sort_values("rows").to_string())
    print(menu.groupby("session")["rows"].agg(["size", "sum", "median"]).to_string())

    src = pd.read_parquet(RUNS / "rebuttal-eqrows-rows300.parquet")
    lists = src.drop_duplicates("design_key").set_index("design_key")["chosen_experiments"]
    for cap in (300, 1500):
        uni, eq = frames(cap)
        d = uni.merge(
            eq, on=["design_key", "agent_name", "chamber", "budget_k"], suffixes=("_u", "_e")
        )
        d = d[d["chamber"] == "wt"].copy()
        desc = pd.DataFrame(
            [describe(parse_selection(lists[k]), menu, cap) for k in d["design_key"]], index=d.index
        )
        d = pd.concat([d, desc], axis=1)
        d["delta"] = d["f1_e"] - d["f1_u"]
        d.to_parquet(RUNS / f"rebuttal-eqrows-diagnose-rows{cap}.parquet", index=False)
        print(f"\n== cap {cap}: per arm, WT ==")
        cols = ["f1_u", "f1_e", "delta", *desc.columns]
        print(d.groupby(["budget_k", "agent_name"])[cols].mean().round(3).to_string())
        k21 = d[d["budget_k"] == 21]
        print(f"\n== cap {cap}, k=21: correlation with F1 across all {len(k21)} lists ==")
        print(
            k21[cols].corr().loc[list(desc.columns), ["f1_u", "f1_e", "delta"]].round(2).to_string()
        )
        # Per-entry: mean F1 of lists that bought it minus lists that did not.
        rows = []
        for entry in menu.index:
            has = k21["design_key"].map(lambda k, e=entry: e in parse_selection(lists[k]))
            if 5 <= has.sum() <= len(k21) - 5:
                rows.append(
                    {
                        "entry": entry,
                        "rows": menu.loc[entry, "rows"],
                        "session": menu.loc[entry, "session"],
                        "buy_rate": has.mean(),
                        "rule_rate": has[k21["agent_name"] == "wt_coverage_max"].mean(),
                        "random_rate": has[k21["agent_name"] == "random"].mean(),
                        "eff_u": k21.loc[has, "f1_u"].mean() - k21.loc[~has, "f1_u"].mean(),
                        "eff_e": k21.loc[has, "f1_e"].mean() - k21.loc[~has, "f1_e"].mean(),
                    }
                )
        print(f"\n== cap {cap}, k=21: per-entry effect on F1 (bought minus not) ==")
        print(pd.DataFrame(rows).sort_values("eff_u").round(3).to_string(index=False))

        account(k21, lists, menu, cap)


def account(k21: pd.DataFrame, lists: pd.Series, menu: pd.DataFrame, cap: int) -> None:
    """Split rule - random at k=21 by entry group, under each sampling.

    Per-entry effects are OLS coefficients fitted on the NON-rule lists only
    (the rule's composition is nearly fixed, so including it would confound
    every entry it always buys with being the rule); each is multiplied by the
    rule-minus-random buy-rate difference. Additive and correlational: read
    the split, not the third decimal.
    """
    x = pd.DataFrame(
        [{e: int(e in parse_selection(lists[k])) for e in menu.index} for k in k21["design_key"]],
        index=k21.index,
    )
    multi = menu["variable"].map(menu["variable"].value_counts()) > 1
    group = pd.Series("other single-entry variable", index=menu.index)
    group[menu["variable"].str.startswith("osr")] = "OSR setting (one edge each)"
    group[multi] = "repeat entry on a hub, small"
    group[multi & (menu["rows"] >= 3000)] = "repeat entry on a hub, 3000-4000 rows"
    is_rule = k21["agent_name"] == "wt_coverage_max"
    is_random = k21["agent_name"] == "random"
    rate = x[is_rule].mean() - x[is_random].mean()
    design = np.column_stack([np.ones(int((~is_rule).sum())), x[~is_rule].to_numpy()])
    out = {"entries, rule - random": rate.groupby(group).sum()}
    for y in ("f1_u", "f1_e"):
        coef = np.linalg.lstsq(design, k21.loc[~is_rule, y].to_numpy(), rcond=None)[0][1:]
        part = (pd.Series(coef, index=menu.index) * rate).groupby(group).sum()
        part["TOTAL predicted"] = part.sum()
        part["measured"] = k21.loc[is_rule, y].mean() - k21.loc[is_random, y].mean()
        out[y] = part
    slope = {
        y: np.polyfit(k21.loc[~is_rule, "distinct_vars"], k21.loc[~is_rule, y], 1)[0]
        for y in ("f1_u", "f1_e")
    }
    print(f"\n== cap {cap}, k=21: rule - random by entry group ==")
    print(pd.DataFrame(out).round(3).to_string())
    print(
        "F1 per distinct variable, non-rule lists: "
        f"uniform {slope['f1_u']:+.4f}, equal {slope['f1_e']:+.4f}"
    )


if __name__ == "__main__":
    main()
