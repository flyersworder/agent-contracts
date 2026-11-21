"""Why does the coverage rule gain least from JCI-PC? (reviewer xQHG, C4)

Table 5 of the paper: PC and JCI-PC (one indicator per regime) at 5000 rows on
the light tunnel. Per purchase list, gain = F1(JCI-PC) - F1(PC). This sets
the gain beside what each list bought. Same files as the table
(`paper_alllow_tables.pool5000`), all scored on one machine. No PC run.

    uv run python -m evaluation.chamber_pipeline.rebuttal_jci_gain_diagnose
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .menu_taxonomy import experiment_strength, experiment_variable
from .oracle_probe import LT_SOURCE_VARIABLES
from .paper_alllow_tables import design_frame
from .rescore import parse_selection

REFERENCE = "uniform_reference"
RULE = "coverage_max_ms"


def frame() -> pd.DataFrame:
    parts = {}
    for est, tag in (("pc", "pc5000"), ("jci", "jcireg5000")):
        low = design_frame(f"lownego-lt-{tag}")
        ref = pd.concat([design_frame(f"rescored-t4-{s}-{tag}") for s in ("k30", "ends")])
        ref = ref[ref["agent_name"].isin([RULE, "random"])]
        parts[est] = pd.concat([low, ref]).drop_duplicates(["design_key", "agent_name"])
    keys = ["design_key", "agent_name", "budget_k", "chosen_experiments"]
    d = parts["pc"][[*keys, "f1"]].merge(
        parts["jci"][["design_key", "agent_name", "f1"]],
        on=["design_key", "agent_name"],
        suffixes=("_pc", "_jci"),
    )
    d["gain"] = d["f1_jci"] - d["f1_pc"]
    rows = []
    for raw in d["chosen_experiments"]:
        names = parse_selection(raw)
        inter = [n for n in names if n != REFERENCE]
        var = pd.Series([experiment_variable(n) for n in inter])
        strength = pd.Series([experiment_strength(n) for n in inter])
        counts = var.value_counts()
        source = var.isin(LT_SOURCE_VARIABLES).to_numpy()
        rows.append(
            {
                "distinct_vars": var.nunique(),
                "repeat_vars": int((counts > 1).sum()),
                "n_strong": int((strength == "strong").sum()),
                "n_mid": int((strength == "mid").sum()),
                "n_weak": int((strength == "weak").sum()),
                "strong_source": int(((strength == "strong").to_numpy() & source).sum()),
                "n_source": int(source.sum()),
                "has_reference": int(REFERENCE in names),
            }
        )
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def ols(d: pd.DataFrame, y: str, xs: list[str]) -> pd.Series:
    a = np.column_stack([np.ones(len(d)), d[xs].to_numpy(dtype=float)])
    coef = np.linalg.lstsq(a, d[y].to_numpy(), rcond=None)[0]
    return pd.Series(coef, index=["intercept", *xs])


def main() -> None:
    pd.set_option("display.width", 220)
    d = frame()
    desc = [
        "distinct_vars",
        "repeat_vars",
        "n_strong",
        "n_mid",
        "n_weak",
        "strong_source",
        "n_source",
        "has_reference",
    ]
    print("== per arm and budget ==")
    print(
        d.groupby(["budget_k", "agent_name"])[["f1_pc", "f1_jci", "gain", *desc]]
        .mean()
        .round(3)
        .to_string()
    )
    for k in (30, 45):
        s = d[d["budget_k"] == k]
        print(f"\n== k={k}: correlation with gain, all {len(s)} lists / non-rule lists ==")
        nr = s[s["agent_name"] != RULE]
        print(
            pd.DataFrame(
                {
                    "all: gain": s[["f1_pc", *desc]].corrwith(s["gain"]),
                    "non-rule: gain": nr[["f1_pc", *desc]].corrwith(nr["gain"]),
                    "non-rule: f1_pc": nr[desc].corrwith(nr["f1_pc"]),
                    "non-rule: f1_jci": nr[desc].corrwith(nr["f1_jci"]),
                }
            )
            .round(2)
            .to_string()
        )
        xs = ["distinct_vars", "n_strong", "n_weak", "strong_source", "has_reference"]
        for y in ("f1_pc", "f1_jci", "gain"):
            c = ols(nr, y, xs)
            rule = s[s["agent_name"] == RULE]
            pred = c["intercept"] + (rule[xs].mean() * c[xs]).sum()
            print(
                f"  OLS on non-rule lists, {y}: " + ", ".join(f"{n} {v:+.4f}" for n, v in c.items())
            )
            print(f"    predicted for the rule's lists {pred:.3f}, measured {rule[y].mean():.3f}")
        entry_ridge(s)
    d.to_parquet("runs/rebuttal-jci-gain-diagnose.parquet", index=False)


def entry_ridge(s: pd.DataFrame, lam: float = 10.0) -> None:
    """Per-experiment ridge fit of the gain on the NON-rule lists, applied to the rule.

    The summaries above (coverage, repeats, strength counts) do not predict
    the rule's gain; which experiments a list holds does. Contributions to
    rule - loop are grouped by strength and by whether the variable is a
    light source.
    """
    entries = sorted({e for x in s["chosen_experiments"] for e in parse_selection(x)})

    def x(f: pd.DataFrame) -> pd.DataFrame:
        rows = [[float(e in parse_selection(c)) for e in entries] for c in f["chosen_experiments"]]
        return pd.DataFrame(rows, columns=entries)

    nr, rule = s[s["agent_name"] != RULE], s[s["agent_name"] == RULE]
    loop = s[s["agent_name"] == "llm_pc"]
    xn = x(nr)
    mu = xn.mean(0)
    c = (xn - mu).to_numpy()
    y = nr["gain"].to_numpy()
    w = pd.Series(
        np.linalg.solve(c.T @ c + lam * np.eye(len(entries)), c.T @ (y - y.mean())), index=entries
    )
    pred = y.mean() + ((x(rule) - mu).to_numpy() @ w.to_numpy()).mean()
    contrib = (x(rule).mean() - x(loop).mean()) * w
    group = pd.Series(
        [
            "observational"
            if e == REFERENCE
            else f"{experiment_strength(e)}, "
            + ("light source" if experiment_variable(e) in LT_SOURCE_VARIABLES else "other")
            for e in entries
        ],
        index=entries,
    )
    print(
        f"  per-experiment ridge (lambda {lam}) on non-rule lists: rule gain predicted {pred:.3f}, measured {rule['gain'].mean():.3f}"
    )
    print(
        f"  rule - loop gain: predicted {contrib.sum():+.3f}, measured {rule['gain'].mean() - loop['gain'].mean():+.3f}; by group:"
    )
    print(contrib.groupby(group).sum().round(3).to_string())


if __name__ == "__main__":
    main()
