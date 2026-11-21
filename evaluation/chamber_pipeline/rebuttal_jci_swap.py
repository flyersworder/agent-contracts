"""Does intervention strength on the light sources drive the JCI-PC gain? (xQHG, C4)

Pre-registered in docs/superpowers/specs/2026-11-21-jci-gain-strength-swap-prereg.md.
In the coverage rule's light-tunnel k=30 lists, every `mid` experiment on a
light-source variable is replaced, in position, by the `strong` experiment on
the SAME variable where the menu has one the list lacks. Coverage, budget and
list order are unchanged. Original and swapped lists are scored with PC and
JCI-PC (one indicator per regime) at 5000 rows (`rebuttal_jci_swap.sh`).

    uv run python -m evaluation.chamber_pipeline.rebuttal_jci_swap build
    uv run python -m evaluation.chamber_pipeline.rebuttal_jci_swap analyze
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

from .menu_taxonomy import experiment_strength, experiment_variable
from .oracle_probe import LT_SOURCE_VARIABLES
from .rescore import design_key, parse_selection

RUNS = Path(__file__).resolve().parents[2] / "runs"
SOURCE = RUNS / "rebuttal-jci-gain-diagnose.parquet"
OUT_SRC = RUNS / "rebuttal-jci-swap-src.parquet"
REFERENCE = "uniform_reference"


def build() -> None:
    d = pd.read_parquet(SOURCE)
    rule = d[(d["agent_name"] == "coverage_max_ms") & (d["budget_k"] == 30)]
    menu = set(
        create_contracted_chamber_agent(
            chamber="lt", configuration="standard", intervention_budget=1
        ).available_experiments()
    )
    rows = []
    for pair, raw in enumerate(rule["chosen_experiments"]):
        names = parse_selection(raw)
        new, swapped = [], 0
        for e in names:
            strong = e[: -len("mid")] + "strong"
            if (
                e != REFERENCE
                and experiment_strength(e) == "mid"
                and experiment_variable(e) in LT_SOURCE_VARIABLES
                and strong in menu
                and strong not in names
            ):
                new.append(strong)
                swapped += 1
            else:
                new.append(e)
        if swapped == 0:
            continue
        assert len(set(new)) == len(names)
        assert {experiment_variable(e) for e in new if e != REFERENCE} == {
            experiment_variable(e) for e in names if e != REFERENCE
        }
        for label, lst in (("original", names), ("strong", new)):
            rows.append(
                {
                    "chamber": "lt",
                    "configuration": "standard",
                    "status": "ok",
                    "budget_k": 30,
                    "agent_name": f"jciswap_{label}",
                    "pair": pair,
                    "n_swapped": swapped,
                    "chosen_experiments": ",".join(lst),
                }
            )
    out = pd.DataFrame(rows)
    out.to_parquet(OUT_SRC, index=False)
    print(out.groupby("agent_name")["n_swapped"].agg(["size", "mean", "min", "max"]).to_string())


def analyze() -> None:
    src = pd.read_parquet(OUT_SRC)
    src["design_key"] = [
        design_key("lt", "standard", parse_selection(x)) for x in src["chosen_experiments"]
    ]
    for est in ("pc", "jci"):
        by = pd.read_parquet(RUNS / f"rebuttal-jci-swap-{est}5000-bykey.parquet")
        assert set(by["rescore_pc_max_rows"]) == {5000} and by["blas_backend"].nunique() == 1
        src[est] = src["design_key"].map(by.groupby("design_key")["f1"].mean())
    assert src[["pc", "jci"]].notna().all().all()
    src["gain"] = src["jci"] - src["pc"]
    print(src.groupby("agent_name")[["pc", "jci", "gain"]].mean().round(3).to_string())
    for y in ("gain", "jci", "pc"):
        w = src.pivot(index="pair", columns="agent_name", values=y)
        diff = w["jciswap_strong"] - w["jciswap_original"]
        half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        print(
            f"strong - original, {y}: {diff.mean():+.3f} [{diff.mean() - half:+.3f}, {diff.mean() + half:+.3f}] n={len(diff)}"
        )


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
