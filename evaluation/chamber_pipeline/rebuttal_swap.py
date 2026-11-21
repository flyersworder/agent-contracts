"""The swap intervention behind the rebuttal's "cause" sentence (reviewer xQHG, C3).

Pre-registered in docs/superpowers/specs/2026-11-21-equal-rows-swap-prereg.md.
Takes the wind-tunnel rule's purchase lists at k=21 and, in place of seeded
one-edge sensor-setting (OSR) entries, swaps in

  big    the 3000- and 4000-row repeat entries on two hub variables
  small  a small repeat entry on the SAME hub variables (the size control)

then scores original, big and small under both row samplings at both caps
(`rebuttal_swap.sh`). The budget stays at 21 and each swap keeps the position
of the entry it replaces, because pooling order moves the row subsample.

    uv run python -m evaluation.chamber_pipeline.rebuttal_swap build
    uv run python -m evaluation.chamber_pipeline.rebuttal_swap analyze
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .rescore import design_key, parse_selection

RUNS = Path(__file__).resolve().parents[2] / "runs"
SOURCE = RUNS / "rebuttal-eqrows-src.parquet"
OUT_SRC = RUNS / "rebuttal-swap-src.parquet"
SEED = 20261121

#: hub variable -> (the big repeat, small repeats in order of preference)
HUBS = {
    "hatch": ("validate_hatch_mic", ("validate_hatch_rpms", "validate_hatch_pressures")),
    "load_out": (
        "validate_load_out_pressure_intake",
        ("validate_load_out_mic", "validate_load_out_current_in", "validate_load_out"),
    ),
}
#: One-edge sensor settings the rule always buys; `osr_ambient` (8000 rows) is
#: excluded so that removing an entry never changes the pool's largest member.
OSR = tuple(
    f"validate_osr_{s}" for s in ("intake", "upwind", "in", "downwind", "mic", "1", "2", "out")
)


def build() -> None:
    src = pd.read_parquet(SOURCE)
    rule = src[(src["agent_name"] == "wt_coverage_max") & (src["budget_k"] == 21)]
    rng = np.random.default_rng(SEED)
    rows = []
    for pair, raw in enumerate(rule["chosen_experiments"]):
        names = parse_selection(raw)
        assert len(names) == len(set(names)) == 21
        hubs = [h for h, (big, _) in HUBS.items() if big not in names]
        removed = [str(x) for x in rng.choice([n for n in names if n in OSR], len(hubs), False)]
        variants = {"original": list(names)}
        if hubs:
            for label in ("big", "small"):
                new = list(names)
                for hub, out in zip(hubs, removed, strict=True):
                    big, smalls = HUBS[hub]
                    entry = big if label == "big" else next(s for s in smalls if s not in names)
                    new[new.index(out)] = entry
                assert len(set(new)) == 21
                variants[label] = new
        for label, new in variants.items():
            rows.append(
                {
                    "chamber": "wt",
                    "configuration": "standard",
                    "status": "ok",
                    "budget_k": 21,
                    "agent_name": f"swap_{label}",
                    "pair": pair,
                    "n_swapped": len(hubs),
                    "removed": ",".join(removed),
                    "chosen_experiments": ",".join(new),
                }
            )
    out = pd.DataFrame(rows)
    out.to_parquet(OUT_SRC, index=False)
    print(out.groupby(["agent_name", "n_swapped"]).size().unstack(fill_value=0).to_string())


def _paired(d: pd.DataFrame, a: str, b: str) -> str:
    w = d.pivot(index="pair", columns="agent_name", values="f1").dropna(subset=[a, b])
    diff = w[a] - w[b]
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = diff.mean() - half, diff.mean() + half
    sign = "above 0" if lo > 0 else ("below 0" if hi < 0 else "contains 0")
    return f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(diff)} ({sign})"


def analyze() -> None:
    src = pd.read_parquet(OUT_SRC)
    src["design_key"] = [
        design_key("wt", "standard", parse_selection(x)) for x in src["chosen_experiments"]
    ]
    for cap in (1500, 300):
        for sampling in ("uniform", "equal"):
            by = pd.read_parquet(RUNS / f"rebuttal-swap-{sampling}-rows{cap}-bykey.parquet")
            assert set(by["rescore_row_sampling"]) == {sampling}
            assert set(by["rescore_pc_max_rows"]) == {cap}
            assert by["blas_backend"].nunique() == 1
            f1 = by.groupby("design_key")["f1"].mean().rename("f1")
            d = src.join(f1, on="design_key")
            assert d["f1"].notna().all()
            print(f"\n== {cap} rows, {sampling} ({by['blas_backend'].iloc[0]}) ==")
            for name, sub in (
                ("both hubs swapped", d[d.n_swapped == 2]),
                ("one or both", d[d.n_swapped >= 1]),
            ):
                print(f"  {name}:")
                print(f"    big - original   {_paired(sub, 'swap_big', 'swap_original')}")
                print(f"    small - original {_paired(sub, 'swap_small', 'swap_original')}")
                print(f"    big - small      {_paired(sub, 'swap_big', 'swap_small')}")


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
