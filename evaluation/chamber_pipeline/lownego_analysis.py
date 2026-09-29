"""Registered analysis for the all-low negotiation re-run (2026-09-29-low-negotiation-prereg).

Same-day contrasts from one sweep's VPS re-score: 95% Welch CI of the
difference in mean directed-edge F1 over DISTINCT designs (a design's F1 is its
mean over the re-score's PC seeds), unequal-n MDE, both row caps. Verdicts are
keyed on the interval, as registered:

- ``team - loop`` (P1): below 0 -> "team loses"; contains 0 -> "not
  established"; above 0 -> "FALSIFIED".
- ``varsplit - team`` (P2): above 0 -> "repair holds"; otherwise "not
  established" (below 0 -> "reversed").
- ``varsplit - loop`` (P3 with P1): above 0 falsifies the headline.

Descriptive, no verdict: cells, distinct designs, distinct variables,
overlap, selection fallbacks, negotiation failures, certification.

    python -m evaluation.chamber_pipeline.lownego_analysis <rescored stem> <lt|wt>

``<rescored stem>`` is the path before ``-rows{300,1500}[-bykey].parquet``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import coverage_curve_probe as lt_probe
from . import wt_coverage_curve_probe as wt_probe

CAPS = (300, 1500)
ARMS = ("llm_pc", "team", "team_varsplit")
CONTRASTS = (
    ("team - loop", "team", "llm_pc"),
    ("varsplit - team", "team_varsplit", "team"),
    ("varsplit - loop", "team_varsplit", "llm_pc"),
)


def designs(stem: str, rows: int, chamber: str, var_of) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """One row per (budget, arm, distinct design); OpenBLAS asserted."""
    b = pd.read_parquet(f"{stem}-rows{rows}-bykey.parquet")
    assert b["blas_backend"].unique().tolist() == ["scipy-openblas"], b["blas_backend"].unique()
    c = pd.read_parquet(f"{stem}-rows{rows}.parquet")
    c = c[(c["status"] == "ok") & c["chosen_experiments"].notna() & (c["chamber"] == chamber)]
    assert (c["reasoning_effort"] == "low").all(), c["reasoning_effort"].unique()
    d = c.drop_duplicates(["budget_k", "agent_name", "design_key"])[
        ["budget_k", "agent_name", "design_key", "chosen_experiments"]
    ]
    d = d.join(b.groupby("design_key")["f1"].mean(), on="design_key")
    assert d["f1"].notna().all(), "design without a re-scored F1"
    d["nv"] = [len({var_of(e) for e in s.split(",")}) for s in d["chosen_experiments"]]
    return d.reset_index(drop=True)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Difference of means, its 95% Welch CI, and the unequal-n MDE."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    diff = a.mean() - b.mean()
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    mde = 2.8 * pooled * np.sqrt(1 / len(a) + 1 / len(b))
    return float(diff), float(diff - half), float(diff + half), float(mde)


def verdict(name: str, lo: float, hi: float) -> str:
    """The registered reading of one interval."""
    if name == "varsplit - team":
        return "repair holds" if lo > 0 else "reversed" if hi < 0 else "not established"
    if lo > 0:
        return "FALSIFIED: multi-agent above loop"
    if hi < 0:
        return "team loses" if name == "team - loop" else "below loop"
    return "not established"


def descriptive(stem: str, chamber: str) -> pd.DataFrame:
    c = pd.read_parquet(f"{stem}-rows300.parquet")
    c = c[(c["status"] == "ok") & (c["chamber"] == chamber)]
    return c.groupby(["budget_k", "agent_name"]).agg(
        cells=("seed", "size"),
        designs=("design_key", "nunique"),
        overlap=("overlap_frac", "mean"),
        fallbacks=("n_selection_fallbacks", "sum"),
        neg_fail=("n_negotiation_failures", "sum"),
        uncertified=("conservation_certified", lambda s: int((s == False).sum())),  # noqa: E712
    )


def main(stem: str, chamber: str) -> None:
    var_of = lt_probe._var if chamber == "lt" else wt_probe._wt_resolver()
    frames = {rows: designs(stem, rows, chamber, var_of) for rows in CAPS}
    desc = descriptive(stem, chamber)
    nv = frames[300].groupby(["budget_k", "agent_name"])["nv"].mean().rename("distinct_vars")
    print(desc.join(nv).round(3).to_string(), "\n")
    for k in sorted(frames[300]["budget_k"].unique()):
        for name, a, b in CONTRASTS:
            for rows in CAPS:
                f = frames[rows][frames[rows]["budget_k"] == k]
                x = f.loc[f["agent_name"] == a, "f1"].to_numpy()
                y = f.loc[f["agent_name"] == b, "f1"].to_numpy()
                diff, lo, hi, mde = welch(x, y)
                print(
                    f"{chamber} k={k:>2} {name:16s} {rows:>4} rows  {diff:+.3f} "
                    f"[{lo:+.3f}, {hi:+.3f}]  MDE {mde:.3f}  n={len(x)},{len(y)}  "
                    f"-> {verdict(name, lo, hi)}"
                )
        print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "lt")
