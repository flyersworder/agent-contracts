"""AAMAS 2027 rebuttal, reviewer xQHG: does an equal-rows cap change a verdict?

Compares, on the SAME purchase lists and the same machine (VPS/OpenBLAS), the
uniform row cap of record against `--row-sampling equal`
(`rebuttal_equal_rows.sh`). Contrast and MDE code is the paper's own
(`paper/aamas2027/figures/make_figures.py`), so a verdict here is computed
exactly as the verdicts in the paper.

    uv run python -m evaluation.chamber_pipeline.rebuttal_equal_rows_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
_FIGURES = ROOT / "paper" / "aamas2027" / "figures"
if not (_FIGURES / "make_figures.py").exists():
    raise SystemExit(f"needs the paper repo checked out at {_FIGURES.parent}")
sys.path.insert(0, str(_FIGURES))
import make_figures as mf  # noqa: E402

BUDGETS = {"lt": [6, 30, 45], "wt": [7, 14, 21]}
RULE = {"lt": "coverage_max_ms", "wt": "wt_coverage_max"}
LABEL = {
    "llm_pc": "loop",
    "one_shot": "single call",
    "team": "team",
    "team_varsplit": "varsplit team",
    "random": "random",
    "RULE": "rule",
}
#: (first, second): first minus second. "RULE" resolves per chamber.
CONTRASTS = [
    ("random", "RULE"),
    ("one_shot", "RULE"),
    ("llm_pc", "RULE"),
    ("team", "RULE"),
    ("team_varsplit", "RULE"),
    ("team", "llm_pc"),
    ("team_varsplit", "llm_pc"),
    ("team_varsplit", "team"),
]
UNIFORM_SOURCES = ("lownego-lt-rescored", "lownego-wt-rescored", "rescored-vps")


def frames(rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(uniform, equal) design-level frames over the same designs."""
    cells = pd.read_parquet(RUNS / f"rebuttal-eqrows-rows{rows}.parquet")
    designs = cells.drop_duplicates(["design_key", "agent_name"])[
        ["design_key", "agent_name", "chamber", "budget_k"]
    ]
    eq = pd.read_parquet(RUNS / f"rebuttal-eqrows-rows{rows}-bykey.parquet")
    assert set(eq["rescore_row_sampling"]) == {"equal"}
    assert set(eq["rescore_pc_max_rows"]) == {rows}
    uni = pd.concat(
        [pd.read_parquet(RUNS / f"{s}-rows{rows}-bykey.parquet") for s in UNIFORM_SOURCES]
    ).drop_duplicates(["design_key", "pc_seed"])
    uni = uni[uni["design_key"].isin(set(designs["design_key"]))]
    # One machine, one estimator, one cap on both sides.
    assert set(uni["blas_backend"]) == set(eq["blas_backend"]) == {"scipy-openblas"}
    assert set(uni["rescore_pc_max_rows"]) == {rows}
    assert set(uni.get("rescore_estimator", pd.Series(["pc"]))) == {"pc"}
    out = []
    for scored in (uni, eq):
        f1 = scored.groupby("design_key")["f1"].mean().rename("f1")
        d = designs.join(f1, on="design_key")
        assert d["f1"].notna().all(), "a design has no score on one side"
        out.append(d)
    return out[0], out[1]


def verdict(c: dict[str, float]) -> str:
    return "tie" if abs(c["d"]) <= c["mde"] else ("R+" if c["d"] > 0 else "R-")


def main() -> None:
    records = []
    for rows in (300, 1500):
        uni, eq = frames(rows)
        for ch, ks in BUDGETS.items():
            for k in ks:
                for a, b in CONTRASTS:
                    an, bn = (RULE[ch] if x == "RULE" else x for x in (a, b))
                    cu = mf.contrast(mf.arm(uni, an, ch, k), mf.arm(uni, bn, ch, k))
                    ce = mf.contrast(mf.arm(eq, an, ch, k), mf.arm(eq, bn, ch, k))
                    records.append(
                        {
                            "rows": rows,
                            "chamber": ch,
                            "k": k,
                            "contrast": f"{LABEL[a]} - {LABEL[b]}",
                            "uniform": cu["d"],
                            "mde_u": cu["mde"],
                            "v_u": verdict(cu),
                            "equal": ce["d"],
                            "mde_e": ce["mde"],
                            "v_e": verdict(ce),
                        }
                    )
        # Level shift: equal minus uniform on the same designs, per chamber and budget.
        both = uni.merge(
            eq, on=["design_key", "agent_name", "chamber", "budget_k"], suffixes=("_u", "_e")
        )
        shift = (
            (both["f1_e"] - both["f1_u"])
            .groupby([both["chamber"], both["budget_k"]])
            .agg(["mean", "std", "size"])
        )
        print(f"\n== {rows} rows: F1(equal) - F1(uniform), per design ==")
        print(shift.round(4).to_string())
        arm_means = both.groupby(["chamber", "budget_k", "agent_name"])[["f1_u", "f1_e"]].mean()
        print(arm_means.round(3).to_string())

    t = pd.DataFrame(records)
    t["changed"] = t["v_u"] != t["v_e"]
    t["flip"] = t.apply(lambda r: {r["v_u"], r["v_e"]} == {"R+", "R-"}, axis=1)
    pd.set_option("display.width", 200)
    print("\n== all contrasts ==")
    print(t.round(3).to_string(index=False))
    for ch in ("lt", "wt"):
        s = t[t["chamber"] == ch]
        print(
            f"\n{ch}: {len(s)} verdicts, {int(s['changed'].sum())} change, "
            f"{int(s['flip'].sum())} resolved sign flips; "
            f"max |shift in contrast| {float((s['equal'] - s['uniform']).abs().max()):.3f}"
        )
    print("\n== verdicts that change ==")
    print(t[t["changed"]].round(3).to_string(index=False))
    t.to_csv(RUNS / "rebuttal-eqrows-contrasts.csv", index=False)


if __name__ == "__main__":
    main()
