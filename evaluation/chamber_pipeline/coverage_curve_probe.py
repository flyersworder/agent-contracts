"""Coverage-curve probe: is the coverage law's SIZE miss a calibration gap?

Pre-registered in `docs/superpowers/specs/2026-09-28-coverage-curve-prereg.md`.
Nothing in `analyze` may change after that file is committed.

The coverage law predicts an arm gap as rate x (difference in distinct
variables), with one straight-line rate calibrated on LLM-free lists. At 300
rows those lists sit at ~11, 15, ~21 and 30 variables -- none in 25-29, which
is where the arms operate. This probe fills the gap with LLM-free lists that
cover EXACTLY m variables and re-runs the law with the measured curve.

    python -m evaluation.chamber_pipeline.coverage_curve_probe generate
    python -m evaluation.chamber_pipeline.coverage_curve_probe analyze
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .menu_taxonomy import experiment_variable

RUNS = Path("runs")
LISTS = RUNS / "probe-cc-lists.parquet"
#: (budget k, coverage levels m). m counts `reference` as a variable, exactly as
#: the paper's `lt_variable_count` does, and every list includes it -- as every
#: LLM list does.
GRID = {30: range(13, 31), 45: range(25, 31)}
LISTS_PER_LEVEL = 30
LLM_ARMS = ("llm_pc", "team", "team_varsplit", "one_shot")
#: Locally re-scored DeepSeek sources (same machine as the new lists).
DS_SOURCES = ("m7-varsplit", "m7-phase1", "m7-p2-lt", "m7-p2-ref", "fig2-fill-lt")


def _var(entry: str) -> str:
    return "reference" if entry == "uniform_reference" else experiment_variable(entry)


def exact_coverage_list(menu: list[str], k: int, m: int, rng: random.Random) -> list[str]:
    """k entries covering exactly m distinct variables, reference always included."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in menu:
        groups[_var(e)].append(e)
    others = sorted(v for v in groups if v != "reference")
    for _ in range(10_000):
        chosen = ["reference", *rng.sample(others, m - 1)]
        if sum(len(groups[v]) for v in chosen) >= k:
            break
    else:  # pragma: no cover - infeasible (k, m) would be a design error
        raise ValueError(f"no feasible list for k={k}, m={m}")
    picks = [rng.choice(groups[v]) for v in chosen]
    rest = [e for v in chosen for e in groups[v] if e not in picks]
    picks += rng.sample(rest, k - m)
    rng.shuffle(picks)
    return picks


def generate(menu: list[str]) -> pd.DataFrame:
    rows = []
    for k, levels in GRID.items():
        for m in levels:
            for s in range(LISTS_PER_LEVEL):
                buy = exact_coverage_list(menu, k, m, random.Random(f"cc:{k}:{m}:{s}"))
                assert len(buy) == k and len({_var(e) for e in buy}) == m
                rows.append(
                    {
                        "chamber": "lt",
                        "configuration": "standard",
                        "agent_name": f"exact_m{m}",
                        "budget_k": k,
                        "seed": s,
                        "m": m,
                        "status": "ok",
                        "chosen_experiments": ",".join(buy),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis
def _n_vars(s: str) -> int:
    return len({_var(e) for e in s.split(",")})


def _design_frame(stem: str, rows: int) -> pd.DataFrame:
    """One row per (design, arm); F1 = mean over the 9 PC seeds."""
    c = pd.read_parquet(RUNS / f"{stem}-rows{rows}.parquet")
    c = c[(c["status"] == "ok") & c["chosen_experiments"].notna() & (c["chamber"] == "lt")]
    b = (
        pd.read_parquet(RUNS / f"{stem}-rows{rows}-bykey.parquet")
        .groupby("design_key")["f1"]
        .mean()
    )
    d = c.drop_duplicates(["design_key", "agent_name"])[
        ["design_key", "agent_name", "budget_k", "source_file", "chosen_experiments"]
    ]
    d = d.join(b, on="design_key").dropna(subset=["f1"])
    d["nv"] = d["chosen_experiments"].map(_n_vars)
    return d.reset_index(drop=True)


# (label, frame, k, arm A, source A, arm B, source B)
CONTRASTS = (
    ("DS LT30 team-loop", "ds", 30, "team", "m7-varsplit", "llm_pc", "m7-varsplit"),
    ("DS LT30 varsplit-team", "ds", 30, "team_varsplit", "m7-varsplit", "team", "m7-varsplit"),
    ("DS LT45 team-loop", "ds", 45, "team", "fig2-fill-lt", "llm_pc", "fig2-fill-lt"),
    ("DS LT45 varsplit-team", "ds", 45, "team_varsplit", "fig2-fill-lt", "team", "fig2-fill-lt"),
    ("GLM LT30 team-loop", "glm", 30, "team", None, "llm_pc", None),
    ("GLM LT30 varsplit-team", "glm", 30, "team_varsplit", None, "team", None),
)


def _curve(cal: pd.DataFrame, k: int):
    means = cal[cal["budget_k"] == k].groupby("nv")["f1"].mean()
    xs, ys = means.index.to_numpy(float), means.to_numpy()
    return lambda nv: np.interp(nv, xs, ys)  # flat beyond the measured range


def _linear_rate(cal: pd.DataFrame) -> float:
    # the paper's form: one straight line with budget intercepts
    design = np.column_stack([np.ones(len(cal)), cal["nv"], cal["budget_k"] == 45]).astype(float)
    return float(np.linalg.lstsq(design, cal["f1"].to_numpy(), rcond=None)[0][1])


def _groups(fr: dict[str, pd.DataFrame]) -> dict[tuple, np.ndarray]:
    g = {}
    for _, f, k, a, sa, b, sb in CONTRASTS:
        for arm, src in ((a, sa), (b, sb)):
            d = fr[f]
            mask = (d["agent_name"] == arm) & (d["budget_k"] == k)
            if src:
                mask &= d["source_file"] == src
            g[(f, arm, k, src)] = np.flatnonzero(mask.to_numpy())
    return g


def _evaluate(fr, cal, groups, idx):
    """Predicted (curve and linear) and measured gap per contrast, for one resample."""
    curves = {k: _curve(cal, k) for k in GRID}
    rate = _linear_rate(cal)
    out = []
    for _, f, k, a, sa, b, sb in CONTRASTS:
        arm_a = fr[f].iloc[idx[(f, a, k, sa)]]
        arm_b = fr[f].iloc[idx[(f, b, k, sb)]]
        out.append(
            (
                curves[k](arm_a["nv"]).mean() - curves[k](arm_b["nv"]).mean(),
                rate * (arm_a["nv"].mean() - arm_b["nv"].mean()),
                arm_a["f1"].mean() - arm_b["f1"].mean(),
            )
        )
    return np.array(out)


def _slope(pred, meas, w):
    return (w * pred * meas).sum() / (w * pred * pred).sum()


def _verdict(lo: float, hi: float) -> str:
    if lo <= 1.0 <= hi:
        return "A: calibration gap (CI contains 1)"
    return (
        "B: LLM lists gain more per variable (CI above 1)"
        if lo > 1.0
        else "C: LLM lists gain less (CI below 1)"
    )


def analyze(n_boot: int = 2000) -> None:
    rng = np.random.default_rng(20260928)
    for rows in (300, 1500):
        cal = _design_frame("probe-cc-lists-rescored", rows)
        fr = {
            "ds": _design_frame("probe-cc-ds-rescored", rows),
            "glm": _design_frame("rescored-xv-glm-lt", rows),
        }
        groups = _groups(fr)
        cal_groups = [
            np.flatnonzero(((cal["budget_k"] == k) & (cal["nv"] == m)).to_numpy())
            for k in GRID
            for m in GRID[k]
        ]
        point = _evaluate(fr, cal, groups, groups)
        se = []
        for _, f, k, a, sa, b, sb in CONTRASTS:
            arm_a, arm_b = (
                fr[f].iloc[groups[(f, a, k, sa)]]["f1"],
                fr[f].iloc[groups[(f, b, k, sb)]]["f1"],
            )
            se.append(np.sqrt(arm_a.var() / len(arm_a) + arm_b.var() / len(arm_b)))
        w = 1 / np.array(se) ** 2
        boots = []
        for _ in range(n_boot):
            idx = {key: rng.choice(v, len(v)) for key, v in groups.items()}
            cal_b = cal.iloc[np.concatenate([rng.choice(g, len(g)) for g in cal_groups])]
            boots.append(_evaluate(fr, cal_b, groups, idx))
        boots = np.array(boots)

        print(f"\n===== {rows} rows =====")
        curve = cal.groupby(["budget_k", "nv"])["f1"].mean().unstack(0).round(3)
        print("LLM-free curve (mean F1 of exact-m lists):\n" + curve.to_string())
        print(f"{'contrast':24s} {'pred(curve)':>11s} {'pred(line)':>10s} {'meas':>7s}")
        for (lab, *_), (pc, pl, ms) in zip(CONTRASTS, point, strict=True):
            print(f"{lab:24s} {pc:+11.3f} {pl:+10.3f} {ms:+7.3f}")
        for name, mask in (("DS (primary at 300)", [0, 1, 2, 3]), ("GLM", [4, 5])):
            mask = np.array(mask)
            for col, lab in ((0, "curve"), (1, "line")):
                s0 = _slope(point[mask, col], point[mask, 2], w[mask])
                sb = np.array([_slope(b[mask, col], b[mask, 2], w[mask]) for b in boots])
                lo, hi = np.percentile(sb, [2.5, 97.5])
                tag = _verdict(lo, hi) if col == 0 else "(old law, for comparison)"
                print(f"  {name:20s} {lab:5s} slope {s0:.2f} [{lo:.2f}, {hi:.2f}]  {tag}")
        # descriptive: LLM lists minus the LLM-free curve at their own coverage
        for f, lab in (("ds", "DeepSeek"), ("glm", "GLM")):
            for k in GRID:
                d = fr[f][(fr[f]["budget_k"] == k) & fr[f]["agent_name"].isin(LLM_ARMS)]
                if len(d) == 0:
                    continue
                gap = (d["f1"] - _curve(cal, k)(d["nv"])).mean()
                bs = []
                for _ in range(500):
                    cb = cal.iloc[np.concatenate([rng.choice(g, len(g)) for g in cal_groups])]
                    db = d.iloc[rng.choice(len(d), len(d))]
                    bs.append((db["f1"] - _curve(cb, k)(db["nv"])).mean())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                print(f"  LLM minus LLM-free curve, {lab} LT{k}: {gap:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("generate", "analyze"))
    a = p.parse_args(argv)
    if a.step == "generate":
        from .oracle_probe import _load

        menu = list(_load("lt")[1])
        df = generate(menu)
        df.to_parquet(LISTS, index=False)
        print(f"wrote {len(df)} lists to {LISTS}")
    else:
        analyze()


if __name__ == "__main__":
    main(sys.argv[1:])
