"""Figure for the coverage-curve probe (docs/superpowers/specs/2026-09-28-coverage-curve-prereg.md).

Old vs calibrated coverage law, both row caps (all scored locally, Accelerate).

Row = row cap (300, 1500). Columns:
  (1) the coverage curve: exact-m random lists (mean, 95% CI) vs the old
      calibration lists and their straight line; arm means marked.
  (2) old law check: predicted = paper rate x coverage gap.
  (3) calibrated law check: predicted = curve(coverage_A) - curve(coverage_B).
Contrasts: team - loop and varsplit - team at DeepSeek LT30, DeepSeek LT45, GLM LT30.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "paper/aamas2027/figures")
import make_figures as mf

from evaluation.chamber_pipeline import coverage_curve_probe as cc

OUT = Path("/Users/qingye/Desktop")
RUNS = Path("runs")
RNG = np.random.default_rng(1)
B = 2000
OLD_RATE = {300: 0.0045, 1500: 0.0131}  # the paper's rates
ARMC = {"llm_pc": mf.INK2, "team": mf.ORANGE, "team_varsplit": mf.BLUE}
MARK = {"DS LT30": ("o", True), "DS LT45": ("^", True), "GLM LT30": ("s", False)}


def free_old(rows):
    c = pd.read_parquet(RUNS / f"probe-cc-oldcal-rescored-rows{rows}.parquet")
    b = (
        pd.read_parquet(RUNS / f"probe-cc-oldcal-rescored-rows{rows}-bykey.parquet")
        .groupby("design_key")["f1"]
        .mean()
    )
    d = c.drop_duplicates(["design_key", "agent_name"]).join(b, on="design_key", rsuffix="_r")
    d["f1"] = d["f1_r"]
    d["nv"] = d.chosen_experiments.map(cc._n_vars)
    return d


fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
for row, rows in enumerate((300, 1500)):
    cal = cc._design_frame("probe-cc-lists-rescored", rows)
    fr = {
        "ds": cc._design_frame("probe-cc-ds-rescored", rows),
        "glm": cc._design_frame("rescored-xv-glm-lt", rows),
    }
    groups = cc._groups(fr)
    cal_groups = [
        np.flatnonzero(((cal.budget_k == k) & (cal.nv == m)).to_numpy())
        for k in cc.GRID
        for m in cc.GRID[k]
    ]

    # ---------------------------------------------------------- (1) the curve
    ax = axes[row, 0]
    c30 = cal[cal.budget_k == 30].groupby("nv").f1.agg(["mean", "std", "count"])
    h = 1.96 * c30["std"] / np.sqrt(c30["count"])
    ax.fill_between(c30.index, c30["mean"] - h, c30["mean"] + h, color=mf.INK, alpha=0.12, lw=0)
    ax.plot(
        c30.index,
        c30["mean"],
        color=mf.INK,
        lw=1.2,
        marker="o",
        ms=2.5,
        label="random lists, exact coverage (new)",
    )
    old = free_old(rows)
    og = old.groupby("agent_name").agg(nv=("nv", "mean"), f1=("f1", "mean"))
    ax.plot(
        og.nv, og.f1, "D", color=mf.GRID, mec=mf.INK2, ms=5, zorder=3, label="old calibration lists"
    )
    xs = np.array([10.5, 30.5])
    b1, b0 = np.polyfit(old.nv, old.f1, 1)
    ax.plot(
        xs, b0 + b1 * xs, color=mf.INK2, lw=0.9, ls="--", label=f"old straight line ({b1:.4f}/var)"
    )
    for f, lab, src in (("ds", "DS LT30", "m7-varsplit"), ("glm", "GLM LT30", None)):
        d = fr[f][(fr[f].budget_k == 30)]
        if src:
            d = d[d.source_file == src]
        for a, col in ARMC.items():
            x = d[d.agent_name == a]
            mk, filled = MARK[lab]
            ax.plot(
                x.nv.mean(),
                x.f1.mean(),
                mk,
                ms=6.5,
                mfc=col if filled else "white",
                mec=col,
                mew=1.2,
                zorder=4,
            )
    ax.set_xlim(10, 31)
    ax.set_title(f"coverage curve, LT k=30, {rows} rows", fontsize=8, loc="left")
    ax.set_xlabel("distinct variables covered", fontsize=7.5)
    ax.set_ylabel("directed-edge F1", fontsize=7.5)
    if row == 0:
        ax.legend(fontsize=6.3, frameon=False, loc="upper left")

    # ------------------------------------------------ (2)/(3) the law checks
    def preds(cal_, idx, fr=fr, rows=rows):
        cur = {k: cc._curve(cal_, k) for k in cc.GRID}
        out = []
        for _, f, k, a, sa, b_, sb in cc.CONTRASTS:
            arm_a, arm_b = fr[f].iloc[idx[(f, a, k, sa)]], fr[f].iloc[idx[(f, b_, k, sb)]]
            out.append(
                (
                    OLD_RATE[rows] * (arm_a.nv.mean() - arm_b.nv.mean()),
                    cur[k](arm_a.nv).mean() - cur[k](arm_b.nv).mean(),
                    arm_a.f1.mean() - arm_b.f1.mean(),
                )
            )
        return np.array(out)

    P = preds(cal, groups)
    se = []
    for _, f, k, a, sa, b_, sb in cc.CONTRASTS:
        arm_a, arm_b = fr[f].iloc[groups[(f, a, k, sa)]].f1, fr[f].iloc[groups[(f, b_, k, sb)]].f1
        se.append(np.sqrt(arm_a.var() / len(arm_a) + arm_b.var() / len(arm_b)))
    se = np.array(se)
    w = 1 / se**2
    boots = np.array(
        [
            preds(
                cal.iloc[np.concatenate([RNG.choice(g, len(g)) for g in cal_groups])],
                {kk: RNG.choice(v, len(v)) for kk, v in groups.items()},
            )
            for _ in range(B)
        ]
    )
    lim = (-0.085, 0.085)
    for col_i, (ax, name) in enumerate(
        (
            (axes[row, 1], "old law: paper rate x coverage gap"),
            (axes[row, 2], "calibrated law: coverage curve"),
        )
    ):
        ax.plot(lim, lim, color=mf.GRID, lw=0.8, zorder=0)
        ax.axhline(0, color=mf.GRID, lw=0.5)
        ax.axvline(0, color=mf.GRID, lw=0.5)
        for i, (lab, *_) in enumerate(cc.CONTRASTS):
            key = " ".join(lab.split()[:2])
            mk, filled = MARK[key]
            color = mf.INK
            ax.plot(
                [P[i, col_i]] * 2,
                [P[i, 2] - 1.96 * se[i], P[i, 2] + 1.96 * se[i]],
                color=color,
                lw=0.8,
                zorder=2,
            )
            ax.plot(
                P[i, col_i],
                P[i, 2],
                mk,
                ms=6,
                mfc=color if filled else "white",
                mec=color,
                mew=1.0,
                zorder=3,
            )
        txt = []
        for vend, mask in (("DeepSeek", np.array([0, 1, 2, 3])), ("GLM", np.array([4, 5]))):
            s0 = cc._slope(P[mask, col_i], P[mask, 2], w[mask])
            sb = np.array([cc._slope(b[mask, col_i], b[mask, 2], w[mask]) for b in boots])
            lo, hi = np.percentile(sb, [2.5, 97.5])
            txt.append(f"{vend} slope {s0:.2f} [{lo:.2f}, {hi:.2f}]")
            xs = np.array(lim)
            ax.plot(
                xs,
                s0 * xs,
                color=mf.INK if vend == "DeepSeek" else mf.INK2,
                lw=0.8,
                ls="--" if vend == "DeepSeek" else ":",
                zorder=1,
            )
        ax.text(
            0.03, 0.97, "\n".join(txt), transform=ax.transAxes, fontsize=6.5, va="top", ha="left"
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name}, {rows} rows", fontsize=8, loc="left")
        ax.set_xlabel("predicted ΔF1", fontsize=7.5)
        ax.set_ylabel("measured ΔF1, 95% CI", fontsize=7.5)

for ax in axes.flat:
    ax.grid(color=mf.GRID, lw=0.4, zorder=0)
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
handles = (
    [
        Line2D([], [], marker="o", ls="", color=c, ms=6, label=lbl)
        for lbl, c in (("loop", mf.INK2), ("team", mf.ORANGE), ("varsplit team", mf.BLUE))
    ]
    + [
        Line2D([], [], marker=m, ls="", mfc=mf.INK if f else "white", mec=mf.INK, ms=6, label=lbl)
        for lbl, (m, f) in MARK.items()
    ]
    + [
        Line2D([], [], color=mf.INK, ls="--", lw=0.8, label="fitted slope, DeepSeek"),
        Line2D([], [], color=mf.INK2, ls=":", lw=0.8, label="fitted slope, GLM"),
        Line2D([], [], color=mf.GRID, lw=0.8, label="diagonal (perfect prediction)"),
    ]
)
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=5,
    fontsize=6.8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle(
    "Coverage law: old calibration vs coverage curve (LT, scored locally)",
    fontsize=9,
    x=0.01,
    ha="left",
)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"coverage_law_old_vs_calibrated.{ext}", dpi=220, bbox_inches="tight")
print("saved")
