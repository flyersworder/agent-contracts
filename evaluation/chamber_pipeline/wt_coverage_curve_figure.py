"""Figure for the pre-registered WT coverage-curve probe (WT k=21, VPS/OpenBLAS scores).

x = distinct variables short of the coverage rule; y = F1 minus the rule's F1.
The dashed line is the registered prediction: the mean F1 of random lists of
exactly that coverage (interpolated), minus the rule. Points are the five
DeepSeek arms with 95% bootstrap CIs of their gap to the rule.

    python -m evaluation.chamber_pipeline.wt_coverage_curve_figure [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import wt_coverage_curve_probe as probe

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#d9d8d3"
COL = {300: BLUE, 1500: ORANGE}
LABEL = {
    "llm_pc": "loop",
    "one_shot": "single call",
    "team": "team",
    "team_varsplit": "varsplit",
    "shared_blackboard": "blackboard",
}
MULTI = {"team", "team_varsplit", "shared_blackboard"}
#: label offsets in points, placed by hand so the five 300-row labels do not collide
OFFSET = {
    "llm_pc": (-4, 9),
    "team_varsplit": (-44, 2),
    "shared_blackboard": (6, 4),
    "one_shot": (-52, -3),
    "team": (6, -4),
}


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    var_of = probe._wt_resolver()
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.axhline(0, color=INK2, lw=0.6, zorder=0)
    for rows in probe.CAPS:
        cal = probe._design_frame(probe.VPS_DIR / probe.LISTS_STEM, rows, var_of)
        corpus = probe._design_frame(probe.PAPER_RUNS / probe.CORPUS_STEM, rows, var_of)
        rule = corpus.loc[corpus["agent_name"] == probe.RULE, "f1"]
        r = rule.mean()
        means = cal.groupby("nv")["f1"].mean()
        ax.plot(21 - means.index, means.to_numpy() - r, color=COL[rows], lw=1.0, ls="--", zorder=1)
        ax.plot(
            21 - means.index,
            means.to_numpy() - r,
            "o",
            ms=2.5,
            color=COL[rows],
            alpha=0.6,
            zorder=1,
        )
        for arm in probe.ARMS:
            d = corpus[corpus["agent_name"] == arm]
            short = 21 - d["nv"].mean()
            gap = d["f1"].mean() - r
            boot = [
                d["f1"].to_numpy()[rng.integers(0, len(d), len(d))].mean()
                - rule.to_numpy()[rng.integers(0, len(rule), len(rule))].mean()
                for _ in range(2000)
            ]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mk = "^" if arm in MULTI else "s"
            ax.plot([short, short], [lo, hi], color=COL[rows], lw=0.7, alpha=0.8, zorder=2)
            ax.plot(short, gap, mk, ms=5.5, mfc=COL[rows], mec="white", mew=0.5, zorder=3)
            if rows == 300:
                ax.annotate(
                    LABEL[arm],
                    xy=(short, gap),
                    xytext=OFFSET[arm],
                    textcoords="offset points",
                    fontsize=6,
                    color=INK2,
                )
    ax.set_xlim(-0.2, 7.3)
    ax.set_xlabel("distinct variables short of the coverage rule")
    ax.set_ylabel("F1 minus the coverage rule")
    ax.set_title("Wind tunnel, k=21 (pre-registered)", fontsize=8, loc="left")
    ax.grid(color=GRID, lw=0.4, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [
        Line2D([], [], marker="s", ls="", mfc=BLUE, mec=BLUE, ms=5, label="300 rows"),
        Line2D([], [], marker="s", ls="", mfc=ORANGE, mec=ORANGE, ms=5, label="1500 rows"),
        Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=5, label="single agent"),
        Line2D([], [], marker="^", ls="", mfc=INK2, mec=INK2, ms=5, label="multi-agent"),
        Line2D(
            [],
            [],
            color=INK2,
            lw=1.0,
            ls="--",
            marker="o",
            ms=2.5,
            label="coverage-only prediction:\nrandom lists of that coverage",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/probe-wtcc-figure.png"))
