"""Predicted vs measured gap to the coverage rule, both chambers on one scale.

One panel per row cap. x = the coverage-only prediction for an arm: the mean,
over the arm's lists, of the F1 that random lists of exactly that coverage
score (interpolated, as both pre-registrations specify), minus the rule's F1.
y = the arm's measured F1 minus the rule's. The diagonal is "coverage explains
the whole gap". Arms at LT k=30 (DeepSeek and GLM, post hoc) and WT k=21
(DeepSeek, pre-registered); colour = single vs multi-agent, marker = chamber.
Every score is VPS/OpenBLAS (asserted by the loader).

    python -m evaluation.chamber_pipeline.coverage_prediction_figure [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import coverage_curve_probe as lt_probe
from . import wt_coverage_curve_probe as wt_probe
from .coverage_two_chamber_figure import (
    ARMS,
    MIN_DESIGNS,
    MULTI,
    PANELS,
    PAPER,
    gap_ci,
    load,
)

SINGLE_C, MULTI_C, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#d9d8d3"
MARKER = {"lt": "o", "wt": "D"}
CAPS = (300, 1500)


def points(rows: int, rng: np.random.Generator) -> list[dict]:
    resolvers = {"lt": lt_probe._var, "wt": wt_probe._wt_resolver()}
    out = []
    for _, ch, k, _nvars, rule_name, lists, sources, _ in PANELS:
        var_of = resolvers[ch]
        corpus = load(PAPER / "rescored-vps", rows, ch, k, var_of)
        rule = corpus.loc[corpus["agent_name"] == rule_name, "f1"].to_numpy()
        cal = load(lists, rows, ch, k, var_of)
        means = cal.groupby("nv")["f1"].mean()
        curve = lambda nv, m=means: np.interp(nv, m.index.to_numpy(float), m.to_numpy())  # noqa: E731
        for vendor, stem in sources:
            d = corpus if stem == PAPER / "rescored-vps" else load(stem, rows, ch, k, var_of)
            for arm in ARMS:
                x = d[d["agent_name"] == arm]
                if len(x) < MIN_DESIGNS:
                    continue
                lo, hi = gap_ci(x["f1"].to_numpy(), rule, rng)
                out.append(
                    {
                        "chamber": ch,
                        "vendor": vendor,
                        "arm": arm,
                        "pred": float(curve(x["nv"].to_numpy()).mean() - rule.mean()),
                        "meas": float(x["f1"].mean() - rule.mean()),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return out


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharex=True, sharey=True)
    lim = (-0.125, 0.025)
    for ax, rows in zip(axes, CAPS, strict=True):
        pts = points(rows, rng)
        ax.plot(lim, lim, color=INK2, lw=0.8, zorder=0)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        for p in pts:
            col = MULTI_C if p["arm"] in MULTI else SINGLE_C
            ax.plot(
                [p["pred"], p["pred"]], [p["lo"], p["hi"]], color=col, lw=0.7, alpha=0.7, zorder=2
            )
            ax.plot(
                p["pred"],
                p["meas"],
                MARKER[p["chamber"]],
                ms=5,
                mfc=col,
                mec="white",
                mew=0.5,
                zorder=3,
            )
        for ch, name in (("lt", "LT"), ("wt", "WT")):
            r = [p["meas"] - p["pred"] for p in pts if p["chamber"] == ch]
            print(f"{rows} rows {name}: mean residual {np.mean(r):+.3f} over {len(r)} arms")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{rows} rows", fontsize=8, loc="left")
        ax.set_xlabel("predicted gap (coverage alone)", fontsize=7.5)
        ax.grid(color=GRID, lw=0.4, zorder=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("measured gap", fontsize=7.5)
    axes[0].annotate(
        "on the line: coverage\nexplains the whole gap",
        xy=(-0.1, -0.1),
        xytext=(-0.075, -0.118),
        fontsize=6,
        color=INK2,
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": INK2},
    )
    handles = [
        Line2D([], [], marker="s", ls="", mfc=SINGLE_C, mec=SINGLE_C, ms=5, label="single agent"),
        Line2D([], [], marker="s", ls="", mfc=MULTI_C, mec=MULTI_C, ms=5, label="multi-agent"),
        Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=5, label="light tunnel, k=30"),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc=INK2,
            mec=INK2,
            ms=4.5,
            label="wind tunnel, k=21 (pre-registered)",
        ),
    ]
    fig.suptitle("F1 gap to the coverage rule", fontsize=8.5, x=0.08, ha="left")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/coverage-prediction.png"))
