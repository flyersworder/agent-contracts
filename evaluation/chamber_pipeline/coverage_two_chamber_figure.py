"""Arm gaps to the coverage rule against the coverage-only prediction, LT k=30 and WT k=21.

Each panel: x = distinct variables short of the coverage rule; y = F1 minus the
rule's F1. Dots joined by a dashed line are the raw mean F1 of random lists of
exactly that coverage (30 lists per level, no language model), minus the rule --
the prediction both pre-registrations interpolate. Markers are arms (pooled over
their sources, as in the paper's Figure 2) with 95% bootstrap CIs; filled =
DeepSeek, open = GLM (LT only). Every score is VPS/OpenBLAS; the loader asserts it.

The WT panel is the pre-registered test (2026-09-29-wt-coverage-curve-prereg);
the LT panel is the post-hoc analysis that motivated it.

    python -m evaluation.chamber_pipeline.coverage_two_chamber_figure [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import coverage_curve_probe as lt_probe
from . import wt_coverage_curve_probe as wt_probe

if TYPE_CHECKING:
    from collections.abc import Callable

VPS = Path("runs-vps/cc")
PAPER = Path("paper/aamas2027/runs")
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#d9d8d3"
COL = {300: BLUE, 1500: ORANGE}
CAPS = (300, 1500)
ARMS = ("llm_pc", "one_shot", "team", "team_varsplit", "shared_blackboard")
MULTI = {"team", "team_varsplit", "shared_blackboard"}
MIN_DESIGNS = 5

#: (panel title, chamber, k, variables, rule, random lists, arm sources [(vendor, stem)], x max)
PANELS = (
    (
        "Light tunnel, k=30 (post hoc)",
        "lt",
        30,
        30,
        "coverage_max_ms",
        VPS / "probe-cc-lists-rescored",
        [("DeepSeek", PAPER / "rescored-vps"), ("GLM", VPS / "rescored-xv-glm-lt-vps")],
        11.3,
    ),
    (
        "Wind tunnel, k=21 (pre-registered)",
        "wt",
        21,
        21,
        "wt_coverage_max",
        VPS / "probe-wtcc-lists-rescored",
        [("DeepSeek", PAPER / "rescored-vps")],
        7.3,
    ),
)
#: hand-placed labels (panel chamber, vendor, arm) -> offset in points
LABELS = {
    ("lt", "DeepSeek", "team"): ("team", (6, -2)),
    ("lt", "GLM", "one_shot"): ("single call (GLM)", (-30, 12)),
    ("wt", "DeepSeek", "team"): ("team", (6, -4)),
}


def load(stem: Path, rows: int, chamber: str, k: int, var_of: Callable[[str], str]) -> pd.DataFrame:
    """One row per (design, arm) at (chamber, k); F1 = mean over 9 PC seeds; OpenBLAS only."""
    b = pd.read_parquet(f"{stem}-rows{rows}-bykey.parquet")
    assert b["blas_backend"].unique().tolist() == ["scipy-openblas"], (
        stem,
        b["blas_backend"].unique(),
    )
    c = pd.read_parquet(f"{stem}-rows{rows}.parquet")
    c = c[
        (c["status"] == "ok")
        & c["chosen_experiments"].notna()
        & (c["chamber"] == chamber)
        & (c["budget_k"] == k)
    ]
    d = c.drop_duplicates(["design_key", "agent_name"])[
        ["design_key", "agent_name", "chosen_experiments"]
    ]
    d = d.join(b.groupby("design_key")["f1"].mean(), on="design_key").dropna(subset=["f1"])
    d["nv"] = [len({var_of(e) for e in s.split(",")}) for s in d["chosen_experiments"]]
    return d.reset_index(drop=True)


def gap_ci(arm: np.ndarray, rule: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    boot = [
        arm[rng.integers(0, len(arm), len(arm))].mean()
        - rule[rng.integers(0, len(rule), len(rule))].mean()
        for _ in range(2000)
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    resolvers = {"lt": lt_probe._var, "wt": wt_probe._wt_resolver()}
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.1), sharey=True, gridspec_kw={"width_ratios": [11.5, 7.5]}
    )
    for ax, (title, ch, k, nvars, rule_name, lists, sources, xmax) in zip(
        axes, PANELS, strict=True
    ):
        var_of = resolvers[ch]
        ax.axhline(0, color=INK2, lw=0.6, zorder=0)
        for rows in CAPS:
            corpus = load(PAPER / "rescored-vps", rows, ch, k, var_of)
            rule = corpus.loc[corpus["agent_name"] == rule_name, "f1"].to_numpy()
            assert (
                len(rule) and (corpus.loc[corpus["agent_name"] == rule_name, "nv"] == nvars).all()
            )
            cal = load(lists, rows, ch, k, var_of)
            means = cal.groupby("nv")["f1"].mean()
            xs = nvars - means.index.to_numpy()
            keep = xs <= xmax
            ax.plot(
                xs[keep],
                means.to_numpy()[keep] - rule.mean(),
                "--",
                color=COL[rows],
                lw=0.9,
                zorder=1,
            )
            ax.plot(
                xs[keep],
                means.to_numpy()[keep] - rule.mean(),
                "o",
                ms=2.3,
                color=COL[rows],
                alpha=0.6,
                zorder=1,
            )
            for vendor, stem in sources:
                d = corpus if stem == PAPER / "rescored-vps" else load(stem, rows, ch, k, var_of)
                for arm in ARMS:
                    x = d[d["agent_name"] == arm]
                    if len(x) < MIN_DESIGNS:
                        continue
                    short = nvars - x["nv"].mean()
                    gap = x["f1"].mean() - rule.mean()
                    lo, hi = gap_ci(x["f1"].to_numpy(), rule, rng)
                    filled = vendor == "DeepSeek"
                    mk = "^" if arm in MULTI else "s"
                    ax.plot([short, short], [lo, hi], color=COL[rows], lw=0.7, alpha=0.75, zorder=2)
                    ax.plot(
                        short,
                        gap,
                        mk,
                        ms=5.2,
                        mfc=COL[rows] if filled else "white",
                        mec=COL[rows],
                        mew=1.0,
                        zorder=3,
                    )
                    if rows == 300 and (ch, vendor, arm) in LABELS:
                        text, off = LABELS[(ch, vendor, arm)]
                        ax.annotate(
                            text,
                            xy=(short, gap),
                            xytext=off,
                            textcoords="offset points",
                            fontsize=6.3,
                            color=INK2,
                        )
        ax.set_xlim(-0.3, xmax)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("distinct variables short of the coverage rule", fontsize=7.5)
        ax.grid(color=GRID, lw=0.4, zorder=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("F1 minus the coverage rule", fontsize=7.5)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=5, label="300 rows"),
        Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=ORANGE, ms=5, label="1500 rows"),
        Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=5, label="single agent"),
        Line2D([], [], marker="^", ls="", mfc=INK2, mec=INK2, ms=5, label="multi-agent"),
        Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=5, label="DeepSeek (filled)"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=5, label="GLM (open)"),
        Line2D(
            [],
            [],
            color=INK2,
            lw=0.9,
            ls="--",
            marker="o",
            ms=2.3,
            label="coverage-only prediction (random lists)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=6.3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.3,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/coverage-two-chamber.png"))
