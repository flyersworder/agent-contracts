"""WT coverage-curve probe: do wind-tunnel arm gaps match random lists of equal coverage?

Pre-registered in `docs/superpowers/specs/2026-09-29-wt-coverage-curve-prereg.md`.
Nothing in `analyze` may change after that file is committed.

The light-tunnel k=30 analysis (arm gap to the coverage rule against what
LLM-free lists of the same coverage lose) was chosen after seeing the data.
This repeats it out of sample on the second chamber, at WT k=21 -- the only WT
budget where arms fall several variables short of the rule. The WT menu has 28
entries over 21 variables and only `hatch`, `load_in` and `load_out` have more
than one entry, so a 21-entry list covers between 14 and 21 variables.

    python -m evaluation.chamber_pipeline.wt_coverage_curve_probe generate
    python -m evaluation.chamber_pipeline.wt_coverage_curve_probe analyze
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

RUNS = Path("runs")
LISTS = RUNS / "probe-wtcc-lists.parquet"
#: The random lists are re-scored on the VPS and pulled here.
VPS_DIR = Path("runs-vps/cc")
LISTS_STEM = "probe-wtcc-lists-rescored"
#: The paper's VPS corpus re-score: WT arms and the rule.
PAPER_RUNS = Path("paper/aamas2027/runs")
CORPUS_STEM = "rescored-vps"

K = 21
LEVELS = range(14, 22)  # every feasible coverage at k=21
LISTS_PER_LEVEL = 30
RULE = "wt_coverage_max"
#: The paper's arms at WT k=21 (DeepSeek). `critique` is not defined in the paper.
ARMS = ("llm_pc", "one_shot", "team", "team_varsplit", "shared_blackboard")
#: Equivalence margin on the mean residual, fixed before scoring.
MARGIN = 0.015
CAPS = (300, 1500)


def exact_coverage_list(
    menu: list[str], var_of: Callable[[str], str], k: int, m: int, rng: random.Random
) -> list[str]:
    """k distinct menu entries covering exactly m distinct variables, order shuffled."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in menu:
        groups[var_of(e)].append(e)
    variables = sorted(groups)
    for _ in range(10_000):
        chosen = rng.sample(variables, m)
        if sum(len(groups[v]) for v in chosen) >= k:
            break
    else:  # pragma: no cover - infeasible (k, m) would be a design error
        raise ValueError(f"no feasible list for k={k}, m={m}")
    picks = [rng.choice(groups[v]) for v in chosen]
    rest = [e for v in chosen for e in groups[v] if e not in picks]
    picks += rng.sample(rest, k - m)
    rng.shuffle(picks)
    return picks


def generate(menu: list[str], var_of: Callable[[str], str]) -> pd.DataFrame:
    rows = []
    for m in LEVELS:
        for s in range(LISTS_PER_LEVEL):
            buy = exact_coverage_list(menu, var_of, K, m, random.Random(f"wtcc:{K}:{m}:{s}"))
            assert len(buy) == K == len(set(buy)) and len({var_of(e) for e in buy}) == m
            rows.append(
                {
                    "chamber": "wt",
                    "configuration": "standard",
                    "agent_name": f"exact_m{m}",
                    "budget_k": K,
                    "seed": s,
                    "m": m,
                    "status": "ok",
                    "chosen_experiments": ",".join(buy),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis
def _wt_resolver() -> Callable[[str], str]:
    from .analyze_headroom import variable_resolver
    from .oracle_probe import _load

    return variable_resolver("wt", _load("wt")[3])


def _design_frame(path_stem: Path, rows: int, var_of: Callable[[str], str]) -> pd.DataFrame:
    """One row per (design, arm) at WT k=21; F1 = mean over the 9 PC seeds (OpenBLAS only)."""
    b = pd.read_parquet(f"{path_stem}-rows{rows}-bykey.parquet")
    assert b["blas_backend"].unique().tolist() == ["scipy-openblas"], b["blas_backend"].unique()
    c = pd.read_parquet(f"{path_stem}-rows{rows}.parquet")
    c = c[
        (c["status"] == "ok")
        & c["chosen_experiments"].notna()
        & (c["chamber"] == "wt")
        & (c["budget_k"] == K)
    ]
    d = c.drop_duplicates(["design_key", "agent_name"])[
        ["design_key", "agent_name", "chosen_experiments"]
    ]
    d = d.join(b.groupby("design_key")["f1"].mean(), on="design_key").dropna(subset=["f1"])
    d["nv"] = [len({var_of(e) for e in s.split(",")}) for s in d["chosen_experiments"]]
    return d.reset_index(drop=True)


def _curve(cal: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    means = cal.groupby("nv")["f1"].mean()
    xs, ys = means.index.to_numpy(float), means.to_numpy()
    return lambda nv: np.interp(nv, xs, ys)  # flat beyond the measured range


def _evaluate(
    cal: pd.DataFrame, rule: pd.Series, arms: dict[str, pd.DataFrame]
) -> dict[str, tuple[float, float]]:
    """Per arm: (predicted gap, measured gap) to the rule. The prediction never uses an arm's F1."""
    curve = _curve(cal)
    r = float(rule.mean())
    return {
        a: (float(curve(d["nv"].to_numpy()).mean()) - r, float(d["f1"].mean()) - r)
        for a, d in arms.items()
    }


def _verdict(lo: float, hi: float) -> str:
    if lo >= -MARGIN and hi <= MARGIN:
        return f"A: coverage accounts for the gaps (CI inside +/-{MARGIN})"
    if lo > 0:
        return "B: arms beat random lists of equal coverage (CI above 0)"
    if hi < 0:
        return "C: arms lose more than coverage predicts (CI below 0)"
    return f"D: inconclusive (CI contains 0 but leaves +/-{MARGIN})"


def analyze(n_boot: int = 2000) -> None:
    rng = np.random.default_rng(20260929)
    var_of = _wt_resolver()
    for rows in CAPS:
        cal = _design_frame(VPS_DIR / LISTS_STEM, rows, var_of)
        corpus = _design_frame(PAPER_RUNS / CORPUS_STEM, rows, var_of)
        rule = corpus.loc[corpus["agent_name"] == RULE, "f1"]
        arms = {a: corpus[corpus["agent_name"] == a].reset_index(drop=True) for a in ARMS}
        levels = [np.flatnonzero((cal["nv"] == m).to_numpy()) for m in LEVELS]
        assert all(len(g) == LISTS_PER_LEVEL for g in levels), [len(g) for g in levels]

        point = _evaluate(cal, rule, arms)
        boots = []
        for _ in range(n_boot):
            cal_b = cal.iloc[np.concatenate([rng.choice(g, len(g)) for g in levels])]
            rule_b = rule.iloc[rng.choice(len(rule), len(rule))]
            arms_b = {a: d.iloc[rng.choice(len(d), len(d))] for a, d in arms.items()}
            boots.append(_evaluate(cal_b, rule_b, arms_b))

        print(f"\n===== WT k={K}, {rows} rows =====")
        print(
            "curve (mean F1 of exact-m lists):", cal.groupby("nv")["f1"].mean().round(3).to_dict()
        )
        print(
            f"rule {RULE}: F1 {rule.mean():.3f} (n={len(rule)}), coverage {corpus.loc[corpus['agent_name'] == RULE, 'nv'].mean():.1f}"
        )

        # informativeness: does coverage move F1 over the arms' range at all?
        def span(c: pd.DataFrame) -> float:
            hi_lo = _curve(c)(np.array([21.0, 16.0]))
            return float(hi_lo[0] - hi_lo[1])

        spans = [
            span(cal.iloc[np.concatenate([rng.choice(g, len(g)) for g in levels])])
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(spans, [2.5, 97.5])
        print(
            f"curve(21) - curve(16) = {span(cal):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            + ("" if lo > 0 else "  -> UNINFORMATIVE: coverage does not move F1 here")
        )
        print(
            f"{'arm':18s} {'n':>4s} {'vars':>5s} {'pred':>7s} {'meas':>7s} {'resid':>7s}  95% CI of resid"
        )
        resid_boot = {a: np.array([b[a][1] - b[a][0] for b in boots]) for a in ARMS}
        for a in ARMS:
            p, m = point[a]
            lo_a, hi_a = np.percentile(resid_boot[a], [2.5, 97.5])
            print(
                f"{a:18s} {len(arms[a]):4d} {arms[a]['nv'].mean():5.1f} {p:+7.3f} {m:+7.3f} {m - p:+7.3f}  [{lo_a:+.3f}, {hi_a:+.3f}]"
            )
        mean_resid = float(np.mean([point[a][1] - point[a][0] for a in ARMS]))
        mb = np.mean([resid_boot[a] for a in ARMS], axis=0)
        lo, hi = np.percentile(mb, [2.5, 97.5])
        print(
            f"PRIMARY mean residual over {len(ARMS)} arms: {mean_resid:+.3f} [{lo:+.3f}, {hi:+.3f}]  {_verdict(lo, hi)}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("generate", "analyze"))
    a = p.parse_args(argv)
    if a.step == "generate":
        from .oracle_probe import _load

        df = generate(list(_load("wt")[0]), _wt_resolver())
        df.to_parquet(LISTS, index=False)
        print(f"wrote {len(df)} lists to {LISTS}")
    else:
        analyze()


if __name__ == "__main__":
    main(sys.argv[1:])
