"""How much of the optimum is reachable by ANY selection policy? An oracle probe.

The top-ranked threat to the topology result is "the task is coverage-shaped,
so a coverage rule tying every LLM arm is a benchmark artefact". The reply so
far has been to scope it. This module measures it instead.

If the value of buying an experiment barely depends on what was already
bought, then no sequential, coordinated or adaptive policy can beat a fixed
ranking, and the ties in the results table are a CEILING property of the task,
not a failure of the agents. If the value does depend on context, then there
is headroom that every topology we built failed to find, which is a stronger
negative result. Either answer is reportable; the current draft has neither.

Two measurements, both LLM-free and both using the ground truth (which is
exactly what makes them oracles rather than agents):

1. **Greedy oracle curve.** Forward greedy selection on the TRUE mean F1
   (averaged over PC subsample seeds). At every budget this is a lower bound
   on the best achievable F1 by any policy. The gap between this curve and
   the coverage rule is the most any agent could have gained over the rule;
   the gap between it and the best LLM arm is the most any topology could
   have gained over what we built. A one-swap local search from the greedy
   prefix at each reported budget tightens the bound.

2. **Context dependence of marginal value.** For random contexts S of size
   k-1 and every candidate e, the marginal gain g(e | S). Decomposed into a
   between-candidate component (intrinsic value of e) and a within-candidate
   component (dependence on S), against PC noise measured at the same seeds.
   Near-zero within-candidate variance means the value function is close to
   modular and a static ranking is optimal — which is what the coverage rule
   is.

Runs on one machine and one BLAS backend by construction; the numbers are
comparable with the Accelerate re-scored corpus only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

from .inference import pool_experiment_data, run_pc
from .menu_taxonomy import coverage_ordered
from .scoring import f1_edges
from .wt_menu_taxonomy import coverage_ordered as wt_coverage_ordered

if TYPE_CHECKING:
    from collections.abc import Sequence

PC_ALPHA = 0.05

# Per-process cache: the full menu's data, fetched once. Workers are forked
# so each builds its own on first use.
_CACHE: dict[str, tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame, list[str]]] = {}


def _load(chamber: str) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    if chamber not in _CACHE:
        adapter = create_contracted_chamber_agent(
            chamber=chamber,  # type: ignore[arg-type]
            configuration="standard",
            intervention_budget=10_000,
        )
        menu = list(adapter.available_experiments())
        data = {name: adapter.query_intervention(name) for name in menu}
        truth = adapter.ground_truth()
        nodes = list(truth.index)
        _CACHE[chamber] = (menu, data, truth, nodes)
    return _CACHE[chamber]


def score(chamber: str, names: Sequence[str], seeds: Sequence[int]) -> float:
    """Mean directed-edge F1 of the ORDERED buy `names` over PC seeds."""
    _, data, truth, nodes = _load(chamber)
    pooled = pool_experiment_data([data[n] for n in names], nodes)
    return float(
        np.mean([f1_edges(run_pc(pooled, nodes, alpha=PC_ALPHA, seed=s), truth) for s in seeds])
    )


def _score_task(task: tuple[str, tuple[str, ...], tuple[int, ...]]) -> float:
    chamber, names, seeds = task
    return score(chamber, list(names), list(seeds))


def greedy_oracle(
    chamber: str,
    k_max: int,
    *,
    seeds: Sequence[int],
    workers: int,
    log: Any = print,
) -> list[dict[str, Any]]:
    menu, _, _, _ = _load(chamber)
    chosen: list[str] = []
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for step in range(1, k_max + 1):
            candidates = [n for n in menu if n not in chosen]
            t0 = time.time()
            scores = list(
                pool.map(
                    _score_task,
                    [(chamber, (*chosen, c), tuple(seeds)) for c in candidates],
                )
            )
            best = int(np.argmax(scores))
            chosen.append(candidates[best])
            rows.append(
                {
                    "chamber": chamber,
                    "k": step,
                    "f1_greedy": scores[best],
                    "chosen": json.dumps(chosen),
                    "n_candidates": len(candidates),
                    "cand_min": float(min(scores)),
                    "cand_median": float(np.median(scores)),
                }
            )
            log(
                f"[{chamber}] k={step:2d} greedy F1={scores[best]:.4f} "
                f"(median cand {np.median(scores):.4f}) +{candidates[best]} {time.time() - t0:.0f}s"
            )
    return rows


def one_swap_improve(
    chamber: str,
    chosen: list[str],
    *,
    seeds: Sequence[int],
    workers: int,
    max_passes: int = 2,
    log: Any = print,
) -> tuple[list[str], float]:
    """Local search: replace one bought experiment by one unbought, accept if better."""
    menu, _, _, _ = _load(chamber)
    current = list(chosen)
    best = score(chamber, current, seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in range(max_passes):
            improved = False
            outside = [n for n in menu if n not in current]
            tasks = []
            for i in range(len(current)):
                for o in outside:
                    trial = list(current)
                    trial[i] = o
                    tasks.append((i, o, trial))
            scores = list(
                pool.map(_score_task, [(chamber, tuple(t[2]), tuple(seeds)) for t in tasks])
            )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-9:
                best = scores[j]
                current = tasks[j][2]
                improved = True
                log(f"[{chamber}] swap k={len(current)}: {tasks[j][1]} in -> F1 {best:.4f}")
            if not improved:
                break
    return current, best


def context_dependence(
    chamber: str,
    k: int,
    *,
    n_contexts: int,
    seeds: Sequence[int],
    workers: int,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Marginal gain of every candidate in each of `n_contexts` random contexts of size k-1."""
    menu, _, _, _ = _load(chamber)
    rng = random.Random(rng_seed)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for ctx in range(n_contexts):
            base = rng.sample(menu, k - 1)
            base_score = score(chamber, base, seeds)
            cands = [n for n in menu if n not in base]
            gains = list(
                pool.map(_score_task, [(chamber, (*base, c), tuple(seeds)) for c in cands])
            )
            for c, g in zip(cands, gains, strict=True):
                rows.append(
                    {
                        "chamber": chamber,
                        "k": k,
                        "context": ctx,
                        "candidate": c,
                        "base_f1": base_score,
                        "gain": g - base_score,
                    }
                )
    return pd.DataFrame(rows)


def reference_policies(
    chamber: str, budgets: Sequence[int], *, seeds: Sequence[int], n_rule_seeds: int = 10
) -> list[dict[str, Any]]:
    """Coverage rule and random at the same PC seeds, for a same-machine comparison."""
    menu, _, _, nodes = _load(chamber)
    rows: list[dict[str, Any]] = []
    for k in budgets:
        for s in range(n_rule_seeds):
            if chamber == "lt":
                rule = coverage_ordered(
                    list(menu), k, s, maximize=True, exclude_strengths=("weak",)
                )
            else:
                rule = wt_coverage_ordered(list(menu), k, s, nodes, maximize=True)
            rows.append(
                {
                    "chamber": chamber,
                    "k": k,
                    "policy": "coverage_rule",
                    "seed": s,
                    "f1": score(chamber, rule, seeds),
                }
            )
            rnd = random.Random(1000 + s).sample(menu, k)
            rows.append(
                {
                    "chamber": chamber,
                    "k": k,
                    "policy": "random",
                    "seed": s,
                    "f1": score(chamber, rnd, seeds),
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chamber", default="lt")
    p.add_argument("--k-max", type=int, default=45)
    p.add_argument("--budgets", default="6,30,45")
    p.add_argument("--seeds", type=int, default=5, help="PC seeds during search")
    p.add_argument("--report-seeds", type=int, default=9, help="PC seeds for reported figures")
    p.add_argument("--contexts", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/oracle-probe")
    a = p.parse_args(argv)

    budgets = [int(b) for b in a.budgets.split(",")]
    search_seeds = list(range(a.seeds))
    report_seeds = list(range(a.report_seeds))
    out = a.out

    greedy = greedy_oracle(a.chamber, a.k_max, seeds=search_seeds, workers=a.workers)
    pd.DataFrame(greedy).to_parquet(f"{out}-{a.chamber}-greedy.parquet", index=False)

    summary: list[dict[str, Any]] = []
    for k in budgets:
        prefix = json.loads(greedy[k - 1]["chosen"])
        g9 = score(a.chamber, prefix, report_seeds)
        swapped, _ = one_swap_improve(a.chamber, prefix, seeds=search_seeds, workers=a.workers)
        s9 = score(a.chamber, swapped, report_seeds)
        summary.append(
            {
                "chamber": a.chamber,
                "k": k,
                "policy": "greedy_oracle",
                "f1": g9,
                "chosen": json.dumps(prefix),
            }
        )
        summary.append(
            {
                "chamber": a.chamber,
                "k": k,
                "policy": "greedy+swap_oracle",
                "f1": s9,
                "chosen": json.dumps(swapped),
            }
        )
        print(f"[{a.chamber}] k={k}: greedy {g9:.4f}  greedy+swap {s9:.4f}")
    refs = reference_policies(a.chamber, budgets, seeds=report_seeds)
    pd.DataFrame(summary + refs).to_parquet(f"{out}-{a.chamber}-summary.parquet", index=False)
    ref = pd.DataFrame(refs).groupby(["k", "policy"]).f1.agg(["mean", "std"])
    print(ref)

    ctx_frames = [
        context_dependence(
            a.chamber, k, n_contexts=a.contexts, seeds=search_seeds, workers=a.workers
        )
        for k in budgets
        if k > 1
    ]
    ctx = pd.concat(ctx_frames, ignore_index=True)
    ctx.to_parquet(f"{out}-{a.chamber}-context.parquet", index=False)
    for k, sub in ctx.groupby("k"):
        piv = sub.pivot(index="context", columns="candidate", values="gain")
        between = float(piv.mean(axis=0).var(ddof=1))
        within = float(piv.var(axis=0, ddof=1).mean())
        print(
            f"[{a.chamber}] k={k}: gain sd between candidates {np.sqrt(between):.4f}, "
            f"within candidate across contexts {np.sqrt(within):.4f}, "
            f"mean gain {np.nanmean(piv.values):+.4f}, frac contexts with any negative-gain best "
            f"{(piv.max(axis=1) < 0).mean():.2f}"
        )


if __name__ == "__main__":
    main()
