"""How much of the optimum is reachable by ANY selection policy? An oracle probe.

The top-ranked threat to the topology result was "the task is coverage-shaped,
so a coverage rule tying every LLM arm is a benchmark artefact". The reply had
been to scope it. This module measures it instead, LLM-free and with the
ground truth as an oracle (which is what makes these oracles, not agents):

1. **Greedy oracle curve.** Forward greedy on the TRUE mean F1 over PC
   subsample seeds, then a one-swap local search whose acceptance threshold
   is the MEASURED noise floor of a gain at that seed count (a swap that
   clears 1e-9 on noisy means is a max over noise, not an improvement). A
   lower bound on the best reachable set.
2. **Static ranking.** Every experiment's mean marginal gain over random
   contexts of size k-1; the top-k by that ranking is a policy. If it
   matches the greedy set, the value function is close to modular and a
   fixed ranking is all an agent would need to know.
3. **Reference policies at the same seeds** -- the coverage rule and random
   -- plus describable rule variants on LT (mid-only; no strong-source
   experiments).
4. **Arm purchases scored on the oracle scale.** Mean marginal gain of the
   experiments each recorded arm actually bought, against random's.

**Seeds are disjoint by construction.** Search uses `range(seeds)`; every
reported figure is re-scored at `report_seed_offset + range(report_seeds)`.
An oracle's own search score is a max over noisy candidates (winner's curse):
at LT k=30 the greedy+swap set read 0.474 in-search and 0.432 fresh. The
tool refuses overlapping seed sets rather than trusting the caller.

**Provenance.** Every emitted row carries the pipeline's standard stamp
(`chosen_experiments`, `design_key`, `selection_key`, `pc_alpha`,
`pc_max_rows`, `blas_backend`, `platform_tag`) so the frames can be fed to
`rescore_selections` and caught by the backend-mismatch guards. Numbers from
one backend are comparable only with corpus files on that backend.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

from .inference import pc_call_defaults, pool_experiment_data, run_pc, runtime_fingerprint
from .menu_taxonomy import coverage_ordered, experiment_variable
from .rescore import LT_CASE_STUDY_NODES, design_key, parse_selection, selection_key
from .scoring import f1_edges
from .wt_menu_taxonomy import coverage_ordered as wt_coverage_ordered

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIGURATION = "standard"
#: Budgets per chamber, the same grid the ladders and Phase 2 used.
DEFAULT_BUDGETS: dict[str, tuple[int, ...]] = {"lt": (6, 30, 45), "wt": (7, 14, 21)}
#: LT variables whose STRONG interventions the ranking penalises (light
#: sources and polarisers; strong pushes sensors off the linear regime).
LT_SOURCE_VARIABLES = frozenset(
    {
        "red",
        "green",
        "blue",
        "pol_1",
        "pol_2",
        "l_11",
        "l_12",
        "l_21",
        "l_22",
        "l_31",
        "l_32",
        "reference",
    }
)

# Per-process cache of the full menu's data; workers populate it via the
# pool initializer, the parent on first use.
_CACHE: dict[str, tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame, list[str]]] = {}


def _load(chamber: str) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    if chamber not in _CACHE:
        adapter = create_contracted_chamber_agent(
            chamber=chamber,  # type: ignore[arg-type]
            configuration=CONFIGURATION,
            intervention_budget=10_000,
        )
        menu = list(adapter.available_experiments())
        data = {name: adapter.query_intervention(name) for name in menu}
        truth = adapter.ground_truth()
        _CACHE[chamber] = (menu, data, truth, list(truth.index))
    return _CACHE[chamber]


def _core_nodes(nodes: Sequence[str]) -> list[str]:
    core = [n for n in LT_CASE_STUDY_NODES if n in nodes]
    return core if len(core) == len(LT_CASE_STUDY_NODES) else []


def score(chamber: str, names: Sequence[str], seeds: Sequence[int]) -> tuple[float, float]:
    """`(f1, f1_core)` of the ORDERED buy, each a mean over PC seeds; core is NaN off LT."""
    _, data, truth, nodes = _load(chamber)
    pooled = pool_experiment_data([data[n] for n in names], nodes)
    core = _core_nodes(nodes)
    alpha = float(pc_call_defaults()["alpha"])
    f1s: list[float] = []
    cores: list[float] = []
    for s in seeds:
        predicted = run_pc(pooled, nodes, alpha=alpha, seed=s, max_rows=_pc_max_rows())
        f1s.append(f1_edges(predicted, truth))
        cores.append(
            f1_edges(predicted.loc[core, core], truth.loc[core, core]) if core else float("nan")
        )
    return float(np.mean(f1s)), float(np.mean(cores))


def _pc_max_rows() -> int | None:
    """PC row cap for every scoring call, overridable via ORACLE_PC_MAX_ROWS.

    An environment variable rather than an argument because `Scorer`'s
    workers are spawned processes: the environment reaches them, a module
    global set in `main` does not. Measured 2026-09-09: the best selection
    CHANGES with this cap (register §34), so it is stamped into every output.
    """
    raw = os.environ.get("ORACLE_PC_MAX_ROWS")
    if raw is None:
        return pc_call_defaults()["max_rows"]  # type: ignore[no-any-return]
    return None if raw.lower() == "none" else int(raw)


def _score_task(task: tuple[str, tuple[str, ...], tuple[int, ...]]) -> tuple[float, float]:
    chamber, names, seeds = task
    return score(chamber, list(names), list(seeds))


class Scorer:
    """One process pool for the whole run, workers pre-loaded with the menu data."""

    def __init__(self, chamber: str, workers: int) -> None:
        self.chamber = chamber
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_load, initargs=(chamber,))

    def many(
        self, buys: Sequence[Sequence[str]], seeds: Sequence[int]
    ) -> list[tuple[float, float]]:
        return list(
            self.pool.map(_score_task, [(self.chamber, tuple(b), tuple(seeds)) for b in buys])
        )

    def f1(self, buys: Sequence[Sequence[str]], seeds: Sequence[int]) -> list[float]:
        return [f for f, _ in self.many(buys, seeds)]

    def close(self) -> None:
        self.pool.shutdown()


def _stamp(chamber: str, names: Sequence[str]) -> dict[str, Any]:
    fp = runtime_fingerprint()
    defaults = pc_call_defaults()
    return {
        "chamber": chamber,
        "configuration": CONFIGURATION,
        "chosen_experiments": ",".join(names),
        "design_key": design_key(chamber, CONFIGURATION, names),
        "selection_key": selection_key(chamber, CONFIGURATION, names),
        "n_experiments": len(names),
        "pc_alpha": float(defaults["alpha"]),
        "pc_max_rows": _pc_max_rows(),
        "blas_backend": fp["blas"],
        "platform_tag": fp["platform"],
    }


def noise_floor(scorer: Scorer, seeds: Sequence[int], k: int, *, n_sets: int = 30) -> float:
    """SD of a GAIN (difference of two `len(seeds)`-seed means) from PC noise alone.

    Same random sets scored at `seeds` and at a shifted, disjoint seed block.
    """
    menu, _, _, _ = _load(scorer.chamber)
    sets = [random.Random(5000 + i).sample(menu, k) for i in range(n_sets)]
    shifted = [s + 1000 for s in seeds]
    a = np.array(scorer.f1(sets, seeds))
    b = np.array(scorer.f1(sets, shifted))
    return float((a - b).std(ddof=1))


def greedy_oracle(
    scorer: Scorer, k_max: int, *, seeds: Sequence[int], log: Any = print
) -> list[dict[str, Any]]:
    menu, _, _, _ = _load(scorer.chamber)
    k_max = min(k_max, len(menu))
    chosen: list[str] = []
    rows: list[dict[str, Any]] = []
    for step in range(1, k_max + 1):
        candidates = [n for n in menu if n not in chosen]
        t0 = time.time()
        scores = scorer.f1([[*chosen, c] for c in candidates], seeds)
        best = int(np.argmax(scores))
        chosen.append(candidates[best])
        rows.append(
            {
                **_stamp(scorer.chamber, chosen),
                "k": step,
                "f1_search": scores[best],
                "cand_median_search": float(np.median(scores)),
            }
        )
        log(
            f"[{scorer.chamber}] greedy k={step:2d} search F1={scores[best]:.4f} "
            f"+{candidates[best]} {time.time() - t0:.0f}s"
        )
    return rows


def one_swap_improve(
    scorer: Scorer,
    chosen: Sequence[str],
    *,
    seeds: Sequence[int],
    threshold: float,
    max_passes: int = 2,
    log: Any = print,
) -> list[str]:
    """Replace one bought experiment by one unbought while the gain clears `threshold`."""
    menu, _, _, _ = _load(scorer.chamber)
    current = list(chosen)
    best = scorer.f1([current], seeds)[0]
    for _ in range(max_passes):
        outside = [n for n in menu if n not in current]
        trials = [
            [*current[:i], o, *current[i + 1 :]] for i in range(len(current)) for o in outside
        ]
        if not trials:
            break
        scores = scorer.f1(trials, seeds)
        j = int(np.argmax(scores))
        if scores[j] - best <= threshold:
            break
        best, current = scores[j], trials[j]
        log(f"[{scorer.chamber}] swap k={len(current)} -> search F1 {best:.4f}")
    return current


def context_dependence(
    scorer: Scorer, k: int, *, n_contexts: int, seeds: Sequence[int], rng_seed: int = 0
) -> pd.DataFrame:
    """Marginal gain of every candidate in each of `n_contexts` random contexts of size k-1."""
    menu, _, _, _ = _load(scorer.chamber)
    rng = random.Random(rng_seed)
    bases = [rng.sample(menu, k - 1) for _ in range(n_contexts)]
    base_scores = scorer.f1(bases, seeds)
    rows: list[dict[str, Any]] = []
    for ctx, (base, base_score) in enumerate(zip(bases, base_scores, strict=True)):
        cands = [n for n in menu if n not in base]
        gains = scorer.f1([[*base, c] for c in cands], seeds)
        rows.extend(
            {
                "chamber": scorer.chamber,
                "k": k,
                "context": ctx,
                "candidate": c,
                "base_f1": base_score,
                "gain": g - base_score,
            }
            for c, g in zip(cands, gains, strict=True)
        )
    return pd.DataFrame(rows)


def ranking_from_contexts(context: pd.DataFrame, k: int | None = None) -> pd.Series:
    """Experiments ordered by mean marginal gain, at one budget or pooled over all."""
    sub = context if k is None else context[context.k == k]
    return sub.groupby("candidate").gain.mean().sort_values(ascending=False)


def reference_buys(chamber: str, k: int, seed: int) -> dict[str, list[str]]:
    """The LLM-free policies, at one budget and rule seed."""
    menu, _, _, nodes = _load(chamber)
    out: dict[str, list[str]] = {}
    if chamber == "lt":
        out["coverage_rule"] = coverage_ordered(
            menu, k, seed, maximize=True, exclude_strengths=("weak",)
        )
        out["rule_mid_only"] = coverage_ordered(
            menu, k, seed, maximize=True, exclude_strengths=("weak", "strong")
        )
        keep = [
            n
            for n in menu
            if not (experiment_variable(n) in LT_SOURCE_VARIABLES and n.endswith("_strong"))
        ]
        out["rule_no_strong_sources"] = coverage_ordered(
            keep, k, seed, maximize=True, exclude_strengths=("weak",)
        )
    else:
        out["coverage_rule"] = wt_coverage_ordered(menu, k, seed, nodes, maximize=True)
    out["random"] = random.Random(1000 + seed).sample(menu, k)
    return out


def arm_purchase_gains(corpus: pd.DataFrame, chamber: str, ranking: pd.Series) -> pd.DataFrame:
    """Mean oracle marginal gain of the experiments each recorded arm bought, per cell."""
    sub = corpus[(corpus.chamber == chamber) & corpus.chosen_experiments.notna()]
    rows = []
    for _, r in sub.iterrows():
        names = parse_selection(r["chosen_experiments"])
        if not names:
            continue
        rows.append(
            {
                "chamber": chamber,
                "agent_name": r["agent_name"],
                "budget_k": int(r["budget_k"]),
                "seed": r.get("seed"),
                "mean_oracle_gain": float(np.mean([ranking.get(n, np.nan) for n in names])),
            }
        )
    return pd.DataFrame(rows)


def mde(sd_a: float, n_a: int, sd_b: float, n_b: int) -> float:
    """2.8 x pooled sd x sqrt(1/n_a + 1/n_b) -- the unequal-n form."""
    pooled = float(np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / max(n_a + n_b - 2, 1)))
    return 2.8 * pooled * float(np.sqrt(1 / n_a + 1 / n_b))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chamber", default="lt", choices=("lt", "wt"))
    p.add_argument("--budgets", default=None, help="comma list; default is the chamber's grid")
    p.add_argument("--k-max", type=int, default=None, help="greedy horizon; default max(budgets)")
    p.add_argument("--seeds", type=int, default=5, help="PC seeds during search")
    p.add_argument("--report-seeds", type=int, default=9, help="PC seeds for every reported figure")
    p.add_argument("--report-seed-offset", type=int, default=100)
    p.add_argument("--contexts", type=int, default=20)
    p.add_argument("--rule-seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--corpus", default="runs/rescored-single-backend.parquet")
    p.add_argument("--out", default="runs/oracle-probe")
    p.add_argument(
        "--pc-max-rows",
        default=None,
        help="PC row cap for every score (default: the pipeline's 300). Sets ORACLE_PC_MAX_ROWS.",
    )
    a = p.parse_args(argv)
    if a.pc_max_rows is not None:
        os.environ["ORACLE_PC_MAX_ROWS"] = str(a.pc_max_rows)

    chamber = a.chamber
    budgets = (
        [int(b) for b in a.budgets.split(",")] if a.budgets else list(DEFAULT_BUDGETS[chamber])
    )
    search_seeds = list(range(a.seeds))
    report_seeds = [a.report_seed_offset + i for i in range(a.report_seeds)]
    if set(search_seeds) & set(report_seeds):
        raise SystemExit("search and report seed sets overlap; reported figures would be biased")
    menu, _, _, _ = _load(chamber)
    k_max = min(a.k_max or max(budgets), len(menu))
    if max(budgets) > k_max:
        raise SystemExit(f"budgets {budgets} exceed k_max={k_max} (menu has {len(menu)})")
    out = f"{a.out}-{chamber}"
    scorer = Scorer(chamber, a.workers)
    try:
        # 1. greedy curve, then swap at each budget with a noise-gated acceptance
        greedy = greedy_oracle(scorer, k_max, seeds=search_seeds)
        gframe = pd.DataFrame(greedy)
        gframe["f1_fresh"] = scorer.f1(
            [parse_selection(r) for r in gframe.chosen_experiments], report_seeds
        )
        gframe.to_parquet(f"{out}-greedy.parquet", index=False)

        policies: list[dict[str, Any]] = []
        floors: dict[int, float] = {}
        for k in budgets:
            floors[k] = noise_floor(scorer, search_seeds, k)
            print(
                f"[{chamber}] k={k}: noise sd of a {len(search_seeds)}-seed gain = {floors[k]:.4f}"
            )
            prefix = parse_selection(gframe.loc[gframe.k == k, "chosen_experiments"].iloc[0])
            swapped = one_swap_improve(scorer, prefix, seeds=search_seeds, threshold=floors[k])
            policies.append(
                {**_stamp(chamber, prefix), "k": k, "policy": "greedy_oracle", "seed": 0}
            )
            policies.append(
                {**_stamp(chamber, swapped), "k": k, "policy": "greedy+swap_oracle", "seed": 0}
            )

        # 2. context dependence -> static rankings
        context = pd.concat(
            [
                context_dependence(scorer, k, n_contexts=a.contexts, seeds=search_seeds)
                for k in budgets
            ],
            ignore_index=True,
        )
        context["noise_sd_gain"] = context.k.map(floors)
        context.to_parquet(f"{out}-context.parquet", index=False)
        pooled_rank = ranking_from_contexts(context)
        for k in budgets:
            at_k = ranking_from_contexts(context, k)
            for s in range(a.rule_seeds):
                for name, rank in (("rank_at_k", at_k), ("rank_pooled", pooled_rank)):
                    order = list(rank.index[:k])
                    random.Random(s).shuffle(order)  # pooling order is a nuisance; average it out
                    policies.append({**_stamp(chamber, order), "k": k, "policy": name, "seed": s})
                for name, buy in reference_buys(chamber, k, s).items():
                    policies.append({**_stamp(chamber, buy), "k": k, "policy": name, "seed": s})

        # 3. everything reported is scored at the disjoint report seeds
        scored = scorer.many(
            [parse_selection(r["chosen_experiments"]) for r in policies], report_seeds
        )
        pframe = pd.DataFrame(policies)
        pframe["f1"] = [f for f, _ in scored]
        pframe["f1_core"] = [c for _, c in scored]
        pframe["report_seeds"] = json.dumps(report_seeds)
        pframe.to_parquet(f"{out}-policies.parquet", index=False)
        print(
            pframe.groupby(["k", "policy"])[["f1", "f1_core"]]
            .agg(["mean", "std", "size"])
            .round(4)
            .to_string()
        )

        # 4. recorded arms on the oracle scale
        try:
            corpus = pd.read_parquet(a.corpus)
        except FileNotFoundError:
            print(f"[{chamber}] no corpus at {a.corpus}; skipping arm purchase scoring")
        else:
            gains = arm_purchase_gains(corpus, chamber, pooled_rank)
            gains.to_parquet(f"{out}-arm-gain.parquet", index=False)
            table = gains.groupby(["agent_name", "budget_k"]).mean_oracle_gain.mean().unstack()
            print((table * 1000).round(1).to_string())
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
