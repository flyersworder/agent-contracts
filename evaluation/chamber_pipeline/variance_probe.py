"""Separate PC measurement noise from selection-induced variance.

Every chamber cell's `seed` currently controls two things at once: WHICH
experiments a selection-free agent buys, and WHICH 300 rows `run_pc`
subsamples. The spread of `random` cells at a given budget is therefore
`selection variance + PC noise` with no way to tell the two apart, and the
whole-corpus claim that rests on it -- "at k=M selection freedom is zero, so
the spread there is pure PC noise" -- is an argument, not a measurement.

This probe crosses the two seeds instead of tying them. For each budget it
draws `n_selections` independent random buys, and scores each one under
`n_pc_seeds` different subsample seeds. That is a one-way ANOVA layout:

    between-group (selection) variance  =  what the CHOICE is worth
    within-group (pc seed) variance     =  what the MEASUREMENT costs

No LLM is involved, so the only stochastic inputs are the two seeds.

Why it matters: the reading that selection-attributable spread is roughly
constant across budgets assumes PC noise does not grow as the pooled data
shrinks. At k=6 the pooled frame is a tenth the size it is at k=59, and if
noise grows there, an apparent selection effect at small k is measurement.
This measures it rather than assuming it either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import random as _random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from agent_contracts.integrations.causalchamber import (
    ChamberId,
    ConfigId,
    create_contracted_chamber_agent,
)

from .inference import pc_call_defaults, pool_experiment_data, run_pc
from .orchestrator import (
    _PcCollinearHandler,
    _PcDegeneracyHandler,
    _PcZeroVarianceHandler,
)
from .scoring import f1_edges, shd


@dataclass(frozen=True)
class ProbeRecord:
    """One (budget, selection, pc_seed) scored triple."""

    chamber: str
    configuration: str
    budget_k: int
    selection_seed: int
    pc_seed: int
    f1: float
    shd: float
    n_rows_pooled: int
    chosen_experiments: str
    # PC's three degradation paths, counted per run. Recorded because a
    # degradation rate that varies with the budget would show up in this
    # probe as between-selection variance and be misread as the choice
    # mattering more -- the register's recurring failure, one axis over.
    n_collinear_dropped: int
    n_zero_variance_dropped: int
    n_pc_degeneracies: int


def run_probe(
    *,
    chamber: ChamberId = "lt",
    configuration: ConfigId = "standard",
    budgets: tuple[int, ...] = (6, 30, 59),
    n_selections: int = 20,
    n_pc_seeds: int = 10,
    pc_alpha: float = 0.05,
    pc_max_rows: int | None = None,
) -> list[ProbeRecord]:
    """Cross selection seeds against PC subsample seeds; score every cell.

    The adapter is rebuilt per (budget, selection) because
    `create_contracted_chamber_agent` meters `query_intervention` against
    the budget -- reusing one across selections would exhaust it.
    """
    # Resolve the effective row cap ONCE, off the BOUND signature rather
    # than the module constant -- `run_pc`'s docstring is explicit that a
    # reassigned `DEFAULT_MAX_ROWS` would describe a run that never happened.
    effective_max_rows: int | None = (
        pc_call_defaults()["max_rows"] if pc_max_rows is None else pc_max_rows
    )

    records: list[ProbeRecord] = []
    for budget_k in budgets:
        for sel in range(n_selections):
            adapter = create_contracted_chamber_agent(
                chamber=chamber,
                configuration=configuration,
                intervention_budget=budget_k,
            )
            nodes = list(adapter.ground_truth().index)
            menu = adapter.available_experiments()
            k = min(budget_k, len(menu))
            chosen = _random.Random(sel).sample(menu, k)
            # Pool ONCE per selection: the data is a pure function of the
            # buy, so re-querying per pc_seed would only burn budget and
            # invite a per-tool violation.
            pooled = pool_experiment_data(
                [adapter.query_intervention(name) for name in chosen], nodes
            )
            truth = adapter.ground_truth()
            for pc_seed in range(n_pc_seeds):
                # Serial by construction: these handlers attach to the
                # module-global inference logger, so concurrent runs in one
                # process would cross-contaminate the counts.
                inference_logger = logging.getLogger("evaluation.chamber_pipeline.inference")
                collinear = _PcCollinearHandler()
                zero_var = _PcZeroVarianceHandler()
                degenerate = _PcDegeneracyHandler()
                for handler in (collinear, zero_var, degenerate):
                    inference_logger.addHandler(handler)
                try:
                    predicted = run_pc(
                        pooled,
                        nodes,
                        alpha=pc_alpha,
                        seed=pc_seed,
                        max_rows=effective_max_rows,
                    )
                finally:
                    for handler in (collinear, zero_var, degenerate):
                        inference_logger.removeHandler(handler)
                records.append(
                    ProbeRecord(
                        chamber=chamber,
                        configuration=configuration,
                        budget_k=budget_k,
                        selection_seed=sel,
                        pc_seed=pc_seed,
                        f1=float(f1_edges(predicted, truth)),
                        shd=float(shd(predicted, truth)),
                        n_rows_pooled=len(pooled),
                        chosen_experiments=json.dumps(sorted(chosen)),
                        n_collinear_dropped=collinear.count,
                        n_zero_variance_dropped=zero_var.count,
                        n_pc_degeneracies=degenerate.count,
                    )
                )
    return records


def decompose(frame: pd.DataFrame) -> pd.DataFrame:
    """One-way variance decomposition of F1 by selection, within each budget.

    Returns per-budget: the within-selection sd (PC measurement noise), the
    between-selection sd of group MEANS, and the bias-corrected estimate of
    the true between-selection sd. The correction matters: a group mean over
    `m` draws still carries `sigma_within^2 / m` of measurement noise, so the
    raw between-sd overstates what the choice is worth. Subtracting it can
    push the estimate below zero, which is reported as 0.0 and read as "no
    detectable selection effect", never as a negative variance.
    """
    rows = []
    for budget_k, sub in frame.groupby("budget_k"):
        groups = [g.f1.to_numpy() for _, g in sub.groupby("selection_seed")]
        m = min(len(g) for g in groups)
        within_var = float(
            sum(((g - g.mean()) ** 2).sum() for g in groups)
            / (sum(len(g) for g in groups) - len(groups))
        )
        means = pd.Series([g.mean() for g in groups])
        between_var_raw = float(means.var(ddof=1))
        corrected = max(0.0, between_var_raw - within_var / m)
        rows.append(
            {
                "budget_k": int(budget_k),
                "n_selections": len(groups),
                "n_pc_seeds": m,
                "mean_f1": float(sub.f1.mean()),
                "sd_total": float(sub.f1.std(ddof=1)),
                "sd_within_pc": within_var**0.5,
                "sd_between_raw": between_var_raw**0.5,
                "sd_selection": corrected**0.5,
                "n_rows_pooled": int(sub.n_rows_pooled.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("budget_k").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chamber", default="lt")
    parser.add_argument("--configuration", default="standard")
    parser.add_argument("--budgets", default="6,30,59")
    parser.add_argument("--n-selections", type=int, default=20)
    parser.add_argument("--n-pc-seeds", type=int, default=10)
    parser.add_argument(
        "--pc-max-rows",
        type=int,
        default=None,
        help="Override run_pc's max_rows. Omit to use the bound default (300), "
        "which is the configuration every sweep in the corpus ran under.",
    )
    parser.add_argument("--out", default="runs/variance-probe.parquet")
    args = parser.parse_args()

    records = run_probe(
        chamber=cast("ChamberId", args.chamber),
        configuration=cast("ConfigId", args.configuration),
        budgets=tuple(int(b) for b in args.budgets.split(",")),
        n_selections=args.n_selections,
        n_pc_seeds=args.n_pc_seeds,
        pc_max_rows=args.pc_max_rows,
    )
    frame = pd.DataFrame([asdict(r) for r in records])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"wrote {len(frame)} rows to {out}\n")
    print(decompose(frame).to_string(index=False))


if __name__ == "__main__":
    main()
