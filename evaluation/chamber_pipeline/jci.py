"""JCI-PC: the plain PC estimator with one context indicator per intervention
target (Joint Causal Inference; Mooij, Magliacane & Claassen, JMLR 2020).

Why it exists (register §34). `run_pc` pools every bought experiment into one
table and treats it as an i.i.d. sample. An experiment that shifts an input's
range (`uniform_red_strong`: red 171-255 instead of 0-85) is then an
unexplained mixture, and Fisher-Z reads the mixture as spurious dependence or
loses the real input→sensor edge. JCI's remedy is to add a binary column per
intervened variable, marking the rows that came from an experiment on it, and
to forbid edges INTO those columns. The indicator explains the shift, and its
out-edges say what the intervention reached.

This is the same PC, the same independence test, the same row cap and the
same collinearity policy; only the input table and the background knowledge
differ. Context columns are stripped before scoring, so the ground truth is
untouched.

Two decisions, recorded rather than argued:

* One indicator per TARGET VARIABLE, not per experiment. JCI's context
  variable is "which regime", and two experiments on the same variable at
  different strengths are one regime family; per-experiment indicators would
  add up to 45 nodes at LT k=45 and be pairwise near-collinear with each other.
* Only the exogeneity assumption (JCI0: nothing points into a context
  variable). No assumption about dependence among context variables — they
  are left for PC to decide, and in practice they are mutually exclusive
  indicators over disjoint row blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from .inference import pool_experiment_data, run_pc
from .menu_taxonomy import experiment_variable as lt_experiment_variable
from .wt_menu_taxonomy import experiment_variable as wt_experiment_variable

CONTEXT_PREFIX = "ctx__"


def intervention_target(chamber: str, experiment_name: str, node_names: list[str]) -> str | None:
    """The chamber variable `experiment_name` intervenes on, or None if it is
    observational (LT's `uniform_reference` parses to a non-node stem)."""
    if chamber == "lt":
        target = lt_experiment_variable(experiment_name)
        return target if target in node_names else None
    if chamber == "wt":
        return wt_experiment_variable(experiment_name, node_names)
    raise ValueError(f"unknown chamber {chamber!r}; expected 'lt' or 'wt'")


def pool_with_context(
    experiment_dfs: list[pd.DataFrame],
    targets: list[str | None],
    node_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """`pool_experiment_data` plus one 0/1 column per distinct target.

    Context columns follow the order in which their target first appears in
    `targets`, so the augmented column order is a function of the ORDERED buy
    like everything else in the re-scorer. Returns the augmented table and
    the list of context column names (possibly empty).
    """
    if len(experiment_dfs) != len(targets):
        raise ValueError(
            f"{len(experiment_dfs)} experiments but {len(targets)} targets; one target per experiment"
        )
    node_set = set(node_names)
    for t in targets:
        if t is not None and t not in node_set:
            raise ValueError(f"intervention target {t!r} is not a node of this chamber")

    pooled = pool_experiment_data(experiment_dfs, node_names)
    ordered_targets: list[str] = []
    for t in targets:
        if t is not None and t not in ordered_targets:
            ordered_targets.append(t)

    lengths = [len(df) for df in experiment_dfs]
    starts = np.cumsum([0, *lengths[:-1]])
    context_names = [CONTEXT_PREFIX + t for t in ordered_targets]
    for t, name in zip(ordered_targets, context_names, strict=True):
        col = np.zeros(len(pooled), dtype=int)
        for start, n, target in zip(starts, lengths, targets, strict=True):
            if target == t:
                col[start : start + n] = 1
        pooled[name] = col
    return pooled, context_names


def run_jci_pc(
    pooled_with_context: pd.DataFrame,
    node_names: list[str],
    context_names: list[str],
    *,
    keep_context: bool = False,
    **run_pc_kwargs: Any,
) -> pd.DataFrame:
    """PC on the augmented table with edges into context nodes forbidden.

    Delegates to `run_pc`, so the row cap, zero-variance and collinearity
    handling and the singular-matrix fallback are exactly the plain
    estimator's. Returns the adjacency restricted to `node_names` unless
    `keep_context` is set (for inspecting what each indicator reached).
    """
    all_names = list(node_names) + list(context_names)
    if list(pooled_with_context.columns) != all_names:
        raise ValueError("columns must be node_names followed by context_names, in order")
    if not context_names:
        return run_pc(pooled_with_context, node_names, **run_pc_kwargs)

    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    # Any node -> any context node is forbidden (JCI exogeneity). Patterns
    # are regexes over causal-learn's node names; `run_pc` forwards the
    # column names to `pc` whenever background knowledge is supplied, since
    # its default `X1..Xn` names would match nothing.
    bk.add_forbidden_by_pattern(".*", f"^{CONTEXT_PREFIX}")

    full = run_pc(
        pooled_with_context,
        all_names,
        background_knowledge=bk,
        **run_pc_kwargs,
    )
    if keep_context:
        return full
    return full.loc[node_names, node_names].copy()


__all__ = ["CONTEXT_PREFIX", "intervention_target", "pool_with_context", "run_jci_pc"]
