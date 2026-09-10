"""UT-IGSP: the estimator that never pools (Squires, Wang & Uhler, UAI 2020).

The chamber authors' interventional method (register §28). It takes ONE
observational sample and ONE sample per intervention with its (possibly
unknown) target, and searches permutations using two families of tests:
conditional independence within the observational sample, and INVARIANCE of
each variable's conditional distribution between the observational sample
and each interventional one. No table is ever concatenated, so the pooled-
regime misfit that drives register §34 (results doc "WHY STRONG
INTERVENTIONS HURT") cannot arise. It is the only estimator here that can
say whether strong interventions rank last when the judge never mixes
regimes.

Scope, stated up front rather than discovered later:

* **Needs an observational sample.** LT has one (`uniform_reference`,
  10,000 rows); the WT validation menu has none, so the re-scorer refuses
  WT for this estimator.
* **Core variables only, by construction.** A column constant within every
  sample (the 18 LT apparatus settings — a setting is fixed for the whole
  experiment) has no covariance for a Gaussian test to use, so it is dropped
  and padded back with zeros, exactly as `run_pc` does. UT-IGSP therefore
  scores the 20-variable graph the chamber authors score, and its numbers
  belong beside the `f1_core` column, not the 38-node one.
* **Alpha is a nuisance parameter** (two of them: CI and invariance). The
  authors grid-search it; we sweep it on neutral designs before quoting any
  chamber number, as §34 requires.

Packages: the old `causaldag` split into `graphical-model-learning`,
`conditional-independence` and `graphical-models` (all alpha releases; the
`igsp` extra). Their import chain wants an OpenMP runtime, absent on a stock
macOS, so `IGSP_AVAILABLE` gates every test and the VPS is where it runs.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from .inference import (
    DEFAULT_COLLINEARITY_THRESHOLD,
    select_noncollinear_columns,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

try:
    from conditional_independence import (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        partial_correlation_suffstat,
        partial_correlation_test,
    )
    from graphical_model_learning import unknown_target_igsp

    IGSP_AVAILABLE = True
except Exception:
    IGSP_AVAILABLE = False

DEFAULT_DEPTH = 4
DEFAULT_NRUNS = 5


def run_utigsp(
    observational: pd.DataFrame,
    interventional: list[tuple[pd.DataFrame, str | None]],
    node_names: list[str],
    *,
    alpha: float = 0.05,
    alpha_inv: float = 0.05,
    seed: int = 0,
    max_rows: int | None = None,
    collinearity_threshold: float | None = DEFAULT_COLLINEARITY_THRESHOLD,
    depth: int = DEFAULT_DEPTH,
    nruns: int = DEFAULT_NRUNS,
) -> pd.DataFrame:
    """Directed adjacency on `node_names` from one observational and k
    interventional samples. `interventional` pairs each sample with the name
    of the variable it intervened on, or None if unknown/observational.

    `max_rows` caps EVERY sample separately (never a pool) under `seed`;
    None uses all rows. The library's own search uses the global RNGs, which
    are seeded here so one seed gives one graph.
    """
    import pandas as pd

    if not interventional:
        raise ValueError("UT-IGSP needs at least one interventional sample")
    if not IGSP_AVAILABLE:
        raise ImportError("UT-IGSP needs the `igsp` extra and an OpenMP runtime")
    missing = set(node_names) - set(observational.columns)
    if missing:
        raise ValueError(f"observational sample is missing node columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)

    def _cap(df: pd.DataFrame) -> pd.DataFrame:
        if max_rows is not None and len(df) > max_rows:
            return df.sample(n=max_rows, random_state=int(rng.integers(2**31 - 1)))
        return df

    obs = _cap(observational[node_names])
    ivs = [(_cap(df[node_names]), target) for df, target in interventional]

    # Column policy, as `run_pc`: a variable constant in the observational
    # sample cannot enter a Gaussian CI test, and a duplicate column makes the
    # covariance singular. Dropped columns are padded back as zeros.
    variances = obs.var()
    kept = [n for n in node_names if variances.get(n, 0.0) > 1e-12]
    if collinearity_threshold is not None and len(kept) > 1:
        kept, dropped = select_noncollinear_columns(obs, kept, collinearity_threshold)
        if dropped:
            logger.warning(
                "UT-IGSP dropped %d collinear column(s): %s", len(dropped), ", ".join(dropped)
            )
    zero_variance = [n for n in node_names if n not in set(kept)]
    if zero_variance:
        logger.warning("UT-IGSP dropped %d zero-variance column(s)", len(zero_variance))

    full = pd.DataFrame(
        np.zeros((len(node_names), len(node_names)), dtype=int),
        index=node_names,
        columns=node_names,
    )
    if len(kept) < 2:
        return full

    index = {n: i for i, n in enumerate(kept)}
    obs_arr = obs[kept].to_numpy(dtype=float)
    iv_arrs = [df[kept].to_numpy(dtype=float) for df, _ in ivs]
    settings: list[dict[str, Any]] = []
    for _, target in ivs:
        known = [index[target]] if target in index else []
        settings.append({"interventions": known, "known_interventions": known})

    random.seed(seed)
    np.random.seed(seed)
    try:
        ci = MemoizedCI_Tester(
            partial_correlation_test, partial_correlation_suffstat(obs_arr), alpha=alpha
        )
        inv = MemoizedInvarianceTester(
            gauss_invariance_test, gauss_invariance_suffstat(obs_arr, iv_arrs), alpha=alpha_inv
        )
        dag, _targets = unknown_target_igsp(
            settings, set(range(len(kept))), ci, inv, depth=depth, nruns=nruns
        )
    except np.linalg.LinAlgError as exc:
        logger.warning("UT-IGSP fell back to all-zeros adjacency (%s)", exc)
        return full

    for i, j in dag.arcs:
        full.loc[kept[i], kept[j]] = 1
    return full


__all__ = ["DEFAULT_DEPTH", "DEFAULT_NRUNS", "IGSP_AVAILABLE", "run_utigsp"]
