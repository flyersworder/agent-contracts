"""GES as a second estimator for the re-scorer (Chickering 2002; causal-learn).

Score-based, BIC, no alpha and no independence tests — the observational
method the chambers' own LT case study uses (register §28). It reads the SAME
pooled table under the SAME row cap and column policy as `run_pc`, so a
verdict that holds under both is robust across test families, and one that
does not is estimator-sensitive. It shares the pooling, so it cannot say
whether pooling itself is the problem; that is the rescale probe's job.

Preprocessing mirrors `run_pc` step for step (subsample under the seed, drop
zero-variance columns, drop collinear columns, pad back with zeros) rather
than calling into it, so the plain PC path stays byte-for-byte what it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .inference import (
    DEFAULT_COLLINEARITY_THRESHOLD,
    DEFAULT_MAX_ROWS,
    cpdag_to_directed_adjacency,
    select_noncollinear_columns,
)

if TYPE_CHECKING:
    import pandas as pd

GES_SCORE = "local_score_BIC"


def run_ges(
    pooled_data: pd.DataFrame,
    node_names: list[str],
    *,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    seed: int = 0,
    collinearity_threshold: float | None = DEFAULT_COLLINEARITY_THRESHOLD,
) -> pd.DataFrame:
    """GES on pooled chamber data; directed adjacency on `node_names`.

    The CPDAG is read through `cpdag_to_directed_adjacency`, the same
    encoding as PC (an undirected edge counts in both directions), so F1
    and SHD are comparable across the two estimators.
    """
    import pandas as pd
    from causallearn.search.ScoreBased.GES import ges

    if list(pooled_data.columns) != list(node_names):
        raise ValueError("pooled_data columns must match node_names in order")

    if max_rows is not None and len(pooled_data) > max_rows:
        pooled_data = pooled_data.sample(n=max_rows, random_state=seed)

    variances = pooled_data.var()
    valid_cols = [n for n in node_names if variances.get(n, 0.0) > 1e-12]
    if collinearity_threshold is not None and len(valid_cols) > 1:
        valid_cols, _ = select_noncollinear_columns(pooled_data, valid_cols, collinearity_threshold)

    full = pd.DataFrame(
        np.zeros((len(node_names), len(node_names)), dtype=int),
        index=node_names,
        columns=node_names,
    )
    if not valid_cols:
        return full

    result = ges(pooled_data[valid_cols].to_numpy(dtype=float), score_func=GES_SCORE)
    valid_adj = cpdag_to_directed_adjacency(result["G"].graph, valid_cols)
    full.loc[valid_cols, valid_cols] = valid_adj.values
    return full


__all__ = ["GES_SCORE", "run_ges"]
