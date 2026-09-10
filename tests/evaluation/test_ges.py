"""GES as a second estimator: score-based (BIC), no alpha, no independence
tests — the observational method of the chambers' own LT case study. Same
pooling, same row cap, same column policy as `run_pc`, so a verdict that
holds under both PC and GES is robust across test families.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.ges import run_ges
from evaluation.chamber_pipeline.inference import CAUSAL_LEARN_AVAILABLE

requires_causal_learn = pytest.mark.skipif(
    not CAUSAL_LEARN_AVAILABLE, reason="causal-learn not installed"
)

NODES = ["a", "b", "c", "d"]


def _chain(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """a -> b -> c, d constant (an unbought apparatus setting)."""
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(scale=0.5, size=n)
    c = -1.5 * b + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": 0.0})


@requires_causal_learn
def test_output_is_a_directed_adjacency_on_node_names() -> None:
    adj = run_ges(_chain(np.random.default_rng(0), 500), NODES, max_rows=None)
    assert list(adj.index) == NODES and list(adj.columns) == NODES
    assert set(np.unique(adj.to_numpy())) <= {0, 1}
    assert int(np.diag(adj.to_numpy()).sum()) == 0


@requires_causal_learn
def test_recovers_the_chain_skeleton_and_pads_the_constant_column() -> None:
    adj = run_ges(_chain(np.random.default_rng(1), 500), NODES, max_rows=None)
    sym = ((adj + adj.T) > 0).astype(int)
    assert sym.loc["a", "b"] == 1 and sym.loc["b", "c"] == 1
    assert sym.loc["a", "c"] == 0
    assert adj.loc["d"].sum() == 0 and adj["d"].sum() == 0


@requires_causal_learn
def test_row_cap_subsamples_under_the_seed() -> None:
    """Two seeds draw different rows and may disagree; one seed is deterministic."""
    data = _chain(np.random.default_rng(2), 3000)
    once = run_ges(data, NODES, max_rows=300, seed=0)
    again = run_ges(data, NODES, max_rows=300, seed=0)
    pd.testing.assert_frame_equal(once, again)


def test_column_order_must_match_node_names() -> None:
    data = _chain(np.random.default_rng(3), 50)[["b", "a", "c", "d"]]
    with pytest.raises(ValueError, match="node_names"):
        run_ges(data, NODES, max_rows=None)
