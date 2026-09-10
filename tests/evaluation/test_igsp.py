"""UT-IGSP as the estimator that never pools (Squires et al. 2020).

One observational sample plus one sample per bought experiment with its
known target; the invariance tests read each regime on its own. This is the
chamber authors' interventional method (register §28) and the only estimator
here that can say whether strong interventions rank last when the judge
never mixes regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.igsp import IGSP_AVAILABLE, run_utigsp

requires_igsp = pytest.mark.skipif(
    not IGSP_AVAILABLE, reason="graphical-model-learning not importable"
)

NODES = ["a", "b", "c", "d"]


def _sample(rng: np.random.Generator, n: int, a_shift: float = 0.0) -> pd.DataFrame:
    """a -> b -> c; d constant (an apparatus setting nobody perturbed)."""
    a = rng.normal(size=n) + a_shift
    b = 2.0 * a + rng.normal(scale=0.5, size=n)
    c = -1.5 * b + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": 0.0, "meta": 1})


@requires_igsp
def test_output_is_a_directed_adjacency_on_node_names() -> None:
    rng = np.random.default_rng(0)
    adj = run_utigsp(_sample(rng, 500), [(_sample(rng, 500, 6.0), "a")], NODES, seed=0)
    assert list(adj.index) == NODES and list(adj.columns) == NODES
    assert set(np.unique(adj.to_numpy())) <= {0, 1}
    assert int(np.diag(adj.to_numpy()).sum()) == 0


@requires_igsp
def test_recovers_the_chain_and_pads_the_constant_column() -> None:
    rng = np.random.default_rng(1)
    adj = run_utigsp(_sample(rng, 800), [(_sample(rng, 800, 6.0), "a")], NODES, seed=0)
    assert adj.loc["a", "b"] == 1 and adj.loc["b", "c"] == 1
    assert adj.loc["a", "c"] == 0
    assert adj.loc["d"].sum() == 0 and adj["d"].sum() == 0


@requires_igsp
def test_a_target_outside_the_kept_columns_is_treated_as_unknown() -> None:
    """An apparatus experiment targets a column that is constant in every
    sample; UT-IGSP still gets the sample, with no declared target."""
    rng = np.random.default_rng(2)
    adj = run_utigsp(_sample(rng, 500), [(_sample(rng, 500, 6.0), "d")], NODES, seed=0)
    assert list(adj.index) == NODES


@requires_igsp
def test_same_seed_same_graph() -> None:
    rng = np.random.default_rng(3)
    obs, iv = _sample(rng, 2000), _sample(rng, 2000, 6.0)
    once = run_utigsp(obs, [(iv, "a")], NODES, seed=0, max_rows=300)
    again = run_utigsp(obs, [(iv, "a")], NODES, seed=0, max_rows=300)
    pd.testing.assert_frame_equal(once, again)


def test_requires_at_least_one_interventional_sample() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="interventional"):
        run_utigsp(_sample(rng, 50), [], NODES)
