"""JCI-PC: PC on pooled interventional data with one context indicator per
intervention target (Mooij, Magliacane & Claassen, JMLR 2020).

The plain estimator pools every bought experiment into one table and treats
it as i.i.d., so an experiment that shifts an input's range is an unexplained
mixture. JCI adds a binary column per intervened variable and forbids edges
INTO those columns; the indicator then explains the shift instead of
corrupting the correlations. Register §34 is the reason this exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.inference import CAUSAL_LEARN_AVAILABLE
from evaluation.chamber_pipeline.jci import (
    CONTEXT_PREFIX,
    intervention_target,
    pool_with_context,
    run_jci_pc,
)

requires_causal_learn = pytest.mark.skipif(
    not CAUSAL_LEARN_AVAILABLE, reason="causal-learn not installed"
)

NODES = ["a", "b", "c"]


def _experiment(rng: np.random.Generator, n: int, a_shift: float = 0.0) -> pd.DataFrame:
    """a -> b -> c, with `a` optionally shifted (a range intervention)."""
    a = rng.normal(size=n) + a_shift
    b = 2.0 * a + rng.normal(scale=0.5, size=n)
    c = -1.5 * b + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "meta": 0})


# ---------------------------------------------------------------------------
# intervention_target
# ---------------------------------------------------------------------------


def test_lt_target_is_the_taxonomy_variable() -> None:
    assert intervention_target("lt", "uniform_red_strong", ["red", "ir_1"]) == "red"


def test_lt_reference_is_observational() -> None:
    """`uniform_reference` intervenes on nothing — no context column."""
    assert intervention_target("lt", "uniform_reference", ["red", "ir_1"]) is None


def test_wt_target_uses_the_wt_taxonomy() -> None:
    assert intervention_target("wt", "validate_load_out_mic", ["load_in", "load_out"]) == "load_out"


def test_unknown_chamber_raises() -> None:
    with pytest.raises(ValueError, match="chamber"):
        intervention_target("xx", "anything", NODES)


# ---------------------------------------------------------------------------
# pool_with_context
# ---------------------------------------------------------------------------


def test_one_context_column_per_distinct_target_marking_its_rows() -> None:
    rng = np.random.default_rng(0)
    dfs = [_experiment(rng, 10), _experiment(rng, 20, a_shift=5.0), _experiment(rng, 30)]
    targets = [None, "a", "a"]  # observational, then two experiments on `a`
    pooled, context = pool_with_context(dfs, targets, NODES)

    assert context == [CONTEXT_PREFIX + "a"]
    assert list(pooled.columns) == NODES + context
    col = pooled[CONTEXT_PREFIX + "a"].to_numpy()
    assert col[:10].sum() == 0
    assert col[10:].sum() == 50
    assert set(col) == {0, 1}


def test_context_columns_follow_first_appearance_order() -> None:
    rng = np.random.default_rng(1)
    dfs = [_experiment(rng, 5) for _ in range(3)]
    _, context = pool_with_context(dfs, ["c", "a", "c"], NODES)
    assert context == [CONTEXT_PREFIX + "c", CONTEXT_PREFIX + "a"]


def test_all_observational_yields_no_context_columns() -> None:
    rng = np.random.default_rng(2)
    pooled, context = pool_with_context([_experiment(rng, 5)], [None], NODES)
    assert context == []
    assert list(pooled.columns) == NODES


def test_target_outside_the_node_set_is_rejected() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="not a node"):
        pool_with_context([_experiment(rng, 5)], ["zz"], NODES)


# ---------------------------------------------------------------------------
# run_jci_pc
# ---------------------------------------------------------------------------


@requires_causal_learn
def test_output_is_on_the_chamber_nodes_only() -> None:
    rng = np.random.default_rng(4)
    dfs = [_experiment(rng, 300), _experiment(rng, 300, a_shift=6.0)]
    pooled, context = pool_with_context(dfs, [None, "a"], NODES)
    adj = run_jci_pc(pooled, NODES, context, max_rows=None)
    assert list(adj.index) == NODES
    assert list(adj.columns) == NODES


@requires_causal_learn
def test_context_nodes_never_receive_an_edge() -> None:
    """The JCI exogeneity assumption, enforced through background knowledge."""
    rng = np.random.default_rng(5)
    dfs = [_experiment(rng, 300), _experiment(rng, 300, a_shift=6.0)]
    pooled, context = pool_with_context(dfs, [None, "a"], NODES)
    full = run_jci_pc(pooled, NODES, context, max_rows=None, keep_context=True)
    assert list(full.index) == NODES + context
    # column = target; nothing may point INTO a context node
    assert full[context].to_numpy().sum() == 0
    # and the indicator must point at the variable it intervened on
    assert full.loc[CONTEXT_PREFIX + "a", "a"] == 1


@requires_causal_learn
def test_plain_pc_and_jci_pc_agree_when_nothing_was_intervened() -> None:
    """With no context columns JCI-PC IS PC — same graph, byte for byte."""
    from evaluation.chamber_pipeline.inference import run_pc

    rng = np.random.default_rng(6)
    dfs = [_experiment(rng, 300)]
    pooled, context = pool_with_context(dfs, [None], NODES)
    assert context == []
    plain = run_pc(pooled, NODES, max_rows=None)
    jci = run_jci_pc(pooled, NODES, context, max_rows=None)
    pd.testing.assert_frame_equal(plain, jci)
