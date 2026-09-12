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
    SESSION_CONTEXT,
    SESSION_GAP_SECONDS,
    add_session_context,
    intervention_target,
    pool_with_context,
    run_jci_pc,
    session_ids,
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


@requires_causal_learn
def test_context_nodes_are_pairwise_non_adjacent() -> None:
    """JCI assumption 0 only. Assumption 3 (declare mutually exclusive
    indicators adjacent) is NOT enforceable: causal-learn honours required
    edges in orientation only, so this pins the behaviour we actually have —
    context pairs are removed from the skeleton before any test."""
    rng = np.random.default_rng(7)
    dfs = [_experiment(rng, 300), _experiment(rng, 300, a_shift=6.0), _experiment(rng, 300)]
    dfs[2]["c"] = dfs[2]["c"] + 6.0
    pooled, context = pool_with_context(dfs, [None, "a", "c"], NODES)
    full = run_jci_pc(pooled, NODES, context, max_rows=None, keep_context=True)
    ca, cc = context
    assert full.loc[ca, cc] == 0 and full.loc[cc, ca] == 0
    assert full.loc[NODES, context].to_numpy().sum() == 0


# ---------------------------------------------------------------------------
# regime labels: one indicator per (variable, strength), not per variable
# ---------------------------------------------------------------------------


def test_regime_label_names_the_variable_and_the_strength() -> None:
    from evaluation.chamber_pipeline.jci import regime_label

    assert regime_label("lt", "uniform_red_strong", ["red"]) == "red@strong"
    assert regime_label("lt", "uniform_red_mid", ["red"]) == "red@mid"
    assert regime_label("lt", "uniform_reference", ["red"]) is None
    assert (
        regime_label("wt", "validate_load_out_mic", ["load_in", "load_out"])
        == "load_out@validate_load_out_mic"
    )


def test_regime_context_gives_one_column_per_regime_of_the_same_variable() -> None:
    """Two strengths of one variable are two regimes; merging them into one
    indicator leaves a mixture inside the indicator's own block (measured
    2026-09-10: -0.010 F1 per merged pair at LT k=45)."""
    rng = np.random.default_rng(9)
    dfs = [
        _experiment(rng, 10),
        _experiment(rng, 20, a_shift=3.0),
        _experiment(rng, 30, a_shift=6.0),
    ]
    pooled, context = pool_with_context(dfs, [None, "a@mid", "a@strong"], NODES)
    assert context == [CONTEXT_PREFIX + "a@mid", CONTEXT_PREFIX + "a@strong"]
    assert pooled[CONTEXT_PREFIX + "a@mid"].to_numpy()[10:30].all()
    assert pooled[CONTEXT_PREFIX + "a@mid"].to_numpy()[30:].sum() == 0
    assert pooled[CONTEXT_PREFIX + "a@strong"].to_numpy()[30:].all()


def test_regime_label_must_still_name_a_node() -> None:
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="not a node"):
        pool_with_context([_experiment(rng, 5)], ["zz@strong"], NODES)


# ---------------------------------------------------------------------------
# Register §37: a recording session is a regime nobody bought. One indicator
# per session, derived from each experiment's own timestamps.


def _stamped(rng: np.random.Generator, n: int, t0: float) -> pd.DataFrame:
    df = _experiment(rng, n)
    df["timestamp"] = t0 + np.arange(n, dtype=float)
    return df


def test_session_ids_split_on_a_large_timestamp_gap_only() -> None:
    rng = np.random.default_rng(0)
    dfs = [_stamped(rng, 50, 1_000.0), _stamped(rng, 50, 5_000.0), _stamped(rng, 50, 400_000.0)]
    assert session_ids(dfs, gap=SESSION_GAP_SECONDS) == [0, 0, 1]
    # ids follow time order, not buy order
    assert session_ids(dfs[::-1], gap=SESSION_GAP_SECONDS) == [1, 0, 0]
    # frames without a timestamp column are one session
    assert session_ids([_experiment(rng, 10), _experiment(rng, 10)], gap=SESSION_GAP_SECONDS) == [
        0,
        0,
    ]


def test_add_session_context_marks_rows_and_is_absent_for_one_session() -> None:
    rng = np.random.default_rng(1)
    dfs = [_stamped(rng, 40, 1_000.0), _stamped(rng, 60, 400_000.0)]
    pooled, ctx = pool_with_context(dfs, ["a", None], NODES)
    pooled2, ctx2 = add_session_context(pooled, ctx, dfs, session_ids(dfs, gap=SESSION_GAP_SECONDS))
    assert ctx2 == [*ctx, SESSION_CONTEXT]
    assert list(pooled2[SESSION_CONTEXT]) == [0] * 40 + [1] * 60
    one = [_stamped(rng, 40, 1_000.0), _stamped(rng, 60, 2_000.0)]
    pooled3, ctx3 = pool_with_context(one, ["a", None], NODES)
    pooled4, ctx4 = add_session_context(
        pooled3, ctx3, one, session_ids(one, gap=SESSION_GAP_SECONDS)
    )
    assert ctx4 == ctx3 and SESSION_CONTEXT not in pooled4.columns


def test_wt_menu_has_two_sessions_and_lt_one() -> None:
    from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

    wt = create_contracted_chamber_agent(chamber="wt", intervention_budget=28)
    names = list(wt.available_experiments())
    ids = session_ids([wt.query_intervention(n) for n in names], gap=SESSION_GAP_SECONDS)
    early = {n for n, i in zip(names, ids, strict=True) if i == 0}
    assert early == {
        "validate_osr_ambient",
        "validate_hatch_mic",
        "validate_load_out_mic",
        "validate_load_in_current_out",
        "validate_load_out_current_in",
        "validate_load_out_pressure_intake",
    }
    lt = create_contracted_chamber_agent(chamber="lt", intervention_budget=59)
    lt_names = list(lt.available_experiments())
    assert set(
        session_ids([lt.query_intervention(n) for n in lt_names], gap=SESSION_GAP_SECONDS)
    ) == {0}


@requires_causal_learn
def test_rescore_session_context_runs_and_labels_records() -> None:
    from evaluation.chamber_pipeline.rescore import _rescore_one_design

    names = ["validate_osr_ambient", "validate_v_out", "validate_load_in"]
    recs = _rescore_one_design(("k", "wt", "standard", names, 2, 0.05, 300, "jci_pc", "session"))
    assert len(recs) == 2 and all(r["rescore_context"] == "session" for r in recs)
    assert all(0.0 <= r["f1"] <= 1.0 for r in recs)
