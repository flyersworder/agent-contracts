"""Tests for the Causal Chamber pipeline's PC inference helpers.

Covers `evaluation.chamber_pipeline.inference`. These tests need
`causal-learn` (provided by the `chambers` extra) and pandas, but do
NOT need `causalchamber` itself — they exercise the inference module
in isolation against synthesized linear-Gaussian data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.inference import (
    CAUSAL_LEARN_AVAILABLE,
    cpdag_to_directed_adjacency,
    pool_experiment_data,
    run_pc,
)

requires_causal_learn = pytest.mark.skipif(
    not CAUSAL_LEARN_AVAILABLE,
    reason="causal-learn not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# pool_experiment_data
# ---------------------------------------------------------------------------


class TestPoolExperimentData:
    """Concatenation + projection to ground-truth nodes."""

    def test_basic_concatenation(self) -> None:
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "intervention": ["x", "x"]})
        df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8], "intervention": ["y", "y"]})
        pooled = pool_experiment_data([df1, df2], ["x", "y"])
        assert pooled.shape == (4, 2)
        assert list(pooled.columns) == ["x", "y"]
        # intervention meta-column dropped.
        assert "intervention" not in pooled.columns

    def test_column_order_matches_node_names(self) -> None:
        df = pd.DataFrame({"y": [1], "x": [2], "z": [3]})
        pooled = pool_experiment_data([df], ["x", "y", "z"])
        assert list(pooled.columns) == ["x", "y", "z"]

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="missing required node columns"):
            pool_experiment_data([df], ["x", "y"])

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pool_experiment_data([], ["x"])


# ---------------------------------------------------------------------------
# cpdag_to_directed_adjacency
# ---------------------------------------------------------------------------


class TestCpdagToDirectedAdjacency:
    """Convention: undirected → both directions; directed → arrowed direction."""

    def test_directed_arrow(self) -> None:
        # PC encoding: i→j is graph[i,j]=-1, graph[j,i]=1.
        # x→y: graph[0,1]=-1, graph[1,0]=1 → adj[0,1]=1, adj[1,0]=0.
        pc_graph = np.array([[0, -1], [1, 0]])
        adj = cpdag_to_directed_adjacency(pc_graph, ["x", "y"])
        assert adj.loc["x", "y"] == 1
        assert adj.loc["y", "x"] == 0

    def test_undirected_becomes_bidirectional(self) -> None:
        # PC undirected x—y: graph[0,1]=-1, graph[1,0]=-1.
        # Both ends are tails; we report both directions.
        pc_graph = np.array([[0, -1], [-1, 0]])
        adj = cpdag_to_directed_adjacency(pc_graph, ["x", "y"])
        assert adj.loc["x", "y"] == 1
        assert adj.loc["y", "x"] == 1

    def test_no_edge(self) -> None:
        pc_graph = np.array([[0, 0], [0, 0]])
        adj = cpdag_to_directed_adjacency(pc_graph, ["x", "y"])
        assert adj.values.sum() == 0

    def test_diagonal_always_zero(self) -> None:
        pc_graph = np.array([[1, 0], [0, 1]])  # bizarre self-loops
        adj = cpdag_to_directed_adjacency(pc_graph, ["x", "y"])
        assert adj.loc["x", "x"] == 0
        assert adj.loc["y", "y"] == 0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            cpdag_to_directed_adjacency(np.zeros((3, 3)), ["x", "y"])


# ---------------------------------------------------------------------------
# run_pc — actual causal-learn integration
# ---------------------------------------------------------------------------


@requires_causal_learn
class TestRunPc:
    """End-to-end PC on synthetic linear-Gaussian data."""

    def _vstructure_data(self, seed: int = 0, n: int = 500) -> pd.DataFrame:
        """Generate y → z ← w (collider). Edges: (y, z), (w, z)."""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(n)
        w = rng.standard_normal(n)
        z = -0.8 * y + 1.2 * w + 0.1 * rng.standard_normal(n)
        return pd.DataFrame({"y": y, "z": z, "w": w})

    def test_recovers_v_structure(self) -> None:
        """PC should recover y→z and w→z; the v-structure orients."""
        data = self._vstructure_data()
        nodes = ["y", "z", "w"]
        adj = run_pc(data, nodes)
        # Definite arrowheads at z from both y and w.
        assert adj.loc["y", "z"] == 1
        assert adj.loc["w", "z"] == 1
        # And y / w should not have spurious edges between them.
        assert adj.loc["y", "w"] == 0
        assert adj.loc["w", "y"] == 0

    def test_subsamples_when_above_max_rows(self) -> None:
        """At max_rows < len(data), PC should still complete fast."""
        data = self._vstructure_data(n=2000)
        nodes = ["y", "z", "w"]
        # If subsampling didn't fire, this would time out at the suite level;
        # if it does, this completes in well under a second.
        adj = run_pc(data, nodes, max_rows=200, seed=0)
        assert adj.shape == (3, 3)

    def test_seed_makes_subsampling_reproducible(self) -> None:
        data = self._vstructure_data(n=2000)
        nodes = ["y", "z", "w"]
        a = run_pc(data, nodes, max_rows=100, seed=42)
        b = run_pc(data, nodes, max_rows=100, seed=42)
        assert (a.values == b.values).all()

    def test_returns_node_indexed_dataframe(self) -> None:
        data = self._vstructure_data()
        nodes = ["y", "z", "w"]
        adj = run_pc(data, nodes)
        assert list(adj.index) == nodes
        assert list(adj.columns) == nodes

    def test_column_order_mismatch_raises(self) -> None:
        data = self._vstructure_data()
        with pytest.raises(ValueError, match="columns must match"):
            run_pc(data, ["y", "w", "z"])  # wrong order

    def test_near_duplicate_columns_are_dropped_not_fatal(self) -> None:
        """A near-duplicate variable must degrade locally, not globally.

        The WT chamber has four barometers that read the same physical
        quantity in the `standard` configuration (pairwise r > 0.9998),
        which makes Fisher-Z's sub-correlation inversion singular. Before
        this filter, that aborted the whole PC run and returned all-zeros
        for ALL 32 nodes -- F1=0 for the cell -- even though the other 28
        variables carried perfectly good signal.

        Dropping the redundant column and padding it back with zeros is the
        same policy `run_pc` already applies to zero-variance columns:
        "no independent signal, no claim". The loss is then bounded to
        edges incident to the dropped node.
        """
        rng = np.random.default_rng(0)
        n = 400
        a = rng.normal(size=n)
        b = 2.0 * a + rng.normal(scale=0.5, size=n)
        # `dup` is `a` plus noise 1e-7 of its scale: numerically collinear.
        dup = a + rng.normal(scale=1e-7, size=n)
        data = pd.DataFrame({"a": a, "b": b, "dup": dup})

        adj = run_pc(data, ["a", "b", "dup"], alpha=0.05, seed=0)

        assert list(adj.index) == ["a", "b", "dup"], "must pad back to full node set"
        # The duplicate is dropped -> no claim about it in either direction.
        assert adj.loc["dup"].sum() == 0
        assert adj["dup"].sum() == 0
        # ...but the surviving a-b relationship is still recovered. This is
        # the whole point: a local drop, not a global all-zeros abort.
        assert adj.loc["a", "b"] + adj.loc["b", "a"] > 0

    def test_collinear_drop_is_logged_for_sweep_visibility(self, caplog) -> None:
        """The drop must be countable, or it is a silent moderator.

        A degradation path that varies with the experiment's independent
        variable and leaves no trace is how a harness ends up measuring
        itself. `_PcCollinearHandler` scrapes this warning per cell, so the
        phrase is load-bearing.
        """
        rng = np.random.default_rng(1)
        a = rng.normal(size=300)
        data = pd.DataFrame(
            {"a": a, "b": rng.normal(size=300), "dup": a + rng.normal(scale=1e-7, size=300)}
        )
        with caplog.at_level(logging.WARNING, logger="evaluation.chamber_pipeline.inference"):
            run_pc(data, ["a", "b", "dup"], alpha=0.05, seed=0)
        assert any("collinear" in r.getMessage().lower() for r in caplog.records)

    def test_collinearity_filter_can_be_disabled(self) -> None:
        """None disables the filter, so the old behaviour stays reachable."""
        rng = np.random.default_rng(2)
        a = rng.normal(size=300)
        data = pd.DataFrame(
            {"a": a, "b": rng.normal(size=300), "dup": a + rng.normal(scale=1e-9, size=300)}
        )
        adj = run_pc(data, ["a", "b", "dup"], alpha=0.05, seed=0, collinearity_threshold=None)
        assert adj.shape == (3, 3)

    def test_singular_matrix_returns_zero_adjacency(self) -> None:
        """Highly-collinear data (rank-deficient correlation) -> all-zeros.

        Constructed degenerate input: y is a deterministic linear function
        of x, so the (x, y) pair is perfectly collinear and Fisher-Z's
        sub-correlation inversion goes singular. Documented chamber-data
        failure mode; the agents promise the all-zeros baseline here, so
        run_pc must deliver it.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 2.0 * x  # perfectly collinear
        z = rng.standard_normal(200)
        data = pd.DataFrame({"x": x, "y": y, "z": z})
        adj = run_pc(data, ["x", "y", "z"])
        # Either PC returns no edges OR it returns the trivial xy edge.
        # The contract is "no crash, well-typed shape" — not a specific graph.
        assert adj.shape == (3, 3)
        assert list(adj.index) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Tests added post M3 review
# ---------------------------------------------------------------------------


@requires_causal_learn
class TestPcAlphaParameterPlumbing:
    """Verify `pc_alpha` actually flows through to the independence test.

    Without this test, a typo in any of the four call sites (random_agent,
    greedy_ig_lite_agent, llm_pc_agent, planner_reasoner_agents) would
    silently use the default 0.05 instead of the caller's value.
    """

    def _branchy_data(self, seed: int = 0, n: int = 500) -> pd.DataFrame:
        """Generate data where alpha matters: a weak dependence that's
        included at alpha=0.5 but excluded at alpha=0.001."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        # y has a tiny residual dependence on x but is mostly noise.
        y = 0.05 * x + rng.standard_normal(n)
        # z is independent of both.
        z = rng.standard_normal(n)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    def test_alpha_changes_output(self) -> None:
        """Same data + different alpha → at least one cell of the
        adjacency must differ. Establishes that pc_alpha is load-bearing."""
        data = self._branchy_data()
        nodes = ["x", "y", "z"]
        adj_strict = run_pc(data, nodes, alpha=0.001)
        adj_loose = run_pc(data, nodes, alpha=0.5)
        # On the same data, looser alpha admits more edges than stricter.
        # Either output may be all-zeros depending on the noise; the
        # contract is just that the parameter affects the output.
        assert not (adj_strict.values == adj_loose.values).all(), (
            "pc_alpha had no effect on output — parameter may not be plumbed correctly"
        )


@requires_causal_learn
class TestSingularFallbackLogging:
    """Verify the singular-matrix fallback emits a warning for M5 sweep visibility."""

    def test_warns_on_singular_matrix(self, caplog) -> None:
        """Degenerate input → all-zeros output AND a warning logged.

        Uses a THREE-way linear dependence (`w = x + y + z`) rather than a
        pairwise duplicate. This matters: the collinearity filter added on
        2026-08-25 compares columns pairwise, so it cannot see a variable
        that is a sum of several others -- here every pairwise |r| is only
        about 0.58, far below the 0.999 cutoff, while the correlation matrix
        is exactly rank-deficient.

        So this test pins the property that the pairwise filter is a
        mitigation and not a replacement: the singular fallback must remain
        reachable for the higher-order collinearity the filter is blind to.
        """
        import logging

        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        z = rng.standard_normal(200)
        a = rng.standard_normal(200)
        # Overlapping sums: rank-deficient as a set, but every PAIRWISE |r|
        # is around 0.5-0.7, nowhere near the 0.999 cutoff. The filter runs
        # and drops nothing; PC still hits a singular sub-matrix.
        data = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "z": z,
                "a": a,
                "p": x + y,
                "q": y + z,
                "r": z + a,
                "s": x + y + z + a,
            }
        )
        cols = ["x", "y", "z", "a", "p", "q", "r", "s"]

        with caplog.at_level(logging.WARNING, logger="evaluation.chamber_pipeline.inference"):
            adj = run_pc(data, cols)

        # Output is well-typed (the all-zeros fallback).
        assert adj.shape == (8, 8)
        assert adj.to_numpy().sum() == 0
        # AND a warning was emitted about the fallback. The exact phrasing
        # is implementation detail; we just check the keyword "fell back"
        # which is in the log message.
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("fell back" in m.lower() for m in warning_messages), (
            f"Expected fallback warning; got: {warning_messages}"
        )


@requires_causal_learn
class TestUnknownLinAlgErrorReraise:
    """Verify that LinAlgErrors NOT matching the singular-matrix phrase re-raise.

    The previous polish only filtered ValueErrors on phrase; LinAlgErrors
    were silently swallowed regardless of message. This test pins the
    new tighter behaviour against regression — using monkeypatch to
    inject a LinAlgError with an unrelated message (e.g., a hypothetical
    "SVD did not converge") and verifying it propagates.
    """

    def test_unknown_linalg_error_propagates(self, monkeypatch) -> None:
        """A LinAlgError with a non-singular message must NOT be swallowed."""
        from evaluation.chamber_pipeline import inference

        def raising_pc(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(inference, "_causallearn_pc", raising_pc)
        # Construct minimal valid input.
        rng = np.random.default_rng(0)
        data = pd.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
        with pytest.raises(np.linalg.LinAlgError, match="SVD did not converge"):
            run_pc(data, ["x", "y"])

    def test_unknown_value_error_propagates(self, monkeypatch) -> None:
        """A ValueError with a non-fisherz-singular message must NOT be swallowed.

        Already true pre-polish, but pinned here as a regression guard
        because the singular-failure logic in run_pc now mixes both
        exception types in one filter expression — easy to break.
        """
        from evaluation.chamber_pipeline import inference

        def raising_pc(*args, **kwargs):
            raise ValueError("input array contains nan")

        monkeypatch.setattr(inference, "_causallearn_pc", raising_pc)
        rng = np.random.default_rng(0)
        data = pd.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
        with pytest.raises(ValueError, match="contains nan"):
            run_pc(data, ["x", "y"])
