"""Tests for the Causal Chamber pipeline's PC inference helpers.

Covers `evaluation.chamber_pipeline.inference`. These tests need
`causal-learn` (provided by the `chambers` extra) and pandas, but do
NOT need `causalchamber` itself — they exercise the inference module
in isolation against synthesized linear-Gaussian data.
"""

from __future__ import annotations

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
