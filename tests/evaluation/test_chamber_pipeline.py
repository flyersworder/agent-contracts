"""Tests for the Causal Chamber pipeline scoring functions.

Covers `evaluation.chamber_pipeline.scoring`. These tests validate the
scoring contract independently of the integration adapter — pure-function
tests that need only pandas (already a transitive dep of the evaluation
modules) and don't need `causalchamber` itself.

The integration-side ground-truth round-trip ("oracle agent → SHD=0,
F1=1") lives in `tests/integrations/test_causalchamber.py`.
"""

import pandas as pd
import pytest

from evaluation.chamber_pipeline.scoring import ci_coverage, f1_edges, shd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def adj(rows: list[list[int]], nodes: list[str]) -> pd.DataFrame:
    """Build an adjacency-matrix DataFrame from rows + node names."""
    return pd.DataFrame(rows, index=nodes, columns=nodes)


# A small reference graph: A → B, A → C, B → C
REF_NODES = ["A", "B", "C"]
REF = adj(
    [
        [0, 1, 1],  # A: → B, → C
        [0, 0, 1],  # B: → C
        [0, 0, 0],  # C
    ],
    REF_NODES,
)
EMPTY = adj([[0] * 3 for _ in range(3)], REF_NODES)


# ---------------------------------------------------------------------------
# SHD
# ---------------------------------------------------------------------------


class TestShd:
    """Structural Hamming Distance."""

    def test_identical_graphs_have_shd_zero(self) -> None:
        assert shd(REF, REF) == 0

    def test_empty_vs_full_reference_equals_edge_count(self) -> None:
        # Reference has 3 edges; predicting empty graph misses all 3.
        assert shd(EMPTY, REF) == 3

    def test_one_extra_edge_costs_one(self) -> None:
        predicted = REF.copy()
        predicted.loc["C", "A"] = 1  # spurious C → A
        assert shd(predicted, REF) == 1

    def test_one_missing_edge_costs_one(self) -> None:
        predicted = REF.copy()
        predicted.loc["A", "B"] = 0  # drop A → B
        assert shd(predicted, REF) == 1

    def test_reversed_edge_costs_two(self) -> None:
        """Cell-wise Hamming: reversing A→B becomes B→A, flipping two cells."""
        predicted = REF.copy()
        predicted.loc["A", "B"] = 0
        predicted.loc["B", "A"] = 1
        assert shd(predicted, REF) == 2

    def test_reorders_predicted_to_match_reference(self) -> None:
        """Predicted with shuffled rows/cols should still score correctly."""
        predicted = REF.loc[["C", "A", "B"], ["C", "A", "B"]].copy()
        # The matrix content is identical to REF after reindexing → SHD=0
        assert shd(predicted, REF) == 0

    def test_nonzero_is_treated_as_edge(self) -> None:
        """Edge weights other than 1 still count as a present edge."""
        # pandas 3+ refuses to put a float into an int64 column without an
        # explicit upcast, so build a float-dtype copy first.
        predicted = REF.astype(float).copy()
        predicted.loc["A", "B"] = 0.7  # nonzero, so still an edge
        assert shd(predicted, REF) == 0

    def test_missing_node_raises(self) -> None:
        predicted = REF.loc[["A", "B"], ["A", "B"]].copy()
        with pytest.raises(ValueError, match="missing"):
            shd(predicted, REF)

    def test_extra_node_raises(self) -> None:
        predicted = adj(
            [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ["A", "B", "C", "D"],
        )
        with pytest.raises(ValueError, match="unknown"):
            shd(predicted, REF)

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(ValueError, match="DataFrames"):
            shd([[0]], REF)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


class TestF1Edges:
    """F1 score on edge presence."""

    def test_perfect_match_is_one(self) -> None:
        assert f1_edges(REF, REF) == pytest.approx(1.0)

    def test_both_empty_is_one(self) -> None:
        """Perfect agreement on the empty graph is F1=1.0 by convention."""
        assert f1_edges(EMPTY, EMPTY) == pytest.approx(1.0)

    def test_predicted_empty_against_nonempty_reference_is_zero(self) -> None:
        assert f1_edges(EMPTY, REF) == pytest.approx(0.0)

    def test_predicted_nonempty_against_empty_reference_is_zero(self) -> None:
        assert f1_edges(REF, EMPTY) == pytest.approx(0.0)

    def test_partial_recovery(self) -> None:
        """Recover 2 of 3 edges, with 0 false positives → F1 = 0.8."""
        predicted = REF.copy()
        predicted.loc["A", "B"] = 0  # drop one true edge
        # tp=2, fp=0, fn=1 → P=1.0, R=2/3 → F1 = 2*1.0*0.667/(1.0+0.667) ≈ 0.8
        assert f1_edges(predicted, REF) == pytest.approx(0.8)

    def test_one_fp_one_fn(self) -> None:
        """Drop A→B, add C→A: tp=2, fp=1, fn=1 → F1 = 2*(2/3)/(2/3+2/3+? )."""
        predicted = REF.copy()
        predicted.loc["A", "B"] = 0
        predicted.loc["C", "A"] = 1
        # tp=2, fp=1, fn=1 → P=2/3, R=2/3 → F1 = 2*(2/3)*(2/3)/((2/3)+(2/3)) = 2/3
        assert f1_edges(predicted, REF) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# CI coverage
# ---------------------------------------------------------------------------


class TestCiCoverage:
    """Calibration coverage on edge-presence intervals."""

    def test_perfect_intervals_yield_full_coverage(self) -> None:
        # Place [1, 1] on true edges and [0, 0] on non-edges → coverage=1, width=0
        intervals: dict[tuple[str, str], tuple[float, float]] = {}
        for i in REF_NODES:
            for j in REF_NODES:
                truth = float(REF.loc[i, j] != 0)
                intervals[(i, j)] = (truth, truth)
        cov, width = ci_coverage(intervals, REF)
        assert cov == pytest.approx(1.0)
        assert width == pytest.approx(0.0)

    def test_uniform_interval_pegs_coverage_at_one_with_max_width(self) -> None:
        """The trivial agent: report [0, 1] everywhere."""
        intervals = {(i, j): (0.0, 1.0) for i in REF_NODES for j in REF_NODES}
        cov, width = ci_coverage(intervals, REF)
        assert cov == pytest.approx(1.0)
        assert width == pytest.approx(1.0)

    def test_zero_width_misaligned_yields_zero_coverage(self) -> None:
        # Predict the wrong binary value with zero width everywhere.
        intervals = {
            (i, j): (1.0 - float(REF.loc[i, j] != 0), 1.0 - float(REF.loc[i, j] != 0))
            for i in REF_NODES
            for j in REF_NODES
        }
        cov, width = ci_coverage(intervals, REF)
        assert cov == pytest.approx(0.0)
        assert width == pytest.approx(0.0)

    def test_partial_reporting_uses_only_reported_pairs(self) -> None:
        """Unreported (i, j) pairs are excluded from numerator and denominator."""
        intervals = {("A", "B"): (1.0, 1.0), ("A", "C"): (0.0, 0.0)}
        cov, width = ci_coverage(intervals, REF)
        # ("A","B") covered (truth=1, in [1,1]); ("A","C") not covered (truth=1, in [0,0])
        assert cov == pytest.approx(0.5)
        assert width == pytest.approx(0.0)

    def test_empty_intervals_returns_zero_zero(self) -> None:
        cov, width = ci_coverage({}, REF)
        assert cov == 0.0
        assert width == 0.0

    def test_unknown_node_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown node"):
            ci_coverage({("A", "Z"): (0.0, 1.0)}, REF)

    def test_malformed_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Malformed"):
            ci_coverage({("A", "B"): (0.7, 0.3)}, REF)

    def test_out_of_unit_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Malformed"):
            ci_coverage({("A", "B"): (-0.1, 0.5)}, REF)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ci_coverage({("A", "B"): (0.0, 1.0)}, REF, alpha=1.5)
