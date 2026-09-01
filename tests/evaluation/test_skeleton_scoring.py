"""Tests for undirected (skeleton) scoring.

Skeleton F1 exists to separate "did the method find this adjacency" from
"did it orient it the way the reference happens to", because orientation
inside a Markov equivalence class is not identifiable and flips on numerical
noise. These tests pin the two properties that make it meaningful: it must
ignore direction, and it must symmetrize BOTH sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.scoring import f1_edges, f1_skeleton, skeletonize

NODES = ["a", "b", "c"]


def g(*edges: tuple[str, str]) -> pd.DataFrame:
    m = pd.DataFrame(0, index=NODES, columns=NODES)
    for src, dst in edges:
        m.loc[src, dst] = 1
    return m


def test_skeletonize_symmetrizes() -> None:
    out = skeletonize(g(("a", "b")))
    assert out.loc["a", "b"] == 1
    assert out.loc["b", "a"] == 1


def test_skeletonize_clears_the_diagonal() -> None:
    """A self-loop is not an adjacency and must not score as a true positive."""
    m = g(("a", "b"))
    m.loc["a", "a"] = 1
    assert skeletonize(m).loc["a", "a"] == 0


def test_skeletonize_does_not_mutate_its_input() -> None:
    m = g(("a", "b"))
    before = m.to_numpy().copy()
    skeletonize(m)
    assert np.array_equal(m.to_numpy(), before)


def test_reversed_edge_is_perfect_undirected_and_zero_directed() -> None:
    """The whole point: orientation is ignored, presence is not.

    a->b scored against b->a has NO cell overlap, so directed F1 is 0. As a
    skeleton the two graphs are identical, so F1 is 1.
    """
    assert f1_edges(g(("a", "b")), g(("b", "a"))) == pytest.approx(0.0)
    assert f1_skeleton(g(("a", "b")), g(("b", "a"))) == pytest.approx(1.0)


def test_missing_edge_still_penalised_undirected() -> None:
    """Ignoring direction must not mean ignoring absence."""
    assert f1_skeleton(g(("a", "b")), g(("a", "b"), ("b", "c"))) < 1.0


def test_both_sides_are_symmetrised() -> None:
    """Scoring a symmetrised prediction against a DIRECTED reference would
    count each true edge twice and halve recall. Identical graphs must score
    1.0 under the skeleton metric, which fails if only one side is doubled.
    """
    truth = g(("a", "b"), ("b", "c"))
    assert f1_skeleton(truth, truth) == pytest.approx(1.0)


def test_unoriented_edge_is_penalised_directed_but_not_undirected() -> None:
    """The case this metric exists for.

    `cpdag_to_directed_adjacency` encodes an UNDIRECTED CPDAG edge as both
    directions, so a correctly-found but unoriented edge scores one true
    positive AND one false positive under `f1_edges` — charged for a
    coin flip the data cannot settle. As a skeleton it is simply correct.
    """
    truth = g(("a", "b"))
    unoriented = g(("a", "b"), ("b", "a"))
    assert f1_edges(unoriented, truth) == pytest.approx(2 / 3)
    assert f1_skeleton(unoriented, truth) == pytest.approx(1.0)


def test_skeleton_is_not_uniformly_higher_than_directed() -> None:
    """Documents a real property, so nobody re-derives the wrong invariant.

    "Ignoring direction can only help" is FALSE, and an earlier version of
    this file asserted it. Symmetrising MERGES a mutual pair into one edge
    while EXPANDING each single arrow into two cells, so a prediction with
    many one-way edges can lose more to inflated false positives than it
    gains from merging. Concretely, below: directed 4/7, skeleton 2/5.

    Consequence for reporting: skeleton F1 is a DIFFERENT metric, not a
    relaxation of the directed one, and the two cannot be compared as though
    one bounded the other.
    """
    nodes = ["n0", "n1", "n2", "n3"]

    def m(cells: list[tuple[int, int]]) -> pd.DataFrame:
        out = pd.DataFrame(0, index=nodes, columns=nodes)
        for i, j in cells:
            out.iloc[i, j] = 1
        return out

    predicted = m([(0, 3), (1, 2), (3, 0), (3, 1)])
    reference = m([(0, 2), (0, 3), (3, 0)])
    assert f1_edges(predicted, reference) == pytest.approx(0.5714, abs=1e-4)
    assert f1_skeleton(predicted, reference) == pytest.approx(0.4000, abs=1e-4)
    assert f1_skeleton(predicted, reference) < f1_edges(predicted, reference)
