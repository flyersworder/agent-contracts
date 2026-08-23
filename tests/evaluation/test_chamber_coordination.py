"""Coordination metrics for the M6 ladder.

See docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md
"""

from __future__ import annotations

from evaluation.chamber_pipeline.coordination import overlap_fraction


def test_overlap_fraction_disjoint_is_zero():
    assert overlap_fraction(["a", "b"], ["c", "d"]) == 0.0


def test_overlap_fraction_identical_is_one():
    assert overlap_fraction(["a", "b"], ["a", "b"]) == 1.0


def test_overlap_fraction_uses_min_denominator():
    """A subset relationship is total overlap for the smaller selection."""
    assert overlap_fraction(["a", "b", "c"], ["a"]) == 1.0


def test_overlap_fraction_empty_is_none_not_zero():
    """Undefined, not zero.

    Zero is the H-B success case -- perfectly disjoint scouts. A cell where a
    scout got no picks at all must not be indistinguishable from it, or the
    analyzer averages the artifact into the headline.
    """
    assert overlap_fraction([], ["a"]) is None
    assert overlap_fraction(["a"], []) is None
    assert overlap_fraction([], []) is None


def test_overlap_fraction_ignores_within_scout_duplicates():
    """The metric is over sets: a scout repeating itself is not agreement."""
    assert overlap_fraction(["a", "a", "b"], ["a"]) == 1.0
    assert overlap_fraction(["a", "a"], ["b", "b"]) == 0.0


def test_overlap_fraction_is_symmetric():
    for a, b in ((["a", "b", "c"], ["b", "d"]), (["x"], ["x", "y", "z"])):
        assert overlap_fraction(a, b) == overlap_fraction(b, a)


def test_overlap_fraction_is_bounded():
    assert 0.0 <= overlap_fraction(["a", "b", "c"], ["b", "c", "d"]) <= 1.0
