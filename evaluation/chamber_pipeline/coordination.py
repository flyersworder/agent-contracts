"""Coordination metrics for the M6 ladder.

Rungs 1 and 2 both run two scouts against a shared budget. What separates
them is whether role differentiation buys exploration diversity that plain
ensembling does not, so the ladder needs a measure of how much the two scouts
actually diverged. That is what this module provides.

See docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md
"""

from __future__ import annotations


def overlap_fraction(chosen_a: list[str], chosen_b: list[str]) -> float | None:
    """Fraction of the smaller selection that also appears in the other.

    ``|A ∩ B| / min(|A|, |B|)``. The minimum is the denominator so that a
    subset relationship reads as total overlap: a scout whose every pick was
    also made by its partner contributed no new experiments, whatever the
    partner did besides.

    Returns ``None`` -- not ``0.0`` -- when either side is empty. Zero is the
    H-B success case, two perfectly disjoint scouts; a cell where a scout got
    no picks at all is a degenerate run and must stay distinguishable from it,
    or the analyzer averages the artifact into the headline.

    Compares sets, so a scout repeating itself within its own loop does not
    register as agreement with its partner.
    """
    if not chosen_a or not chosen_b:
        return None
    shared = len(set(chosen_a) & set(chosen_b))
    return shared / min(len(set(chosen_a)), len(set(chosen_b)))
