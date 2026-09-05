"""`_C95_NEGOTIATE` is per chamber, and the calibrated set is derived from it.

WT `team` cells ran on LT's negotiate constant with their conservation voided
-- 300 cells at `conservation_certified = None` -- because the figure had never
been isolated on WT. The fix is a per-chamber dict; these guards are about the
failure mode that made the single constant survive, which is that a wrong
calibration figure does NOT fail loudly. It yields plausible conservation
numbers that are really statements about provisioning.
"""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline import orchestrator
from evaluation.chamber_pipeline.orchestrator import (
    _C95_NEGOTIATE_BY_CHAMBER,
    _NEGOTIATE_CALIBRATED_CHAMBERS,
    _PROVISION_MULTIPLE,
    SweepConfigurationError,
    _ladder_calibration,
    get_spec,
    is_provisional_calibration,
)


def test_calibrated_set_is_derived_from_the_measurements():
    """One source of truth, so the two cannot disagree.

    The previous design held the constant in one place and the
    "which chambers are calibrated" set in another, with a comment warning
    that an entry left behind after measurement "silently voids conservation
    for the whole sweep". Deriving the set removes the possibility rather
    than documenting it.
    """
    assert frozenset(_C95_NEGOTIATE_BY_CHAMBER) == _NEGOTIATE_CALIBRATED_CHAMBERS


def test_wt_negotiation_is_measured_and_differs_from_lt():
    """The measurement that unblocks WT H-C.

    Asserting it DIFFERS from LT is the point: had WT's figure landed on
    LT's, carrying LT's over would have been harmless and the 300 voided
    cells would have been a formality. It does not -- WT's menu is 28
    experiments against LT's 59, and the negotiate prompt renders the menu.
    """
    assert "wt" in _C95_NEGOTIATE_BY_CHAMBER
    assert _C95_NEGOTIATE_BY_CHAMBER["wt"] != _C95_NEGOTIATE_BY_CHAMBER["lt"]


def test_negotiating_arm_on_a_measured_chamber_is_not_provisional():
    assert is_provisional_calibration("wt", 14, negotiates=True) is False
    assert is_provisional_calibration("lt", 30, negotiates=True) is False


def test_negotiating_arm_on_an_unmeasured_chamber_stays_provisional():
    """The safety property, checked on a chamber that does not exist.

    A new chamber must void conservation for its negotiating arm until its
    negotiate cost is measured, exactly as WT did.
    """
    assert is_provisional_calibration("zz", 14, negotiates=True) is True
    assert is_provisional_calibration("zz", 14, negotiates=False) is False


def test_overhead_uses_the_chambers_own_figure():
    """The bug this whole change exists to fix.

    `_ladder_calibration` already took `chamber`; the negotiate term ignored
    it, so a WT team cell was provisioned with LT's number.
    """
    spec = get_spec("team")
    _, _, _, overhead_wt = _ladder_calibration(spec, 14, chamber="wt")
    _, _, _, overhead_lt = _ladder_calibration(spec, 30, chamber="lt")
    assert overhead_wt == (
        spec.negotiation_rounds * _PROVISION_MULTIPLE * _C95_NEGOTIATE_BY_CHAMBER["wt"]
    )
    assert overhead_lt == (
        spec.negotiation_rounds * _PROVISION_MULTIPLE * _C95_NEGOTIATE_BY_CHAMBER["lt"]
    )
    assert overhead_wt != overhead_lt


def test_non_negotiating_arm_gets_no_negotiate_overhead():
    """`c95_negotiate` must not leak into the blind fan-in rungs.

    They are the arms whose conservation was already being reported; adding
    overhead to them would silently loosen their grants and inflate H-C.
    """
    for name in ("fan_in_homog", "fan_in_spec"):
        spec = get_spec(name)
        assert spec.negotiation_rounds == 0
        assert _ladder_calibration(spec, 14, chamber="wt")[3] == 0


def test_unmeasured_chamber_raises_rather_than_borrowing_lt(monkeypatch):
    """Same policy as `_A95_RECONCILE_BY_K`: measure, do not extrapolate.

    The unmeasured chamber is simulated by removing WT from the negotiate
    table rather than by passing an unknown chamber id. An unknown id trips
    the `_A95_RECONCILE_BY_K` guard first, so the test would pass without the
    negotiate guard existing at all -- and a test that passes against the
    missing implementation is exactly the failure this suite has hit before.
    """
    spec = get_spec("team")
    monkeypatch.setitem(orchestrator._C95_NEGOTIATE_BY_CHAMBER, "wt", 6102)
    monkeypatch.delitem(orchestrator._C95_NEGOTIATE_BY_CHAMBER, "wt")
    with pytest.raises(SweepConfigurationError, match="negotiation cost is not calibrated"):
        _ladder_calibration(spec, 14, chamber="wt")
