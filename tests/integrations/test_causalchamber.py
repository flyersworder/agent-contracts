"""Tests for the Causal Chamber integration.

This file holds two kinds of tests:

1. **M1 contract checks** — verify the import surface and constructor shape
   that `docs/causal_chamber_M1_decisions.md` §2.1 fixes. These run today
   (no causalchamber install required).
2. **M2 smoke test** — ground-truth round-trip ("load lt/standard, fake
   agent returns ground truth, score reports SHD=0 and F1=1") per the M2
   acceptance criterion in `docs/causal_chamber_validation_plan.md` §9. This
   test is `xfail(strict=True)` against the M1 stub; when M2 lands the test
   will pass and `strict=True` will force-fail CI as a forcing function to
   remove the marker.

The xfail-strict pattern is the correct shape for "failing smoke test
exists" (the M1 acceptance criterion) without breaking CI today.
"""

from __future__ import annotations

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.integrations import (
    CAUSAL_CHAMBER_AVAILABLE,
    ContractedChamberAgent,
    create_contracted_chamber_agent,
)

# ---------------------------------------------------------------------------
# M1 contract checks: API shape exists, irrespective of whether the
# causalchamber package is installed.
# ---------------------------------------------------------------------------


class TestM1ApiSurface:
    """API-shape checks fixed by docs/causal_chamber_M1_decisions.md §2.1."""

    def test_availability_flag_is_bool(self) -> None:
        """Convention: every integration exposes a `<NAME>_AVAILABLE` bool."""
        assert isinstance(CAUSAL_CHAMBER_AVAILABLE, bool)

    def test_class_and_factory_are_exported(self) -> None:
        """M1 Q1 §2.1: both class and factory are exported."""
        if CAUSAL_CHAMBER_AVAILABLE:
            assert ContractedChamberAgent is not None
            assert create_contracted_chamber_agent is not None
        else:
            # When the optional dep is missing, both should be None — same
            # convention used by langchain / langgraph / google_adk blocks.
            assert ContractedChamberAgent is None
            assert create_contracted_chamber_agent is None

    def test_class_name_follows_contracted_x_convention(self) -> None:
        """M1 Q1 §2.1: noun is `Contracted<X>`, not `<X>Contract`."""
        # Imported by name; that the name resolves at module level is the test.
        from agent_contracts.integrations import causalchamber as cc

        assert hasattr(cc, "ContractedChamberAgent")
        assert not hasattr(cc, "ChamberContract"), (
            "M1 Q1 §2.1: class should be ContractedChamberAgent, not "
            "ChamberContract — that name appeared in the §4.2 sketch but "
            "deviates from the codebase convention. See "
            "docs/causal_chamber_M1_decisions.md §2.2."
        )

    def test_class_is_not_dataclass(self) -> None:
        """M1 Q1 §2.1: regular class with __init__, no @dataclass."""
        from dataclasses import is_dataclass

        from agent_contracts.integrations import causalchamber as cc

        assert not is_dataclass(cc.ContractedChamberAgent), (
            "M1 Q1 §2.1: integrations are regular classes, not dataclasses. "
            "See docs/causal_chamber_M1_decisions.md §2.2."
        )


# ---------------------------------------------------------------------------
# Tests below this line need the causalchamber package installed.
# ---------------------------------------------------------------------------

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


@requires_causalchamber
class TestConstructorShape:
    """Constructor signature checks (M1, runnable when causalchamber is installed)."""

    def _make_contract(self) -> Contract:
        return Contract(
            id="m1-stub",
            name="M1 Stub Contract",
            resources=ResourceConstraints(per_tool_limits={"intervene": 1}),
        )

    def test_first_param_is_contract(self) -> None:
        """M1 Q1 §2.1: caller constructs the Contract; adapter takes it as input."""
        contract = self._make_contract()
        agent = ContractedChamberAgent(contract=contract, chamber="lt")
        assert agent.contract is contract

    def test_default_configuration_is_standard(self) -> None:
        contract = self._make_contract()
        agent = ContractedChamberAgent(contract=contract, chamber="lt")
        assert agent.configuration == "standard"

    def test_strict_mode_default_true(self) -> None:
        contract = self._make_contract()
        agent = ContractedChamberAgent(contract=contract, chamber="lt")
        assert agent.strict_mode is True

    def test_monitors_and_enforcer_wired(self) -> None:
        """M1 §2.3: hand-wired ResourceMonitor / TemporalMonitor / ContractEnforcer."""
        contract = self._make_contract()
        agent = ContractedChamberAgent(contract=contract, chamber="lt")
        assert agent._resource_monitor is not None
        assert agent._temporal_monitor is not None
        assert agent._enforcer is not None


@requires_causalchamber
class TestM1StubRaises:
    """Verify the stub fails for the right reason — NotImplementedError, not import errors.

    This is what makes the smoke-test below meaningful: when we get
    NotImplementedError today, we know the stub is correctly placed and
    M2 can fill in the bodies without refactoring the surface.
    """

    def _agent(self) -> ContractedChamberAgent:
        return ContractedChamberAgent(
            contract=Contract(
                id="stub",
                name="M1 Stub",
                resources=ResourceConstraints(per_tool_limits={"intervene": 1}),
            ),
            chamber="lt",
        )

    def test_query_intervention_raises_notimplemented(self) -> None:
        with pytest.raises(NotImplementedError, match="M2"):
            self._agent().query_intervention("any")

    def test_query_observation_raises_notimplemented(self) -> None:
        with pytest.raises(NotImplementedError, match="M2"):
            self._agent().query_observation()

    def test_ground_truth_raises_notimplemented(self) -> None:
        with pytest.raises(NotImplementedError, match="M2"):
            self._agent().ground_truth()

    def test_factory_raises_notimplemented(self) -> None:
        with pytest.raises(NotImplementedError, match="M2"):
            create_contracted_chamber_agent(chamber="lt", intervention_budget=1)


# ---------------------------------------------------------------------------
# M2 smoke test — the milestone-gating round-trip from the validation plan.
# ---------------------------------------------------------------------------


@requires_causalchamber
@pytest.mark.xfail(
    strict=True,
    reason="M2 implementation pending — see docs/causal_chamber_validation_plan.md §9 milestone M2. "
    "When the test passes, xfail(strict=True) will fail CI; remove the marker at that point.",
)
class TestM2SmokeRoundTrip:
    """The M2 acceptance test, encoded today and gated by xfail-strict.

    Per `docs/causal_chamber_validation_plan.md` §9 M2 acceptance criterion:
        "Smoke test passes: load lt/standard graph, run a fake agent that
        returns the ground truth, score reports SHD=0 and F1=1"
    """

    def test_perfect_recovery_yields_shd_zero_and_f1_one(self) -> None:
        """Ground-truth round-trip: oracle agent → SHD=0, F1=1."""
        agent = create_contracted_chamber_agent(
            chamber="lt",
            configuration="standard",
            intervention_budget=59,  # full LT menu
        )

        # Oracle agent: cheats by returning the ground-truth graph directly.
        # In M3+ we replace this with the five real baselines.
        ground_truth = agent.ground_truth()
        predicted = ground_truth.copy()

        # Imported here (not at module top) because evaluation/chamber_pipeline/
        # doesn't exist until §8 of the plan is implemented in M2.
        from evaluation.chamber_pipeline.scoring import (  # type: ignore[import-not-found]
            f1_edges,
            shd,
        )

        assert shd(predicted, ground_truth) == 0
        assert f1_edges(predicted, ground_truth) == pytest.approx(1.0)
