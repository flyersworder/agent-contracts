"""Tests for the Causal Chamber integration.

This file holds three kinds of tests:

1. **M1 contract checks** — verify the import surface and constructor shape
   that `docs/causal_chamber_M1_decisions.md` §2.1 fixes. These run today
   (no causalchamber install required).
2. **M2 behavioural tests** — exercise the real adapter implementation:
   ground-truth retrieval, per-tool budget enforcement (strict + lenient),
   factory contract, run-loop dispatch.
3. **M2 smoke test** — ground-truth round-trip ("load lt/standard, oracle
   agent returns ground truth, score reports SHD=0 and F1=1") per the M2
   acceptance criterion in `docs/causal_chamber_validation_plan.md` §9.
   Was `xfail(strict=True)` during M1; xfail removed once implementation
   made it XPASS.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.core.wrapper import ContractViolationError
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

# `evaluation/` is research apparatus and is deliberately absent from the sdist
# (see the sdist allowlist in pyproject.toml). Tests that reach into it must
# therefore skip rather than fail when running from an installed archive.
requires_evaluation_pkg = pytest.mark.skipif(
    importlib.util.find_spec("evaluation") is None,
    reason="`evaluation/` is not shipped in the sdist; run from a repository checkout",
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


# ---------------------------------------------------------------------------
# M2 behavioural tests — the integration is now implemented.
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestM2DataLoading:
    """Lazy-load + ground-truth retrieval (M2)."""

    def _agent(self, **kwargs: Any) -> ContractedChamberAgent:
        kwargs.setdefault("intervention_budget", 1)
        return create_contracted_chamber_agent(chamber="lt", **kwargs)

    def test_load_is_idempotent(self) -> None:
        a = self._agent()
        a.load()
        first_dataset = a._dataset
        a.load()
        # Second load should reuse the same handle, not redownload.
        assert a._dataset is first_dataset

    def test_ground_truth_returns_square_dataframe(self) -> None:
        gt = self._agent().ground_truth()
        # LT/standard is documented as 38 nodes, 57 edges.
        assert gt.shape == (38, 38)
        assert list(gt.index) == list(gt.columns)
        assert int((gt.values != 0).sum()) == 57

    def test_available_experiments_returns_full_menu(self) -> None:
        # LT/standard has M=59 interventional experiments.
        names = self._agent().available_experiments()
        assert len(names) == 59
        assert all(isinstance(n, str) for n in names)


@requires_causalchamber
class TestM2BudgetEnforcement:
    """Per-tool budget gating + audit emission (M2)."""

    def test_query_intervention_charges_one_unit(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        names = a.available_experiments()
        a.query_intervention(names[0])
        assert a._resource_monitor.usage.get_tool_usage("intervene") == 1

    def test_query_intervention_returns_real_dataframe(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        names = a.available_experiments()
        df = a.query_intervention(names[0])
        # LT experiments are 1000 samples by 46 columns per the plan §3.3.
        assert df.shape == (1000, 46)
        assert "intervention" in df.columns

    def test_strict_mode_raises_when_budget_exhausted(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        names = a.available_experiments()
        a.query_intervention(names[0])  # spend the budget
        with pytest.raises(ContractViolationError, match="intervention budget exhausted"):
            a.query_intervention(names[1])

    def test_lenient_mode_does_not_raise_on_overshoot(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=1, strict_mode=False)
        names = a.available_experiments()
        a.query_intervention(names[0])
        # Should complete without raising; emits a tool_blocked event but proceeds.
        a.query_intervention(names[1])
        assert any(e["type"] == "tool_blocked" for e in a.events)

    def test_failed_tool_call_does_not_charge_budget(self) -> None:
        """Charge-on-success: bad name raises before usage is incremented."""
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=5)
        with pytest.raises(KeyError):
            a.query_intervention("nonexistent-experiment-xyz")
        assert a._resource_monitor.usage.get_tool_usage("intervene") == 0

    def test_query_observation_enforces_separate_budget(self) -> None:
        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=0, observation_budget=2
        )
        df = a.query_observation(n_samples=10)
        assert df.shape[0] == 10
        # Intervene budget untouched; observe charged once.
        assert a._resource_monitor.usage.get_tool_usage("observe") == 1
        assert a._resource_monitor.usage.get_tool_usage("intervene") == 0

    def test_query_observation_rejects_non_positive_n_samples(self) -> None:
        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=0, observation_budget=1
        )
        with pytest.raises(ValueError, match="n_samples"):
            a.query_observation(n_samples=0)

    def test_audit_log_records_tool_use_events(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        names = a.available_experiments()
        a.query_intervention(names[0])
        tool_use_events = [e for e in a.events if e["type"] == "tool_use"]
        assert len(tool_use_events) == 1
        assert tool_use_events[0]["data"]["tool_name"] == "intervene"
        assert tool_use_events[0]["data"]["experiment_name"] == names[0]


@requires_causalchamber
class TestM2Factory:
    """`create_contracted_chamber_agent` factory contract (M2)."""

    def test_default_contract_id_includes_budget(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=15)
        assert a.contract.id == "chamber-lt-standard-k15"

    def test_explicit_contract_id_wins(self) -> None:
        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=1, contract_id="custom"
        )
        assert a.contract.id == "custom"

    def test_per_tool_limits_set_correctly(self) -> None:
        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=10, observation_budget=5
        )
        assert a.contract.resources.per_tool_limits == {"intervene": 10, "observe": 5}

    def test_zero_observation_budget_is_emitted_as_a_hard_zero(self) -> None:
        """An omitted key means *unconstrained*, so zero must be emitted.

        This test previously asserted the opposite -- that the key is omitted
        -- which codified the defect rather than catching it.
        """
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=10)
        assert a.contract.resources.per_tool_limits["observe"] == 0

    def test_extra_resources_merged(self) -> None:
        extras = ResourceConstraints(tokens=50_000, cost_usd=2.0)
        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=10, extra_resources=extras
        )
        assert a.contract.resources.tokens == 50_000
        assert a.contract.resources.cost_usd == 2.0
        # And per-tool limits still flow through.
        assert a.contract.resources.per_tool_limits["intervene"] == 10


@requires_causalchamber
class TestM2RunLoop:
    """Bound-agent execution under enforcement (M2)."""

    def test_run_dispatches_to_agent_with_self(self) -> None:
        captured: dict[str, Any] = {}

        def my_agent(adapter: ContractedChamberAgent, label: str) -> str:
            captured["adapter"] = adapter
            captured["label"] = label
            return "done"

        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=1, agent=my_agent)
        result = a.run("hello")
        assert result == "done"
        assert captured["adapter"] is a
        assert captured["label"] == "hello"

    def test_run_without_agent_raises(self) -> None:
        a = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        with pytest.raises(RuntimeError, match="agent"):
            a.run()

    def test_run_stops_enforcer_even_on_exception(self) -> None:
        def failing_agent(adapter: ContractedChamberAgent) -> None:
            raise RuntimeError("boom")

        a = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=1, agent=failing_agent
        )
        with pytest.raises(RuntimeError, match="boom"):
            a.run()
        # contract_started + contract_stopped events both present.
        types = [e["type"] for e in a.events]
        assert "contract_started" in types
        assert "contract_stopped" in types


# ---------------------------------------------------------------------------
# M2 smoke test — the milestone-gating round-trip from the validation plan.
#
# Per `docs/causal_chamber_validation_plan.md` §9 M2 acceptance criterion:
#     "Smoke test passes: load lt/standard graph, run a fake agent that
#     returns the ground truth, score reports SHD=0 and F1=1"
#
# Implemented in M2 (was xfail-strict during M1; xfail removed once
# implementation made it XPASS).
# ---------------------------------------------------------------------------


@requires_causalchamber
@requires_evaluation_pkg
class TestM2SmokeRoundTrip:
    """M2 acceptance round-trip: oracle agent → SHD=0, F1=1."""

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
        from evaluation.chamber_pipeline.scoring import (  # type: ignore[import-not-found]
            f1_edges,
            shd,
        )

        assert shd(predicted, ground_truth) == 0
        assert f1_edges(predicted, ground_truth) == pytest.approx(1.0)


@requires_causalchamber
def test_default_observation_budget_is_enforced_as_zero():
    """`observation_budget=0` means zero, not unconstrained.

    `can_use_tool` treats an *absent* per-tool key as unconstrained, so
    emitting the limit only when positive inverted the documented default:
    every default-constructed agent could call `query_observation` without
    bound. Regression guard for that inversion.
    """
    agent = create_contracted_chamber_agent(
        chamber="lt", configuration="standard", intervention_budget=5
    )
    limits = agent.contract.resources.per_tool_limits
    assert limits["observe"] == 0
    assert agent._resource_monitor.can_use_tool("observe") is False
    assert agent._resource_monitor.can_use_tool("intervene") is True


# --------------------------------------------------------------------------
# M6 Task 1: additive per-node metering and token attribution
# --------------------------------------------------------------------------


@requires_causalchamber
def test_as_node_blocks_on_the_node_limit():
    agg = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=10, node_monitors={"aggregator": agg}
    )
    name = adapter.available_experiments()[0]
    with adapter.as_node("aggregator"), pytest.raises(ContractViolationError):
        adapter.query_intervention(name)


@requires_causalchamber
def test_aggregate_cap_still_binds_inside_as_node():
    """Routing is additive: the k cap must stay live, not be bypassed."""
    scout = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 99}))
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=2, node_monitors={"scout_a": scout}
    )
    menu = adapter.available_experiments()
    with adapter.as_node("scout_a"):
        adapter.query_intervention(menu[0])
        adapter.query_intervention(menu[1])
        with pytest.raises(ContractViolationError):  # aggregate k=2 exhausted
            adapter.query_intervention(menu[2])


@requires_causalchamber
def test_as_node_attributes_token_delta_to_the_node():
    counter = {"n": 0}
    mon = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 5}))
    adapter = create_contracted_chamber_agent(
        chamber="lt",
        intervention_budget=5,
        node_monitors={"scout_a": mon},
        token_meter=lambda: counter["n"],
    )
    with adapter.as_node("scout_a"):
        counter["n"] += 1234  # stands in for _CountingLLM's totals
    assert mon.usage.tokens == 1234


@requires_causalchamber
def test_as_node_rejects_an_unregistered_node():
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=5)
    with pytest.raises(KeyError), adapter.as_node("nobody"):
        pass


@requires_causalchamber
def test_as_node_refuses_to_nest():
    """Nesting would charge the inner block's tokens to both nodes."""
    mons = {
        "scout_a": ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 5})),
        "scout_b": ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 5})),
    }
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=5, node_monitors=mons
    )
    with adapter.as_node("scout_a"), pytest.raises(RuntimeError):  # noqa: SIM117
        with adapter.as_node("scout_b"):
            pass


@requires_causalchamber
def test_default_none_preserves_aggregate_behaviour():
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
    menu = adapter.available_experiments()
    adapter.query_intervention(menu[0])
    adapter.query_intervention(menu[1])
    with pytest.raises(ContractViolationError):
        adapter.query_intervention(menu[2])


def test_wrong_key_exp_leaves_intervene_unconstrained():
    """Pins the trap: a zero-grant on an unknown key is silent."""
    m = ResourceMonitor(ResourceConstraints(per_tool_limits={"exp": 0}))
    assert m.can_use_tool("intervene") is True  # NOT blocked
    m2 = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    assert m2.can_use_tool("intervene") is False


def test_observe_absent_means_unlimited():
    m = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    assert m.can_use_tool("observe") is True  # the side channel
    m2 = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0, "observe": 0}))
    assert m2.can_use_tool("observe") is False


class TestDatasetConfigurationPairing:
    """The dataset must be chosen by (chamber, configuration), not chamber.

    `configuration` selects the ground-truth GRAPH. The wind tunnel in
    `pressure-control` is physically a different system — the hatch is
    servo-driven, which adds mediators `standard` does not have — so pairing
    its graph with `standard` data scores against the wrong truth and reports
    a plausible number with no error anywhere.
    """

    def test_wired_pairs_resolve(self) -> None:
        from agent_contracts.integrations.causalchamber import dataset_for

        assert dataset_for("lt", "standard") == "lt_interventions_standard_v1"
        assert dataset_for("wt", "standard") == "wt_validate_v1"

    def test_unwired_configuration_raises_rather_than_falling_back(self) -> None:
        from agent_contracts.integrations.causalchamber import dataset_for

        with pytest.raises(ValueError, match="no dataset is wired"):
            dataset_for("wt", "pressure-control")

    def test_error_names_the_wired_pairs(self) -> None:
        """The message has to be actionable — it is read at 2am mid-sweep."""
        from agent_contracts.integrations.causalchamber import dataset_for

        with pytest.raises(ValueError) as excinfo:
            dataset_for("lt", "pressure-control")
        assert "wired pairs are" in str(excinfo.value)
        assert "standard" in str(excinfo.value)

    def test_legacy_chamber_view_still_matches_the_standard_pairs(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            DATASET_FOR_CHAMBER,
            dataset_for,
        )

        for chamber in ("lt", "wt"):
            assert DATASET_FOR_CHAMBER[chamber] == dataset_for(chamber, "standard")
