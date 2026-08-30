"""Tests for the Causal Chamber pipeline's M3a baseline agents.

Covers `evaluation.chamber_pipeline.agents.random_agent` and
`greedy_ig_lite_agent` (M3a). The LLM-bearing variants live in
`test_chamber_llm_agents.py` (M3b) and `test_chamber_planner_reasoner.py`
(M3c) — this file deliberately stays focused on the M3a non-LLM agents
plus the cross-chamber compatibility smoke (added post M3 review).

These tests need `causalchamber` (the chambers extra) for the
ContractedChamberAgent end-to-end path.
"""

from __future__ import annotations

import pytest

from agent_contracts.core.wrapper import ContractViolationError
from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from agent_contracts.integrations.causalchamber import (
    ContractedChamberAgent,
    create_contracted_chamber_agent,
)
from evaluation.chamber_pipeline.agents import (
    _parse_target,
    greedy_ig_lite_agent,
    random_agent,
)
from tests.evaluation.conftest import RecordingLLM, _menu_from

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# Parser unit tests (no chamber data required)
# ---------------------------------------------------------------------------


class TestParseTarget:
    """Experiment-name → target-variable parsing."""

    def test_simple_target_mid(self) -> None:
        assert _parse_target("uniform_red_mid") == "red"

    def test_underscore_in_target(self) -> None:
        # The non-greedy `(.+?)_(weak|mid|strong)$` correctly leaves the
        # multi-segment target intact.
        assert _parse_target("uniform_t_ir_1_mid") == "t_ir_1"
        assert _parse_target("uniform_diode_vis_1_strong") == "diode_vis_1"
        assert _parse_target("uniform_osr_angle_2_weak") == "osr_angle_2"

    def test_unparseable_returns_none(self) -> None:
        # LT's `uniform_reference` is the chamber's no-intervention baseline
        # and is intentionally unparseable here — we treat None as a
        # distinct target so it doesn't preempt real-target slots.
        assert _parse_target("uniform_reference") is None
        assert _parse_target("malformed") is None
        assert _parse_target("") is None


# ---------------------------------------------------------------------------
# Random agent (M3a)
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestRandomAgent:
    """Pareto-floor baseline: pick k uniformly at random, run PC."""

    def test_returns_node_aligned_dataframe(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        adj = random_agent(adapter, seed=0)
        gt = adapter.ground_truth()
        # Same node set, same ordering.
        assert list(adj.index) == list(gt.index)
        assert list(adj.columns) == list(gt.columns)
        assert adj.shape == gt.shape

    def test_spends_full_budget(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        random_agent(adapter, seed=0)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 3

    def test_zero_budget_returns_empty_adjacency(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=0)
        adj = random_agent(adapter, seed=0)
        # No data → no inference → all-zeros adjacency on the right shape.
        assert (adj.values == 0).all()
        assert adj.shape == adapter.ground_truth().shape

    def test_seed_is_deterministic(self) -> None:
        """Same seed + adapter state → same intervention selection."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter1 = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        adapter2 = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        random_agent(adapter1, seed=99)
        random_agent(adapter2, seed=99)
        # Both adapters should have spent on the same experiment names.
        events1 = [e["data"]["experiment_name"] for e in adapter1.events if e["type"] == "tool_use"]
        events2 = [e["data"]["experiment_name"] for e in adapter2.events if e["type"] == "tool_use"]
        assert events1 == events2

    def test_does_not_overshoot_budget_in_strict_mode(self) -> None:
        """Budget=2 should never spend a 3rd query — pure agent contract."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        random_agent(adapter, seed=0)
        # No ContractViolationError should have been raised, since the
        # agent stays within budget by construction.
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 2


# ---------------------------------------------------------------------------
# GreedyIG-lite agent (M3a)
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestGreedyIgLiteAgent:
    """Principled non-LLM baseline via greedy target-coverage."""

    def test_prefers_distinct_targets(self) -> None:
        """Tier-1 selection should hit each parsed target at most once."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        # LT has 29 distinct parseable targets + 1 reference. Budget=5
        # comfortably stays in tier 1 (one per target), so all 5 chosen
        # experiments should have distinct targets.
        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=5)
        greedy_ig_lite_agent(adapter, seed=0)

        chosen = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
        assert len(chosen) == 5
        targets = [_parse_target(name) for name in chosen]
        # All-distinct (allowing one None for the reference experiment).
        assert len(set(targets)) == 5

    def test_returns_aligned_dataframe(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        adj = greedy_ig_lite_agent(adapter, seed=0)
        assert adj.shape == adapter.ground_truth().shape
        assert list(adj.index) == list(adapter.ground_truth().index)

    def test_zero_budget_returns_empty_adjacency(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=0)
        adj = greedy_ig_lite_agent(adapter, seed=0)
        assert (adj.values == 0).all()


# ---------------------------------------------------------------------------
# All five plan-§5.1 baseline variants are now implemented:
#
#   - random_agent / greedy_ig_lite_agent : tested above (M3a, no LLM)
#   - llm_only_agent / llm_pc_agent       : tests/evaluation/test_chamber_llm_agents.py (M3b)
#   - planner_reasoner_agents             : tests/evaluation/test_chamber_planner_reasoner.py (M3c)
#
# This file deliberately stays focused on the M3a non-LLM agents.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spot-check that ContractViolationError surfaces from agents too
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestAgentBudgetContract:
    """If an agent overshoots its declared per_tool_limits, the framework
    raises — this is a contract-level guarantee, not an agent-level one,
    and we verify it works through the agent dispatch surface."""

    def test_overshooting_raises_via_query_intervention(self) -> None:
        """Construct a degenerate 'agent' that intentionally overshoots."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        def overshooting_agent(adapter: ContractedChamberAgent) -> None:
            names = adapter.available_experiments()
            adapter.query_intervention(names[0])  # ok
            adapter.query_intervention(names[1])  # should raise

        adapter = create_contracted_chamber_agent(
            chamber="lt", intervention_budget=1, agent=overshooting_agent
        )
        with pytest.raises(ContractViolationError, match="intervention budget exhausted"):
            adapter.run()


# ---------------------------------------------------------------------------
# Chamber-parameterized smoke tests (added post M3 review)
#
# Catches the kind of bug where an agent silently degrades on one chamber
# but works on another because a regex / parser was tested only against
# LT's naming conventions. Random is robust by design (no parsing); GIG
# requires parseable target names and so MUST raise on chambers without
# them, per plan §5.1 variant 2 ("LT-only" footnote).
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestChamberCompatibility:
    """Cross-chamber compatibility for the M3a non-LLM agents."""

    def test_random_agent_works_on_lt(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        adj = random_agent(adapter, seed=0)
        assert adj.shape == adapter.ground_truth().shape

    def test_random_agent_works_on_wt(self) -> None:
        """Random doesn't parse menu names, so it works on any chamber."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=2)
        adj = random_agent(adapter, seed=0)
        assert adj.shape == adapter.ground_truth().shape

    def test_greedy_ig_lite_works_on_lt(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        adj = greedy_ig_lite_agent(adapter, seed=0)
        assert adj.shape == adapter.ground_truth().shape

    def test_greedy_ig_lite_raises_on_wt(self) -> None:
        """WT's experimental design has no discrete intervention targets,
        so target-coverage degenerates to random selection. GIG-lite
        must refuse to run rather than silently mimic Random — otherwise
        the §5.3 Pareto plot on WT would show GIG ≈ Random with no
        explanation, and a reviewer would (rightly) ask why."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=2)
        with pytest.raises(NotImplementedError, match=r"WT|wt|target-coverage|LT-only"):
            greedy_ig_lite_agent(adapter, seed=0)

    def test_greedy_ig_lite_error_names_chamber_and_offers_remedy(self) -> None:
        """The error message must be loud enough for an M5 sweep
        runner to know which cells to skip and why."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=2)
        with pytest.raises(NotImplementedError) as exc:
            greedy_ig_lite_agent(adapter, seed=0)
        msg = str(exc.value)
        # Chamber identifier in message → easy to grep in sweep logs.
        assert "wt" in msg.lower()
        # Remedy spelled out → orchestrator author knows what to do.
        assert "skip" in msg.lower()
        # Plan-doc reference → anyone confused has a place to read.
        assert "5.1" in msg


# ---------------------------------------------------------------------------
# The UNCONTRACTED control -- `llm_pc` with the contract removed
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestUncontractedAgent:
    """Every other registered arm is contracted, so nothing measured what
    governance costs. This arm is the other half of that comparison."""

    def test_stops_when_the_agent_says_done(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )
        from evaluation.chamber_pipeline.agents import uncontracted_agent
        from tests.evaluation.conftest import RecordingLLM, _menu_from

        def responder(idx: int, msgs: list[dict[str, str]]) -> str:
            menu = _menu_from(msgs)
            if idx >= 2:
                return "DONE"
            return menu[idx % len(menu)] if menu else ""

        llm = RecordingLLM(responder)
        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=59)
        uncontracted_agent(adapter, seed=0, llm=llm)
        stats = adapter.coordination_stats
        assert stats["n_experiments_distinct"] == 2, stats
        assert stats["agg_hit_safety_stop"] == 0

    def test_a_never_stopping_agent_is_flagged_not_silently_capped(self) -> None:
        """Running the whole menu by choice and by exhaustion are different
        findings, and the experiment count alone cannot tell them apart."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )
        from evaluation.chamber_pipeline.agents import uncontracted_agent
        from tests.evaluation.conftest import RecordingLLM, _menu_from

        llm = RecordingLLM(lambda i, m: (_menu_from(m) or [""])[0])
        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=59)
        uncontracted_agent(adapter, seed=0, llm=llm)
        stats = adapter.coordination_stats
        assert stats["agg_hit_safety_stop"] == 1, stats

    def test_the_prompt_states_no_budget(self) -> None:
        """The arm removes the contract; if the prompt still quotes a number
        it removes enforcement only, which is a different experiment."""
        from evaluation.chamber_pipeline.llm_planner import (
            build_uncontracted_select_prompt,
        )

        msgs = build_uncontracted_select_prompt(["a_x", "b_y"], 59, [])
        blob = " ".join(m["content"] for m in msgs)
        assert "59" not in blob
        assert "budget" not in blob.lower() or "no budget" in blob.lower()
        assert "DONE" in blob


class TestSharedBlackboard:
    """The axis's top rung: two voices, one undivided record and menu.

    Its whole value is being the arm that SHOULD collapse onto the loop, so the
    properties that make that a fair test are the ones worth pinning: the
    record must be shared and complete, the menu must never be partitioned, and
    the turns must actually alternate. A blackboard that quietly handed each
    voice half the menu would be `fan_in_spec` under another name, and would
    "fail" to collapse for a reason having nothing to do with the axis.
    """

    @staticmethod
    def _run(k: int, llm: RecordingLLM) -> ContractedChamberAgent:
        from evaluation.chamber_pipeline.agents import shared_blackboard_agents

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=k)
        shared_blackboard_agents(adapter, seed=0, llm=llm)
        return adapter

    def test_it_spends_the_whole_budget_on_distinct_experiments(
        self, fake_llm: RecordingLLM
    ) -> None:
        adapter = self._run(6, fake_llm)
        assert len(adapter.purchased) == 6
        assert len(set(adapter.purchased)) == 6

    def test_turns_alternate_between_two_distinct_voices(self, fake_llm: RecordingLLM) -> None:
        """Different system prompts on odd and even steps, else it is one voice."""
        self._run(6, fake_llm)
        systems = [c["messages"][0]["content"] for c in fake_llm.calls]
        even, odd = {systems[0], systems[2], systems[4]}, {systems[1], systems[3], systems[5]}
        assert len(even) == 1, "voice A must be consistent across its turns"
        assert len(odd) == 1, "voice B must be consistent across its turns"
        assert even != odd, "the two voices must actually differ"

    def test_every_call_sees_the_complete_shared_record(self, fake_llm: RecordingLLM) -> None:
        """The defining property: each prompt carries every prior pick,
        including the ones the OTHER voice made. That is what separates this
        arm from every partitioned rung on the ladder."""
        adapter = self._run(6, fake_llm)
        for step, call in enumerate(fake_llm.calls):
            body = str(call["messages"])
            for prior in adapter.purchased[:step]:
                assert prior in body, f"step {step} cannot see earlier pick {prior}"

    def test_the_menu_is_never_partitioned(self, fake_llm: RecordingLLM) -> None:
        """Both voices must be able to reach every experiment not yet bought."""
        adapter = self._run(4, fake_llm)
        for step, call in enumerate(fake_llm.calls):
            offered = set(_menu_from(call["messages"]))
            bought = set(adapter.purchased[:step])
            assert not offered & bought, f"step {step} was re-offered a bought name"
            assert len(offered) + len(bought) == 59, (
                f"step {step} saw {len(offered)} of the 59-entry menu — partitioned"
            )

    def test_a_budget_over_the_menu_stops_rather_than_spinning(
        self, fake_llm: RecordingLLM
    ) -> None:
        adapter = self._run(70, fake_llm)
        assert len(adapter.purchased) == 59
