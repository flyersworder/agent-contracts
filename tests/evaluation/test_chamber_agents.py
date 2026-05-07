"""Tests for the Causal Chamber pipeline's M3a baseline agents.

Covers `evaluation.chamber_pipeline.agents.random_agent` and
`greedy_ig_lite_agent` (M3a). The LLM-bearing variants land in M3b/M3c
and have placeholder tests verifying they raise NotImplementedError
with a clear M3 reference.

These tests need `causalchamber` (the chambers extra) for the
ContractedChamberAgent end-to-end path.
"""

from __future__ import annotations

import pytest

from agent_contracts.core.wrapper import ContractViolationError
from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline.agents import (
    _parse_target,
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    planner_reasoner_agents,
    random_agent,
)

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
# LLM-bearing variants — M3b/M3c stubs
# ---------------------------------------------------------------------------


class TestM3bM3cStubs:
    """Verify the LLM-bearing variants raise with a clear M3 reference."""

    def test_llm_only_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="M3b"):
            llm_only_agent(adapter=None)  # type: ignore[arg-type]

    def test_llm_pc_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="M3b"):
            llm_pc_agent(adapter=None)  # type: ignore[arg-type]

    def test_planner_reasoner_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="M3c"):
            planner_reasoner_agents(
                adapter=None,  # type: ignore[arg-type]
                planner_budget=1,
                reasoner_budget=1,
            )


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
            ContractedChamberAgent,
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
