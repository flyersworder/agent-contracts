"""Rungs 1 and 2 of the coordination ladder: two blind scouts, one aggregator.

Rung 1 (`differentiate=False`) is a homogeneous ensemble; rung 2
(`differentiate=True`) gives the scouts distinct roles. Neither scout may
learn the other exists -- that blindness is what makes the pair a test of
role differentiation rather than of communication.
"""

from __future__ import annotations

import pandas as pd

from evaluation.chamber_pipeline.agents import fan_in_agents
from tests.evaluation.conftest import requires_causalchamber


@requires_causalchamber
def test_fan_in_returns_square_adjacency_over_node_names(make_ladder_adapter, fake_llm):
    adapter = make_ladder_adapter(fake_llm)
    out = fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm)
    nodes = adapter.ground_truth().columns.tolist()
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == nodes
    assert list(out.columns) == nodes


@requires_causalchamber
def test_scouts_use_decorrelated_seeds(make_ladder_adapter, fake_llm, monkeypatch):
    """2*seed and 2*seed+1, never seed and seed+1.

    M4b seeds are contiguous 0..29, so `seed + 1` collides with the NEXT
    cell's scout_a and the two arms would share draws across cells.
    """
    import evaluation.chamber_pipeline.agents as agents_mod

    seen: list[int] = []
    real = agents_mod._llm_select_loop

    def spy(*args, **kwargs):
        seen.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(agents_mod, "_llm_select_loop", spy)
    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(adapter, seed=7, scout_a_budget=1, scout_b_budget=1, llm=fake_llm)
    assert seen[:2] == [14, 15]


@requires_causalchamber
def test_aggregator_consumes_tokens_via_reconciliation(make_ladder_adapter, counting_llm):
    """Without this call the fan-in edges carry budget nothing spends.

    PC is not an LLM call, so absent reconciliation the aggregator consumes
    nothing, `DelegationGraph._consumed()` reads zero, and verify() is
    vacuously true -- H-2 would be unfalsifiable and P2 would have no
    empirical form.
    """
    from evaluation.chamber_pipeline.agents import (
        _RECONCILE_MAX_TOKENS,
        _SELECTION_MAX_TOKENS,
    )

    adapter = make_ladder_adapter(counting_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm)

    reconcile = [c for c in counting_llm.calls if c["max_tokens"] == _RECONCILE_MAX_TOKENS]
    assert len(reconcile) == 1
    assert reconcile[0]["max_tokens"] != _SELECTION_MAX_TOKENS
    # Making the call is not enough: its tokens must REACH the node monitor.
    assert adapter.delegation_graph.monitor_for("aggregator").usage.tokens > 0


@requires_causalchamber
def test_overlap_and_distinct_count_recorded(make_ladder_adapter, fake_llm):
    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm)
    stats = adapter.coordination_stats
    assert 0.0 <= stats["overlap_frac"] <= 1.0
    assert stats["n_experiments_distinct"] >= 1


@requires_causalchamber
def test_coordination_stats_exist_even_on_the_empty_budget_path(make_ladder_adapter, fake_llm):
    """Task 8 reads this in run_cell; a happy-path-only attribute raises."""
    adapter = make_ladder_adapter(fake_llm, k=0)
    fan_in_agents(adapter, seed=0, scout_a_budget=0, scout_b_budget=0, llm=fake_llm)
    assert adapter.coordination_stats["overlap_frac"] is None
    assert adapter.coordination_stats["n_experiments_distinct"] == 0


@requires_causalchamber
def test_colliding_scouts_report_full_overlap(make_ladder_adapter, conflict_llm):
    """The degeneracy rung 1 risks: identical scouts buy no diversity."""
    adapter = make_ladder_adapter(conflict_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=conflict_llm)
    assert adapter.coordination_stats["overlap_frac"] == 1.0
    assert adapter.coordination_stats["n_experiments_distinct"] == 1


@requires_causalchamber
def test_differentiate_selects_the_role_prompts(make_ladder_adapter, fake_llm):
    """Rung 2 must actually use distinct role framings, and stay blind."""
    from evaluation.chamber_pipeline.llm_planner import (
        _SCOUT_BROAD_SYSTEM_MESSAGE,
        _SCOUT_TARGETED_SYSTEM_MESSAGE,
    )

    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=1,
        scout_b_budget=1,
        differentiate=True,
        llm=fake_llm,
    )
    systems = {c["messages"][0]["content"] for c in fake_llm.calls}
    assert _SCOUT_BROAD_SYSTEM_MESSAGE in systems
    assert _SCOUT_TARGETED_SYSTEM_MESSAGE in systems
    for text in systems:
        assert "planner" not in text.lower()
        assert "other agent" not in text.lower()


@requires_causalchamber
def test_homogeneous_mode_gives_both_scouts_the_same_prompt(make_ladder_adapter, fake_llm):
    """Rung 1's scouts differ only by sampling, never by instruction."""
    from evaluation.chamber_pipeline.agents import _SELECTION_MAX_TOKENS

    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=1,
        scout_b_budget=1,
        differentiate=False,
        llm=fake_llm,
    )
    selection = [c for c in fake_llm.calls if c["max_tokens"] == _SELECTION_MAX_TOKENS]
    assert len({c["messages"][0]["content"] for c in selection}) == 1


@requires_causalchamber
def test_conservation_holds_after_the_arm_runs(make_ladder_adapter, counting_llm):
    """H-C: the graph still verifies once every node has spent."""
    adapter = make_ladder_adapter(counting_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=counting_llm)
    adapter.delegation_graph.verify()


@requires_causalchamber
def test_scout_budgets_are_respected_per_node(make_ladder_adapter, fake_llm):
    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm)
    graph = adapter.delegation_graph
    for scout in ("scout_a", "scout_b"):
        used = graph.monitor_for(scout).usage.get_tool_usage("intervene")
        assert used <= graph.in_flow(scout).per_tool["intervene"]
    assert graph.monitor_for("aggregator").usage.get_tool_usage("intervene") == 0
