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
    from evaluation.chamber_pipeline.agents import _RECONCILE_MAX_TOKENS

    adapter = make_ladder_adapter(counting_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm)

    from tests.evaluation.conftest import call_kind

    reconcile = [c for c in counting_llm.calls if call_kind(c["messages"]) == "reconcile"]
    assert len(reconcile) == 1
    # This used to assert `!= _SELECTION_MAX_TOKENS`, standing in for "do not
    # reuse a selection-sized cap on a reasoning call". That inequality became
    # meaningless when both were raised to 32768 -- and the underlying concern
    # was never the difference, it was that the cap must clear the reasoning
    # load. Assert the substantive property instead.
    assert reconcile[0]["max_tokens"] == _RECONCILE_MAX_TOKENS
    assert reconcile[0]["max_tokens"] >= 2 * 11690  # worst measured late-loop call
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
    from tests.evaluation.conftest import call_kind

    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=1,
        scout_b_budget=1,
        differentiate=False,
        llm=fake_llm,
    )
    # Filter by prompt kind. Filtering on `max_tokens == _SELECTION_MAX_TOKENS`
    # also matched the aggregator once selection and reconcile were both raised
    # to 32768, so this compared a scout's system prompt against the
    # aggregator's and failed for the wrong reason.
    selection = [c for c in fake_llm.calls if call_kind(c["messages"]) == "select"]
    assert selection
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


# ---------------------------------------------------------------------------
# `honor_aggregator` -- the ablation answering "your aggregator is a no-op"
# ---------------------------------------------------------------------------


@requires_causalchamber
def test_honoring_the_aggregator_restricts_the_pooled_set(make_ladder_adapter):
    """With honor_aggregator, the aggregator's answer selects what PC sees.

    Default behaviour discards that answer and pools the dedup'd union, so a
    negative result for fan-in could be dismissed as an artifact of a null
    aggregator. This is the arm that makes the claim measurable instead of
    arguable.
    """
    from tests.evaluation.conftest import RecordingLLM, call_kind

    picked: list[str] = []

    def responder(idx: int, msgs: list[dict[str, str]]) -> str:
        from tests.evaluation.conftest import _menu_from

        menu = _menu_from(msgs)
        if call_kind(msgs) == "reconcile":
            # Name exactly ONE of the scouts' picks: a strict subset, so
            # "honored" and "discarded" cannot agree by accident.
            return picked[0]
        choice = menu[idx % len(menu)] if menu else ""
        picked.append(choice)
        return choice

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=2,
        scout_b_budget=2,
        honor_aggregator=True,
        llm=llm,
    )
    stats = adapter.coordination_stats
    assert stats["n_experiments_distinct"] == 1, stats
    assert stats["agg_named"] == 1
    assert stats["agg_hallucinated"] == 0
    assert stats["agg_dropped"] >= 1
    assert "agg_fallback" not in stats


@requires_causalchamber
def test_default_discards_the_aggregator_and_pools_the_union(make_ladder_adapter):
    """The control for the test above: same LLM, honor_aggregator off."""
    from tests.evaluation.conftest import RecordingLLM, _menu_from, call_kind

    picked: list[str] = []

    def responder(idx: int, msgs: list[dict[str, str]]) -> str:
        menu = _menu_from(msgs)
        if call_kind(msgs) == "reconcile":
            return picked[0]
        choice = menu[idx % len(menu)] if menu else ""
        picked.append(choice)
        return choice

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=llm)
    stats = adapter.coordination_stats
    assert stats["n_experiments_distinct"] > 1, stats
    assert not any(k.startswith("agg_") for k in stats)


@requires_causalchamber
def test_a_hallucinating_aggregator_cannot_fabricate_data(make_ladder_adapter):
    """`_parse_name_list` matches the MENU, not what was purchased.

    The reconcile prompt carries only the two scouts' lists -- no menu -- so
    an aggregator can only name an unpurchased experiment by inventing it
    from its own knowledge of the chamber. Rare, but pooling it would
    fabricate data the budget never bought, so the arm intersects against
    actual purchases and falls back rather than pooling an empty set.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from, call_kind

    full_menu: list[str] = []
    picked: list[str] = []

    def responder(idx: int, msgs: list[dict[str, str]]) -> str:
        menu = _menu_from(msgs)
        if call_kind(msgs) == "reconcile":
            # A real menu name the scouts never bought.
            unbought = [m for m in full_menu if m not in picked]
            return unbought[-1] if unbought else ""
        if menu and not full_menu:
            full_menu.extend(menu)
        choice = menu[idx % len(menu)] if menu else ""
        picked.append(choice)
        return choice

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=2,
        scout_b_budget=2,
        honor_aggregator=True,
        llm=llm,
    )
    stats = adapter.coordination_stats
    assert stats["agg_named"] == 1, stats
    assert stats["agg_hallucinated"] == 1, stats
    assert stats.get("agg_fallback") == 1, stats
    # Fell back to the union rather than pooling an empty set.
    assert stats["n_experiments_distinct"] > 1


@requires_causalchamber
def test_an_empty_aggregator_response_falls_back_to_the_union(make_ladder_adapter):
    """Truncation and empty content are live failure modes on this stack.

    Scoring the parser instead of the topology is the error to avoid: an
    empty reconcile must not pool nothing and report F1 on no data.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from, call_kind

    def responder(idx: int, msgs: list[dict[str, str]]) -> str:
        if call_kind(msgs) == "reconcile":
            return ""
        menu = _menu_from(msgs)
        return menu[idx % len(menu)] if menu else ""

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm)
    fan_in_agents(
        adapter,
        seed=0,
        scout_a_budget=2,
        scout_b_budget=2,
        honor_aggregator=True,
        llm=llm,
    )
    stats = adapter.coordination_stats
    assert stats["agg_named"] == 0
    assert stats.get("agg_fallback") == 1
    assert stats["n_experiments_distinct"] > 1
