"""Rung 4: two scouts that know about each other and negotiate.

The ladder's top rung. Rungs 1 and 2 keep the scouts blind so they isolate
role differentiation; here the scouts exchange proposals before executing,
which is the first rung where coordination is explicit rather than emergent.

The scout-to-scout channel is a Python variable, NOT a DelegationGraph edge.
Control flow may cycle; budget flow may not (whitepaper §4.6 P3).
"""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline.agents import team_agents
from tests.evaluation.conftest import requires_causalchamber


@requires_causalchamber
def test_team_makes_exactly_four_negotiation_calls(make_ladder_adapter, counting_llm):
    """One upfront round -- propose then revise, per scout. O(1) in k."""
    adapter = make_ladder_adapter(counting_llm)
    team_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm)
    from tests.evaluation.conftest import is_negotiation

    negotiation = [c for c in counting_llm.calls if is_negotiation(c["messages"])]
    assert len(negotiation) == 4


@requires_causalchamber
def test_team_backstop_removes_contested_picks_from_scout_b(make_ladder_adapter, conflict_llm):
    """Both scouts name the same experiment; scout_a keeps it, scout_b re-picks.

    The backstop must apply to the EXECUTED selections, not just the
    proposals -- the scouts execute blind after negotiating, so filtering only
    the proposals leaves them free to re-collide.
    """
    adapter = make_ladder_adapter(conflict_llm)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=conflict_llm)
    assert adapter.coordination_stats["overlap_frac"] == 0.0


@requires_causalchamber
def test_backstop_does_not_leak_the_exclusion_into_scout_bs_prompt(
    make_ladder_adapter, conflict_llm
):
    """Exclusion narrows the menu; it must not appear as "Already spent".

    Routing it through `starting_chosen` would render the contested names into
    scout_b's prompt, destroying the blindness of the execution phase, and
    would also shrink `actual_spend` so scout_b silently under-spends.
    """
    from evaluation.chamber_pipeline.agents import _SELECTION_MAX_TOKENS

    adapter = make_ladder_adapter(conflict_llm)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=conflict_llm)
    selection = [c for c in conflict_llm.calls if c["max_tokens"] == _SELECTION_MAX_TOKENS]
    assert selection, "no selection calls were made"

    # Each scout's loop starts clean. A scout legitimately sees its OWN prior
    # picks from round 2 onward -- that is its loop history, not the peer's --
    # so the guard is on the first call of each scout's loop, which is where a
    # `starting_chosen`-based exclusion would show up.
    starts = [c for c in selection if "Already spent: (none yet)" in c["messages"][-1]["content"]]
    assert len(starts) == 2, "expected one clean loop start per scout"


@requires_causalchamber
def test_scout_b_still_spends_its_full_budget_after_exclusion(make_ladder_adapter, conflict_llm):
    """Matched budget survives the backstop: exclusion must not under-spend."""
    adapter = make_ladder_adapter(conflict_llm)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=conflict_llm)
    graph = adapter.delegation_graph
    used = graph.monitor_for("scout_b").usage.get_tool_usage("intervene")
    assert used == 2


def test_team_channel_cannot_be_a_bidirectional_graph_edge():
    """Only the SECOND edge of the pair is a cycle, and only while unsealed.

    `allocate()` calls `_require_unsealed()` before any cycle check, so a
    sealed graph raises the sealed error rather than CycleError. And
    scout_a -> scout_b alone is not a cycle: it is the return edge that closes
    the loop. Hence message passing, not graph edges.
    """
    from agent_contracts.core.contract import Contract, ResourceConstraints
    from agent_contracts.core.delegation_graph import CycleError, DelegationGraph

    root = Contract(id="t", name="t", resources=ResourceConstraints(tokens=1000))
    graph = DelegationGraph(root)
    for n in ("scout_a", "scout_b"):
        graph.add_node(n)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=100)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=100)
    graph.allocate("scout_a", "scout_b", tokens=0)  # not yet a cycle
    with pytest.raises(CycleError):
        graph.allocate("scout_b", "scout_a", tokens=0)  # closes it


@requires_causalchamber
def test_team_conservation_holds_and_stats_are_recorded(make_ladder_adapter, counting_llm):
    adapter = make_ladder_adapter(counting_llm)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=counting_llm)
    adapter.delegation_graph.verify()
    stats = adapter.coordination_stats
    assert stats["n_experiments_distinct"] >= 1
    assert stats["overlap_frac"] is None or 0.0 <= stats["overlap_frac"] <= 1.0


@requires_causalchamber
def test_exclude_narrows_the_menu_without_touching_the_prompt(make_ladder_adapter, fake_llm):
    """Unit-level guard on the new `_llm_select_loop` parameter."""
    from evaluation.chamber_pipeline.agents import _llm_select_loop

    adapter = make_ladder_adapter(fake_llm)
    banned = set(adapter.available_experiments()[:3])
    chosen, _ = _llm_select_loop(adapter, fake_llm, "m", seed=0, spend=2, exclude=banned)
    assert not (set(chosen) & banned)
    # The loop starts clean; from round 2 the scout legitimately sees its own
    # prior pick, which is its loop history and not the exclusion.
    assert "Already spent: (none yet)" in fake_llm.calls[0]["messages"][-1]["content"]
    # And the excluded names appear in no menu, in any round.
    for call in fake_llm.calls:
        rendered_menu = call["messages"][-1]["content"].partition("Menu:\n")[2]
        for name in banned:
            assert name not in rendered_menu


@requires_causalchamber
def test_scout_b_spends_its_full_budget_at_a_large_budget(make_ladder_adapter, conflict_llm):
    """Regression: the exclusion set must not starve scout_b.

    `exclude` narrows the menu and so feeds `actual_spend = min(spend,
    len(available))`. An earlier guard checked only `contested` against the
    full menu, ignoring `set(chosen_a)` -- the larger half of the exclusion --
    so scout_b returned 14 picks against a budget of 22 while conservation
    still passed and nothing flagged it.
    """
    adapter = make_ladder_adapter(conflict_llm, k=20)
    team_agents(adapter, seed=0, scout_a_budget=10, scout_b_budget=10, llm=conflict_llm)
    graph = adapter.delegation_graph
    assert graph.monitor_for("scout_a").usage.get_tool_usage("intervene") == 10
    assert graph.monitor_for("scout_b").usage.get_tool_usage("intervene") == 10


def test_parse_name_list_does_not_match_prefixes():
    """WT has `actuators_random_walk_1` through `_16`.

    Plain substring containment invents a claim the scout never made, which
    inflates `contested` and over-excludes the other scout.
    """
    from evaluation.chamber_pipeline.agents import _parse_name_list

    menu = [f"actuators_random_walk_{i}" for i in (1, 10, 12, 16)]
    resp = {
        "choices": [
            {"message": {"content": "actuators_random_walk_10 and actuators_random_walk_12"}}
        ]
    }
    assert _parse_name_list(resp, menu) == [
        "actuators_random_walk_10",
        "actuators_random_walk_12",
    ]


@requires_causalchamber
def test_negotiation_calls_carry_a_temperature(make_ladder_adapter, counting_llm):
    """The propose prompts differ only by the letter A/B.

    Without a temperature both scouts return the same claim list, and the
    negotiation contributes noise instead of a split.
    """
    from evaluation.chamber_pipeline.agents import (
        _SCOUT_TEMPERATURE,
    )

    adapter = make_ladder_adapter(counting_llm)
    team_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm)
    from tests.evaluation.conftest import is_negotiation

    negotiation = [c for c in counting_llm.calls if is_negotiation(c["messages"])]
    assert negotiation
    for call in negotiation:
        assert call["temperature"] == _SCOUT_TEMPERATURE


@requires_causalchamber
def test_negotiation_outcome_is_recorded(make_ladder_adapter, conflict_llm):
    """`n_contested` makes rung 4's mechanism measurable.

    Without it a team whose scouts never agree on a split is indistinguishable
    from one whose negotiation worked.
    """
    adapter = make_ladder_adapter(conflict_llm)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=conflict_llm)
    assert adapter.coordination_stats["n_contested"] >= 0


def _queried(adapter) -> set[str]:
    return {
        e["data"]["experiment_name"]
        for e in adapter.events
        if e.get("type") == "tool_use" and "experiment_name" in e["data"]
    }


@requires_causalchamber
def test_the_selection_loop_changes_which_experiments_are_queried(
    make_ladder_adapter,
):
    """Varying ONLY the selection response must change the queried set.

    The negotiation answer is held fixed. An earlier version of this test
    varied both, so the pools differed between runs and the sets differed for
    that reason -- it passed even when selection was made fully inert by
    truncating each pool to exactly its budget.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from, is_negotiation

    def queried(sel):
        holder: list = []

        def responder(idx, msgs):
            names = _menu_from(msgs)
            if not names:
                return ""
            # Classify by the call's own max_tokens, not by menu size. A size
            # threshold had zero margin: the largest selection pool is
            # `budget + ceil((M-k)/2)` = 30 against a `> 30` guard, so any
            # pool-sizing change that reached 31 would silently reclassify
            # selection prompts as negotiation and both tests would keep
            # passing while testing something else.
            if is_negotiation(msgs):
                return "\n".join(names[:5])  # negotiation: FIXED
            return sel(names)

        llm = RecordingLLM(responder)
        holder.append(llm)
        adapter = make_ladder_adapter(llm, k=20)
        team_agents(adapter, seed=0, scout_a_budget=10, scout_b_budget=10, llm=llm)
        return _queried(adapter)

    first, last = queried(lambda n: n[0]), queried(lambda n: n[-1])
    assert len(first) == 20 and len(last) == 20  # matched budget holds
    assert first != last, "selection is inert: identical queries either way"


@requires_causalchamber
def test_matched_budget_holds_at_the_largest_ladder_budget(make_ladder_adapter):
    """k=45 is where the pool split fell short.

    A plain `rest[0::2]` / `rest[1::2]` gave 23 + 18 = 41 of 45 once the
    claims were large, with `status=ok` and conservation certified. k=6 and
    k=30 never triggered it, so a test at a smaller budget proves nothing.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from, is_negotiation

    holder: list = []

    def responder(idx, msgs):
        names = _menu_from(msgs)
        if not names:
            return ""
        # max_tokens, not menu size -- see the sibling test for why.
        if is_negotiation(msgs):
            return "\n".join(names[:23])  # negotiation: a full budget's worth
        return names[0]

    llm = RecordingLLM(responder)
    holder.append(llm)
    adapter = make_ladder_adapter(llm, k=45)
    team_agents(adapter, seed=0, scout_a_budget=23, scout_b_budget=22, llm=llm)
    assert len(_queried(adapter)) == 45


@requires_causalchamber
def test_claim_is_capped_so_a_verbose_scout_cannot_starve_its_partner(
    make_ladder_adapter,
):
    """Uncapped, a scout reasoning over most of the menu swallows the pool.

    Measured before the cap: 10 + 4 against a 20 budget when `claim_a`
    reached 55 names, reported `status=ok` with conservation certified.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from

    def responder(idx, msgs):
        names = _menu_from(msgs)
        if not names:
            return ""
        # A negotiation reply that names almost the entire menu.
        return "\n".join(names[:55]) if len(names) > 40 else names[0]

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm, k=20)
    team_agents(adapter, seed=0, scout_a_budget=10, scout_b_budget=10, llm=llm)
    graph = adapter.delegation_graph
    a = graph.monitor_for("scout_a").usage.get_tool_usage("intervene")
    b = graph.monitor_for("scout_b").usage.get_tool_usage("intervene")
    assert (a, b) == (10, 10), f"matched budget broken: {a} + {b}"


@requires_causalchamber
def test_unparseable_negotiation_is_recorded(make_ladder_adapter):
    """An unusable negotiation round must leave a trace.

    It degrades to a menu-order partition -- every seed queries the identical
    experiments, zero between-seed variance -- while `overlap_frac` reads 0.0
    and `n_contested` reads 0, i.e. indistinguishable from a perfect split.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from

    def responder(idx, msgs):
        # Negotiation prompts get prose with no menu names; selection works.
        names = _menu_from(msgs)
        if len(names) > 40:  # the negotiation prompts render the whole menu
            return "I would prefer to defer this decision."
        return names[0] if names else ""

    llm = RecordingLLM(responder)
    adapter = make_ladder_adapter(llm, k=4)
    team_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=llm)
    assert adapter.coordination_stats["n_negotiation_failures"] > 0


def test_static_kwargs_cannot_be_mutated_through_the_registry():
    """`get_spec(...).static_kwargs[k] = v` used to persist globally."""
    from evaluation.chamber_pipeline.orchestrator import get_spec

    with pytest.raises(TypeError):
        get_spec("fan_in_spec").static_kwargs["differentiate"] = False


@requires_causalchamber
def test_the_fallback_partition_varies_with_the_seed(make_ladder_adapter):
    """The `rest` shuffle is seeded, so the split is not the same every seed.

    Before the shuffle, `rest` was sliced by parity. The menu is ordered by
    variable family and intervention strength, so a parity slice handed
    scout_a zero `osr_c` and zero `red` experiments -- the identical blind
    spot in all 30 seeds, because nothing about the slice depended on the
    seed. Deleting the shuffle leaves every other test green.
    """
    from evaluation.chamber_pipeline import agents as _agents
    from tests.evaluation.conftest import RecordingLLM, _menu_from, is_negotiation

    real_loop = _agents._llm_select_loop

    def pool_a_for(seed: int) -> frozenset:
        holder: list = []
        captured: list = []

        def responder(idx, msgs):
            names = _menu_from(msgs)
            if not names:
                return ""
            # Negotiation held FIXED across seeds: claims cannot be the
            # source of any difference this test observes.
            if is_negotiation(msgs):
                return "\n".join(names[:3])
            return names[0]

        llm = RecordingLLM(responder)
        holder.append(llm)
        adapter = make_ladder_adapter(llm, k=20)

        def spy(*a, **kw):
            captured.append(kw.get("exclude"))
            return real_loop(*a, **kw)

        _agents._llm_select_loop = spy
        try:
            team_agents(adapter, seed=seed, scout_a_budget=10, scout_b_budget=10, llm=llm)
        finally:
            _agents._llm_select_loop = real_loop
        all_names = set(adapter.available_experiments())
        return frozenset(all_names - captured[0])

    pools = {pool_a_for(s) for s in range(5)}
    assert len(pools) > 1, "scout_a's pool is identical in every seed: shuffle is inert"


@requires_causalchamber
def test_negotiation_failures_counts_rounds_not_scouts(make_ladder_adapter):
    """One unparseable revision is one failure, and the split still stands.

    The counter previously summed over the two SCOUTS (`revised or proposed`
    empty), which reports 0 for the likelier and more damaging case: both
    propose rounds parse but both REVISE rounds return prose, reducing rung 4
    to one-shot proposals with no negotiation at all.
    """
    from tests.evaluation.conftest import RecordingLLM, _menu_from, is_negotiation

    holder: list = []
    negotiation_seen: list = []

    def responder(idx, msgs):
        names = _menu_from(msgs)
        if not names:
            return ""
        if is_negotiation(msgs):
            negotiation_seen.append(idx)
            # Rounds are propose_a, propose_b, revise_a, revise_b. Break the
            # FOURTH only: scout_b falls back to its own proposal, so the
            # split survives intact and the old scout-wise count reports 0.
            if len(negotiation_seen) == 4:
                return "I would prefer not to commit to a split at this time."
            return "\n".join(names[:3])
        return names[0]

    llm = RecordingLLM(responder)
    holder.append(llm)
    adapter = make_ladder_adapter(llm, k=20)
    team_agents(adapter, seed=0, scout_a_budget=10, scout_b_budget=10, llm=llm)

    assert len(negotiation_seen) == 4
    assert adapter.coordination_stats["n_negotiation_failures"] == 1
