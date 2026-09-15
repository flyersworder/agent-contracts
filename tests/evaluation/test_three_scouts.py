"""The three-agent ablation (2026-09-15): blind fan-in arms with n scouts.

A coauthor asked whether the gain from diversifying the search space becomes
visible with more agents. Every multi-agent arm in the corpus was two scouts,
hard-wired in the graph, the spec, the calibration, the record and the
prompts. These tests pin the n-scout generalisation of the BLIND fan-in family
only: the negotiating `team` stays pairwise.
"""

from __future__ import annotations

import json

import pytest

from evaluation.chamber_pipeline.coordination import (
    build_fan_in_graph,
    mean_pairwise_overlap,
)
from evaluation.chamber_pipeline.llm_planner import build_reconcile_prompt
from evaluation.chamber_pipeline.menu_taxonomy import (
    partition_pools_by_variable,
    partition_pools_by_variable_n,
)
from evaluation.chamber_pipeline.orchestrator import (
    _build_agent_kwargs,
    _ladder_nodes,
    _scout_c95s,
    get_spec,
    run_cell,
)
from tests.evaluation.conftest import requires_causalchamber

# ---------------------------------------------------------------------------
# metrics and prompts
# ---------------------------------------------------------------------------


def test_mean_pairwise_overlap_reduces_to_overlap_fraction_for_two():
    assert mean_pairwise_overlap([["a", "b"], ["b", "c"]]) == 0.5


def test_mean_pairwise_overlap_averages_the_three_pairs():
    # ab: 1/2, ac: 0, bc: 1/2
    assert mean_pairwise_overlap([["a", "b"], ["b", "c"], ["c", "d"]]) == pytest.approx(1 / 3)


def test_mean_pairwise_overlap_is_none_if_any_scout_is_empty():
    assert mean_pairwise_overlap([["a"], [], ["a"]]) is None


def test_reconcile_prompt_lists_every_designer():
    msgs = build_reconcile_prompt(["x1"], ["x2"], ["x3"])
    user = msgs[1]["content"]
    assert "Designer A selected:\nx1" in user
    assert "Designer B selected:\nx2" in user
    assert "Designer C selected:\nx3" in user
    # The system message is the call-kind marker; it must still classify as
    # reconcile and must not lie about the count.
    assert "independent designers" in msgs[0]["content"]
    assert "two independent" not in msgs[0]["content"]


def test_reconcile_prompt_two_lists_is_unchanged():
    old = build_reconcile_prompt(["x1"], ["x2"])[1]["content"]
    assert "Designer A selected:\nx1\n\nDesigner B selected:\nx2" in old
    assert "Designer C" not in old


# ---------------------------------------------------------------------------
# variable partition into n pools
# ---------------------------------------------------------------------------


def _menu() -> list[str]:
    names = []
    for v in ("red", "green", "blue", "pol_1", "pol_2", "osr_c", "v_c", "diode_1"):
        for s in ("weak", "mid", "strong"):
            names.append(f"uniform_{v}_{s}")
    return names  # 8 variables x 3 = 24 entries


def test_n_pools_are_disjoint_cover_the_menu_and_keep_variables_whole():
    from evaluation.chamber_pipeline.menu_taxonomy import experiment_variable

    pools = partition_pools_by_variable_n(_menu(), (2, 2, 2), 0)
    assert len(pools) == 3
    assert set().union(*pools) == set(_menu())
    assert sum(len(p) for p in pools) == len(_menu())
    for pool in pools:
        for name in pool:
            assert all(
                (experiment_variable(other) != experiment_variable(name)) or other in pool
                for other in _menu()
            )


def test_two_pools_match_the_claimless_two_scout_partition():
    a, b = partition_pools_by_variable(_menu(), [], [], 2, 2, 0)
    assert partition_pools_by_variable_n(_menu(), (2, 2), 0) == (a, b)


def test_n_pools_raise_when_a_pool_cannot_exceed_its_budget():
    with pytest.raises(ValueError, match="pool of"):
        partition_pools_by_variable_n(_menu(), (8, 8, 8), 0)


def test_n_pools_are_seeded():
    assert partition_pools_by_variable_n(_menu(), (2, 2, 2), 0) != partition_pools_by_variable_n(
        _menu(), (2, 2, 2), 1
    )


# ---------------------------------------------------------------------------
# spec, kwargs, nodes, calibration
# ---------------------------------------------------------------------------


def test_three_scout_specs_declare_three_plain_roles():
    for name in ("fan_in_homog3", "fan_in_varsplit3"):
        spec = get_spec(name)
        assert spec.is_ladder_arm
        assert spec.scout_roles == ("plain", "plain", "plain")
        assert spec.extra_kwargs == ("scout_budgets",)
        assert spec.negotiation_rounds == 0


def test_blind_varsplit_specs_are_lt_only():
    assert get_spec("fan_in_varsplit").chambers == ("lt",)
    assert get_spec("fan_in_varsplit3").chambers == ("lt",)
    assert get_spec("fan_in_varsplit").scout_roles == ("plain", "plain")


def test_scout_budgets_split_k_with_remainder_to_the_earlier_scouts():
    kw = _build_agent_kwargs(get_spec("fan_in_homog3"), 31, 0, 0.05, None)
    assert kw["scout_budgets"] == (11, 10, 10)
    assert "scout_a_budget" not in kw
    kw2 = _build_agent_kwargs(get_spec("fan_in_homog"), 31, 0, 0.05, None)
    assert (kw2["scout_a_budget"], kw2["scout_b_budget"]) == (16, 15)


def test_ladder_nodes_follow_the_role_count():
    assert _ladder_nodes(get_spec("fan_in_homog")) == ("scout_a", "scout_b", "aggregator")
    assert _ladder_nodes(get_spec("fan_in_homog3")) == (
        "scout_a",
        "scout_b",
        "scout_c",
        "aggregator",
    )


def test_scout_c95s_are_one_per_role():
    assert _scout_c95s(get_spec("fan_in_homog3"), "lt") == (2205, 2205, 2205)
    assert _scout_c95s(get_spec("fan_in_spec"), "lt") == (3003, 10379)


# ---------------------------------------------------------------------------
# the agent and the cell
# ---------------------------------------------------------------------------


@pytest.fixture
def make_three_scout_adapter():
    from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

    def _make(llm, k: int = 6):
        graph = build_fan_in_graph(k=k, c95=1350, a95=21163, scout_c95s=(1350,) * 3)
        adapter = create_contracted_chamber_agent(
            chamber="lt",
            intervention_budget=k,
            node_monitors={
                n: graph.monitor_for(n) for n in ("scout_a", "scout_b", "scout_c", "aggregator")
            },
            token_meter=lambda: llm.total_tokens,
        )
        adapter.delegation_graph = graph
        return adapter

    return _make


@requires_causalchamber
def test_three_scouts_each_buy_their_share_and_all_spend_tokens(make_three_scout_adapter, fake_llm):
    from evaluation.chamber_pipeline.agents import fan_in_agents

    adapter = make_three_scout_adapter(fake_llm)
    out = fan_in_agents(adapter, seed=0, scout_budgets=(2, 2, 2), llm=fake_llm)
    assert out.shape[0] == out.shape[1] > 0
    g = adapter.delegation_graph
    for scout in ("scout_a", "scout_b", "scout_c"):
        assert g.monitor_for(scout).usage.tool_usage_by_name.get("intervene") == 2
        assert g.monitor_for(scout).usage.tokens > 0
    assert g.monitor_for("aggregator").usage.tokens > 0
    assert adapter.coordination_stats["n_scouts"] == 3
    assert adapter.coordination_stats["n_experiments_distinct"] > 0


@requires_causalchamber
def test_three_scout_seeds_are_decorrelated(make_three_scout_adapter, fake_llm, monkeypatch):
    """n*seed + i, never seed + i: contiguous seeds must not share draws."""
    import evaluation.chamber_pipeline.agents as agents_mod
    from evaluation.chamber_pipeline.agents import fan_in_agents

    seen: list[int] = []
    real = agents_mod._llm_select_loop

    def spy(*args, **kwargs):
        seen.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(agents_mod, "_llm_select_loop", spy)
    fan_in_agents(make_three_scout_adapter(fake_llm), seed=7, scout_budgets=(1, 1, 1), llm=fake_llm)
    assert seen[:3] == [21, 22, 23]


@requires_causalchamber
def test_legacy_two_scout_kwargs_still_work(make_ladder_adapter, fake_llm):
    from evaluation.chamber_pipeline.agents import fan_in_agents

    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm)
    assert adapter.coordination_stats["n_scouts"] == 2
    assert adapter.coordination_stats["overlap_frac"] is not None


@requires_causalchamber
def test_blind_varsplit_scouts_never_share_a_variable(make_three_scout_adapter, fake_llm):
    from evaluation.chamber_pipeline.agents import fan_in_agents
    from evaluation.chamber_pipeline.menu_taxonomy import experiment_variable

    adapter = make_three_scout_adapter(fake_llm)
    fan_in_agents(adapter, seed=0, scout_budgets=(2, 2, 2), llm=fake_llm, partition="variable")
    picks = adapter.coordination_stats["picks_by_scout"]
    assert len(picks) == 3
    vars_by_scout = [{experiment_variable(n) for n in p} for p in picks]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (vars_by_scout[i] & vars_by_scout[j])


@requires_causalchamber
def test_run_cell_records_three_scouts(fake_llm):
    record = run_cell(
        spec=get_spec("fan_in_homog3"),
        chamber="lt",
        configuration="standard",
        budget_k=6,
        seed=0,
        llm=fake_llm,
    )
    assert record.status == "ok", record.error_message
    tokens = json.loads(record.scout_tokens_json)
    assert set(tokens) == {"scout_a", "scout_b", "scout_c"}
    assert record.scout_a_tokens == tokens["scout_a"]
    assert record.scout_b_tokens == tokens["scout_b"]
    assert record.n_scouts == 3
    assert record.conservation_certified is not None
    assert record.n_experiments_distinct is not None


@requires_causalchamber
def test_run_cell_two_scout_record_gains_the_json_without_losing_the_columns(fake_llm):
    record = run_cell(
        spec=get_spec("fan_in_homog"),
        chamber="lt",
        configuration="standard",
        budget_k=6,
        seed=0,
        llm=fake_llm,
    )
    assert record.status == "ok", record.error_message
    assert set(json.loads(record.scout_tokens_json)) == {"scout_a", "scout_b"}
    assert record.n_scouts == 2
