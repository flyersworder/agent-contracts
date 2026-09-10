"""The adaptive-feedback arm: the loop with the data's verdict in the prompt."""

from __future__ import annotations

import pytest

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent
from evaluation.chamber_pipeline.agents import adaptive_feedback_agent, summarize_estimate
from evaluation.chamber_pipeline.llm_planner import FEEDBACK_HEADER, build_feedback_select_prompt
from evaluation.chamber_pipeline.orchestrator import get_spec

from .conftest import RecordingLLM as FakeLLM


def _first_on_menu(_idx: int, messages: list[dict[str, str]]) -> str:
    body = messages[-1]["content"]
    menu = body.split("Menu:\n", 1)[1].split("\n\n")[0]
    return menu.splitlines()[0].split(". ", 1)[-1].strip()


def test_prompt_without_feedback_only_adds_the_header() -> None:
    plain = build_feedback_select_prompt(["uniform_a", "uniform_b"], 2, None, None)
    assert plain[1]["content"].startswith(f"{FEEDBACK_HEADER}: (no estimate yet)")
    with_fb = build_feedback_select_prompt(["uniform_a", "uniform_b"], 2, None, "3 edges")
    assert "3 edges" in with_fb[1]["content"]


def test_summary_is_keyed_by_menu_entries_and_partitions_them() -> None:
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    est = adapter.ground_truth().copy() * 0
    est.loc["red", "ir_1"] = 1  # one edge: red and ir_1 are "connected"
    text = summarize_estimate(est, menu, [menu[0]], nodes)
    assert "finds 1 directed edge" in text
    unreached = text.split("NO edge has reached yet: ")[1].split("\n")[0]
    reached = text.split("already connected: ")[1]
    assert "uniform_red_mid" in reached and "uniform_red_mid" not in unreached
    assert menu[0] not in unreached and menu[0] not in reached  # spent entries excluded
    for token in unreached.replace(", ... (", ",").split(", "):
        assert token.startswith("uniform_") or token.endswith("more)") or token == "(none)"


@pytest.mark.parametrize("chamber", ["lt", "wt"])
def test_agent_spends_exactly_the_budget_and_feeds_back_on_schedule(chamber: str) -> None:
    adapter = create_contracted_chamber_agent(chamber=chamber, intervention_budget=7)
    llm = FakeLLM(responder=_first_on_menu)
    adj = adaptive_feedback_agent(adapter, llm=llm, feedback_interval=3)
    spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
    assert len(spent) == 7 and len(set(spent)) == 7
    assert adj.shape == adapter.ground_truth().shape
    bodies = [c["messages"][-1]["content"] for c in llm.calls]
    assert len(bodies) == 7
    # No estimate before the first interval; an estimate on every prompt after it.
    assert all("(no estimate yet)" in b for b in bodies[:3])
    assert all("PC on the" in b for b in bodies[3:])
    # The estimate is recomputed at 3 and 6, and the prompt at 6 reports 6 bought.
    assert "PC on the 3 experiments" in bodies[3] and "PC on the 6 experiments" in bodies[6]


def test_registered_as_a_single_llm_arm_on_both_chambers() -> None:
    spec = get_spec("adaptive_feedback")
    assert spec.kind == "llm_single" and spec.accepts_llm and spec.chambers == ("lt", "wt")
    assert spec.scout_roles is None and spec.negotiation_rounds == 0
