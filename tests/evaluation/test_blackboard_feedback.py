"""The combined arm: two voices over one shared record AND one shared estimate.

The design a co-author sketched on 2026-09-14 (two agents in a ring, each
passing its output plus state to the other) is `shared_blackboard` plus
`adaptive_feedback`. Neither existing arm is changed; the blackboard gains an
optional `feedback_interval` and the registry gets `blackboard_feedback`.
"""

from __future__ import annotations

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent
from evaluation.chamber_pipeline.agents import shared_blackboard_agents
from evaluation.chamber_pipeline.llm_planner import FEEDBACK_HEADER
from evaluation.chamber_pipeline.orchestrator import get_spec

from .conftest import RecordingLLM


def _first_on_menu(_idx: int, messages: list[dict[str, str]]) -> str:
    body = messages[-1]["content"]
    menu = body.split("Menu:\n", 1)[1].split("\n\n")[0]
    return menu.splitlines()[0].split(". ", 1)[-1].strip()


def _run(k: int, interval: int | None) -> tuple[RecordingLLM, list[str]]:
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=k)
    llm = RecordingLLM(responder=_first_on_menu)
    shared_blackboard_agents(adapter, seed=0, llm=llm, feedback_interval=interval)
    return llm, list(adapter.purchased)


def test_without_an_interval_the_blackboard_is_byte_identical_to_before() -> None:
    llm, bought = _run(6, None)
    assert len(bought) == 6 and len(set(bought)) == 6
    assert all(FEEDBACK_HEADER not in str(c["messages"]) for c in llm.calls)


def test_the_estimate_is_computed_on_the_shared_record_and_shown_to_both_voices() -> None:
    llm, bought = _run(7, 3)
    assert len(bought) == 7 and len(set(bought)) == 7
    bodies = [c["messages"][-1]["content"] for c in llm.calls]
    assert len(bodies) == 7
    # No estimate before the first interval, an estimate on every prompt after it,
    # computed on the SHARED record: the count is the whole board, not one voice's.
    assert all("(no estimate yet)" in b for b in bodies[:3])
    assert "PC on the 3 experiments" in bodies[3] and "PC on the 3 experiments" in bodies[4]
    assert "PC on the 6 experiments" in bodies[6]


def test_both_voices_keep_their_briefs_and_the_complete_record() -> None:
    llm, bought = _run(6, 2)
    systems = [c["messages"][0]["content"] for c in llm.calls]
    assert len({systems[0], systems[2], systems[4]}) == 1
    assert len({systems[1], systems[3], systems[5]}) == 1
    assert systems[0] != systems[1]
    for step, call in enumerate(llm.calls):
        body = str(call["messages"])
        for prior in bought[:step]:
            assert prior in body


def test_registered_as_a_multi_agent_arm_with_the_feedback_arm_interval() -> None:
    spec = get_spec("blackboard_feedback")
    assert spec.run is shared_blackboard_agents
    assert spec.kind == "llm_multi" and spec.accepts_llm and spec.chambers == ("lt", "wt")
    assert dict(spec.static_kwargs) == {"feedback_interval": 5}
    assert spec.scout_roles is None
