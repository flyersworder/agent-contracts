"""The adaptive-feedback arm: the loop with the data's verdict in the prompt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent
from evaluation.chamber_pipeline.agents import (
    adaptive_feedback_agent,
    sensor_configured_by,
    summarize_estimate,
)
from evaluation.chamber_pipeline.inference import run_pc
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


# --- register §36: the feedback must know what the estimator threw away ---


def _frame_with_constant_and_duplicate() -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    b = rng.normal(size=400)
    nodes = ["a", "b", "b_dup", "const"]
    frame = pd.DataFrame({"a": a, "b": b, "b_dup": b * 2.0 + 1e-9, "const": 1.0})
    return frame, nodes


def test_run_pc_reports_the_columns_it_dropped() -> None:
    frame, nodes = _frame_with_constant_and_duplicate()
    dropped: dict[str, list[str]] = {}
    run_pc(frame, nodes, dropped_out=dropped)
    assert dropped == {"zero_variance": ["const"], "collinear": ["b_dup"]}


def test_run_pc_reports_empty_lists_when_nothing_is_dropped() -> None:
    frame, _nodes = _frame_with_constant_and_duplicate()
    clean = frame[["a", "b"]]
    dropped: dict[str, list[str]] = {}
    run_pc(clean, ["a", "b"], dropped_out=dropped)
    assert dropped == {"zero_variance": [], "collinear": []}


def test_summary_relays_removed_variables_and_stops_nominating_them() -> None:
    adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=1)
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    est = adapter.ground_truth().copy() * 0
    text = summarize_estimate(est, menu, [], nodes, collinear_dropped=["pressure_ambient", "pot_2"])
    removed = text.split("REMOVED from the estimate")[1].split("\n")[0]
    assert "pressure_ambient" in removed and "pot_2" in removed
    unreached = text.split("NO edge has reached yet: ")[1].split("\n")[0]
    # the entry that perturbs a removed variable is no longer "unexplored"
    assert "validate_pot_2" not in unreached
    assert "validate_pot_2" in removed
    # the line says what it means for the buyer
    assert "cannot be connected" in text


def test_summary_is_byte_identical_when_nothing_was_dropped() -> None:
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    est = adapter.ground_truth().copy() * 0
    est.loc["red", "ir_1"] = 1
    before = summarize_estimate(est, menu, [menu[0]], nodes)
    after = summarize_estimate(est, menu, [menu[0]], nodes, collinear_dropped=[])
    assert before == after
    assert "REMOVED" not in after


def test_agent_feeds_the_dropped_columns_into_the_prompt_on_wt() -> None:
    # WT's four barometers are collinear in `standard`; PC drops three of them
    # on every estimate, so the feedback line must appear once an estimate exists.
    adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=6)
    llm = FakeLLM(responder=_first_on_menu)
    adaptive_feedback_agent(adapter, llm=llm, feedback_interval=3)
    bodies = [c["messages"][-1]["content"] for c in llm.calls]
    assert all("REMOVED from the estimate" not in b for b in bodies[:3])
    assert all("REMOVED from the estimate" in b for b in bodies[3:])
    assert "pressure_" in bodies[3].split("REMOVED from the estimate")[1].split("\n")[0]


# --- register §36, second fix: a setting that configures a removed sensor is excluded ---


@pytest.mark.parametrize("chamber", ["lt", "wt"])
def test_sensor_configured_by_matches_the_manual_and_the_truth(chamber: str) -> None:
    # The rule is name-based (the chamber manual: `osr_X` is the oversampling
    # rate and `v_X` the reference voltage of sensor X). The test, not the
    # agent, checks it against the ground truth: every such setting has
    # exactly one child and the rule names it.
    adapter = create_contracted_chamber_agent(chamber=chamber, intervention_budget=1)
    truth = adapter.ground_truth()
    nodes = list(truth.index)
    settings = [n for n in nodes if n.startswith(("osr_", "v_"))]
    assert settings
    for s in settings:
        children = list(truth.columns[truth.loc[s] > 0])
        assert len(children) == 1, (s, children)
        assert sensor_configured_by(s, nodes) == children[0], s
    # non-settings map to nothing
    assert sensor_configured_by(nodes[-1], nodes) is None
    assert sensor_configured_by("hatch" if chamber == "wt" else "red", nodes) is None


def test_summary_excludes_settings_that_configure_a_removed_sensor() -> None:
    adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=1)
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    est = adapter.ground_truth().copy() * 0
    text = summarize_estimate(
        est, menu, [], nodes, collinear_dropped=["pressure_ambient", "pressure_intake"]
    )
    unreached = text.split("NO edge has reached yet: ")[1].split("\n")[0]
    removed = text.split("REMOVED from the estimate")[1].split("\n")[0]
    for entry in ("validate_osr_ambient", "validate_osr_intake"):
        assert entry not in unreached, entry
        assert entry in removed, entry
    # a setting of a KEPT sensor is still a legitimate unexplored buy
    assert "validate_osr_upwind" in unreached
