"""The effect-feedback arm: the loop told what each experiment CHANGED.

Register §36: coverage-shaped feedback ("what the estimate has not reached")
nominates every unbought variable and overrides the model's prior in both
directions. This arm reports, per bought experiment, which variables moved —
mean shift or variance change against the other bought experiments — with no
estimator in the loop, so it cannot inherit the estimator's column drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent
from evaluation.chamber_pipeline.agents import effect_feedback_agent, summarize_effects
from evaluation.chamber_pipeline.orchestrator import get_spec

from .conftest import RecordingLLM as FakeLLM


def _first_on_menu(_idx: int, messages: list[dict[str, str]]) -> str:
    body = messages[-1]["content"]
    menu = body.split("Menu:\n", 1)[1].split("\n\n")[0]
    return menu.splitlines()[0].split(". ", 1)[-1].strip()


def _synthetic() -> tuple[list[pd.DataFrame], list[str], list[str], list[str]]:
    rng = np.random.default_rng(0)
    nodes = ["a", "b", "y", "z", "w"]
    n = 300

    def frame(a: float, b: float, y_shift: float, z_scale: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": a,
                "b": b,
                "y": rng.normal(y_shift, 1.0, n),
                "z": rng.normal(0.0, z_scale, n),
                "w": rng.normal(0.0, 1.0, n),
            }
        )

    # uniform_a_mid: sets a=1 and shifts y by 2 sd; uniform_b_mid: sets b=1 and
    # doubles z's noise; uniform_a_low: sets a=-1 and moves nothing else.
    dfs = [frame(1.0, 0.0, 2.0, 1.0), frame(0.0, 1.0, 0.0, 3.0), frame(-1.0, 0.0, 0.0, 1.0)]
    names = ["uniform_a_mid", "uniform_b_mid", "uniform_a_low"]
    menu = [*names, "uniform_w_mid", "uniform_z_mid"]
    return dfs, names, nodes, menu


def test_summary_reports_shift_and_noise_per_experiment_and_excludes_own_target() -> None:
    dfs, names, nodes, menu = _synthetic()
    text = summarize_effects(dfs, names, nodes, menu, names)
    lines = {ln.split(":")[0].strip("- "): ln for ln in text.splitlines() if ln.startswith("- ")}
    assert set(lines) == set(names)
    assert "y" in lines["uniform_a_mid"].split("shifted")[1].split(";")[0]
    assert "a" not in lines["uniform_a_mid"].split("shifted")[1].split(";")[0].split(", ")
    assert "z" in lines["uniform_b_mid"].split("noise")[1]
    assert "changed nothing" in lines["uniform_a_low"]
    # the aggregate line names the inert experiment and only it
    inert = text.split("changed nothing measurable:")[1].splitlines()[0]
    assert "uniform_a_low" in inert and "uniform_a_mid" not in inert


def test_summary_is_deterministic_and_needs_two_experiments() -> None:
    dfs, names, nodes, menu = _synthetic()
    assert summarize_effects(dfs, names, nodes, menu, names) == summarize_effects(
        dfs, names, nodes, menu, names
    )
    assert summarize_effects(dfs[:1], names[:1], nodes, menu, names[:1]) is None


def test_summary_mentions_only_bought_entries_and_node_names() -> None:
    adapter = create_contracted_chamber_agent(chamber="wt", intervention_budget=5)
    nodes = list(adapter.ground_truth().index)
    menu = list(adapter.available_experiments())
    names = menu[:5]
    dfs = [adapter.query_intervention(n) for n in names]
    text = summarize_effects(dfs, names, nodes, menu, names)
    assert text is not None
    tokens = set(text.replace(":", " ").replace(",", " ").replace(";", " ").split())
    for entry in menu[5:]:
        assert entry not in tokens  # unbought entries are never named: nothing is known about them
    for entry in names:
        assert entry in tokens


@pytest.mark.parametrize("chamber", ["lt", "wt"])
def test_agent_spends_the_budget_and_feeds_back_on_schedule(chamber: str) -> None:
    adapter = create_contracted_chamber_agent(chamber=chamber, intervention_budget=7)
    llm = FakeLLM(responder=_first_on_menu)
    adj = effect_feedback_agent(adapter, llm=llm, feedback_interval=3)
    spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
    assert len(spent) == 7 and len(set(spent)) == 7
    assert adj.shape == adapter.ground_truth().shape
    bodies = [c["messages"][-1]["content"] for c in llm.calls]
    assert len(bodies) == 7
    assert all("(no estimate yet)" in b for b in bodies[:3])
    assert all("shifted" in b for b in bodies[3:])
    assert "PC on the" not in bodies[3]  # no estimator in the loop


def test_registered_as_a_single_llm_arm_on_both_chambers() -> None:
    spec = get_spec("effect_feedback")
    assert spec.kind == "llm_single" and spec.accepts_llm and spec.chambers == ("lt", "wt")
    assert spec.scout_roles is None and spec.negotiation_rounds == 0
