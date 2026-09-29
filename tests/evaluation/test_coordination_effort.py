"""The coordination-call reasoning effort is a run-level knob, recorded per cell.

Every negotiate / reconcile / critique call in the corpus ran at ``high``, pinned
as a drift fix and never matched against the loop, which makes no coordination
call. The knob lets the multi-agent arms be re-run with every call at ``low``.
It rides the environment for the same reason as the selection knob: worker
processes inherit the environment, not a mutated module global.
"""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline import agents
from evaluation.chamber_pipeline.agents import team_agents
from evaluation.chamber_pipeline.run_experiment import (
    COORDINATION_EFFORT_ENV,
    SELECTION_EFFORT_ENV,
    apply_coordination_effort,
    build_arg_parser,
)
from tests.evaluation.conftest import call_kind, requires_causalchamber


class TestCoordinationEffortHelper:
    def test_default_is_high_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(COORDINATION_EFFORT_ENV, raising=False)
        assert agents.coordination_reasoning_effort() == "high"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(COORDINATION_EFFORT_ENV, "low")
        assert agents.coordination_reasoning_effort() == "low"

    def test_invalid_env_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(COORDINATION_EFFORT_ENV, "none")
        with pytest.raises(ValueError, match=r"low|medium|high"):
            agents.coordination_reasoning_effort()


@requires_causalchamber
@pytest.mark.parametrize(("setting", "expected"), [(None, "high"), ("low", "low")])
def test_team_negotiation_calls_carry_the_effort(
    monkeypatch: pytest.MonkeyPatch, make_ladder_adapter, counting_llm, setting, expected
) -> None:
    """The knob must reach every coordination call (negotiate, revise, reconcile) and no selection call."""
    monkeypatch.delenv(SELECTION_EFFORT_ENV, raising=False)
    if setting is None:
        monkeypatch.delenv(COORDINATION_EFFORT_ENV, raising=False)
    else:
        monkeypatch.setenv(COORDINATION_EFFORT_ENV, setting)
    adapter = make_ladder_adapter(counting_llm)
    team_agents(adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm)
    kinds = [(call_kind(c["messages"]), c["effort"]) for c in counting_llm.calls]
    coordination = [
        e for k, e in kinds if k in {"negotiate_propose", "negotiate_revise", "reconcile"}
    ]
    selection = [e for k, e in kinds if k == "select"]
    assert len(coordination) == 5  # propose + revise per scout, then one reconcile
    assert set(coordination) == {expected}
    assert selection and set(selection) == {"low"}


class TestCoordinationEffortFlag:
    def test_flag_default_is_none(self) -> None:
        args = build_arg_parser().parse_args(["--out", "x.parquet"])
        assert args.coordination_effort is None

    def test_flag_rejects_unknown_level(self) -> None:
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--coordination-effort", "max", "--out", "x.parquet"])

    def test_apply_sets_env_only_when_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(COORDINATION_EFFORT_ENV, raising=False)
        apply_coordination_effort(None)
        assert agents.coordination_reasoning_effort() == "high"
        apply_coordination_effort("low")
        assert agents.coordination_reasoning_effort() == "low"
        apply_coordination_effort(None)
        assert agents.coordination_reasoning_effort() == "high"
