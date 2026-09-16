"""The selection-call reasoning effort is a run-level knob, recorded per cell.

Every selection call ran at ``low`` for the whole corpus (agents.py pins it so
a provider default cannot drift under us). The three-agent ablation found GLM
near-deterministic at ``low``; testing whether more reasoning diversifies the
scouts needs the knob at the CLI without touching every agent signature. An
environment variable is the carrier because ``--max-workers`` forks worker
processes, which inherit the environment but not a mutated module global.
"""

from __future__ import annotations

import pytest

from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline import agents
from evaluation.chamber_pipeline.run_experiment import (
    SELECTION_EFFORT_ENV,
    apply_selection_effort,
    build_arg_parser,
)

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE, reason="causalchamber not installed"
)


class TestSelectionEffortHelper:
    def test_default_is_low_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SELECTION_EFFORT_ENV, raising=False)
        assert agents.selection_reasoning_effort() == "low"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SELECTION_EFFORT_ENV, "high")
        assert agents.selection_reasoning_effort() == "high"

    def test_invalid_env_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SELECTION_EFFORT_ENV, "maximal")
        with pytest.raises(ValueError, match=r"low|medium|high"):
            agents.selection_reasoning_effort()

    @requires_causalchamber
    @pytest.mark.parametrize("agent", [agents.llm_pc_agent, agents.one_shot_agent])
    def test_selection_call_carries_the_effort(
        self, monkeypatch: pytest.MonkeyPatch, agent: object
    ) -> None:
        """The knob must reach the wire, not just the helper -- on the loop and the single call."""
        from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

        monkeypatch.setenv(SELECTION_EFFORT_ENV, "high")
        seen: list[dict[str, object]] = []

        def fake_llm(**kwargs: object) -> dict[str, object]:
            seen.append(kwargs)
            return {"choices": [{"message": {"content": "nothing on the menu"}}]}

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        agent(adapter, llm=fake_llm)  # type: ignore[operator]
        assert seen, "no LLM call was made"
        assert seen[0]["extra_body"]["reasoning"]["effort"] == "high"  # type: ignore[index]


class TestSelectionEffortFlag:
    def test_flag_default_is_none(self) -> None:
        args = build_arg_parser().parse_args(["--out", "x.parquet"])
        assert args.selection_effort is None

    def test_flag_rejects_unknown_level(self) -> None:
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--selection-effort", "max", "--out", "x.parquet"])

    def test_apply_sets_env_only_when_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SELECTION_EFFORT_ENV, raising=False)
        apply_selection_effort(None)
        assert agents.selection_reasoning_effort() == "low"
        apply_selection_effort("high")
        assert agents.selection_reasoning_effort() == "high"
