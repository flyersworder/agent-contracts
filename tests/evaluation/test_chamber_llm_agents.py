"""Tests for the M3b LLM-bearing chamber agents.

Covers `evaluation.chamber_pipeline.agents.llm_only_agent` (variant 3)
and `llm_pc_agent` (variant 4). Uses an in-process mock LLM via the
agents' `llm` injection point — no network, no real LiteLLM dependency
beyond import resolution.

Per plan §11 R1 mitigation order, M3b lands after M3a's pure pipeline
and before M3c's multi-agent variant. The mocked-LLM tests here pin
the agent's interaction with the LLM seam (call count, message shape,
fallback behavior on bad output) so that swapping in real DeepSeek v4
Flash for the M4 sweep is a single one-line change.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline.agents import llm_only_agent, llm_pc_agent

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# FakeLLM — synthetic completion callable that records every call
# ---------------------------------------------------------------------------


class FakeLLM:
    """Synthetic LiteLLM-shaped completion callable for tests.

    Drives scripted responses; records every call. Mirrors the
    `litellm.completion(model=..., messages=...)` surface so the agents
    can use it as a drop-in for the real client.

    Two response strategies:
        - `responses=[str, str, ...]` cycles through pre-baked content
          strings. When exhausted, raises AssertionError (catches the
          common bug "agent kept calling LLM beyond expected count").
        - `responder=lambda call_idx, messages: str` lets tests build
          dynamic responses per-call (e.g., always return the first menu
          item from the user message).

    Recorded `calls` is a list of dicts with `model`, `messages`, `idx`.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Any = None,
    ) -> None:
        if (responses is None) == (responder is None):
            raise ValueError("Pass exactly one of `responses` or `responder`")
        self._responses = responses
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> dict:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx})

        if self._responses is not None:
            if idx >= len(self._responses):
                raise AssertionError(
                    f"FakeLLM exhausted: agent made {idx + 1} calls but only "
                    f"{len(self._responses)} responses were scripted. "
                    f"Most recent user message: {messages[-1]['content'][:200]}"
                )
            content = self._responses[idx]
        else:
            content = self._responder(idx, messages)

        return {"choices": [{"message": {"content": content}}]}


def _indexed_menu_responder(idx: int, messages: list[dict[str, str]]) -> str:
    """Responder that picks menu entry at position `idx` from the user prompt.

    Mimics a sane LLM that doesn't repeat itself: call N gets the Nth
    distinct menu entry. Important for test stability — picking the same
    experiment twice causes pooled data to be perfectly redundant, which
    in turn makes PC's Fisher-Z test hit singular sub-correlation
    matrices on highly-collinear LT chamber data.

    Tests that explicitly want to exercise the dedup / fallback path
    use a different responder (or `responses=[same, same, ...]`).
    """
    user_text = messages[-1]["content"]
    lines = [line.strip() for line in user_text.splitlines() if line.strip()]
    menu_entries = [line for line in lines if line.startswith(("uniform_", "exp_"))]
    if not menu_entries:
        raise RuntimeError(f"Test fixture found no menu entries in: {user_text[:300]}")
    return menu_entries[idx % len(menu_entries)]


# ---------------------------------------------------------------------------
# llm_only_agent
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestLlmOnlyAgent:
    """The LLM picks each intervention AND emits the final adjacency."""

    def test_returns_aligned_dataframe(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        # Script: 2 selection responses + 1 adjacency-emission response.
        menu = adapter.available_experiments()
        llm = FakeLLM(responses=[menu[0], menu[1], json.dumps({menu[0]: []})])
        adj = llm_only_agent(adapter, llm=llm)
        assert adj.shape == adapter.ground_truth().shape
        assert list(adj.index) == list(adapter.ground_truth().index)

    def test_makes_budget_plus_one_llm_calls(self) -> None:
        """k selection calls + 1 adjacency-emission call."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        menu = adapter.available_experiments()
        llm = FakeLLM(
            responses=[menu[0], menu[1], menu[2], "{}"]  # 3 picks + 1 graph emission
        )
        llm_only_agent(adapter, llm=llm)
        assert len(llm.calls) == 4

    def test_spends_full_intervention_budget(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        menu = adapter.available_experiments()
        llm = FakeLLM(responses=[menu[0], menu[1], "{}"])
        llm_only_agent(adapter, llm=llm)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 2

    def test_zero_budget_skips_llm_entirely(self) -> None:
        """Budget 0 → empty adjacency, no LLM calls (no API spend on degenerate)."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=0)
        llm = FakeLLM(responses=[])
        adj = llm_only_agent(adapter, llm=llm)
        assert (adj.values == 0).all()
        assert adj.shape == adapter.ground_truth().shape
        assert len(llm.calls) == 0

    def test_falls_back_on_off_menu_response(self) -> None:
        """If the LLM returns junk, the agent picks a random unspent
        experiment so the budget axis stays clean."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        # Two garbage selection responses + a final adjacency response.
        # The agent should still spend both intervention slots via fallback.
        llm = FakeLLM(responses=["???", "completely off menu", "{}"])
        llm_only_agent(adapter, seed=42, llm=llm)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 2

    def test_sends_model_to_llm_callable(self) -> None:
        """Verify the `model` kwarg propagates to the LLM call."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        menu = adapter.available_experiments()
        llm = FakeLLM(responses=[menu[0], "{}"])
        llm_only_agent(adapter, model="custom/test-model", llm=llm)
        assert llm.calls[0]["model"] == "custom/test-model"
        assert llm.calls[1]["model"] == "custom/test-model"

    def test_avoids_repeating_picks(self) -> None:
        """When the LLM keeps returning the same name, the agent's
        already-chosen guard + fallback should pick distinct names."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        menu = adapter.available_experiments()
        # LLM always returns menu[0] — agent should fall back for the
        # second and third picks since menu[0] has already been chosen.
        llm = FakeLLM(responses=[menu[0], menu[0], menu[0], "{}"])
        llm_only_agent(adapter, seed=0, llm=llm)
        spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
        assert len(spent) == 3
        assert len(set(spent)) == 3, f"Expected 3 distinct picks, got {spent}"

    def test_parses_emitted_adjacency(self) -> None:
        """Final-step JSON adjacency is parsed into the returned DataFrame."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        nodes = list(adapter.ground_truth().index)
        # Emit a single edge between the first two ground-truth node names.
        edge_json = json.dumps({nodes[0]: [nodes[1]]})
        menu = adapter.available_experiments()
        llm = FakeLLM(responses=[menu[0], edge_json])
        adj = llm_only_agent(adapter, llm=llm)
        assert adj.loc[nodes[0], nodes[1]] == 1


# ---------------------------------------------------------------------------
# llm_pc_agent
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestLlmPcAgent:
    """LLM picks each intervention; classical PC infers the graph."""

    def test_returns_aligned_dataframe(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        llm = FakeLLM(responder=_indexed_menu_responder)
        adj = llm_pc_agent(adapter, llm=llm)
        assert adj.shape == adapter.ground_truth().shape
        assert list(adj.index) == list(adapter.ground_truth().index)

    def test_makes_exactly_budget_llm_calls(self) -> None:
        """k selection calls — NO final adjacency-emission call (PC handles it)."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        llm = FakeLLM(responder=_indexed_menu_responder)
        llm_pc_agent(adapter, llm=llm)
        assert len(llm.calls) == 4

    def test_spends_full_intervention_budget(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
        llm = FakeLLM(responder=_indexed_menu_responder)
        llm_pc_agent(adapter, llm=llm)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 3

    def test_zero_budget_skips_llm_entirely(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=0)
        llm = FakeLLM(responses=[])
        adj = llm_pc_agent(adapter, llm=llm)
        assert (adj.values == 0).all()
        assert len(llm.calls) == 0

    def test_falls_back_on_off_menu_response(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        # Two garbage responses; agent's fallback should still spend both
        # intervention slots (PC consumes the resulting data).
        llm = FakeLLM(responses=["???", "off menu garbage"])
        llm_pc_agent(adapter, seed=42, llm=llm)
        assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 2


# ---------------------------------------------------------------------------
# Cross-variant invariants
# ---------------------------------------------------------------------------


@requires_causalchamber
class TestSharedSelectionLoopBehavior:
    """Properties both LLM agents must satisfy by construction."""

    def test_neither_agent_overshoots_budget(self) -> None:
        """At budget=2, neither variant may spend a 3rd intervention."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        for runner in (llm_only_agent, llm_pc_agent):
            adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
            menu = adapter.available_experiments()
            # Always answer with the first menu name; the agent should
            # use its already-chosen guard + RNG fallback to spend twice
            # without going over the limit.
            if runner is llm_only_agent:
                llm = FakeLLM(responses=[menu[0], menu[0], "{}"])
            else:
                llm = FakeLLM(responses=[menu[0], menu[0]])
            runner(adapter, llm=llm)
            assert adapter._resource_monitor.usage.get_tool_usage("intervene") == 2

    def test_first_user_message_lists_menu(self) -> None:
        """Plan §5 / module docstring: menu-only at planning time. Verify
        the first selection prompt actually contains menu entries."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        menu = adapter.available_experiments()
        llm = FakeLLM(responses=[menu[0], "{}"])
        llm_only_agent(adapter, llm=llm)

        first_user_msg = llm.calls[0]["messages"][-1]["content"]
        # First three menu entries should appear verbatim in the prompt.
        for name in menu[:3]:
            assert name in first_user_msg

    def test_llm_pc_does_not_send_node_names_to_llm(self) -> None:
        """Plan §5 menu-only stance: the LLM in llm_pc_agent never sees
        ground-truth node names — only experiment menu entries. (Names
        like `uniform_red_mid` reveal `red` indirectly through naming;
        that's the whole point of "menu only" being honest.) What matters
        here is that we don't pass the node-list explicitly."""
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=1)
        nodes = list(adapter.ground_truth().index)
        llm = FakeLLM(responder=_indexed_menu_responder)
        llm_pc_agent(adapter, llm=llm)

        # No prompt to llm_pc_agent should contain a structured
        # `Variables: ...` block — that's an llm_only adjacency-emission
        # construct. We grep for that block header verbatim.
        for call in llm.calls:
            for msg in call["messages"]:
                assert "Variables (use these exact names)" not in msg["content"], (
                    "llm_pc_agent leaked node-name list into LLM prompt"
                )
        # Sanity: ground-truth nodes were available but not reached.
        assert len(nodes) > 0


# ---------------------------------------------------------------------------
# Smoke for FakeLLM itself — catch regressions in the test fixture
# ---------------------------------------------------------------------------


class TestFakeLLM:
    """Sanity that the test harness itself behaves as documented."""

    def test_records_calls(self) -> None:
        llm = FakeLLM(responses=["a", "b"])
        llm(model="m", messages=[{"role": "user", "content": "hello"}])
        llm(model="m", messages=[{"role": "user", "content": "world"}])
        assert len(llm.calls) == 2
        assert llm.calls[0]["messages"][0]["content"] == "hello"

    def test_exhaustion_raises(self) -> None:
        llm = FakeLLM(responses=["only"])
        llm(model="m", messages=[{"role": "user", "content": "first"}])
        with pytest.raises(AssertionError, match="exhausted"):
            llm(model="m", messages=[{"role": "user", "content": "second"}])

    def test_responder_callback(self) -> None:
        llm = FakeLLM(responder=lambda idx, msgs: f"call-{idx}")
        r = llm(model="m", messages=[{"role": "user", "content": "x"}])
        assert r["choices"][0]["message"]["content"] == "call-0"

    def test_init_validates_exactly_one_strategy(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            FakeLLM(responses=None, responder=None)
        with pytest.raises(ValueError, match="exactly one"):
            FakeLLM(responses=["x"], responder=lambda i, m: "y")


# Public re-export check — defensive against future module-shape regressions.
def test_top_level_reexports() -> None:
    from evaluation.chamber_pipeline import llm_only_agent as exported_only
    from evaluation.chamber_pipeline import llm_pc_agent as exported_pc

    assert exported_only is llm_only_agent
    assert exported_pc is llm_pc_agent


# ---------------------------------------------------------------------------
# Attribute-style response coverage (added post M3 review)
#
# LiteLLM may return Pydantic-like response objects in production rather
# than plain dicts. The parser layer (llm_planner._response_text) handles
# both shapes, but the AGENT layer was only tested against dicts. This
# class adds end-to-end coverage on attribute-style responses to catch
# any future regression in the parsing-vs-agent integration.
# ---------------------------------------------------------------------------


class _AttrMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _AttrChoice:
    def __init__(self, content: str) -> None:
        self.message = _AttrMessage(content)


class _AttrResponse:
    """Pydantic-like response shape returned by some LiteLLM versions."""

    def __init__(self, content: str) -> None:
        self.choices = [_AttrChoice(content)]


class FakeAttrLLM:
    """FakeLLM variant that wraps content in attribute-style objects.

    Mirrors `FakeLLM`'s interface but returns `_AttrResponse` instead of
    a dict. Used to verify the agents' parsing layer handles both shapes.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> _AttrResponse:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx})
        if idx >= len(self._responses):
            raise AssertionError(
                f"FakeAttrLLM exhausted: {idx + 1} calls, {len(self._responses)} responses"
            )
        return _AttrResponse(self._responses[idx])


@requires_causalchamber
class TestAgentsHandleAttrStyleResponses:
    """End-to-end agent runs with Pydantic-like LLM responses."""

    def test_llm_only_with_attr_responses(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        menu = adapter.available_experiments()
        # 2 selections + 1 adjacency emission, all attr-style.
        llm = FakeAttrLLM(responses=[menu[0], menu[1], json.dumps({menu[0]: []})])
        adj = llm_only_agent(adapter, llm=llm)
        # If parsing fails on attr shape, the agent silently falls back
        # to RNG selections — but the LLM call count would still be 3,
        # so we can't catch that via call count alone. The smoking gun
        # is whether the actual chosen experiments match the script.
        spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
        assert spent == [menu[0], menu[1]], (
            "Attr-style responses didn't propagate through to selection — "
            "parsing layer silently fell back to RNG."
        )
        # And the adjacency-emission stage must have produced a well-typed result.
        assert adj.shape == adapter.ground_truth().shape

    def test_llm_pc_with_attr_responses(self) -> None:
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
        menu = adapter.available_experiments()
        # No adjacency-emission step for llm_pc — just 2 attr selections.
        llm = FakeAttrLLM(responses=[menu[0], menu[1]])
        adj = llm_pc_agent(adapter, llm=llm)
        spent = [e["data"]["experiment_name"] for e in adapter.events if e["type"] == "tool_use"]
        assert spent == [menu[0], menu[1]]
        assert adj.shape == adapter.ground_truth().shape
