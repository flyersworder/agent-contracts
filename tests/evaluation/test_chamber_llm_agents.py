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
from typing import Any, ClassVar

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

    Recorded `calls` is a list of dicts with `model`, `messages`, `idx`, and
    `kwargs` (everything else the caller passed, e.g. `max_tokens`,
    `temperature`). `kwargs` is what lets a test assert that a parameter was
    *omitted* rather than passed as None -- the distinction reuse safety turns
    on, since rungs 0 and 3 must call the provider byte-identically to M4b.
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

    def __call__(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        idx = len(self.calls)
        self.calls.append({"model": model, "messages": messages, "idx": idx, "kwargs": kwargs})

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


# --------------------------------------------------------------------------
# M6 Task 2: blind role prompts
# --------------------------------------------------------------------------


def test_blind_scout_prompts_never_reference_already_chosen():
    from evaluation.chamber_pipeline.llm_planner import (
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )

    menu = ["uniform_t_ir_1_mid", "uniform_l_12_mid", "uniform_diode_ir_3_mid"]
    for build in (build_scout_broad_prompt, build_scout_targeted_prompt):
        msgs = build(menu, 3, None)
        text = " ".join(m["content"] for m in msgs).lower()
        assert "already_chosen" not in text
        assert "planner" not in text
        assert "reasoner" not in text
        assert "other agent" not in text
        assert "uniform_t_ir_1_mid" in text


def test_scout_roles_differ():
    from evaluation.chamber_pipeline.llm_planner import (
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )

    a = build_scout_broad_prompt(["x"], 1, None)[0]["content"]
    b = build_scout_targeted_prompt(["x"], 1, None)[0]["content"]
    assert a != b


def test_scout_prompts_match_the_prompt_builder_arity():
    """`_llm_select_loop` calls prompt_builder(menu, remaining, all_chosen).

    A two-parameter builder raises TypeError on every differentiate=True run.
    """
    import inspect

    from evaluation.chamber_pipeline.llm_planner import (
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
        build_select_prompt,
    )

    reference = list(inspect.signature(build_select_prompt).parameters)
    for build in (build_scout_broad_prompt, build_scout_targeted_prompt):
        assert list(inspect.signature(build).parameters) == reference


def test_scout_prompts_still_carry_own_prior_picks():
    """`already_chosen` is the scout's OWN loop history, not another agent's."""
    from evaluation.chamber_pipeline.llm_planner import build_scout_broad_prompt

    msgs = build_scout_broad_prompt(["a", "b"], 2, ["a"])
    user = msgs[1]["content"]
    assert "Already spent" in user
    assert "a" in user


def test_reconcile_prompt_carries_both_scouts_selections():
    from evaluation.chamber_pipeline.llm_planner import build_reconcile_prompt

    msgs = build_reconcile_prompt(["exp_a1", "exp_a2"], ["exp_b1", "exp_a2"])
    text = " ".join(m["content"] for m in msgs)
    for name in ("exp_a1", "exp_a2", "exp_b1"):
        assert name in text


def test_every_output_cap_clears_the_measured_reasoning_load():
    """Every cap must exceed what the model actually spends thinking.

    Measured 2026-08-23 on DeepSeek v4 Flash 0423: a *selection* call -- the
    cheapest prompt in the pipeline -- spends 976 reasoning tokens at `high`
    effort and 475 at `low`. Reconciliation and negotiation are strictly
    harder prompts, so their caps must clear that load with margin, and no
    cap may sit near it. An absolute floor, not a multiple of the selection
    cap: the earlier relative form was calibrated against a 200-token
    selection cap that was itself the bug.
    """
    from evaluation.chamber_pipeline.agents import (
        _NEGOTIATE_MAX_TOKENS,
        _RECONCILE_MAX_TOKENS,
        _SELECTION_MAX_TOKENS,
    )

    # Superseded 2026-08-24: 976 was measured on an EMPTY-history selection
    # call, the cheapest step of the loop. A LATE-loop call (25 already chosen)
    # measures 2,175 on flash-0731 and 11,690 on flash -- so the floor must be
    # anchored on the late-loop figure, not the first-call one.
    worst_observed_reasoning = 11690
    for cap in (_SELECTION_MAX_TOKENS, _RECONCILE_MAX_TOKENS, _NEGOTIATE_MAX_TOKENS):
        assert cap >= 2 * worst_observed_reasoning
    # Reconciliation reasons over both scouts' lists. The old rationale here
    # ("never cheaper than one pick") is FALSE at late loop -- one pick reached
    # 11,690 against reconcile's measured 8,557 at k=30 -- but the ordering is
    # kept as a safety property: the aggregator must never be the tightest cap,
    # because a truncated reconcile silently pins its billed spend to the cap
    # and that spend feeds the P2 and H-C measurements.
    assert _RECONCILE_MAX_TOKENS >= _SELECTION_MAX_TOKENS


@requires_causalchamber
def test_select_loop_omits_temperature_by_default():
    """Reuse safety: rungs 0 and 3 must call the provider exactly as in M4b.

    Passing `temperature=None` is not the same as omitting the argument --
    providers may treat an explicit null differently from an absent key -- so
    the default path must not mention temperature at all.
    """
    from agent_contracts.integrations.causalchamber import (
        create_contracted_chamber_agent,
    )
    from evaluation.chamber_pipeline.agents import _llm_select_loop

    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
    menu = adapter.available_experiments()
    llm = FakeLLM(responder=lambda idx, msgs: menu[idx])
    _llm_select_loop(adapter, llm, "m", seed=0)
    assert llm.calls, "loop made no LLM calls"
    for call in llm.calls:
        assert "temperature" not in call["kwargs"]


@requires_causalchamber
def test_select_loop_forwards_an_explicit_temperature():
    """Rung 1's entire diversity mechanism is this number.

    Two homogeneous scouts receive byte-identical messages on the happy path
    -- the seed only feeds the off-menu fallback RNG -- so without an explicit
    temperature a low provider default drives `overlap_frac` to 1.0 and
    degenerates rung 1 into rung 0 at double the budget.
    """
    from agent_contracts.integrations.causalchamber import (
        create_contracted_chamber_agent,
    )
    from evaluation.chamber_pipeline.agents import _SCOUT_TEMPERATURE, _llm_select_loop

    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
    menu = adapter.available_experiments()
    llm = FakeLLM(responder=lambda idx, msgs: menu[idx])
    _llm_select_loop(adapter, llm, "m", seed=0, temperature=_SCOUT_TEMPERATURE)
    assert llm.calls
    for call in llm.calls:
        assert call["kwargs"]["temperature"] == _SCOUT_TEMPERATURE
    assert _SCOUT_TEMPERATURE > 0.0, "a zero temperature cannot decorrelate scouts"


# --------------------------------------------------------------------------
# M6: selection-call reasoning budget (2026-08-23 provider regression)
# --------------------------------------------------------------------------


def test_selection_cap_exceeds_observed_reasoning_load():
    """The 200-token cap could not hold a single selection call.

    Measured 2026-08-23 on the real 59-item LT menu: default effort spends
    821 reasoning tokens, `high` 976, `low` 475, `minimal` 415 -- every level
    over 200. All four pinned providers returned `finish_reason=length` with
    empty content, and `_llm_select_loop` silently fell back to `rng.choice`.
    """
    from evaluation.chamber_pipeline.agents import (
        _SELECTION_MAX_TOKENS,
        _SELECTION_REASONING_EFFORT,
    )

    assert _SELECTION_MAX_TOKENS >= 1500
    assert _SELECTION_REASONING_EFFORT in {"none", "minimal", "low", "medium", "high"}


@requires_causalchamber
def test_selection_calls_pin_the_reasoning_effort():
    """Effort must be explicit, not inherited from a provider default.

    M4b never set it and silently tracked DeepSeek's default; that default
    rose (M4b's observed ~509 output tokens/call vs 821 today) when three
    effort tiers shipped on 2026-08-13. An unset parameter is an unpinned one.
    """
    from agent_contracts.integrations.causalchamber import (
        create_contracted_chamber_agent,
    )
    from evaluation.chamber_pipeline.agents import (
        _SELECTION_REASONING_EFFORT,
        _llm_select_loop,
    )

    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
    menu = adapter.available_experiments()
    llm = FakeLLM(responder=lambda idx, msgs: menu[idx])
    _llm_select_loop(adapter, llm, "m", seed=0)
    assert llm.calls
    for call in llm.calls:
        reasoning = call["kwargs"]["extra_body"]["reasoning"]
        assert reasoning == {"effort": _SELECTION_REASONING_EFFORT}


@requires_causalchamber
def test_selection_fallback_is_counted_not_silent():
    """A degraded selection must leave a trace.

    The fallback exists so a bad response degrades to random rather than
    crashing. That graceful degradation is exactly what hid a 100% failure
    rate, so it now increments a counter the sweep records per cell.
    """
    from agent_contracts.integrations.causalchamber import (
        create_contracted_chamber_agent,
    )
    from evaluation.chamber_pipeline.agents import _llm_select_loop

    class CountingStub(FakeLLM):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.selection_fallbacks = 0

        def record_selection_fallback(self) -> None:
            self.selection_fallbacks += 1

    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
    # Empty content is exactly what a truncated reasoning call returns.
    llm = CountingStub(responder=lambda idx, msgs: "")
    chosen, _ = _llm_select_loop(adapter, llm, "m", seed=0)
    assert len(chosen) == 3  # still spends the budget
    assert llm.selection_fallbacks == 3  # and says so


@requires_causalchamber
def test_no_fallback_recorded_when_selection_parses():
    from agent_contracts.integrations.causalchamber import (
        create_contracted_chamber_agent,
    )
    from evaluation.chamber_pipeline.agents import _llm_select_loop

    class CountingStub(FakeLLM):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.selection_fallbacks = 0

        def record_selection_fallback(self) -> None:
            self.selection_fallbacks += 1

    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=3)
    menu = adapter.available_experiments()
    llm = CountingStub(responder=lambda idx, msgs: menu[idx])
    _llm_select_loop(adapter, llm, "m", seed=0)
    assert llm.selection_fallbacks == 0


class TestAlreadyChosenAreNotSelectable:
    """Duplicates were structurally invited and then punished.

    The rendered menu listed all 59 experiments including every already-spent
    one, the prompt said "do not repeat unless you have a reason", and the
    loop treated any repeat as a failure and replaced it with `rng.choice`.
    Measured on real cells at k=30: 6-10 of 30 selections were random, in both
    model snapshots. Every ladder rung carried the same ~30% random component,
    which shrinks exactly the between-rung differences the ladder measures.
    """

    class _FirstItemLLM:
        """Always names the FIRST experiment in the rendered menu.

        Under the old behaviour this picks the same name every step: one real
        selection then n-1 duplicate fallbacks. Once spent items leave the
        menu, the first item differs each step, so the same trivial policy
        spends its whole budget on distinct experiments.
        """

        def __init__(self) -> None:
            self.fallbacks = 0

        def record_selection_fallback(self) -> None:
            self.fallbacks += 1

        def __call__(self, *, model, messages, **_):  # type: ignore[no-untyped-def]
            body = messages[-1]["content"]
            menu_part = body.split("Menu:\n", 1)[1]
            first = menu_part.splitlines()[0].strip()
            return {"choices": [{"message": {"content": first}}]}

    @staticmethod
    def _adapter(budget: int):  # type: ignore[no-untyped-def]
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )

        return create_contracted_chamber_agent(chamber="lt", intervention_budget=budget)

    def test_a_spent_experiment_is_absent_from_the_rendered_menu(self) -> None:
        from evaluation.chamber_pipeline.agents import _llm_select_loop

        seen_menus: list[list[str]] = []
        llm = self._FirstItemLLM()
        real_call = llm.__call__

        def spy(*, model, messages, **kw):  # type: ignore[no-untyped-def]
            body = messages[-1]["content"]
            seen_menus.append(body.split("Menu:\n", 1)[1].splitlines())
            return real_call(model=model, messages=messages, **kw)

        adapter = self._adapter(5)
        chosen, _ = _llm_select_loop(adapter, spy, "m", 0, spend=5)

        # Each step's menu must omit everything picked before it.
        for step, menu_lines in enumerate(seen_menus):
            for prior in chosen[:step]:
                assert prior not in menu_lines, (
                    f"step {step}: already-spent {prior!r} still offered"
                )

    def test_a_first_item_policy_no_longer_degenerates_to_random(self) -> None:
        from evaluation.chamber_pipeline.agents import _llm_select_loop

        llm = self._FirstItemLLM()
        adapter = self._adapter(5)
        chosen, _ = _llm_select_loop(adapter, llm, "m", 0, spend=5)

        assert len(chosen) == 5
        assert len(set(chosen)) == 5, f"duplicates remain: {chosen}"
        assert llm.fallbacks == 0, (
            f"{llm.fallbacks}/5 selections fell back to rng.choice; the menu "
            "is still offering spent experiments"
        )

    def test_a_response_that_restates_its_history_is_not_thrown_away(self) -> None:
        """Parsing against the full menu mis-reads reasoning as a selection.

        `parse_selection_response` scans the whole response text for any menu
        name, longest first. Reasoning models routinely restate what they have
        already run ("I already did X, so now Y"). Against the full menu the
        spent name X wins the scan whenever it is the longer string, the
        duplicate check then discards it, and the model's ACTUAL pick Y is
        replaced by `rng.choice` -- a real selection thrown away and recorded
        as a fallback.
        """
        from evaluation.chamber_pipeline.llm_planner import parse_selection_response

        spent, wanted = "uniform_red_strong", "uniform_blue_mid"
        assert len(spent) > len(wanted)  # the length-sort is what bites
        text = f"I already ran {spent}, so now I pick {wanted}."
        response = {"choices": [{"message": {"content": text}}]}

        # Old behaviour: the spent name wins and the real pick is lost.
        assert parse_selection_response(response, [wanted, spent]) == spent
        # Filtered: only the offered names can match, so the pick survives.
        assert parse_selection_response(response, [wanted]) == wanted

    class _RestatesHistoryLLM:
        """Picks the SHORTEST offered name, after restating everything spent.

        The shortest pick guarantees that any previously-spent name mentioned
        alongside it is longer, so `parse_selection_response`'s longest-first
        scan prefers the spent one whenever it is still in the list it is
        given. That is the whole failure: real reasoning text, real pick,
        discarded as a duplicate.
        """

        def __init__(self) -> None:
            self.fallbacks = 0

        def record_selection_fallback(self) -> None:
            self.fallbacks += 1

        def __call__(self, *, model, messages, **_):  # type: ignore[no-untyped-def]
            body = messages[-1]["content"]
            menu = [ln.strip() for ln in body.split("Menu:\n", 1)[1].splitlines() if ln.strip()]
            spent_block = body.split("Menu:\n", 1)[0]
            spent = [
                ln.strip() for ln in spent_block.splitlines() if ln.strip().startswith("uniform_")
            ]
            pick = min(menu, key=len)
            preamble = f"I already ran {', '.join(spent)}. " if spent else ""
            return {"choices": [{"message": {"content": f"{preamble}Now I pick {pick}."}}]}

    def test_the_loop_keeps_a_real_pick_that_arrives_with_its_history(self) -> None:
        from evaluation.chamber_pipeline.agents import _llm_select_loop

        llm = self._RestatesHistoryLLM()
        adapter = self._adapter(6)
        chosen, _ = _llm_select_loop(adapter, llm, "m", 0, spend=6)

        assert len(set(chosen)) == 6
        assert llm.fallbacks == 0, (
            f"{llm.fallbacks}/6 real selections were discarded as duplicates "
            "because the response restated its own history"
        )


class TestSelectionCapIsSizedForLateLoopReasoning:
    """The selection cap has now been sized against the wrong workload twice.

    200 was calibrated against nothing; 2048 was calibrated against a call with
    an EMPTY history, which is the first and cheapest step of the loop.
    Reasoning volume scales with the prompt, and the prompt grows by one
    spent-experiment line per step, so both held at k=6 and failed at k=30 --
    where an instrumented cell attributed all 13 of 30 selection failures to
    `finish_reason=length`.
    """

    # Measured 2026-08-24 on a late-loop selection call (25 already chosen,
    # effort=low, production provider order pinned, served by Novita).
    MEASURED_LATE_LOOP_TOKENS: ClassVar[dict[str, int]] = {
        "deepseek-v4-flash-0731": 2175,
        "deepseek-v4-flash": 11690,
    }

    def test_the_cap_clears_the_worst_measured_late_loop_call(self) -> None:
        from evaluation.chamber_pipeline.agents import _SELECTION_MAX_TOKENS

        worst = max(self.MEASURED_LATE_LOOP_TOKENS.values())
        assert worst <= _SELECTION_MAX_TOKENS, (
            f"cap {_SELECTION_MAX_TOKENS} is below the worst measured late-loop "
            f"call ({worst}); selections will truncate to empty content and "
            "degrade to rng.choice, and the rate will grow with k"
        )

    def test_the_cap_keeps_headroom_over_the_worst_measurement(self) -> None:
        """Not merely above the measurement: the tail is heavier than n=2 shows,
        and k=45 prompts are longer than the k=30 ones this was measured on."""
        from evaluation.chamber_pipeline.agents import _SELECTION_MAX_TOKENS

        worst = max(self.MEASURED_LATE_LOOP_TOKENS.values())
        assert 2 * worst <= _SELECTION_MAX_TOKENS, (
            f"cap {_SELECTION_MAX_TOKENS} leaves under 2x headroom over {worst}"
        )

    def test_a_truncated_selection_is_still_recorded_as_a_fallback(self) -> None:
        """The cap is the fix; the counter is the alarm. Keep both.

        Raising the cap must not remove the ability to SEE truncation if it
        recurs at a larger k -- that visibility is what turned a silent 43%
        random-selection rate into a diagnosable defect.
        """
        from agent_contracts.integrations.causalchamber import (
            create_contracted_chamber_agent,
        )
        from evaluation.chamber_pipeline.agents import _llm_select_loop

        class _TruncatingLLM:
            def __init__(self) -> None:
                self.fallbacks = 0

            def record_selection_fallback(self) -> None:
                self.fallbacks += 1

            def __call__(self, **_):  # type: ignore[no-untyped-def]
                return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

        llm = _TruncatingLLM()
        adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=4)
        chosen, _ = _llm_select_loop(adapter, llm, "m", 0, spend=4)
        assert len(chosen) == 4  # still spends the budget
        assert llm.fallbacks == 4  # and says so, every time


def test_call_kind_markers_are_unambiguous():
    """Each prompt builder must map to exactly one kind.

    The classifier replaced a `max_tokens` discriminator that became ambiguous
    the moment two caps were set to the same value, and before that a menu-size
    threshold with zero margin. This guard is what keeps the third version from
    degrading the same way: if a prompt's wording changes so two markers match,
    or none do, this fails instead of six tests quietly measuring the wrong
    calls.
    """
    from evaluation.chamber_pipeline.llm_planner import (
        build_negotiate_propose_prompt,
        build_negotiate_revise_prompt,
        build_reconcile_prompt,
        build_select_prompt,
    )
    from tests.evaluation.conftest import call_kind

    menu = ["uniform_a", "uniform_b", "uniform_c"]
    expected = {
        "select": (build_select_prompt(menu, 3, ["uniform_a"]), "select"),
        "reconcile": (build_reconcile_prompt(["uniform_a"], ["uniform_b"]), "reconcile"),
        "propose": (build_negotiate_propose_prompt("A", 2, menu), "negotiate_propose"),
        "revise": (
            build_negotiate_revise_prompt(menu, 2, ["uniform_a"], ["uniform_b"]),
            "negotiate_revise",
        ),
    }
    for label, (msgs, want) in expected.items():
        assert call_kind(msgs) == want, f"{label} classified as {call_kind(msgs)!r}"
    # And no kind is "unknown", which would silently empty a test's filter.
    assert "unknown" not in {call_kind(m) for m, _ in expected.values()}
