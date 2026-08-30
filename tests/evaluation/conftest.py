"""Shared fixtures for the M6 ladder arms.

Three constraints these fixtures exist to satisfy, each of which silently
degrades a test into measuring the wrong thing if got wrong:

- There is no stub adapter and no toy chamber. The suite builds *real* LT
  adapters behind a `requires_causalchamber` skipif; LT has 38 variables and
  59 experiments.
- Real experiment names look like `uniform_t_ir_1_mid`, not `exp_0`. A
  responder returning an off-menu name sends `_llm_select_loop` into its
  seeded random fallback, so the test measures the fallback, not the agent.
- The adapter must be built with `node_monitors` registered, or `as_node`
  raises KeyError before any agent logic runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE, reason="causalchamber not installed"
)


class RecordingLLM:
    """FakeLLM plus `max_tokens`, so tests can classify calls by their cap."""

    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = 0

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **_: Any,
    ) -> dict:
        idx = len(self.calls)
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "idx": idx,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        self.total_tokens += 100  # stands in for _CountingLLM's accumulation
        return {"choices": [{"message": {"content": self._responder(idx, messages)}}]}


def call_kind(messages: list[dict[str, str]]) -> str:
    """Classify an LLM call by its PROMPT, not by its max_tokens.

    Tests used to discriminate on `max_tokens == _NEGOTIATE_MAX_TOKENS`. That
    broke the moment the selection and negotiate caps were both raised to
    32768 -- the discriminator silently became ambiguous and six tests started
    measuring the wrong calls. An earlier version keyed on menu size (`> 30`)
    and had exactly zero margin against the largest selection pool.

    Markers verified pairwise-exclusive across all four prompt builders by
    `test_call_kind_markers_are_unambiguous`. A bare "designer" test is NOT
    sufficient: the reconcile prompt's system message says "You are one of two
    designers", so it matched too and reconcile calls were counted as
    negotiation.
    """
    body = " ".join(m["content"] for m in messages)
    if "selected:" in body:
        return "reconcile"
    if "You are designer" in body:
        return "negotiate_propose"
    if "other designer proposed" in body:
        return "negotiate_revise"
    if "Remaining budget" in body:
        return "select"
    return "unknown"


def is_negotiation(messages: list[dict[str, str]]) -> bool:
    """Either negotiation round, but never reconciliation."""
    return call_kind(messages).startswith("negotiate")


def _menu_from(messages: list[dict[str, str]]) -> list[str]:
    """Recover the menu from the user message.

    Parses only the text after the `Menu:` marker. `build_select_prompt`
    renders an "Already spent (do not repeat...)" block BEFORE the menu in the
    same message, and those names also start with `uniform_`. Scraping the
    whole body returns the scout's own prior pick from round 2 onward, which
    `_llm_select_loop` rejects as a duplicate and replaces via its seeded
    random fallback -- so the test would silently measure the fallback.
    """
    body = messages[-1]["content"]
    _, _, after = body.partition("Menu:\n")
    return [
        tok
        for line in after.splitlines()
        for tok in [line.strip("- ").strip()]
        if tok.startswith("uniform_")
    ]


@pytest.fixture
def fake_llm() -> RecordingLLM:
    """Cycles through the menu so the two scouts do not trivially collide."""

    def responder(idx: int, msgs: list[dict[str, str]]) -> str:
        menu = _menu_from(msgs)
        return menu[idx % len(menu)] if menu else ""

    return RecordingLLM(responder)


@pytest.fixture
def counting_llm() -> RecordingLLM:
    return RecordingLLM(lambda _i, msgs: (_menu_from(msgs) or [""])[0])


@pytest.fixture
def conflict_llm() -> RecordingLLM:
    """Always names the FIRST menu item, so both scouts collide every round."""
    return RecordingLLM(lambda _i, msgs: (_menu_from(msgs) or [""])[0])


@pytest.fixture
def make_ladder_adapter():
    """Factory: an LT adapter whose token_meter tracks THIS test's LLM.

    A fixture capturing one specific LLM would charge add_tokens(0) on every
    `as_node` exit for tests driving the agent with a different fixture,
    silently zeroing the attribution H-2 depends on -- and passing, because
    those tests do not assert on tokens.
    """
    from evaluation.chamber_pipeline.coordination import build_fan_in_graph

    def _make(llm: RecordingLLM, k: int = 4):
        graph = build_fan_in_graph(k=k, c95=1350, a95=21163)
        adapter = create_contracted_chamber_agent(
            chamber="lt",
            intervention_budget=k,
            node_monitors={n: graph.monitor_for(n) for n in ("scout_a", "scout_b", "aggregator")},
            token_meter=lambda: llm.total_tokens,
        )
        adapter.delegation_graph = graph
        return adapter

    return _make
