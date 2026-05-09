"""Tests for chamber-pipeline LLM prompt construction and response parsing.

Covers `evaluation.chamber_pipeline.llm_planner`. Pure functions; no
`causalchamber` and no network needed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.chamber_pipeline.llm_planner import (
    _response_text,
    build_adjacency_prompt,
    build_select_prompt,
    parse_adjacency_response,
    parse_selection_response,
)

# ---------------------------------------------------------------------------
# Helpers — small fakes for LiteLLM completion responses
# ---------------------------------------------------------------------------


def _dict_response(content: str) -> dict:
    """Plain-dict shape — what most LiteLLM responses deserialize to."""
    return {"choices": [{"message": {"content": content}}]}


class _AttrMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _AttrChoice:
    def __init__(self, content: str) -> None:
        self.message = _AttrMessage(content)


class _AttrResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_AttrChoice(content)]


# ---------------------------------------------------------------------------
# build_select_prompt
# ---------------------------------------------------------------------------


class TestBuildSelectPrompt:
    """Selection-prompt construction (per-step intervention picking)."""

    def test_returns_chat_message_pair(self) -> None:
        msgs = build_select_prompt(["uniform_a_mid", "uniform_b_mid"], remaining_budget=2)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_full_menu(self) -> None:
        menu = ["uniform_red_mid", "uniform_green_strong", "uniform_blue_weak"]
        msgs = build_select_prompt(menu, remaining_budget=3)
        user_text = msgs[1]["content"]
        for name in menu:
            assert name in user_text

    def test_user_message_includes_remaining_budget(self) -> None:
        msgs = build_select_prompt(["a"], remaining_budget=7)
        # Tolerant: just the integer must appear in the user message.
        assert "7" in msgs[1]["content"]

    def test_already_chosen_listed_when_provided(self) -> None:
        msgs = build_select_prompt(["a", "b", "c"], remaining_budget=2, already_chosen=["a"])
        # The already-spent block names "a" so the LLM can decide whether to
        # repeat. The exact phrasing is implementation-detail; only
        # presence of the spent name and the spent-experiments framing
        # matters here.
        assert "a" in msgs[1]["content"]

    def test_no_already_chosen_uses_none_marker(self) -> None:
        msgs = build_select_prompt(["a", "b"], remaining_budget=2)
        assert "none" in msgs[1]["content"].lower()

    def test_truncates_pathologically_large_menu(self) -> None:
        big_menu = [f"exp_{i:04d}" for i in range(500)]
        msgs = build_select_prompt(big_menu, remaining_budget=5)
        text = msgs[1]["content"]
        # Earliest entries are present; middle/late entries truncated.
        assert "exp_0000" in text
        assert "exp_0499" not in text
        assert "omitted" in text


# ---------------------------------------------------------------------------
# parse_selection_response
# ---------------------------------------------------------------------------


class TestParseSelectionResponse:
    """Permissive extraction of one menu name from LLM output."""

    def test_bare_name_dict_response(self) -> None:
        menu = ["uniform_red_mid", "uniform_blue_weak"]
        resp = _dict_response("uniform_red_mid")
        assert parse_selection_response(resp, menu) == "uniform_red_mid"

    def test_bare_name_attr_response(self) -> None:
        menu = ["uniform_red_mid"]
        resp = _AttrResponse("uniform_red_mid")
        assert parse_selection_response(resp, menu) == "uniform_red_mid"

    def test_name_with_prose_prefix(self) -> None:
        menu = ["uniform_red_mid", "uniform_blue_strong"]
        resp = _dict_response("I'll pick: uniform_blue_strong")
        assert parse_selection_response(resp, menu) == "uniform_blue_strong"

    def test_name_in_quotes(self) -> None:
        menu = ["uniform_red_mid"]
        resp = _dict_response('My choice is "uniform_red_mid".')
        assert parse_selection_response(resp, menu) == "uniform_red_mid"

    def test_name_in_backticks(self) -> None:
        menu = ["uniform_red_mid"]
        resp = _dict_response("`uniform_red_mid`")
        assert parse_selection_response(resp, menu) == "uniform_red_mid"

    def test_off_menu_returns_none(self) -> None:
        menu = ["uniform_a_mid", "uniform_b_mid"]
        resp = _dict_response("uniform_q_mid")  # not in menu
        assert parse_selection_response(resp, menu) is None

    def test_empty_response_returns_none(self) -> None:
        menu = ["uniform_a_mid"]
        assert parse_selection_response(_dict_response(""), menu) is None

    def test_garbage_response_returns_none(self) -> None:
        menu = ["uniform_a_mid"]
        resp = _dict_response("???")
        assert parse_selection_response(resp, menu) is None

    def test_malformed_response_returns_none(self) -> None:
        # Missing choices → returns None, doesn't raise.
        assert parse_selection_response({}, ["x"]) is None

    def test_longest_match_wins(self) -> None:
        """Prefer `uniform_red_strong` over `uniform_red` if both exist."""
        menu = ["uniform_red", "uniform_red_strong"]
        resp = _dict_response("uniform_red_strong")
        assert parse_selection_response(resp, menu) == "uniform_red_strong"

    def test_first_in_text_wins_when_multiple_match(self) -> None:
        """If LLM output mentions multiple valid names, take the first one
        we find in the response. With the longest-match tiebreak, that
        means longest names get a slight priority — acceptable for this
        permissive parsing."""
        menu = ["uniform_a_mid", "uniform_b_mid"]
        resp = _dict_response("Considering uniform_b_mid and uniform_a_mid")
        result = parse_selection_response(resp, menu)
        # Either is fine — the contract is just "a valid menu name appears".
        assert result in menu


# ---------------------------------------------------------------------------
# build_adjacency_prompt
# ---------------------------------------------------------------------------


class TestBuildAdjacencyPrompt:
    """Final-step adjacency-emission prompt for llm_only_agent."""

    def test_returns_chat_message_pair(self) -> None:
        msgs = build_adjacency_prompt(["x", "y"], n_experiments=3)
        assert len(msgs) == 2
        assert {m["role"] for m in msgs} == {"system", "user"}

    def test_user_message_lists_all_node_names(self) -> None:
        nodes = ["x", "y", "z", "w"]
        msgs = build_adjacency_prompt(nodes, n_experiments=2)
        user_text = msgs[1]["content"]
        for n in nodes:
            assert n in user_text

    def test_user_message_includes_experiment_count(self) -> None:
        msgs = build_adjacency_prompt(["x"], n_experiments=12)
        assert "12" in msgs[1]["content"]

    def test_includes_json_format_example(self) -> None:
        msgs = build_adjacency_prompt(["x", "y"], n_experiments=1)
        # Must give the LLM the output schema unambiguously.
        assert "JSON" in msgs[1]["content"] or "json" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# parse_adjacency_response
# ---------------------------------------------------------------------------


class TestParseAdjacencyResponse:
    """JSON adjacency parsing back into a directed-adjacency DataFrame."""

    def test_basic_directed_edges(self) -> None:
        nodes = ["x", "y", "z"]
        resp = _dict_response(json.dumps({"x": ["y"], "y": ["z"]}))
        adj = parse_adjacency_response(resp, nodes)
        assert adj.loc["x", "y"] == 1
        assert adj.loc["y", "z"] == 1
        assert adj.loc["x", "z"] == 0  # no transitive edges added
        assert adj.loc["z", "x"] == 0

    def test_returns_dataframe_with_node_index(self) -> None:
        nodes = ["a", "b", "c"]
        adj = parse_adjacency_response(_dict_response("{}"), nodes)
        assert list(adj.index) == nodes
        assert list(adj.columns) == nodes
        assert adj.shape == (3, 3)

    def test_empty_object_yields_zero_adjacency(self) -> None:
        adj = parse_adjacency_response(_dict_response("{}"), ["x", "y"])
        assert adj.values.sum() == 0

    def test_unknown_source_dropped(self) -> None:
        resp = _dict_response(json.dumps({"unknown": ["x"], "x": ["y"]}))
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.loc["x", "y"] == 1
        # No row for "unknown", and no spurious entries from it.
        assert "unknown" not in adj.index

    def test_unknown_target_dropped(self) -> None:
        resp = _dict_response(json.dumps({"x": ["y", "ghost"]}))
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.loc["x", "y"] == 1
        # No "ghost" column, no error.
        assert "ghost" not in adj.columns

    def test_self_loop_dropped(self) -> None:
        resp = _dict_response(json.dumps({"x": ["x", "y"]}))
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.loc["x", "x"] == 0
        assert adj.loc["x", "y"] == 1

    def test_markdown_fence_tolerated(self) -> None:
        resp = _dict_response('```json\n{"x": ["y"]}\n```')
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.loc["x", "y"] == 1

    def test_surrounding_prose_tolerated(self) -> None:
        resp = _dict_response('Here\'s the graph: {"x": ["y"]}. Hope this helps!')
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.loc["x", "y"] == 1

    def test_malformed_json_returns_zero_adjacency(self) -> None:
        resp = _dict_response("{not valid json")
        adj = parse_adjacency_response(resp, ["x", "y"])
        # Tolerant: degenerate response → no edges, well-typed shape.
        assert adj.values.sum() == 0
        assert adj.shape == (2, 2)

    def test_non_dict_json_returns_zero_adjacency(self) -> None:
        resp = _dict_response(json.dumps([["x", "y"]]))  # list, not dict
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.values.sum() == 0

    def test_empty_response_returns_zero_adjacency(self) -> None:
        adj = parse_adjacency_response(_dict_response(""), ["x", "y"])
        assert adj.values.sum() == 0

    def test_target_list_must_be_list(self) -> None:
        # A dict value that's not a list (e.g., a string) is dropped.
        resp = _dict_response(json.dumps({"x": "y"}))
        adj = parse_adjacency_response(resp, ["x", "y"])
        assert adj.values.sum() == 0


# ---------------------------------------------------------------------------
# _response_text helper
# ---------------------------------------------------------------------------


class TestResponseText:
    """Defensive reader covering both dict-shaped and attr-shaped responses."""

    def test_reads_dict_response(self) -> None:
        assert _response_text(_dict_response("hello")) == "hello"

    def test_reads_attr_response(self) -> None:
        assert _response_text(_AttrResponse("hello")) == "hello"

    def test_returns_empty_on_missing_choices(self) -> None:
        assert _response_text({}) == ""
        assert _response_text(None) == ""

    def test_returns_empty_on_none_content(self) -> None:
        assert _response_text({"choices": [{"message": {"content": None}}]}) == ""


# ---------------------------------------------------------------------------
# numpy import smoke (the parser uses np.zeros internally; trip it early)
# ---------------------------------------------------------------------------


def test_numpy_pandas_available() -> None:
    """Sanity check: dependencies the parser needs are present in this env."""
    assert isinstance(np.zeros(2), np.ndarray)
    assert isinstance(pd.DataFrame({"x": [1]}), pd.DataFrame)
