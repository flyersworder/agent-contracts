"""Prompt construction and response parsing for LLM-bearing chamber agents.

Used by `agents.llm_only_agent` (M3b) and `agents.llm_pc_agent` (M3b). Kept
as pure functions in a separate module so unit tests don't need the
`causalchamber` extra installed and don't need network access — the
planner is testable against synthetic LLM responses in isolation.

Design choices (per plan §5 + the M3b "menu only at planning time" decision):

- **Selection prompt is opaque-menu only**: the LLM sees `available_experiments()`
  as a string list. No node-name list, no parsed `target -> experiments`
  mapping. This is the strongest baseline-honesty stance — the LLM must
  infer what the menu encodes from naming alone, exactly as a domain-naive
  agent would. See §6.5 of the validation plan for the comparison rationale.
- **Adjacency-emission prompt does reveal node names**: this is the *output
  schema*, not a planning-time hint. If we hid node names here, `llm_only`
  literally couldn't produce a well-typed answer. The leak is at the output
  stage and applies only to `llm_only_agent` (the LLM-emits-graph variant);
  `llm_pc_agent` never invokes `build_adjacency_prompt`.
- **Failure-tolerant parsing**: malformed selection responses return None,
  malformed adjacency responses return all-zeros. Callers (the agents) are
  responsible for fallback policy (random pick / no-edge baseline). This
  keeps prompt parsing pure and predictable; the agent decides what to do
  with garbage.

The `_response_text` helper accepts both dict-like and Pydantic-like
LiteLLM completion responses, mirroring how `litellm_wrapper.py` already
handles the same shape variation in production.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

# Maximum number of menu entries to pretty-print in the selection prompt
# before truncating with an ellipsis note. LT's menu is 59 entries, well
# under any reasonable limit; this just guards against pathological menus
# blowing up the prompt size at M5 sweep time.
_MAX_MENU_LINES = 200


# ---------------------------------------------------------------------------
# Selection prompt (per-step intervention picking)
# ---------------------------------------------------------------------------


def build_select_prompt(
    menu: list[str],
    remaining_budget: int,
    already_chosen: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages asking the LLM to pick ONE experiment from `menu`.

    The system message frames the role; the user message lists the menu and
    the constraints. The LLM is told to respond with just the experiment
    name on its own line — `parse_selection_response` is permissive enough
    to handle some deviation from this, but the prompt asks for the simple
    form to keep parsing stable.

    Args:
        menu: All available experiment names (the chamber's `available_experiments()`).
        remaining_budget: Number of intervention queries the agent has left,
            including this one. Surfaced so the LLM can pace itself in
            principle (whether it actually does is the M3b empirical
            question).
        already_chosen: Experiments already spent in this run, in order.
            Surfaced so the LLM can avoid duplicates if it cares to. None
            and empty-list are equivalent.

    Returns:
        List of `{role, content}` dicts in OpenAI / LiteLLM chat format.
    """
    chosen = already_chosen or []

    # Build the menu rendering. Truncate only if pathologically large.
    if len(menu) > _MAX_MENU_LINES:
        rendered_menu = "\n".join(menu[:_MAX_MENU_LINES])
        rendered_menu += f"\n... ({len(menu) - _MAX_MENU_LINES} more, omitted for brevity)"
    else:
        rendered_menu = "\n".join(menu)

    chosen_block = (
        "Already spent (do not repeat unless you have a reason):\n" + "\n".join(chosen) + "\n"
        if chosen
        else "Already spent: (none yet)\n"
    )

    system = (
        "You are designing causal-discovery experiments on a physical "
        "chamber. You will be shown a menu of available pre-recorded "
        "interventional experiments. Your task is to pick ONE experiment "
        "to query next, using only the experiment names. The names encode "
        "what each experiment perturbs and how strongly."
    )

    user = (
        f"{chosen_block}\n"
        f"Remaining budget (including this pick): {remaining_budget}\n\n"
        f"Menu:\n{rendered_menu}\n\n"
        "Respond with the exact name of ONE experiment from the menu, on "
        "its own line, with no other commentary."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Two-role variants used by the Planner+Reasoner agents (M3c).
#
# Both phases reuse `build_select_prompt`'s user-message structure (menu,
# remaining budget, already-chosen list) and only override the system
# message to communicate the role. This keeps the diff against M3b small
# and means every parsing-side fix in `parse_selection_response` applies
# uniformly across all three LLM-bearing variants.
# ---------------------------------------------------------------------------


_PLANNER_SYSTEM_MESSAGE = (
    "You are the Planner in a two-agent causal-discovery design. You will "
    "pick interventional experiments first, then a Reasoner agent will "
    "pick additional experiments informed by your choices. Your task is "
    "to pick experiments that give the Reasoner a useful baseline to "
    "build on — prioritize broad coverage of distinct perturbed variables "
    "over depth on any one variable. The experiment names encode what "
    "each one perturbs and how strongly."
)

_REASONER_SYSTEM_MESSAGE = (
    "You are the Reasoner in a two-agent causal-discovery design. The "
    "Planner has already selected the experiments shown in the "
    "'Already spent' block below. Your task is to pick ONE additional "
    "experiment that best complements the Planner's choices — focus on "
    "gaps in coverage or experiments that would help disambiguate the "
    "graph structure suggested by the Planner's picks. The experiment "
    "names encode what each one perturbs and how strongly."
)


def build_planner_select_prompt(
    menu: list[str],
    remaining_budget: int,
    already_chosen: list[str] | None = None,
) -> list[dict[str, str]]:
    """Selection prompt for the Planner phase of `planner_reasoner_agents` (M3c).

    Same user message as `build_select_prompt`; system message is replaced
    to frame the role — pick for coverage, knowing a Reasoner will refine.
    """
    msgs = build_select_prompt(menu, remaining_budget, already_chosen)
    msgs[0]["content"] = _PLANNER_SYSTEM_MESSAGE
    return msgs


def build_reasoner_select_prompt(
    menu: list[str],
    remaining_budget: int,
    already_chosen: list[str] | None = None,
) -> list[dict[str, str]]:
    """Selection prompt for the Reasoner phase of `planner_reasoner_agents` (M3c).

    Same user message as `build_select_prompt`; system message is replaced
    to frame the role — refine based on the Planner's picks (which appear
    in the user message's `already_chosen` block).
    """
    msgs = build_select_prompt(menu, remaining_budget, already_chosen)
    msgs[0]["content"] = _REASONER_SYSTEM_MESSAGE
    return msgs


def parse_selection_response(response: Any, menu: list[str]) -> str | None:
    """Extract one valid experiment name from an LLM completion response.

    Permissive parsing — the LLM may add prefixes ("I pick: ..."), wrap
    the name in quotes/backticks, or surround it with reasoning. We
    search the response text for any exact match against the menu and
    return the first match found (left-to-right scan).

    Args:
        response: LiteLLM completion response (dict or Pydantic-like). The
            content of `choices[0].message.content` is what we parse.
        menu: The menu the LLM was given. Only names appearing here are
            considered valid.

    Returns:
        A menu name found in the response, or None if no menu name
        appears verbatim. Callers (the agents) are responsible for
        fallback when None is returned.
    """
    text = _response_text(response)
    if not text:
        return None

    # Sort by descending length so longer names match before their
    # prefixes (e.g., `uniform_red_strong` matches before `uniform_red`).
    for name in sorted(menu, key=len, reverse=True):
        # Word-boundary match to avoid e.g. matching `red` inside
        # `red_mid`. We escape the name for regex safety.
        pattern = r"(?<![\w-])" + re.escape(name) + r"(?![\w-])"
        if re.search(pattern, text):
            return name
    return None


# ---------------------------------------------------------------------------
# Adjacency-emission prompt (final step of llm_only_agent)
# ---------------------------------------------------------------------------


def build_adjacency_prompt(
    node_names: list[str],
    n_experiments: int,
) -> list[dict[str, str]]:
    """Build the chat messages asking the LLM to emit a directed adjacency.

    Used only by `llm_only_agent` (the LLM-emits-graph variant). Note that
    this prompt necessarily reveals `node_names` — see module docstring
    for why this is not considered a leak under the M3b "menu only at
    planning time" stance.

    Args:
        node_names: The chamber's ground-truth node names (the universe
            of variables the LLM may emit edges over). Order is preserved
            in the prompt.
        n_experiments: How many interventional experiments the LLM saw
            during the selection phase. Reported for context — the
            response data isn't included; this is the LLM-only variant
            and the LLM is being asked to commit a graph based on what it
            already inferred from the menu choices.

    Returns:
        List of `{role, content}` dicts in OpenAI / LiteLLM chat format.
    """
    rendered_nodes = "\n".join(node_names)

    system = (
        "You are now committing to a directed causal graph based on the "
        "experiments you selected. Output the graph as a JSON object "
        "mapping each source variable to the list of variables it directly "
        "causes. Include only edges you have evidence for. It is acceptable "
        "to leave a variable out if it has no outgoing edges."
    )

    user = (
        f"You completed {n_experiments} interventional experiments above.\n\n"
        f"Variables (use these exact names):\n{rendered_nodes}\n\n"
        "Output the directed causal graph as a JSON object on a single "
        'line, e.g. `{"x": ["y", "z"], "y": []}`. No prose, no markdown '
        "fences — just the JSON object."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Tolerant of markdown code fences around the JSON since LLMs frequently
# add them despite being told not to. Captures the largest `{...}` block.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_adjacency_response(response: Any, node_names: list[str]) -> pd.DataFrame:
    """Parse the LLM's adjacency-JSON response into a directed-adjacency DataFrame.

    Robust to:
        - Markdown fences (```json ... ```)
        - Surrounding prose ("Here's the graph: { ... }. Hope this helps.")
        - Edges to/from unknown variables (silently dropped)
        - Empty edge lists (variable contributes no edges)
        - Malformed JSON (returns all-zeros — caller handles)

    Returns the all-zeros adjacency on:
        - Empty / unparseable response text
        - JSON that parses but isn't a dict[str, list[str]]
        - Any exception during parsing

    Args:
        response: LiteLLM completion response.
        node_names: The chamber's ground-truth node names. The output
            DataFrame's rows/columns are indexed by these in this order.

    Returns:
        Square DataFrame `(len(node_names), len(node_names))`, integer
        entries in `{0, 1}`, indexed by `node_names`. `adj.loc[s, t] == 1`
        iff the LLM said `s -> t`.
    """
    n = len(node_names)
    empty = pd.DataFrame(np.zeros((n, n), dtype=int), index=node_names, columns=node_names)

    text = _response_text(response)
    if not text:
        return empty

    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return empty

    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return empty

    if not isinstance(parsed, dict):
        return empty

    node_set = set(node_names)
    adj = empty.copy()

    for source, targets in parsed.items():
        if source not in node_set or not isinstance(targets, list):
            # Drop edges from unknown sources or malformed value types.
            continue
        for target in targets:
            if target in node_set and target != source:
                adj.loc[source, target] = 1

    return adj


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _response_text(response: Any) -> str:
    """Pull the content string out of a LiteLLM completion response.

    Mirrors the dict-or-Pydantic accommodation in
    `litellm_wrapper._extract_response_content`. Returns the empty string
    on any structural deviation rather than raising — the caller-side
    fallback paths (random selection / empty adjacency) are well-defined,
    so a defensive read is more useful here than a strict one.
    """
    try:
        choices = response["choices"] if isinstance(response, dict) else response.choices
        first = choices[0]
        message = first["message"] if isinstance(first, dict) else first.message
        content = message["content"] if isinstance(message, dict) else message.content
        return str(content) if content else ""
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""


__all__ = [
    "build_adjacency_prompt",
    "build_planner_select_prompt",
    "build_reasoner_select_prompt",
    "build_select_prompt",
    "parse_adjacency_response",
    "parse_selection_response",
]
