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
        menu: The experiments still SELECTABLE this step -- the chamber's
            `available_experiments()` minus anything already spent.
        remaining_budget: Number of intervention queries the agent has left,
            including this one. Surfaced so the LLM can pace itself in
            principle (whether it actually does is the M3b empirical
            question).
        already_chosen: Experiments already spent in this run, in order.
            Shown as history only -- callers are expected to have removed
            them from `menu`, so they are not selectable. None and
            empty-list are equivalent.

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

    # Spent experiments are history, not options: `_llm_select_loop` removes
    # them from `menu` before calling this. The old wording ("do not repeat
    # unless you have a reason") invited a repeat that the loop then scored as
    # a failure, so say plainly that they are gone.
    chosen_block = (
        "Already spent (no longer on the menu):\n" + "\n".join(chosen) + "\n"
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


# ---------------------------------------------------------------------------
# Blind scout roles used by the fan-in arms (M6 rungs 1 and 2).
#
# "Blind" is the defining property: neither scout may learn that another
# agent exists. Rung 1 runs two scouts on the SAME prompt and gets its
# diversity from sampling temperature alone; rung 2 differentiates them by
# role. If either prompt leaked the existence of a peer, the two rungs would
# stop isolating role differentiation from mere ensembling, which is the
# comparison the ladder is built to make.
#
# `already_chosen` here is the scout's own prior picks within its own loop --
# never another agent's. It reaches these builders because
# `_llm_select_loop` calls `prompt_builder(menu, remaining, all_chosen)`
# positionally; a two-parameter builder raises TypeError on every call.
# ---------------------------------------------------------------------------


_SCOUT_BROAD_SYSTEM_MESSAGE = (
    "You are designing causal-discovery experiments on a physical "
    "chamber. You will be shown a menu of available pre-recorded "
    "interventional experiments. Your task is to pick ONE experiment "
    "to query next, using only the experiment names. The names encode "
    "what each experiment perturbs and how strongly. Favour BREADTH: "
    "prefer an experiment that perturbs a target no earlier pick has "
    "touched, so that the set you accumulate covers as many distinct "
    "intervention targets as possible."
)


_SCOUT_TARGETED_SYSTEM_MESSAGE = (
    "You are designing causal-discovery experiments on a physical "
    "chamber. You will be shown a menu of available pre-recorded "
    "interventional experiments. Your task is to pick ONE experiment "
    "to query next, using only the experiment names. The names encode "
    "what each experiment perturbs and how strongly. Favour DEPTH: "
    "prefer an experiment that disambiguates variables whose "
    "relationships remain least constrained by what has been picked so "
    "far, even if that means perturbing a target already touched."
)


def build_scout_broad_prompt(
    menu: list[str],
    remaining_budget: int,
    already_chosen: list[str] | None = None,
) -> list[dict[str, str]]:
    """Coverage-seeking scout (rung 2's role A; rung 1 uses it for both).

    Same user message as `build_select_prompt`; the system message asks for
    breadth. Mentions no other agent.
    """
    msgs = build_select_prompt(menu, remaining_budget, already_chosen)
    msgs[0]["content"] = _SCOUT_BROAD_SYSTEM_MESSAGE
    return msgs


def build_scout_targeted_prompt(
    menu: list[str],
    remaining_budget: int,
    already_chosen: list[str] | None = None,
) -> list[dict[str, str]]:
    """Disambiguation-seeking scout (rung 2's role B).

    Same user message as `build_select_prompt`; the system message asks for
    depth. Mentions no other agent.
    """
    msgs = build_select_prompt(menu, remaining_budget, already_chosen)
    msgs[0]["content"] = _SCOUT_TARGETED_SYSTEM_MESSAGE
    return msgs


_RECONCILE_SYSTEM_MESSAGE = (
    "You are aggregating the experiment selections of two independent "
    "designers who worked without knowledge of each other. Their lists "
    "may overlap or conflict. Produce a single deduplicated ordering of "
    "the experiments to run, most informative first, keeping every "
    "distinct experiment exactly once."
)


def build_reconcile_prompt(
    chosen_a: list[str],
    chosen_b: list[str],
) -> list[dict[str, str]]:
    """Aggregator prompt: merge two scouts' selections into one ordering.

    This is the aggregator's single indivisible call -- the one whose size
    makes whitepaper §4.6 P2's fragmentation penalty concrete, since it
    cannot be split across the two scouts' separate grants.
    """
    user = (
        "Designer A selected:\n"
        + ("\n".join(chosen_a) if chosen_a else "(nothing)")
        + "\n\nDesigner B selected:\n"
        + ("\n".join(chosen_b) if chosen_b else "(nothing)")
        + "\n\nRespond with the deduplicated experiment names, one per line, "
        "most informative first, and no other commentary."
    )
    return [
        {"role": "system", "content": _RECONCILE_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Negotiation prompts for the team rung (M6 rung 4).
#
# This is the one rung where the scouts know a peer exists. Rungs 1 and 2 are
# blind by construction, so keeping these prompts separate is what stops the
# ladder's two comparisons -- role differentiation, and explicit coordination
# -- from being confounded into one.
# ---------------------------------------------------------------------------


_NEGOTIATE_SYSTEM_MESSAGE = (
    "You are one of two designers planning causal-discovery experiments on a "
    "physical chamber. You share a fixed total budget with the other "
    "designer, so an experiment one of you runs is one the other cannot. "
    "Your aim is a joint plan that covers as much as possible, not the best "
    "individual plan."
)


def build_negotiate_propose_prompt(
    menu: list[str],
    budget: int,
    role: str,
) -> list[dict[str, str]]:
    """Round 1: state which experiments this scout intends to claim."""
    user = (
        f"You are designer {role}. You may run {budget} experiment(s).\n\n"
        f"Menu:\n" + "\n".join(menu) + "\n\n"
        f"List the {budget} experiment name(s) you intend to claim, one per "
        "line, and no other commentary."
    )
    return [
        {"role": "system", "content": _NEGOTIATE_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def build_negotiate_revise_prompt(
    menu: list[str],
    budget: int,
    own: list[str],
    other: list[str],
) -> list[dict[str, str]]:
    """Round 2: having seen the peer's claim, revise to reduce collisions."""
    user = (
        f"You may run {budget} experiment(s).\n\n"
        "You proposed:\n" + ("\n".join(own) if own else "(nothing)") + "\n\n"
        "The other designer proposed:\n"
        + ("\n".join(other) if other else "(nothing)")
        + "\n\nAny experiment you both named is wasted duplication. Revise "
        f"your claim to {budget} experiment name(s) from the menu below, one "
        "per line, and no other commentary.\n\n"
        "Menu:\n" + "\n".join(menu)
    )
    return [
        {"role": "system", "content": _NEGOTIATE_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


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


def summarize_experiments(
    experiment_dfs: list[pd.DataFrame],
    chosen_names: list[str],
    node_names: list[str],
    decimals: int = 2,
) -> str:
    """Render a compact markdown table of per-experiment per-node means.

    Used by `build_adjacency_prompt` (when invoked via `llm_only_agent`)
    to give the LLM the *data* it asked for via its intervention picks.
    Without this summary, `llm_only_agent` reduces to "commit a graph
    based on names alone," which empirically yields the empty graph
    (verified in the M4b smoke run, 2026-05-13).

    The output is markdown rather than CSV so a curious reader of the
    LLM trace sees something legible. Rows are experiments (preserving
    the order in `chosen_names`); columns are the chamber's graph nodes
    (preserving the order in `node_names`). Non-node columns (timestamp,
    counter, intervention, etc.) are dropped.

    Each cell is the within-experiment mean of that node, rounded to
    `decimals` places. Standard deviations are *not* included in v1 —
    the LLM can implicitly gauge dynamic range by comparing means
    across experiments, which is cheaper in tokens. If smoke results
    show insufficient signal, std can be added behind the same call.

    Args:
        experiment_dfs: Per-experiment measurement DataFrames, in the
            same order as `chosen_names`. Each DataFrame should have
            one row per sample and one column per chamber variable
            (plus metadata columns which are ignored).
        chosen_names: The experiment names spent so far (e.g.,
            `["uniform_red_mid", "uniform_green_mid"]`). The
            intervention target is encoded in the name; the LLM is
            expected to parse it.
        node_names: The chamber's ground-truth graph nodes. Only these
            columns appear in the summary; everything else is dropped.
        decimals: Rounding for the mean values. 2 is usually enough
            to surface intervention effects without bloating tokens.

    Returns:
        Markdown table string, or the empty string if no experiments
        were provided. Shape: `(len(chosen_names) + 2)` rows
        x `(len(node_names) + 1)` columns including header and divider.
    """
    if not experiment_dfs or not chosen_names:
        return ""

    # Header: experiment | node1 | node2 | ...
    header = "| experiment | " + " | ".join(node_names) + " |"
    divider = "|" + "|".join(["---"] * (len(node_names) + 1)) + "|"

    rows: list[str] = [header, divider]
    for name, df in zip(chosen_names, experiment_dfs, strict=False):
        # Only summarize columns that are graph nodes — drop chamber
        # metadata (timestamp, counter, flag, intervention, ...).
        present = [c for c in node_names if c in df.columns]
        means = df[present].mean(numeric_only=True)
        cells = [f"{means[n]:.{decimals}f}" if n in means.index else "—" for n in node_names]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def build_adjacency_prompt(
    node_names: list[str],
    n_experiments: int,
    data_summary: str | None = None,
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
            during the selection phase. Reported for context.
        data_summary: Optional markdown table from `summarize_experiments`
            giving per-experiment per-node means. When provided, the
            prompt instructs the LLM to base its graph on the observed
            data shifts. When None (default), the prompt falls back to
            the pre-M4b "commit a graph based on names alone" behavior
            — kept for backward-compat and unit tests; production
            `llm_only_agent` always passes a summary.

    Returns:
        List of `{role, content}` dicts in OpenAI / LiteLLM chat format.
    """
    rendered_nodes = "\n".join(node_names)

    if data_summary:
        # Data-grounded path: the LLM has actual measurements to reason
        # over, so the system prompt drops the "leave a variable out
        # if it has no outgoing edges" escape hatch that empirically
        # collapses to the empty graph.
        system = (
            "You are inferring a directed causal graph from interventional "
            "data. For each pair of variables, decide whether the row's "
            "intervention target causally affects the column variable by "
            "comparing that column's mean across experiments. Output the "
            "graph as a JSON object mapping each source variable to the "
            "list of variables it directly causes. Include every edge "
            "supported by a clear mean shift; omit only when the evidence "
            "is genuinely absent."
        )
        user = (
            f"You completed {n_experiments} interventional experiments. The "
            "intervention target of each experiment is encoded in its name "
            "(e.g., `uniform_red_mid` intervenes on `red`).\n\n"
            f"Per-experiment per-variable means (graph nodes only):\n\n"
            f"{data_summary}\n\n"
            f"Variables (use these exact names):\n{rendered_nodes}\n\n"
            "Output the directed causal graph as a JSON object on a single "
            'line, e.g. `{"x": ["y", "z"], "y": []}`. No prose, no markdown '
            "fences — just the JSON object."
        )
    else:
        # Legacy path: pre-M4b behavior, kept so existing unit tests
        # (which don't construct a data summary) still exercise the
        # builder without flagging spurious failures.
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
