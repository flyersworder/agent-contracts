"""Causal-discovery agents for the chamber pillar.

Each agent is a callable matching `ContractedChamberAgent.run`'s
expected signature:

    def agent(adapter: ContractedChamberAgent, **kwargs) -> pd.DataFrame

It is passed a contract-wrapped chamber adapter, spends some prefix of
the adapter's `per_tool_limits["intervene"]` budget by calling
`adapter.query_intervention(...)`, and returns a directed adjacency
matrix DataFrame indexed by the chamber's ground-truth node names.

Agents (per plan §5):

| # | Variant            | Architecture | Method               | Status |
|---|--------------------|--------------|----------------------|--------|
| 1 | random             | single-agent | naive                | M3a ✅ |
| 2 | greedy_ig_lite     | single-agent | non-LLM, principled  | M3a ✅ |
| 3 | llm_only           | single-agent | LLM throughout       | M3b ✅ |
| 4 | llm_pc             | single-agent | LLM-orchestrated PC  | M3b ✅ |
| 5 | planner_reasoner   | multi-agent  | LLM, two roles       | M3c ✅ |

The R1 mitigation order from plan §11 is reflected here: Random and
GreedyIG-lite (no network, no API keys, fast unit tests) land first.
LLM-bearing variants land in M3b/M3c with mocked LiteLLM in tests.

All agents return a directed-adjacency DataFrame in the convention
documented in `inference.cpdag_to_directed_adjacency`: undirected
edges from PC are reported in both directions; definite arrows in
their oriented direction.
"""

from __future__ import annotations

import random as _random
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from .inference import pool_experiment_data, run_pc
from .llm_planner import (
    build_adjacency_prompt,
    build_planner_select_prompt,
    build_reasoner_select_prompt,
    build_select_prompt,
    parse_adjacency_response,
    parse_selection_response,
    summarize_experiments,
)

if TYPE_CHECKING:
    from agent_contracts.integrations.causalchamber import ContractedChamberAgent

# Type alias for the LLM callable we accept. Matches `litellm.completion`'s
# kwargs surface: at minimum `model` (str) and `messages` (list of role/content
# dicts). Returns a LiteLLM-shaped completion response (dict or Pydantic-like).
# Tests pass synthetic callables; production passes `litellm.completion`.
LLMCallable = Callable[..., Any]


# Per-LLM-call output cap for the selection step. Without a cap, the
# model (DeepSeek v4 Flash specifically) generates ~1300 output tokens
# of verbose reasoning for what is fundamentally a "pick one item from
# this list" task. With this cap, per-call latency drops from ~37s to
# ~1.5-3s — making the M4 pilot wall-time-feasible. The expected
# response is just one menu name (~15-30 chars), so 200 tokens is
# generous headroom for any reasoning prefix the model insists on.
# Tests can monkey-patch this if they need different behavior.
# Raised 200 -> 2048 on 2026-08-23. The old cap could not hold a single
# selection call: measured on the real 59-item LT menu, DeepSeek v4 Flash 0423
# spends 821 reasoning tokens at the provider default, 976 at `high`, 475 at
# `low`, 415 at `minimal` -- every level over 200. All four pinned providers
# returned `finish_reason=length` with EMPTY content, and the loop below then
# fell back to `rng.choice`, silently turning LLM selection into random
# selection. M4b (2026-05-18) was unaffected: its recorded 509-2376 output
# tokens per call prove the calls ran to completion, because providers did not
# then count reasoning tokens against `max_tokens`.
_SELECTION_MAX_TOKENS = 2048

# Pinned explicitly rather than inherited. M4b never set this and silently
# tracked DeepSeek's default; that default then rose (M4b's ~509 output
# tokens/call against 821 today) when three thinking-effort tiers shipped on
# 2026-08-13, under a model snapshot whose weights never changed. Pinning the
# weights does not pin the behaviour -- only setting the parameter does.
# "low" (475 tokens) is the closest match to M4b's observed profile, which is
# what keeps the reused rung-0 and rung-3 cells comparable.
_SELECTION_REASONING_EFFORT = "low"

# Per-LLM-call output cap for the adjacency-emission step in
# `llm_only_agent`. Larger because the response is a JSON object
# encoding the full directed-adjacency matrix (LT: ~38 nodes,
# WT: ~32 nodes — at worst ~38*38 = 1444 entries, but typically
# only edges-present are encoded so much smaller).
#
# Why 32768 (was 4096): DeepSeek v4 Flash is a *reasoning model* that
# spends most of its `completion_tokens` budget on internal chain-of-
# thought (`reasoning_content`), with only a small fraction left for
# the visible `content` field. Diagnostic on 2026-05-14 showed 95% of
# output tokens were reasoning even on a 2-node prompt; on the 38-node
# LT prompt with 30 experiments of data summary, 4096 was consumed
# entirely by reasoning and `content` came back empty, parsing to the
# all-zeros adjacency. Bumping to 32768 leaves comfortable room for
# both reasoning and the JSON. Cost impact at OpenRouter Flash pricing
# is ~$0.009 per call → ~$1.35 across all 150 LLM-only pilot cells,
# negligible vs the ~$1.40 pilot baseline. The 1M-token context
# accommodates this trivially.
_ADJACENCY_MAX_TOKENS = 32768

# Reasoning calls must never reuse `_SELECTION_MAX_TOKENS`. DeepSeek v4 Flash
# is a reasoning model whose `reasoning_tokens` routinely reach 95% of
# `completion_tokens`, so a 200-token cap is consumed entirely by hidden
# reasoning and the response comes back with empty `content`. That is the M4b
# root-cause bug; these are sized 4-8x expected content to avoid repeating it.
_RECONCILE_MAX_TOKENS = 8192  # aggregator merges two selection lists
_NEGOTIATE_MAX_TOKENS = 4096  # short proposals in the team arm

# Rung 1's two scouts run the SAME prompt, so `seed` cannot decorrelate them:
# `_llm_select_loop` uses it only for the fallback RNG reached on an off-menu
# or duplicate response. On the happy path both scouts receive byte-identical
# messages. Sampling temperature is therefore the entire diversity mechanism
# for the homogeneous fan-in arm, and it is recorded per cell so the result is
# reproducible. Left to the provider default, a low value would drive
# `overlap_frac` to 1.0 and collapse rung 1 into rung 0 at double the budget.
_SCOUT_TEMPERATURE = 1.0


# Pattern matching the LT experiment naming convention `uniform_<TARGET>_<STRENGTH>`
# (and WT's analogous form). The single LT outlier `uniform_reference` is the
# chamber's no-intervention baseline and parses to target=None.
_EXPERIMENT_NAME_RE = re.compile(r"^uniform_(?P<target>.+?)_(?P<strength>weak|mid|strong)$")


def _parse_target(experiment_name: str) -> str | None:
    """Extract the perturbed-variable name from an experiment name.

    Returns None for unparseable names (e.g., LT's `uniform_reference`,
    which is the no-intervention baseline experiment). Treating None as
    a distinct "target" ensures the baseline experiment, if selected,
    contributes one observational sample without bumping any variable's
    target-coverage count.
    """
    m = _EXPERIMENT_NAME_RE.match(experiment_name)
    return m.group("target") if m else None


# ---------------------------------------------------------------------------
# Helpers shared by multiple agents.
# ---------------------------------------------------------------------------


def _intervention_budget(adapter: ContractedChamberAgent) -> int:
    """Return the agent's `per_tool_limits["intervene"]`, default 0.

    Centralized here so agents don't reach into the contract internals
    in five different ways.
    """
    return adapter.contract.resources.per_tool_limits.get("intervene", 0)


def _node_names(adapter: ContractedChamberAgent) -> list[str]:
    """Return the chamber's ground-truth node names, ordered.

    Available without spending budget — `ground_truth()` only loads the
    reference graph, not interventional data.
    """
    return list(adapter.ground_truth().index)


def _empty_adjacency(node_names: list[str]) -> pd.DataFrame:
    """All-zeros adjacency DataFrame on the given node set.

    Used when the agent's budget is 0 (no data to fit) or when PC
    fails — caller decides which is appropriate.
    """
    n = len(node_names)
    return pd.DataFrame(
        [[0] * n for _ in range(n)],
        index=node_names,
        columns=node_names,
    )


# ---------------------------------------------------------------------------
# Variant 1 — Random. Plan §5.1.
# ---------------------------------------------------------------------------


def random_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Pick `k` interventions uniformly at random; infer graph via PC.

    The Pareto floor of plan §6.5 / Figure 6.1. If LLM and principled
    methods don't clear this line, the LLM isn't doing real work.

    Spends exactly `per_tool_limits["intervene"]` interventions (or
    fewer if the menu is smaller). Calls `query_intervention()` for
    each, pools the resulting rows, runs PC, returns the directed
    adjacency.

    Args:
        adapter: Contract-wrapped chamber adapter.
        seed: Seed for the intervention-selection RNG. Pass distinct
            seeds across runs to estimate variance.
        pc_alpha: Significance level for the PC independence test.

    Returns:
        Directed-adjacency DataFrame indexed by chamber node names.
    """
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = adapter.available_experiments()

    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)

    rng = _random.Random(seed)
    k = min(budget, len(menu))
    chosen = rng.sample(menu, k)

    dfs = [adapter.query_intervention(name) for name in chosen]
    pooled = pool_experiment_data(dfs, nodes)
    return run_pc(pooled, nodes, alpha=pc_alpha)


# ---------------------------------------------------------------------------
# Variant 2 — GreedyIG-lite. Plan §5.1 ("greedy by approximate variance
# reduction in the MAP-graph posterior"). The "lite" qualifier is
# load-bearing: full Bayesian posterior over DAGs is deferred to a v2 /
# journal extension. Here we approximate "information gain" by edge-
# churn under refits, which is a defensible greedy proxy.
# ---------------------------------------------------------------------------


def greedy_ig_lite_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Greedy target-coverage intervention selection; PC-infer at the end.

    Plan §5.1 variant 2: "principled non-LLM baseline." The "lite"
    qualifier matters — full Bayesian variance reduction over the DAG
    posterior is deferred to a v2 / journal extension per the plan.
    What we implement here is the cleanest defensible greedy
    information-gain proxy that doesn't need posterior maintenance:

    **Greedy target-coverage**: at each step, prefer interventions
    targeting variables we haven't yet perturbed. Once every variable
    in the menu has been perturbed at least once, fall back to random
    selection over the remaining experiments. Run PC once on the
    pooled data at the end.

    Why this counts as "greedy variance reduction in the MAP-graph
    posterior" (the plan's wording):

    - For a linear-Gaussian SCM, Hauser & Bühlmann (2014) show that
      single-target interventions on previously-unperturbed variables
      strictly reduce the size of the interventional-Markov equivalence
      class (I-MEC). Greedy target coverage is therefore a
      monotone-improving I-MEC reduction policy, which is exactly the
      "approximate variance reduction in the MAP-graph posterior"
      semantics the plan calls out, modulo a constant-factor approximation.
    - It needs no Bayesian machinery — no DAG sampling, no MCMC, no
      score caching. One PC fit per run vs. O(menu_size · budget) for
      naive expected-IG. Honest about being "lite".

    Spending pattern: spends exactly `min(budget, len(menu))`
    interventions, one per call to `query_intervention`. Failed PC
    fits (rare on real chamber data, but possible for degenerate
    pooled inputs) return the all-zeros adjacency so callers always
    get a coherent shape.

    Args:
        adapter: Contract-wrapped chamber adapter.
        seed: RNG seed for shuffling within target-coverage tiers
            (controls tie-breaking when multiple unspent targets
            remain).
        pc_alpha: PC independence-test significance level.

    Returns:
        Directed-adjacency DataFrame indexed by chamber node names.
    """
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = list(adapter.available_experiments())

    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)

    # Guard: target-coverage requires that menu names parse into discrete
    # target variables. WT's experimental design uses random-walk
    # (`actuators_random_walk_N`), regime-jump (`regime_jumps_single`),
    # and load-mix (`loads_hatch_mix_*`) experiments that don't have a
    # discrete intervention target — _parse_target returns None for all
    # of them. Without this guard, GreedyIG-lite would silently degrade
    # to "all targets are None → tier 1 has 1 entry → tier 2 has the
    # rest in random order" (i.e., effectively random selection), which
    # would invisibly skew the §5.3 Pareto plot on WT. Per plan §5.1
    # variant 2, GreedyIG-lite is LT-only; the M5 sweep skips it on WT.
    n_parseable = sum(1 for name in menu if _parse_target(name) is not None)
    if n_parseable == 0:
        chamber_label = getattr(adapter, "chamber", "<unknown>")
        raise NotImplementedError(
            f"GreedyIG-lite cannot run on chamber '{chamber_label}': none of "
            f"the {len(menu)} menu entries match the `uniform_<target>_<strength>` "
            f"naming convention that the target-coverage heuristic requires. "
            f"This chamber's experimental design (e.g., random-walk perturbations) "
            f"does not have discrete intervention targets, so target-coverage has "
            f"no structure to exploit. Per the validation plan §5.1 variant 2, "
            f"GreedyIG-lite is LT-only — skip variant 2 in this chamber's cells "
            f"of the §6.1 sweep. Sample menu entries: {menu[:3]}."
        )

    rng = _random.Random(seed)

    # Group menu by parsed target variable. None bucket holds the
    # observational baseline (LT's `uniform_reference`); we treat it
    # as its own coverage tier so it doesn't preempt real targets.
    by_target: dict[str | None, list[str]] = {}
    for name in menu:
        by_target.setdefault(_parse_target(name), []).append(name)

    # Within each target's bucket, shuffle so seed actually matters.
    for names in by_target.values():
        rng.shuffle(names)

    # Tier 1: one experiment per distinct target (greedy coverage).
    # Tier 2: remaining experiments in random order (fallback).
    tier1: list[str] = []
    tier2: list[str] = []
    for names in by_target.values():
        if names:
            tier1.append(names[0])
            tier2.extend(names[1:])
    rng.shuffle(tier1)
    rng.shuffle(tier2)
    selection_order = tier1 + tier2

    # Spend in priority order until budget is exhausted.
    chosen = selection_order[: min(budget, len(selection_order))]
    dfs = [adapter.query_intervention(name) for name in chosen]

    pooled = pool_experiment_data(dfs, nodes)
    return run_pc(pooled, nodes, alpha=pc_alpha)


# ---------------------------------------------------------------------------
# Variants 3 + 4 — LLM-bearing single-agent variants. Plan §5.1.
# Both share `_llm_select_loop` for the per-step intervention picking
# (k LLM calls) and differ only in what consumes the resulting
# experiments: llm_only asks the LLM to commit a graph; llm_pc routes
# the pooled data through classical PC inference.
#
# Variant 5 (planner_reasoner) — M3c — multi-agent with delegated
# sub-budgets under conservation A + B <= k_intervene. Reuses
# `_llm_select_loop` for both phases, switching only the prompt
# builder (Planner vs Reasoner system messages) and seeding the
# Reasoner with the Planner's picks via `starting_chosen`.
# ---------------------------------------------------------------------------


def _default_llm() -> LLMCallable:
    """Return `litellm.completion`, importing lazily so non-LLM agents stay zero-dep.

    We don't import litellm at module top because the M3a agents (random,
    greedy_ig_lite) don't need it — and chamber-pipeline tests for those
    shouldn't fail at collection time when LLM-stack deps are missing.
    Importing here means `llm_only_agent` and `llm_pc_agent` only require
    litellm at call time, and only when no `llm` kwarg was supplied.
    """
    from litellm import completion

    return completion


# Prompt-builder type alias used by `_llm_select_loop`. Three concrete
# implementations live in `llm_planner.py`: build_select_prompt (M3b
# default), build_planner_select_prompt + build_reasoner_select_prompt
# (M3c). Any callable matching this shape is acceptable; tests can pass
# stand-ins to verify the loop's role-handoff behaviour.
PromptBuilder = Callable[[list[str], int, list[str] | None], list[dict[str, str]]]


def _llm_select_loop(
    adapter: ContractedChamberAgent,
    llm: LLMCallable,
    model: str,
    seed: int,
    *,
    spend: int | None = None,
    starting_chosen: list[str] | None = None,
    prompt_builder: PromptBuilder = build_select_prompt,
    temperature: float | None = None,
) -> tuple[list[str], list[pd.DataFrame]]:
    """Step `spend` times: prompt LLM for one experiment, query, repeat.

    Shared by all LLM-bearing variants — `llm_only_agent` (M3b),
    `llm_pc_agent` (M3b), and the two phases of `planner_reasoner_agents`
    (M3c). Each variant differs in: (a) which `prompt_builder` it passes
    (Planner uses `build_planner_select_prompt` etc.), (b) whether it
    seeds the loop with prior-phase choices via `starting_chosen`, and
    (c) what runs *after* the loop (LLM emits adjacency vs PC infers it).

    Failure-tolerant: if the LLM returns an off-menu / malformed response,
    we deterministically pick a random unspent menu entry (RNG seeded with
    `seed`) and proceed. This guarantees the agent always spends exactly
    `min(spend, len(menu) - len(starting_chosen))` interventions, so
    cross-variant comparisons on the budget axis remain clean.

    Args:
        adapter: Contract-wrapped chamber adapter.
        llm: LLM callable matching `litellm.completion`'s shape.
        model: Model identifier passed through to `llm(model=..., messages=...)`.
        seed: Seed for the fallback RNG (only used on bad LLM outputs).
        spend: Override the per-tool budget (None = use adapter's full
            `per_tool_limits["intervene"]`). Used by `planner_reasoner_agents`
            to limit each phase to its sub-budget. The adapter's per-tool
            enforcement still gates overall spend, so this is the
            "soft" cap; the adapter is the "hard" cap.
        starting_chosen: Experiments already spent by an earlier phase.
            Excluded from this phase's selectable pool AND surfaced to
            the LLM via the prompt's `already_chosen` block. Used by the
            Reasoner phase to inherit the Planner's picks.
        prompt_builder: Callable returning chat messages for the
            selection prompt. Defaults to the M3b opaque-menu prompt.
        temperature: Sampling temperature forwarded to the completion call.
            None (the default) omits the argument entirely rather than
            passing null, so rungs 0 and 3 reach the provider byte-identically
            to M4b. The fan-in arms pass `_SCOUT_TEMPERATURE`; see its comment
            for why the seed alone cannot decorrelate two scouts.

    Returns:
        `(chosen_names, experiment_dfs)` — parallel lists of just THIS
        loop's spend (does not include `starting_chosen`).
    """
    full_budget = _intervention_budget(adapter)
    menu = list(adapter.available_experiments())

    if full_budget <= 0 or not menu:
        return [], []

    spend = full_budget if spend is None else spend
    if spend <= 0:
        return [], []

    starting_chosen = list(starting_chosen or [])
    # Cap by what's still selectable (menu minus prior-phase picks).
    available = [m for m in menu if m not in starting_chosen]
    actual_spend = min(spend, len(available))
    if actual_spend <= 0:
        return [], []

    rng = _random.Random(seed)
    chosen: list[str] = []
    dfs: list[pd.DataFrame] = []
    for step in range(actual_spend):
        remaining = actual_spend - step
        # Compose the "already chosen" view: prior phase + this phase so far.
        all_chosen = starting_chosen + chosen
        messages = prompt_builder(menu, remaining, all_chosen)
        # Cap output to ~200 tokens. The expected response is just one
        # menu name (~15-30 chars / ~5-10 tokens), but DeepSeek v4 Flash
        # without max_tokens generates ~1300 output tokens of verbose
        # reasoning per call (verified empirically during M4b debugging).
        # 200 leaves ample headroom for the model's reasoning prefix
        # while bringing per-call latency from ~37s back down to ~1.5-3s.
        # Callers wanting a different cap can monkey-patch _SELECTION_MAX_TOKENS.
        extra: dict[str, Any] = {
            "extra_body": {"reasoning": {"effort": _SELECTION_REASONING_EFFORT}}
        }
        if temperature is not None:
            extra["temperature"] = temperature
        response = llm(
            model=model,
            messages=messages,
            max_tokens=_SELECTION_MAX_TOKENS,
            **extra,
        )
        name = parse_selection_response(response, menu)

        if name is None or name in all_chosen:
            # Fallback: random unspent. If everything is spent (LLM kept
            # picking duplicates and the menu is exhausted), fall back to
            # random over the full menu so we still spend the slot.
            unspent = [m for m in menu if m not in all_chosen]
            name = rng.choice(unspent) if unspent else rng.choice(menu)
            # Record it. This fallback exists so a bad response degrades to
            # random rather than crashing, and that graceful degradation is
            # precisely what concealed a 100% selection-failure rate for the
            # three months after providers began counting reasoning tokens
            # against `max_tokens`. Degradation must never again be silent.
            recorder = getattr(llm, "record_selection_fallback", None)
            if recorder is not None:
                recorder()

        chosen.append(name)
        dfs.append(adapter.query_intervention(name))

    return chosen, dfs


def llm_only_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """LLM picks each intervention, then emits the final adjacency directly.

    Plan §5.1 variant 3 — the "LLM throughout" cell. DeepSeek v4 Flash via
    OpenRouter through the framework's LiteLLM integration. The LLM never
    sees classical inference output: it is asked to commit a graph based
    on the experiments it chose, full stop. The pooled measurement data
    is *not* fed back to the LLM in this variant; that's the
    `llm_pc_agent` design (where PC consumes the data instead).

    Spending pattern: exactly `min(budget, len(menu))` interventions.
    Final adjacency-emission LLM call is *not* counted against
    `per_tool_limits["intervene"]` (it spends LLM tokens, not chamber
    tools).

    Args:
        adapter: Contract-wrapped chamber adapter.
        model: LiteLLM model identifier. Defaults per plan §5 to
            DeepSeek v4 Flash via OpenRouter.
        seed: RNG seed for the fallback path when the LLM returns
            unparseable selections.
        llm: Injectable LLM callable for testing. Production callers
            leave this None and we resolve `litellm.completion` lazily.

    Returns:
        Directed-adjacency DataFrame indexed by chamber node names.
    """
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)

    if budget <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()
    chosen, dfs = _llm_select_loop(adapter, llm, model, seed)

    # Final step: ask the LLM to commit a graph. We pass a compact
    # per-experiment per-node mean summary (built in `llm_planner`)
    # so the LLM does in-context inference over the data it asked for,
    # rather than reciting priors. The M4b smoke run (2026-05-13)
    # established empirically that without the summary the LLM
    # collapses to the empty graph in every cell.
    #
    # This is NOT the same as `llm_pc_agent`: PC consumes the *raw*
    # pooled data via a classical CI test; here the LLM consumes a
    # numeric *summary* via natural-language reasoning. The two are
    # cleanly distinct ablations on the same data — exactly what the
    # plan §5.3 row for "LLM-only" intended.
    data_summary = summarize_experiments(dfs, chosen, nodes)
    adj_messages = build_adjacency_prompt(
        nodes,
        n_experiments=len(chosen),
        data_summary=data_summary,
    )
    # Cap output for the adjacency-emission step. Larger than the
    # selection cap because the response encodes the full directed-edge
    # JSON map for ~38-node chambers. See _ADJACENCY_MAX_TOKENS docstring.
    response = llm(model=model, messages=adj_messages, max_tokens=_ADJACENCY_MAX_TOKENS)
    return parse_adjacency_response(response, nodes)


def llm_pc_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """LLM plans intervention sequence; classical PC infers the graph.

    Plan §5.1 variant 4 — the "main hybrid" cell. The LLM chooses *what*
    to perturb (selection); PC chooses *what edges those perturbations
    imply* (inference). This is the comparison that most directly
    interrogates the LLM's domain-design value: pull the inference step
    out of the LLM's hands, leave only intervention design.

    Spending pattern: exactly `min(budget, len(menu))` interventions.
    No final LLM call — `run_pc()` consumes the pooled data and returns
    the directed adjacency.

    Args:
        adapter: Contract-wrapped chamber adapter.
        model: LiteLLM model identifier. Defaults per plan §5.
        seed: RNG seed forwarded to the selection-loop fallback and to
            `run_pc()`'s subsampling RNG.
        pc_alpha: PC independence-test significance level.
        llm: Injectable LLM callable for testing.

    Returns:
        Directed-adjacency DataFrame indexed by chamber node names.
    """
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)

    if budget <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()
    _chosen, dfs = _llm_select_loop(adapter, llm, model, seed)

    if not dfs:
        return _empty_adjacency(nodes)

    pooled = pool_experiment_data(dfs, nodes)
    return run_pc(pooled, nodes, alpha=pc_alpha, seed=seed)


def planner_reasoner_agents(
    adapter: ContractedChamberAgent,
    planner_budget: int,
    reasoner_budget: int,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Planner + Reasoner under conservation A + B <= total. Plan §5.1 variant 5.

    The contribution-load-bearing variant. Two LLM-driven phases share
    the chamber adapter's intervention budget under an explicit
    conservation law (A + B <= k_intervene), and a single PC inference
    consumes the union of the experiments both phases queried. The
    headline comparison vs. `llm_pc_agent` (variant 4) is plan §5.3:
    if Planner+Reasoner sits on or above LLM+PC at matched total
    budget, that's direct evidence the framework's conservation laws
    preserve quality under delegation.

    Phase split:
        - Planner (budget A): broad exploration. Sees the chamber menu
          and a planner-framed system message asking it to pick
          experiments that give the Reasoner a useful baseline.
        - Reasoner (budget B): targeted refinement. Sees the menu plus
          the Planner's picks (via the prompt's `already_chosen` block)
          and a reasoner-framed system message asking it to pick
          experiments that complement the Planner's choices.
        - Inference: PC on the pooled data of all (A + B) experiments.
          Same inference step as `llm_pc_agent` to keep the §5.3
          comparison clean — only the *selection policy* differs.

    Conservation enforcement:
        - Both sub-budgets are allocated via
          `ContractingCapability.create_subcontract` BEFORE either
          phase runs. As of the per-tool delegation refactor, the
          framework primitive enforces conservation on
          `per_tool_limits` — so if A + B > k_intervene, the second
          `create_subcontract` raises `ConservationViolationError`
          before any LLM call (no API spend on a contract that can't
          legally execute).
        - This is the AAMAS plan §5 line 76-77 claim — "delegation
          primitives — not just per_tool_limits — are exercised" —
          implemented as a single framework primitive: the same
          method records the audit trail and enforces the
          conservation law.

    Args:
        adapter: Contract-wrapped chamber adapter. Its
            `per_tool_limits["intervene"]` is the total k that
            A + B must satisfy.
        planner_budget: A — interventions allocated to the Planner.
            Non-negative. A=0 means the Reasoner runs alone.
        reasoner_budget: B — interventions allocated to the Reasoner.
            Non-negative. B=0 means the Planner runs alone (the
            variant degenerates to llm_pc with budget A).
        model: LiteLLM model identifier. Defaults per plan §5.
        seed: RNG seed for both phases' fallback paths and PC's
            row-subsampling. Both phases share the seed; randomness
            is only used on bad LLM outputs.
        pc_alpha: PC independence-test significance level.
        llm: Injectable LLM callable for testing.

    Returns:
        Directed-adjacency DataFrame indexed by chamber node names.

    Raises:
        ValueError: If either sub-budget is negative.
        ConservationViolationError: If A + B > k_intervene.
    """
    # Local import to avoid pulling the delegation framework into the
    # module top-level (keeps the M3a non-LLM path zero-dep). The
    # delegation primitives are first used at M3c, not before.
    from agent_contracts.core.delegation import ContractingCapability

    if planner_budget < 0 or reasoner_budget < 0:
        raise ValueError(
            f"Sub-budgets must be non-negative; got planner={planner_budget}, "
            f"reasoner={reasoner_budget}"
        )

    nodes = _node_names(adapter)

    # Build the delegation capability up-front. Both subcontracts are
    # created BEFORE either phase runs, so per-tool conservation
    # (planner_budget + reasoner_budget <= k_intervene) is enforced
    # by the framework primitive itself: `create_subcontract` raises
    # ConservationViolationError on the second call when A + B > k.
    # No manual check needed here — `ContractingCapability` owns
    # conservation semantics for tokens, cost, AND per_tool_limits.
    # This is what the plan §5 line 76-77 calls out: "delegation
    # primitives — not just per_tool_limits — are exercised."
    capability = ContractingCapability(
        parent_contract=adapter.contract,
        parent_monitor=adapter._resource_monitor,
    )
    capability.create_subcontract(
        name="planner",
        per_tool_limits={"intervene": planner_budget},
        description=(f"Chamber Planner: broad-exploration phase, sub-budget A={planner_budget}"),
    )
    capability.create_subcontract(
        name="reasoner",
        per_tool_limits={"intervene": reasoner_budget},
        description=(
            f"Chamber Reasoner: targeted-refinement phase, sub-budget B={reasoner_budget}"
        ),
    )

    if _intervention_budget(adapter) <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()

    # Phase 1 — Planner: broad exploration under sub-budget A.
    planner_chosen, planner_dfs = _llm_select_loop(
        adapter,
        llm,
        model,
        seed,
        spend=planner_budget,
        starting_chosen=None,
        prompt_builder=build_planner_select_prompt,
    )

    # Phase 2 — Reasoner: refines based on Planner's picks under sub-budget B.
    # `starting_chosen` carries the Planner's selections into the
    # Reasoner's prompt (the role-handoff signal) and excludes them from
    # the Reasoner's selectable pool.
    _reasoner_chosen, reasoner_dfs = _llm_select_loop(
        adapter,
        llm,
        model,
        seed,
        spend=reasoner_budget,
        starting_chosen=planner_chosen,
        prompt_builder=build_reasoner_select_prompt,
    )

    # Phase 3 — Inference: PC on pooled data from BOTH phases. Same
    # inference step as llm_pc_agent so the §5.3 comparison reads
    # cleanly (only the selection policy differs across the two cells).
    all_dfs = planner_dfs + reasoner_dfs
    if not all_dfs:
        return _empty_adjacency(nodes)
    pooled = pool_experiment_data(all_dfs, nodes)
    return run_pc(pooled, nodes, alpha=pc_alpha, seed=seed)


__all__ = [
    "greedy_ig_lite_agent",
    "llm_only_agent",
    "llm_pc_agent",
    "planner_reasoner_agents",
    "random_agent",
]
