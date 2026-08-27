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
# this list" task. Tests can monkey-patch this if they need different
# behavior.
# Raised 200 -> 2048 on 2026-08-23, then 2048 -> 32768 the same day. The old
# cap could not hold a single selection call: measured on the real 59-item LT
# menu, DeepSeek v4 Flash 0423 spends 821 reasoning tokens at the provider
# default, 976 at `high`, 475 at `low`, 415 at `minimal` -- every level over
# 200. All four pinned providers returned `finish_reason=length` with EMPTY
# content, and the loop below then fell back to `rng.choice`, silently turning
# LLM selection into random selection. M4b (2026-05-18) was unaffected: its
# recorded 509-2376 output tokens per call prove the calls ran to completion,
# because providers did not then count reasoning tokens against `max_tokens`.
#
# WHY 2048 WAS STILL WRONG, and the methodological error to avoid repeating:
# those 415-976 figures were all measured on a call with an EMPTY history --
# the FIRST and cheapest step of the loop. Reasoning volume scales with the
# prompt, and the prompt grows by one spent-experiment line per step. Measured
# on a late-loop call (25 already chosen, effort=low, providers pinned):
# flash-0731 emits 2,175 tokens and flash 11,690 -- 1.1x and 5.7x the 2048 cap.
#
# The consequence was severe and invisible. An instrumented k=30 cell
# (2026-08-24, 0731, pinned providers) attributed EVERY selection failure to
# truncation: {'length': 13, 'empty': 0, 'offmenu': 0, 'ok': 17} -- 13 of 30
# picks were `rng.choice`. Because the failure rate is a function of history
# length, it was 0/36 at k=6 and ~43% at k=30, which made the harness a
# MODERATOR CORRELATED WITH THE BUDGET: in M4b, `llm_pc` beat `random` by
# +0.034 F1 at k=6 (resolved) and only +0.018 at k=30 (below MDE), so LLM
# selection appeared to stop helping as budget grew. That was this cap, not a
# property of the model.
#
# It also threatened M6 directly: the ladder varies how budget is SPLIT across
# agents, and splitting shortens each agent's history. Two scouts at k=15 each
# truncate less than one loop at k=30, so the fan-in rungs would have scored
# better than the loop for reasons having nothing to do with coordination --
# H-B could have come out positive as a pure `max_tokens` artifact.
#
# 32768 matches `_ADJACENCY_MAX_TOKENS`. `max_tokens` is a CEILING, not a
# reservation -- billing follows tokens actually generated -- so sizing it
# generously costs nothing and removes the failure mode instead of relocating
# it to a larger k. Calibrate any future change on a LATE-loop call.
_SELECTION_MAX_TOKENS = 32768

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
# Raised 8192 -> 32768 on 2026-08-24, alongside the selection cap, for the
# same reason: the old value was calibrated on k=6-ish prompts (reconcile
# median 2,826 / p95 6,375) while `_A95_RECONCILE` already measures 8,557 at
# k=30, so an 8192 cap truncates at the budgets the ladder actually runs.
#
# CAVEAT, and it is not about accuracy. `team_agents`/`fan_in_agents` DISCARD
# the reconcile response -- the merge below is a plain Python dedup plus PC --
# so truncation here never changed a result. What it changed is COST: a
# truncated call still generates and bills `max_tokens` of reasoning, which
# `as_node("aggregator")` books into the aggregator's monitor. At 8192 a
# truncated reconcile pinned aggregator spend to exactly 8192, which sits
# inside P2's incompleteness window (6418, 12836] -- so `tree_would_refuse`
# could report True because the call TRUNCATED, not because reconciliation is
# genuinely indivisible-and-large.
#
# Consequence: `_A95_RECONCILE` and `_C95_NEGOTIATE` in `orchestrator.py` are
# now calibrated against truncated calls and MUST be re-derived from
# untruncated late-loop measurements at k=45 before any sweep whose H-C or P2
# numbers are reported. Otherwise both are calibration artifacts. This is what
# spec §3's c95(45) pre-flight probe measures.
_RECONCILE_MAX_TOKENS = 32768  # aggregator merges two selection lists
_NEGOTIATE_MAX_TOKENS = 32768  # short proposals in the team arm

# Pinned for the same reason as `_SELECTION_REASONING_EFFORT`: an unset
# parameter silently tracks a provider default, and DeepSeek raised that
# default under unchanged weights on 2026-08-13. These two calls DO want real
# deliberation -- reconciliation merges two selection lists, negotiation
# reasons about a peer's claim -- so "high" is the right value, but it has to
# be stated rather than inherited.
_COORDINATION_REASONING_EFFORT = "high"

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
    exclude: set[str] | None = None,
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
        exclude: Experiment names removed from the selectable menu WITHOUT
            appearing in the prompt. Used by the team arm's collision
            backstop. Deliberately not routed through `starting_chosen`,
            which would render the excluded names into the prompt as an
            "Already spent" block, destroying the blindness of the execution
            phase. NOTE that `exclude` narrows the menu and therefore DOES
            feed `actual_spend = min(spend, len(available))`, exactly as
            `starting_chosen` would: the two differ only in prompt rendering,
            never on the spend axis. A caller excluding many names must top up
            the selectable pool itself, or the scout silently under-spends.
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
    menu = [m for m in adapter.available_experiments() if m not in (exclude or set())]

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
        # Offer only what is still unspent. Previously the full menu was
        # rendered every step with the spent items still in it, the prompt
        # said "do not repeat unless you have a reason", and the loop below
        # treated any repeat as a failure and replaced it with `rng.choice`.
        # Duplicates were invited and then punished: measured 6-10 of 30
        # selections falling back to random at k=30, on BOTH model snapshots,
        # which gave every ladder rung the same ~30% random component.
        # `actual_spend <= len(available)` guarantees this is non-empty at
        # every step.
        selectable = [m for m in menu if m not in all_chosen]
        messages = prompt_builder(selectable, remaining, all_chosen)
        # See `_SELECTION_MAX_TOKENS` for why 200 was untenable once
        # providers began counting reasoning tokens against `max_tokens`.
        # Note the cap is NOT a hard bound when `reasoning.effort` is set:
        # responses routinely exceed it and still finish with `stop`, so
        # effort -- not max_tokens -- is the real cost control.
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
        # Validate against the SAME list the model was shown. Parsing against
        # the full menu would accept a spent name as well-formed and then
        # discard it one line later as a duplicate -- the failure this change
        # removes.
        name = parse_selection_response(response, selectable)

        if name is None:
            # Fallback: random unspent. Reachable now only on a genuinely
            # unusable response (empty content from a truncated reasoning
            # budget, or a name that is not on the offered list at all).
            name = rng.choice(selectable) if selectable else rng.choice(menu)
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
    response = llm(
        model=model,
        messages=adj_messages,
        max_tokens=_ADJACENCY_MAX_TOKENS,
        # Pinned like every other call. This is the pipeline's largest
        # reasoning call and the one with a documented history of returning
        # empty content when reasoning consumed the budget, so leaving it to
        # track a provider default is the worst place to do so.
        extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
    )
    # A degenerate all-zero emission -- the M4b failure mode, where the model
    # spends its budget on hidden reasoning and returns empty content -- is
    # already unambiguous in the record as `n_edges_predicted == 0`. Counting
    # it into `n_selection_fallbacks` would make one column mean two
    # different failures.
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


def fan_in_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    scout_a_budget: int,
    scout_b_budget: int,
    differentiate: bool = False,
    honor_aggregator: bool = False,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Two blind scouts fund one aggregator — ladder rungs 1 and 2.

    ``differentiate=False`` is rung 1, a homogeneous ensemble whose only
    source of divergence is sampling temperature. ``differentiate=True`` is
    rung 2, where the scouts carry distinct role framings. Neither scout is
    told the other exists: that blindness is what makes the pair isolate role
    differentiation rather than communication, which is rung 4's business.

    Budget flows through a :class:`DelegationGraph` whose scout and aggregator
    nodes carry real monitors; the adapter routes each chamber call to the
    monitor of whichever node is acting, additively with the aggregate cap.
    """
    from evaluation.chamber_pipeline.coordination import overlap_fraction
    from evaluation.chamber_pipeline.llm_planner import (
        build_reconcile_prompt,
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )

    nodes = _node_names(adapter)
    # Set on EVERY path, including the early returns below. Task 8's scorer
    # reads this in `run_cell`; an attribute that exists only on the happy
    # path raises AttributeError on empty-menu and zero-budget cells.
    adapter.coordination_stats = {"overlap_frac": None, "n_experiments_distinct": 0}
    if _intervention_budget(adapter) <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)
    llm = llm or _default_llm()

    prompt_a = build_scout_broad_prompt if differentiate else build_select_prompt
    prompt_b = build_scout_targeted_prompt if differentiate else build_select_prompt

    # 2*seed and 2*seed+1, never seed and seed+1: M4b seeds are contiguous
    # 0..29, so seed+1 would collide with the next cell's scout_a.
    with adapter.as_node("scout_a"):
        chosen_a, dfs_a = _llm_select_loop(
            adapter,
            llm,
            model,
            2 * seed,
            spend=scout_a_budget,
            starting_chosen=None,
            prompt_builder=prompt_a,
            temperature=_SCOUT_TEMPERATURE,
        )
    with adapter.as_node("scout_b"):
        chosen_b, dfs_b = _llm_select_loop(
            adapter,
            llm,
            model,
            2 * seed + 1,
            spend=scout_b_budget,
            starting_chosen=None,
            prompt_builder=prompt_b,
            temperature=_SCOUT_TEMPERATURE,
        )

    # The aggregator's reconciliation call. REQUIRED, not decorative: PC is
    # not an LLM call, so without it the aggregator consumes nothing, the
    # fan-in edges carry budget nobody spends, `_consumed()` reads zero, and
    # verify() is vacuously true. It is also the single indivisible request
    # that puts this arm inside whitepaper §4.6 P2's incompleteness window.
    with adapter.as_node("aggregator"):
        agg_response = llm(
            model=model,
            messages=build_reconcile_prompt(chosen_a, chosen_b),
            max_tokens=_RECONCILE_MAX_TOKENS,
            extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
        )

    # Duplicates still COST budget — each query_intervention was metered — but
    # are dropped before pooling so PC does not see an inflated n.
    seen: set[str] = set()
    dfs: list[pd.DataFrame] = []
    for name, frame in zip(chosen_a + chosen_b, dfs_a + dfs_b, strict=True):
        if name not in seen:
            seen.add(name)
            dfs.append(frame)

    # `honor_aggregator` is the ablation that answers the obvious review of
    # this arm: "your aggregator's output is discarded, so a negative result
    # about fan-in is an artifact of a null aggregator."
    #
    # By the time the aggregator runs the scouts have ALREADY BOUGHT their
    # experiments, so its only levers are reordering (which reaches PC solely
    # through `run_pc`'s row subsample) and dropping (strictly less data).
    # It cannot un-buy, and it holds no information the scouts lack. That is
    # a property of the architecture, not of this implementation -- but the
    # claim has to be measured rather than argued, which is what this does.
    #
    # Hallucinated names are intersected away: `_parse_name_list` matches
    # against the menu, not against what was purchased, so an aggregator may
    # name an experiment nobody ran. Pooling that would fabricate data.
    agg_diag: dict[str, int] = {}
    if honor_aggregator:
        bought = dict(zip(chosen_a + chosen_b, dfs_a + dfs_b, strict=True))
        named = _parse_name_list(agg_response, list(adapter.available_experiments()))
        kept = [n for n in named if n in bought]
        agg_diag = {
            "agg_named": len(named),
            "agg_hallucinated": len(named) - len(kept),
            "agg_dropped": len(seen) - len(kept),
        }
        # An empty or fully-hallucinated response must not silently pool
        # nothing -- that would score the parser, not the topology.
        if kept:
            dfs = [bought[n] for n in kept]
            seen = set(kept)
        else:
            agg_diag["agg_fallback"] = 1

    adapter.coordination_stats = {
        "overlap_frac": overlap_fraction(chosen_a, chosen_b),
        "n_experiments_distinct": len(seen),
        **agg_diag,
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def _parse_name_list(response: Any, menu: list[str]) -> list[str]:
    """Every menu name appearing in a response, deduplicated, in menu order.

    Matches on word boundaries, longest name first, exactly as
    `parse_selection_response` does. Plain substring containment is wrong on
    any menu with prefix relationships: WT has `actuators_random_walk_1`
    through `_16`, so a response naming only `_10` and `_12` also matches
    `_1` -- inventing a claim the scout never made, inflating `contested`,
    and over-excluding the other scout.
    """
    from evaluation.chamber_pipeline.llm_planner import _response_text

    text = _response_text(response) or ""
    claimed: list[str] = []
    for name in sorted(menu, key=len, reverse=True):
        if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text):
            continue
        if any(name in longer for longer in claimed):
            continue  # a longer name already matched here
        claimed.append(name)
    return [n for n in menu if n in claimed]


def team_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    scout_a_budget: int,
    scout_b_budget: int,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Two scouts negotiate their split before executing — ladder rung 4.

    One upfront round, O(1) in ``k``: each scout proposes, each sees the
    other's proposal and revises once, a deterministic backstop resolves what
    remains contested, and only then do both execute.

    This is the ladder's only rung where a scout knows a peer exists. Rungs 1
    and 2 stay blind so that they isolate role differentiation; making the
    coordination explicit here is what separates the two comparisons.

    The scout-to-scout channel is a Python variable, **not** a graph edge. A
    bidirectional pair raises :class:`CycleError` regardless of carrying zero
    tokens, because ``allocate()`` runs its reachability check before it
    inspects the amount. Control flow may cycle; budget flow may not
    (whitepaper §4.6 P3).
    """
    from evaluation.chamber_pipeline.coordination import overlap_fraction
    from evaluation.chamber_pipeline.llm_planner import (
        build_negotiate_propose_prompt,
        build_negotiate_revise_prompt,
        build_reconcile_prompt,
    )

    nodes = _node_names(adapter)
    adapter.coordination_stats = {"overlap_frac": None, "n_experiments_distinct": 0}
    if _intervention_budget(adapter) <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)
    llm = llm or _default_llm()
    menu = list(adapter.available_experiments())

    def negotiate(role: str, budget: int, node: str) -> list[str]:
        with adapter.as_node(node):
            proposal = llm(
                model=model,
                messages=build_negotiate_propose_prompt(menu, budget, role),
                max_tokens=_NEGOTIATE_MAX_TOKENS,
                # The two propose prompts differ only by the letter A/B, so
                # without a temperature both scouts return the same claim list
                # and the negotiation contributes noise instead of a split --
                # the degeneracy `_SCOUT_TEMPERATURE` exists to prevent.
                temperature=_SCOUT_TEMPERATURE,
                extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
            )
        return _parse_name_list(proposal, menu)

    proposed_a = negotiate("A", scout_a_budget, "scout_a")
    proposed_b = negotiate("B", scout_b_budget, "scout_b")

    def revise(budget: int, own: list[str], other: list[str], node: str) -> list[str]:
        with adapter.as_node(node):
            revised = llm(
                model=model,
                messages=build_negotiate_revise_prompt(menu, budget, own, other),
                max_tokens=_NEGOTIATE_MAX_TOKENS,
                # The two propose prompts differ only by the letter A/B, so
                # without a temperature both scouts return the same claim list
                # and the negotiation contributes noise instead of a split --
                # the degeneracy `_SCOUT_TEMPERATURE` exists to prevent.
                temperature=_SCOUT_TEMPERATURE,
                extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
            )
        return _parse_name_list(revised, menu)

    revised_a = revise(scout_a_budget, proposed_a, proposed_b, "scout_a")
    revised_b = revise(scout_b_budget, proposed_b, proposed_a, "scout_b")

    # Turn the negotiation into the actual split. An earlier version had both
    # scouts run a blind loop and merely removed contested names from
    # scout_b's menu, which was wrong three ways: the negotiated lists had no
    # bearing on what either scout executed, so rung 4 was rung 1 with extra
    # LLM calls; names scout_a claimed but never picked were queried by
    # nobody; and the anti-starvation guard looked only at `contested`, not at
    # the full exclusion, so scout_b silently under-spent -- 14 picks against
    # a budget of 22, measured on the real LT menu.
    source_a = revised_a or proposed_a
    source_b = revised_b or proposed_b
    # Count rounds that produced nothing usable. Unparseable negotiation is
    # the documented empty-content mode, and it drops the affected scout to
    # the seeded fallback partition while `overlap_frac` reads 0.0 and
    # `n_contested` reads 0 -- indistinguishable from a perfect split. Unlike
    # the selection loop, this path had no recorder at all.
    # All four parses, not two scouts. Counting only scouts whose `revised or
    # proposed` is empty misses the likelier and more damaging case: both
    # propose rounds parse but both REVISE rounds return prose, which reduces
    # rung 4 to one-shot proposals with no negotiation at all while reporting
    # zero failures.
    negotiation_failures = sum(
        1 for parsed in (proposed_a, proposed_b, revised_a, revised_b) if not parsed
    )
    # `contested` is measured from the lists actually used, not from
    # `revised_*`: when a revise reply is unparseable the code falls back to
    # the proposals, and reading the discarded list reports 0 conflicts for a
    # round that resolved none.
    contested = set(source_a) & set(source_b)

    # Cap each claim at its budget. Uncapped, a scout that reasons out loud
    # over most of the menu swallows the shared pool and starves its partner:
    # measured 10 + 4 against a 20 budget when `claim_a` reached 55 names.
    claim_a = list(source_a)[:scout_a_budget]
    claim_b = [n for n in source_b if n not in set(claim_a)][:scout_b_budget]

    # Partition the REST of the menu between the scouts, so each selects from
    # a pool strictly larger than its budget. Two constraints, and an earlier
    # version satisfied only the first:
    #
    #  * The pool must EXCEED the budget, or `actual_spend ==
    #    len(available)`, every name in the pool is queried, and the selection
    #    loop is inert -- verified: the queried set was byte-identical whether
    #    the selection LLM returned the first menu item, the last, or
    #    "GARBAGE".
    #  * The pool must never fall SHORT of the budget, or the scout silently
    #    under-spends. A plain `rest[0::2]` / `rest[1::2]` split does fall
    #    short once the claims are large: at k=45 with a full claim, measured
    #    23 + 18 = 41 of 45, reported `status=ok` with conservation certified.
    #    Each scout's shortfall is therefore reserved BEFORE the leftover is
    #    divided.
    #
    # `rest` is shuffled on a seeded RNG rather than sliced by parity. The
    # menu order is fixed and groups by variable family and intervention
    # strength, so `rest[0::2]` handed scout_a 0 of 3 `osr_c` and 0 of 2 `red`
    # experiments on LT -- the same blind spot in all 30 seeds, since nothing
    # about the slice depends on the seed.
    rest = [m for m in menu if m not in set(claim_a) | set(claim_b)]
    _random.Random(seed).shuffle(rest)
    need_a = max(0, scout_a_budget - len(claim_a))
    need_b = max(0, scout_b_budget - len(claim_b))
    take_a, take_b, leftover = (
        rest[:need_a],
        rest[need_a : need_a + need_b],
        rest[need_a + need_b :],
    )
    pool_a = set(claim_a) | set(take_a) | set(leftover[0::2])
    pool_b = set(claim_b) | set(take_b) | set(leftover[1::2])

    all_names = set(menu)
    with adapter.as_node("scout_a"):
        chosen_a, dfs_a = _llm_select_loop(
            adapter,
            llm,
            model,
            2 * seed,
            spend=scout_a_budget,
            prompt_builder=build_select_prompt,
            temperature=_SCOUT_TEMPERATURE,
            exclude=all_names - pool_a,
        )
    with adapter.as_node("scout_b"):
        chosen_b, dfs_b = _llm_select_loop(
            adapter,
            llm,
            model,
            2 * seed + 1,
            spend=scout_b_budget,
            prompt_builder=build_select_prompt,
            temperature=_SCOUT_TEMPERATURE,
            # `| set(chosen_a)` is belt-and-braces only: `pool_a` and
            # `pool_b` are disjoint by construction, so it never removes
            # anything. Kept so the disjointness is not silently load-bearing
            # on one construction alone.
            exclude=(all_names - pool_b) | set(chosen_a),
        )

    with adapter.as_node("aggregator"):
        llm(
            model=model,
            messages=build_reconcile_prompt(chosen_a, chosen_b),
            max_tokens=_RECONCILE_MAX_TOKENS,
            extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
        )

    seen: set[str] = set()
    dfs: list[pd.DataFrame] = []
    for name, frame in zip(chosen_a + chosen_b, dfs_a + dfs_b, strict=True):
        if name not in seen:
            seen.add(name)
            dfs.append(frame)

    adapter.coordination_stats = {
        "overlap_frac": overlap_fraction(chosen_a, chosen_b),
        "n_experiments_distinct": len(seen),
        # How many claims the negotiation failed to resolve. Without this the
        # rung's defining mechanism is unmeasurable: a `team` arm whose
        # scouts never actually agree on a split looks identical to one whose
        # negotiation worked.
        "n_contested": len(contested),
        "n_negotiation_failures": negotiation_failures,
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)
