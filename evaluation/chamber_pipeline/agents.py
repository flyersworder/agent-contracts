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

import contextlib
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
from .menu_taxonomy import coverage_ordered, partition_pools_by_variable
from .wt_menu_taxonomy import coverage_ordered as wt_coverage_ordered

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

# Sampling temperature. `None` means "send no `temperature` field", i.e. the
# provider's default -- which is what every arm did up to 2026-08-30 and is
# therefore what all recorded data was produced under.
#
# It is a REAL source of variance, not a nicety. `llm_pc` and `team` ran
# unpinned through M6 and M7, so the cell seed governs only the fallback RNG and
# PC, never the model: the same seed and config has produced F1 0.330 and 0.482.
# The consequence showed up at the level of ARM MEANS, not just cells -- three
# independent n>=10 estimates of the same `team` - `llm_pc` contrast span -0.023
# to -0.048.
#
# The default stays `None` deliberately. Flipping it silently would make every
# new cell incomparable with 2,000+ recorded ones while every column still
# matched. Pass `--temperature` to pin it, and read `temperature` on the
# RunRecord to know what a cell actually ran under.
#
# NOT applied to the scout roles: `_SCOUT_TEMPERATURE` exists to stop two
# identically-prompted scouts returning the same claim list, and pinning them to
# a shared low value would reintroduce exactly that degeneracy.
_DEFAULT_TEMPERATURE: float | None = None


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
# Coverage-manipulation controls (M7 Phase 1 follow-up). Neither uses an LLM.
#
# Phase 1 found `team` buys 23.4 distinct VARIABLES against the loop's 27.9 at
# an identical 30 experiments, because two scouts unknowingly buy the same
# variable at different strengths while `overlap_frac` reads 0.0 by
# construction. What it could NOT settle is whether that deficit matters: the
# loop's own F1 is flat in its variable count, but over a range of only 25-30
# with n=10, and team's 23.4 sits below that range.
#
# These two arms replace the extrapolation with a direct manipulation. At LT
# k=30 they span 11 to 30 distinct variables -- the full achievable range,
# bracketing both the loop and team -- at an identical budget, identical PC
# settings and no LLM in the loop to add variance. If F1 tracks coverage across
# that span, team's deficit is redundancy after all; if it does not, the
# coordination cost is real and the ladder needs a different instrument.
#
# `seed` is forwarded to `run_pc` as `llm_pc` and `team` do, NOT withheld as
# `random_agent` does -- these are compared against the ladder rungs, so they
# must draw their PC subsample the same way those do.
# ---------------------------------------------------------------------------


def _coverage_agent(
    adapter: ContractedChamberAgent,
    seed: int,
    pc_alpha: float,
    *,
    maximize: bool,
    exclude_strengths: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Shared body: spend the whole budget on a coverage-ordered selection."""
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = adapter.available_experiments()
    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)
    chosen = coverage_ordered(
        list(menu),
        min(budget, len(menu)),
        seed,
        maximize=maximize,
        exclude_strengths=exclude_strengths,
    )
    dfs = [adapter.query_intervention(name) for name in chosen]
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def _wt_coverage_agent(
    adapter: ContractedChamberAgent,
    seed: int,
    pc_alpha: float,
    *,
    maximize: bool,
) -> pd.DataFrame:
    """WT twin of :func:`_coverage_agent`, using the wind-tunnel parse.

    Separate function rather than a branch inside `_coverage_agent` because
    the two parses need different inputs: LT splits on a strength suffix and
    needs only the name, WT resolves a longest node-name prefix and needs the
    node list. Folding them together would mean passing an unused argument on
    one path and silently doing nothing on the other.

    There is no `_ms` variant here: WT entries carry no intervention strength,
    so the strength confound the LT pair was built to close does not exist.
    """
    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = adapter.available_experiments()
    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)
    chosen = wt_coverage_ordered(
        list(menu),
        min(budget, len(menu)),
        seed,
        nodes,
        maximize=maximize,
    )
    dfs = [adapter.query_intervention(name) for name in chosen]
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def wt_coverage_max_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Spend `k` on as many DISTINCT WT variables as `k` allows.

    **Not the same portfolio as its LT namesake.** WT's 28 entries cover 21
    variables, and the only multi-entry ones are `hatch` (3), `load_in` (3)
    and `load_out` (4) -- exactly the three highest out-degree drivers in the
    ground truth (6, 8, 8 edges). Every other entry is a single-entry
    apparatus setting with out-degree 1. So maximising breadth here spends the
    budget on trivial settings and away from the real drivers, where on LT it
    traded intervention strength for breadth. Same rule, opposite portfolio.
    """
    return _wt_coverage_agent(adapter, seed, pc_alpha, maximize=True)


def wt_coverage_min_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Spend `k` on as FEW distinct WT variables as `k` allows.

    Fattest-first, which on WT means `load_out`, `load_in` and `hatch` -- the
    three real drivers.

    **The prediction written here before the run was WRONG and is kept as a
    failed pre-registration.** It read: "expected to do WELL, the reverse of
    the LT case", reasoning that concentrating budget on out-degree 6/8/8
    drivers should beat spreading it over out-degree-1 settings. Measured
    (n=50): 0.124 / 0.165 / 0.229 at k=7/14/21, against `wt_coverage_max`'s
    0.188 / 0.232 / 0.282. Breadth wins on WT too, and by a wide margin.

    The reason is that buying a driver's several menu entries makes that ONE
    variable vary several times -- redundant, in exactly the sense the M7
    mechanism result measures -- while breadth activates a new source each
    time. Out-degree is not what the budget buys; a distinct varying variable
    is. That the same conclusion survives a menu whose fat entries are the
    real drivers, rather than LT's intervention strengths, is the stronger
    form of the coverage finding.
    """
    return _wt_coverage_agent(adapter, seed, pc_alpha, maximize=False)


def coverage_max_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Spend `k` on as many DISTINCT variables as `k` allows — the upper bound.

    One entry from every variable before a second from any, so at LT k=30 this
    touches all 30 variables. Not an agent anyone would deploy: it is the
    high-coverage end of a manipulation, and its only job is to sit opposite
    :func:`coverage_min_agent` at the same budget.
    """
    return _coverage_agent(adapter, seed, pc_alpha, maximize=True)


def coverage_min_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Spend `k` on as FEW distinct variables as `k` allows — the lower bound.

    Fattest variables first, each exhausted before the next, so at LT k=30 this
    touches 11 variables: nine at all three strengths, then one pair and one
    single. Deliberately the worst portfolio the menu permits at that budget.
    """
    return _coverage_agent(adapter, seed, pc_alpha, maximize=False)


def coverage_max_ms_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """`coverage_max` restricted to mid+strong — the DECONFOUNDED upper end.

    The unrestricted pair varies breadth and intervention STRENGTH together
    (see `coverage_ordered`), so it cannot say which one moved F1. Dropping
    `weak` leaves 50 entries over the same 30 variables and closes the strength
    channel: this arm and :func:`coverage_min_ms_agent` both buy zero weak
    interventions.
    """
    return _coverage_agent(adapter, seed, pc_alpha, maximize=True, exclude_strengths=("weak",))


def coverage_min_ms_agent(
    adapter: ContractedChamberAgent,
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """`coverage_min` restricted to mid+strong — the DECONFOUNDED lower end.

    At LT k=30 this touches 15 variables (the 15 fattest, each at both mid and
    strong) against the max arm's 30, so the span is 15-30 rather than the
    confounded 11-30, and neither end buys a weak intervention.
    """
    return _coverage_agent(adapter, seed, pc_alpha, maximize=False, exclude_strengths=("weak",))


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


def _says_stop(response: Any, stop_token: str) -> bool:
    """True iff the agent's LAST non-empty line is the stop token alone.

    Deliberately strict, and the strictness is the point: a false stop is
    unrecoverable. The run simply ends with less data than the agent wanted
    and no error is raised, so the cell silently scores a shorter experiment
    than the one we meant to run.

    A word-boundary search over the whole response is too loose -- it fires
    on "not done yet", which is an agent asking to CONTINUE. Matching the
    final line against the token alone (bare trailing punctuation allowed)
    matches what the prompt asks for and rejects prose.
    """
    from evaluation.chamber_pipeline.llm_planner import _response_text

    text = _response_text(response)
    if not text:
        return False
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    return lines[-1].strip(" .!*`'\"").upper() == stop_token.upper()


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
    stop_token: str | None = None,
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

        # Checked AFTER the parse (a named experiment wins over a stop
        # token) but BEFORE the fallback below. `DONE` is not a menu name,
        # so the parse returns None for it, and the fallback would convert
        # the agent's decision to stop into a random PURCHASE -- turning a
        # self-terminating agent into one that always spends the safety cap,
        # silently.
        if name is None and stop_token is not None and _says_stop(response, stop_token):
            break

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
    temperature: float | None = _DEFAULT_TEMPERATURE,
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
    _chosen, dfs = _llm_select_loop(adapter, llm, model, seed, temperature=temperature)

    if not dfs:
        return _empty_adjacency(nodes)

    pooled = pool_experiment_data(dfs, nodes)
    return run_pc(pooled, nodes, alpha=pc_alpha, seed=seed)


def uncontracted_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """`llm_pc` with the contract removed: the agent decides when to stop.

    The UNCONTRACTED half of the framework's central comparison, and the one
    thing the chamber pillar was missing -- every other arm in the registry
    is contracted, so nothing measured what governance costs or buys.

    Identical to `llm_pc_agent` in mechanism (same iterative loop, same
    fallback handling, same PC afterwards). Two differences, both essential
    and neither cosmetic:

    1. No budget is stated in the prompt, and the agent may answer
       `DONE` instead of naming an experiment.
    2. The adapter is built with `intervention_budget = len(menu)` rather
       than `k`. That cap is a PHYSICAL limit -- there are only so many
       distinct experiments on the menu -- not a governance bound. It exists
       so a non-terminating agent cannot loop forever.

    Because the cap can still bind, the arm records whether it did. An agent
    that runs the whole menu because it never said `DONE` is a different
    finding from one that chose to run the whole menu, and the two are
    indistinguishable from the experiment count alone.
    """
    from evaluation.chamber_pipeline.llm_planner import (
        UNCONTRACTED_STOP_TOKEN,
        build_uncontracted_select_prompt,
    )

    nodes = _node_names(adapter)
    menu = list(adapter.available_experiments())
    if _intervention_budget(adapter) <= 0 or not menu:
        adapter.coordination_stats = {"n_experiments_distinct": 0}
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()
    chosen, dfs = _llm_select_loop(
        adapter,
        llm,
        model,
        seed,
        prompt_builder=build_uncontracted_select_prompt,
        stop_token=UNCONTRACTED_STOP_TOKEN,
    )

    adapter.coordination_stats = {
        "n_experiments_distinct": len(chosen),
        # The safety cap bound iff the agent never volunteered a stop.
        "agg_hit_safety_stop": int(len(chosen) >= len(menu)),
    }

    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


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

    The word boundaries are the WHOLE defence, and there is deliberately no
    second one. A `not any(name in longer for longer in claimed)` guard used
    to follow this loop, for the same prefix threat -- but the regex already
    stops `_1` matching inside `_10`, so the guard had no true positives left
    and every firing removed a real claim. Verified against the live menus:
    WT names three droppable pairs (`validate_load_in`, `validate_load_out`,
    `validate_osr_in`), LT none. Two layers against one bug means the second
    one is only ever wrong.
    """
    from evaluation.chamber_pipeline.llm_planner import _response_text

    text = _response_text(response) or ""
    claimed: list[str] = []
    for name in sorted(menu, key=len, reverse=True):
        if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text):
            continue
        claimed.append(name)
    return [n for n in menu if n in claimed]


def _substring_shadowed(claimed: list[str]) -> int:
    """How many claimed names a substring guard would have discarded.

    Exactly the rule removed on 2026-08-29: `any(name in longer for longer in
    claimed)`. Kept as a COUNTER rather than a filter so the incidence of the
    old defect is measurable on the arm it affected, instead of argued from
    menu structure. WT names three shadowed pairs of 28 (`validate_load_in`,
    `validate_load_out`, `validate_osr_in`); LT names none, so this is
    structurally 0 on LT.
    """
    return sum(1 for n in claimed if any(n != other and n in other for other in claimed))


def _capped_claim(names: list[str], budget: int, stream: str, seed: int) -> list[str]:
    """At most `budget` of `names`, chosen without menu-order bias.

    A scout that reasons out loud over most of the menu claims more than it
    can spend, so the claim has to be cut to size. `names` arrives in MENU
    order -- `_parse_name_list` returns it that way -- and the menu is grouped
    by variable family and intervention strength, so a plain `[:budget]` slice
    keeps the head families and drops the tail ones, identically in every
    seed. That is the same defect the `rest` shuffle below exists to fix,
    where parity-slicing handed scout_a `0 of 3 osr_c and 0 of 2 red`
    experiments on LT.

    It matters most exactly where the effect is measured: the claim is ~10% of
    scout_a's pool at LT k=6 and ~77% at k=45, because a full claim leaves no
    top-up and only half the shuffled leftover as extra freedom.

    A seeded shuffle, NOT the scout's stated order. Preference order would be
    more faithful to the negotiation, but `_parse_name_list` scans the whole
    response, and a revise reply restates the peer's proposals before making
    its own -- so preference-order truncation could keep the PEER's names.
    Fixing that needs answer/restatement separation (spec §11); until then an
    unbiased subset is the honest cut.

    The RNG is keyed by a string naming the stream, so scout_a's and scout_b's
    permutations are independent of each other and of the `rest` shuffle,
    which draws from `_random.Random(seed)`.
    """
    if len(names) <= budget:
        return list(names)
    shuffled = list(names)
    _random.Random(f"team-claim-{stream}:{seed}").shuffle(shuffled)
    # Re-sorted into menu order after the draw: WHICH names survive is now
    # unbiased, and the order they are handed on in stays the deterministic
    # one every other list in this module uses.
    keep = set(shuffled[:budget])
    return [n for n in names if n in keep]


def _maybe_node(adapter: ContractedChamberAgent, name: str) -> Any:
    """`adapter.as_node(name)` when a delegation graph exists, else a no-op.

    Node routing requires a sealed `DelegationGraph`, which `run_cell` builds
    only for arms with measured per-role token costs -- `_ladder_calibration`
    raises rather than extrapolate one. An arm whose accuracy we want before
    its cost is calibrated would otherwise be unrunnable, so the routing
    degrades instead of blocking: the adapter's aggregate monitor still gates
    the intervention budget, and only the per-node token accounting is lost.
    Such a cell is simply not conservation-certified, which `run_cell` already
    records as None rather than as a pass.
    """
    if getattr(adapter, "delegation_graph", None) is None:
        return contextlib.nullcontext()
    return adapter.as_node(name)


def _resolve_batch_selection(
    named: list[str], menu: list[str], budget: int, seed: int, stream: str
) -> tuple[list[str], int, int]:
    """Turn a free-form batch answer into exactly `budget` distinct names.

    A batch answer is unconstrained in a way a one-at-a-time answer is not:
    the model may name more than the budget, fewer, or names it was not
    offered. All three have to resolve to exactly `budget` picks or the arm is
    not budget-comparable with the loop and the ladder stops being a
    controlled comparison.

    Over-long is cut with `_capped_claim`, i.e. a seeded shuffle rather than a
    menu-order slice -- the menu is grouped by variable family, so a slice
    keeps the head families in every seed. Short is topped up from the
    remaining menu on the same seeded shuffle. Both are counted, because a
    silent top-up would let a model that answered with one name score as a
    full-budget arm.

    Returns `(chosen, n_over, n_short)`.
    """
    offered = [n for n in named if n in set(menu)]
    n_over = max(0, len(offered) - budget)
    chosen = _capped_claim(offered, budget, stream, seed)
    n_short = max(0, budget - len(chosen))
    if n_short:
        rest = [m for m in menu if m not in set(chosen)]
        _random.Random(f"batch-topup-{stream}:{seed}").shuffle(rest)
        chosen = chosen + rest[:n_short]
    return [m for m in menu if m in set(chosen)], n_over, n_short


def one_shot_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Pick the whole budget in ONE call, then infer. The no-history control.

    Rung 0 spends `k` calls, each conditioned on everything picked so far.
    This spends one. Every multi-agent rung on the ladder SPLITS that running
    record between agents without anything establishing what an unsplit record
    is worth -- so without this arm the ladder measures the cost of dividing a
    resource whose value was never priced.
    """
    from evaluation.chamber_pipeline.llm_planner import build_batch_select_prompt

    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = list(adapter.available_experiments())
    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()
    budget = min(budget, len(menu))
    response = llm(
        model=model,
        messages=build_batch_select_prompt(menu, budget),
        max_tokens=_SELECTION_MAX_TOKENS,
        extra_body={"reasoning": {"effort": _SELECTION_REASONING_EFFORT}},
    )
    chosen, n_over, n_short = _resolve_batch_selection(
        _parse_name_list(response, menu), menu, budget, seed, "one_shot"
    )
    dfs = [adapter.query_intervention(name) for name in chosen]
    adapter.coordination_stats = {
        "n_experiments_distinct": len(set(chosen)),
        "n_batch_over_budget": n_over,
        "n_batch_topped_up": n_short,
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def shared_blackboard_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Two voices alternating over ONE complete shared record — the axis's top.

    The ladder orders its rungs by how much of the loop's running record
    survives the partition. Every multi-agent rung until now destroys some of
    it: the fan-in rungs split it in half and blind each side, `team` splits it
    by negotiated agreement, the relay leaves one seam. This arm removes the
    partition entirely while keeping two agents — they alternate turns, they
    draw from the SAME undivided menu, and each one sees every pick either has
    made.

    **It should collapse onto the loop, and that is the point.** Two agents
    alternating with a complete shared history IS the loop with two voices, so
    the pre-registered prediction is `shared_blackboard` ~ `llm_pc`. If it does
    NOT collapse, the record axis is wrong: the cost would then be in having
    several agents at all rather than in partitioning what they know, and the
    paper's reframing fails. That is the most informative failure available to
    this plan, which is why the arm is worth its cost even though its expected
    result is "no difference".

    The two voices are the coverage-seeking and disambiguation-seeking framings
    already used by `fan_in_spec` (rung 2). Reusing them is deliberate: rung 2
    gives those same two roles a SPLIT record and loses, so running them here
    over a SHARED record isolates the record from the roles. Any gap between
    the two arms is attributable to the partition and to nothing else.

    Implementation is one `_llm_select_loop` call per pick, with the running
    record threaded through `starting_chosen` — which both renders it into the
    prompt's already-chosen block and removes those names from the selectable
    menu. No new selection machinery, so the arm inherits the loop's tested
    truncation, fallback and accounting behaviour.

    **The board holds picks, not prose.** Neither voice can write a rationale,
    an intention or a note to the other; the shared state is *what was done*,
    never *what anyone thinks*. That is narrower than "blackboard" means in the
    classical sense (Hearsay-II and successors), and the paper has to say so —
    see spec §4. It is deliberate: the axis is how much of the loop's record
    survives, the loop's record IS a list of picks, and sharing more would vary
    two things at once.

    A rationale-passing variant is the natural next arm and is specified in
    spec §7: same topology, each pick carrying one line of reasoning the next
    voice reads. It must be compared against THIS arm rather than against the
    loop, or it varies two things and resolves nothing.
    """
    from evaluation.chamber_pipeline.llm_planner import (
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )

    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = adapter.available_experiments()
    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)
    llm = llm or _default_llm()

    voices = (
        ("voice_a", build_scout_broad_prompt),
        ("voice_b", build_scout_targeted_prompt),
    )
    record: list[str] = []
    dfs: list[pd.DataFrame] = []
    for step in range(min(budget, len(menu))):
        node, builder = voices[step % 2]
        with _maybe_node(adapter, node):
            # A distinct seed per step: `seed` drives only the fallback RNG,
            # and reusing one value would correlate every fallback pick across
            # the whole record. Offset by `step`, not by `step + 1`, so step 0
            # reproduces a plain loop's first call exactly.
            picked, frames = _llm_select_loop(
                adapter,
                llm,
                model,
                seed * 1000 + step,
                spend=1,
                starting_chosen=record,
                prompt_builder=builder,
            )
        if not picked:
            # The menu is exhausted, or the adapter refused the purchase. Either
            # way another turn cannot help, and looping would burn the rest of
            # the budget on calls that buy nothing.
            break
        record.extend(picked)
        dfs.extend(frames)

    adapter.coordination_stats = {
        "overlap_frac": None,  # no partition exists, so there is nothing to overlap
        "n_experiments_distinct": len(set(record)),
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def critique_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Executor-evaluator: propose a set, have it reviewed, revise, then infer.

    The pattern reviewers name most often, and the only shape on the ladder
    where a second agent does not take a share of the budget. Every other
    multi-agent rung DIVIDES the work; this one leaves the proposer holding
    the whole budget and adds an opinion about it.

    Three calls regardless of `k`, so it is also the cheapest multi-agent arm
    by a wide margin -- worth reporting on the cost axis whatever it does to
    accuracy.

    With no feedback available, the critic judges the SET from names alone:
    what is over-covered, what is untouched, which swaps would help. It
    advises and does not decide -- the proposer emits the final list, which is
    what separates this from the team arm's negotiation, where both sides
    hold budget.
    """
    from evaluation.chamber_pipeline.llm_planner import (
        _response_text,
        build_batch_select_prompt,
        build_critique_prompt,
        build_revise_after_critique_prompt,
    )

    nodes = _node_names(adapter)
    budget = _intervention_budget(adapter)
    menu = list(adapter.available_experiments())
    if budget <= 0 or not menu:
        return _empty_adjacency(nodes)

    llm = llm or _default_llm()
    budget = min(budget, len(menu))
    extra: dict[str, Any] = {"extra_body": {"reasoning": {"effort": _SELECTION_REASONING_EFFORT}}}

    with _maybe_node(adapter, "proposer"):
        first = llm(
            model=model,
            messages=build_batch_select_prompt(menu, budget),
            max_tokens=_SELECTION_MAX_TOKENS,
            **extra,
        )
    proposed, over_1, short_1 = _resolve_batch_selection(
        _parse_name_list(first, menu), menu, budget, seed, "propose"
    )

    with _maybe_node(adapter, "critic"):
        review = llm(
            model=model,
            messages=build_critique_prompt(menu, budget, proposed),
            max_tokens=_RECONCILE_MAX_TOKENS,
            extra_body={"reasoning": {"effort": _COORDINATION_REASONING_EFFORT}},
        )
    critique_text = _response_text(review) or ""

    with _maybe_node(adapter, "proposer"):
        final = llm(
            model=model,
            messages=build_revise_after_critique_prompt(menu, budget, proposed, critique_text),
            max_tokens=_SELECTION_MAX_TOKENS,
            **extra,
        )
    # Tested BEFORE resolution, not after. `_resolve_batch_selection` tops a
    # short answer up to the full budget, so `revised` is never empty and a
    # post-hoc `revised or proposed` fallback can never fire -- an unreadable
    # revise would silently become a random basket, scoring the arm on a
    # top-up rather than on its proposal. A review nobody could read leaves
    # the plan standing.
    named_revised = _parse_name_list(final, menu)
    if named_revised:
        revised, over_2, short_2 = _resolve_batch_selection(
            named_revised, menu, budget, seed, "revise"
        )
        chosen, revise_unusable = revised, 0
    else:
        chosen, over_2, short_2, revise_unusable = proposed, 0, 0, 1

    dfs = [adapter.query_intervention(name) for name in chosen]
    adapter.coordination_stats = {
        "n_experiments_distinct": len(set(chosen)),
        # How much the review actually moved the set. 0 means the critic was
        # decorative, which is a result and must not be mistaken for one.
        "n_critique_changed": len(set(chosen) ^ set(proposed)) // 2,
        "n_critique_empty": int(not critique_text.strip()),
        # The revise answer named nothing on the menu, so the proposal stands.
        "n_revise_unusable": revise_unusable,
        "n_batch_over_budget": over_1 + over_2,
        "n_batch_topped_up": short_1 + short_2,
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def team_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    scout_a_budget: int,
    scout_b_budget: int,
    llm: LLMCallable | None = None,
    partition: str = "experiment",
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
    # See `_capped_claim` for why the cut is a seeded shuffle and not a slice.
    uncapped_a = list(source_a)
    claim_a = _capped_claim(uncapped_a, scout_a_budget, "a", seed)
    # Measured against the list the cap was actually applied to. An earlier
    # version excluded `source_a[:scout_a_budget]` -- a MENU-ORDER slice --
    # while the cap ran on the SHUFFLED `claim_a`. Same size, different
    # membership, so the two base lists could differ in length and the counter
    # over-reported: with source_a=[p,x], source_b=[x,y,z] and both budgets 1,
    # it read 2 truncated where 1 had been. A counter lying by being measured
    # against the wrong list is the defect this whole commit series is about.
    uncapped_b = [n for n in source_b if n not in set(claim_a)]
    claim_b = _capped_claim(uncapped_b, scout_b_budget, "b", seed)
    n_claim_truncated = max(0, len(uncapped_a) - len(claim_a)) + max(
        0, len(uncapped_b) - len(claim_b)
    )
    # Measured on the lists actually used, before the cap: this is the
    # incidence of the defect removed with the substring guard, not its
    # effect. Non-zero means the old code silently dropped a real claim here.
    n_substring_conflicts = _substring_shadowed(list(source_a)) + _substring_shadowed(
        list(source_b)
    )

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
    if partition == "variable":
        # Same negotiation, same A-wins-ties rule, same budgets; only the
        # GRANULARITY of the split changes. See
        # `partition_pools_by_variable` for why, and `menu_taxonomy` for the
        # null model that says the name-level split protects nothing.
        pool_a, pool_b = partition_pools_by_variable(
            menu, claim_a, claim_b, scout_a_budget, scout_b_budget, seed
        )
    elif partition == "experiment":
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
    else:
        raise ValueError(f"partition must be 'experiment' or 'variable', got {partition!r}")

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
        # How many claimed names the budget cap discarded. Zero means the
        # scouts claimed within budget and the cap never ran; a large number
        # means the cap, not the negotiation, decided most of the split.
        "n_claim_truncated": n_claim_truncated,
        "n_substring_conflicts": n_substring_conflicts,
        # The larger scout's claim as a share of what it could see. The cap
        # only matters in proportion to this: at 0.1 the shuffled leftover
        # dominates the pool, at 0.77 the claim does.
        "claim_pool_share": max(
            len(claim_a) / len(pool_a) if pool_a else 0.0,
            len(claim_b) / len(pool_b) if pool_b else 0.0,
        ),
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)


def team_varsplit_agents(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    scout_a_budget: int,
    scout_b_budget: int,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """`team`, partitioned by VARIABLE instead of by experiment name.

    The one-change control for M7's mechanism finding. `team` loses ~0.048 F1
    to the loop, and ~two-thirds of that is traced to buying 23.4 distinct
    variables against the loop's 27.9 -- because its two pools are disjoint as
    sets of EXPERIMENTS while a variable can sit in both at different
    strengths. Measured, the cross-scout duplication is at chance: 5.6
    observed against a random-split null of 4.11 +- 1.51.

    This arm keeps the topology, the budgets, the four negotiation calls and
    the A-wins-ties rule, and changes only what gets partitioned. If the
    diagnosis holds it should recover most of the variable deficit and roughly
    two-thirds of the F1 gap; if it does not, the cost is coordination itself
    and the redundancy account is wrong.

    `shared_variables` is 0 by construction here, exactly as `overlap_frac` is
    0 by construction in both arms -- so neither is evidence of anything, and
    the arm has to be judged on F1 and on distinct-variable count.

    **This is not a free win, and the outcome is genuinely open.** Putting
    every entry of a variable in one pool removes cross-scout duplication and
    concentrates within-scout duplication: a pool of ~29 entries now spans only
    ~15 variables, so a scout must pick almost exactly one entry per variable
    to use its budget well. Measured against `team` under `--mock-llm`, where
    selection degrades to seeded random, the two effects cancel almost exactly:

    | | shared vars | per-scout distinct | total distinct |
    |---|---|---|---|
    | `team` | 3.83 | 12.5 / 12.3 | 21.0 |
    | `team_varsplit` | **0.00** | 9.7 / 10.8 | 20.5 |

    So the arm pays off only if the scouts are good at avoiding SELF-repetition
    -- and the real ones are (0.8 and 0.2 repeats over 15 picks). If they hold
    that behaviour inside the narrower pools, total distinct variables should
    approach 29 against `team`'s 23.4. If they do not, this arm converts one
    duplication problem into another and lands no better. The random-selection
    control above is what separates those two outcomes.
    """
    return team_agents(
        adapter,
        model,
        seed,
        pc_alpha,
        scout_a_budget=scout_a_budget,
        scout_b_budget=scout_b_budget,
        llm=llm,
        partition="variable",
    )


# Declared at the END of the module, after every agent it names. The previous
# copy sat mid-file and listed only the five M4b arms, so the three ladder arms
# defined below it were absent from `import *` and from the package's
# re-exports -- a public surface that disagreed with the registry the sweep
# actually runs.
__all__ = [
    "coverage_max_agent",
    "coverage_max_ms_agent",
    "coverage_min_agent",
    "coverage_min_ms_agent",
    "critique_agents",
    "fan_in_agents",
    "greedy_ig_lite_agent",
    "llm_only_agent",
    "llm_pc_agent",
    "one_shot_agent",
    "planner_reasoner_agents",
    "random_agent",
    "shared_blackboard_agents",
    "team_agents",
    "team_varsplit_agents",
    "uncontracted_agent",
]
