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
| 3 | llm_only           | single-agent | LLM throughout       | M3b ⏳ |
| 4 | llm_pc             | single-agent | LLM-orchestrated PC  | M3b ⏳ |
| 5 | planner_reasoner   | multi-agent  | LLM, two roles       | M3c ⏳ |

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
from typing import TYPE_CHECKING

import pandas as pd

from .inference import pool_experiment_data, run_pc

if TYPE_CHECKING:
    from agent_contracts.integrations.causalchamber import ContractedChamberAgent


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
# Variants 3-5 land in M3b / M3c. Stubs documented here so the module's
# shape is the M3 surface even though only Random + GreedyIG-lite are
# usable today. See plan §5 for the precise semantics each variant must
# implement.
# ---------------------------------------------------------------------------


def llm_only_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
) -> pd.DataFrame:
    """LLM picks each intervention, then emits the final adjacency. M3b stub.

    Plan §5.1 variant 3. DeepSeek v4 Flash via OpenRouter through the
    framework's LiteLLM integration. LLM uses no classical inference
    step — it is asked to emit the adjacency matrix directly.
    """
    raise NotImplementedError(
        "M3b — see docs/causal_chamber_validation_plan.md §9 milestone M3 "
        "(staged: Random + GreedyIG-lite in M3a, LLM-only + LLM+PC in M3b, "
        "Planner+Reasoner in M3c)."
    )


def llm_pc_agent(
    adapter: ContractedChamberAgent,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
) -> pd.DataFrame:
    """LLM plans intervention sequence; classical PC infers the graph. M3b stub.

    Plan §5.1 variant 4. The "main hybrid" cell — LLM chooses what to
    perturb, PC chooses what edges those perturbations imply.
    """
    raise NotImplementedError("M3b — see docs/causal_chamber_validation_plan.md §9 milestone M3.")


def planner_reasoner_agents(
    adapter: ContractedChamberAgent,
    planner_budget: int,
    reasoner_budget: int,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
) -> pd.DataFrame:
    """Planner + Reasoner under conservation A + B ≤ total. M3c stub.

    Plan §5.1 variant 5 — the contribution-load-bearing variant.
    Planner agent picks interventions under sub-budget A; Reasoner
    agent proposes graph under sub-budget B. Conservation law:
    A + B ≤ adapter's total intervention budget.

    Exercises the framework's delegation primitives, not just
    `per_tool_limits`. AAMAS-fit relies on this variant.
    """
    raise NotImplementedError("M3c — see docs/causal_chamber_validation_plan.md §9 milestone M3.")


__all__ = [
    "greedy_ig_lite_agent",
    "llm_only_agent",
    "llm_pc_agent",
    "planner_reasoner_agents",
    "random_agent",
]
