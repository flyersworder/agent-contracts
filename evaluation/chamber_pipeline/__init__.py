"""Causal Chamber evaluation pipeline.

This pipeline implements the chamber pillar of the AAMAS / ECAI 2027
mainstream-venue extension. See `docs/causal_chamber_validation_plan.md`
for the full design and `docs/causal_chamber_M1_decisions.md` §3 for the
decision to keep scoring functions in this pipeline (rather than in a
new top-level `validators/` submodule).

Modules:
    scoring: SHD, F1, CI-coverage scoring functions for ground-truth-based
        causal-discovery evaluation. Pure functions; no framework state.
    inference: PC algorithm wrapper (via `causal-learn`) shared by Random,
        GreedyIG-lite, and LLM+PC variants per plan §5.
    llm_planner: Prompt builders + response parsers for the LLM-bearing
        agents. Pure functions; no chamber or network dependencies.
    agents: Five baseline variants from plan §5.1 — random_agent (M3a),
        greedy_ig_lite_agent (M3a), llm_only_agent (M3b), llm_pc_agent
        (M3b), planner_reasoner_agents (M3c).

Future modules (M3+):
    orchestrator:  one experiment cell end-to-end
    run_experiment: CLI entry point for the §6.1 sweep
    analyze_results: aggregation + Pareto figure generation
"""

from .agents import (
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    planner_reasoner_agents,
    random_agent,
)
from .inference import (
    CAUSAL_LEARN_AVAILABLE,
    cpdag_to_directed_adjacency,
    pool_experiment_data,
    run_pc,
)
from .llm_planner import (
    build_adjacency_prompt,
    build_select_prompt,
    parse_adjacency_response,
    parse_selection_response,
)
from .scoring import ci_coverage, f1_edges, shd

__all__ = [
    "CAUSAL_LEARN_AVAILABLE",
    "build_adjacency_prompt",
    "build_select_prompt",
    "ci_coverage",
    "cpdag_to_directed_adjacency",
    "f1_edges",
    "greedy_ig_lite_agent",
    "llm_only_agent",
    "llm_pc_agent",
    "parse_adjacency_response",
    "parse_selection_response",
    "planner_reasoner_agents",
    "pool_experiment_data",
    "random_agent",
    "run_pc",
    "shd",
]
