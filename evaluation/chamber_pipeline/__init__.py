"""Causal Chamber evaluation pipeline.

This pipeline implements the chamber pillar of the AAMAS / ECAI 2027
mainstream-venue extension. See `docs/causal_chamber_validation_plan.md`
for the full design and `docs/causal_chamber_M1_decisions.md` §3 for the
decision to keep scoring functions in this pipeline (rather than in a
new top-level `validators/` submodule).

Modules:
    scoring: SHD, F1, CI-coverage scoring functions for ground-truth-based
        causal-discovery evaluation. Pure functions; no framework state.

Future modules (M3+):
    agents:        Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner
    orchestrator:  one experiment cell end-to-end
    run_experiment: CLI entry point for the §6.1 sweep
    analyze_results: aggregation + Pareto figure generation
"""

from .scoring import ci_coverage, f1_edges, shd

__all__ = [
    "ci_coverage",
    "f1_edges",
    "shd",
]
