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

Agent calling convention (the contract M4's orchestrator dispatches against):
    Each agent is a callable `agent(adapter, **agent_kwargs) -> pd.DataFrame`
    that:
      - Reads its budget from `adapter.contract.resources.per_tool_limits["intervene"]`
      - Spends some prefix of that budget via `adapter.query_intervention(...)`
      - Returns a directed-adjacency DataFrame indexed by the chamber's
        ground-truth node names (`adapter.ground_truth().index`), with
        entries in {0, 1} and the diagonal forced to 0.
    Variant-specific kwargs (e.g., `model`, `llm`, `pc_alpha`,
    `planner_budget`) are documented per-agent. Agents may raise
    `NotImplementedError` when incompatible with a chamber (e.g.,
    GreedyIG-lite on WT — see plan §5.1 variant 2).

M4 modules (added at M4a):
    orchestrator: AgentSpec registry + per-cell runner (`run_cell`) +
        full-sweep runner (`run_sweep`). The orchestrator is the
        contract M4's CLI dispatches against — plan §5.1 chamber
        compatibility lives in the registry, not in callers.
    results: RunRecord dataclass + Parquet/CSV writers. One record
        per cell of the §6.1 grid; never raises mid-sweep so a
        single bad cell doesn't lose the surrounding ones.
    run_experiment: argparse CLI. `--pilot` runs the M4 sweep,
        `--m5` runs the full M5 sweep, `--mock-llm` enables
        offline smoke testing without OpenRouter spend.

Future modules (M5+):
    analyze_results: aggregation + §5.3 Pareto figure generation
"""

from .agents import (
    fan_in_agents,
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    planner_reasoner_agents,
    random_agent,
    team_agents,
    uncontracted_agent,
)
from .inference import (
    CAUSAL_LEARN_AVAILABLE,
    cpdag_to_directed_adjacency,
    pool_experiment_data,
    run_pc,
)
from .llm_planner import (
    build_adjacency_prompt,
    build_planner_select_prompt,
    build_reasoner_select_prompt,
    build_select_prompt,
    parse_adjacency_response,
    parse_selection_response,
)
from .orchestrator import (
    AGENT_REGISTRY,
    MENU_SIZES,
    AgentSpec,
    SweepSpec,
    count_cells,
    get_spec,
    iter_sweep_cells,
    run_cell,
    run_sweep,
)
from .results import (
    RunRecord,
    RunStatus,
    write_records_csv,
    write_records_parquet,
)
from .scoring import ci_coverage, f1_edges, shd

__all__ = [
    "AGENT_REGISTRY",
    "CAUSAL_LEARN_AVAILABLE",
    "MENU_SIZES",
    "AgentSpec",
    "RunRecord",
    "RunStatus",
    "SweepSpec",
    "build_adjacency_prompt",
    "build_planner_select_prompt",
    "build_reasoner_select_prompt",
    "build_select_prompt",
    "ci_coverage",
    "count_cells",
    "cpdag_to_directed_adjacency",
    "f1_edges",
    "fan_in_agents",
    "get_spec",
    "greedy_ig_lite_agent",
    "iter_sweep_cells",
    "llm_only_agent",
    "llm_pc_agent",
    "parse_adjacency_response",
    "parse_selection_response",
    "planner_reasoner_agents",
    "pool_experiment_data",
    "random_agent",
    "run_cell",
    "run_pc",
    "run_sweep",
    "shd",
    "team_agents",
    "uncontracted_agent",
    "write_records_csv",
    "write_records_parquet",
]
