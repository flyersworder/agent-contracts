# Agent Contracts Project Memory

This document tracks development progress and key decisions for the Agent Contracts framework.

## Project Overview

**Agent Contracts** is a formal framework for governing autonomous AI agents through explicit resource constraints and temporal boundaries.

- **Repository**: https://github.com/flyersworder/agent-contracts
- **Started**: November 1, 2025
- **Primary Developer**: qingye
- **AI Assistant**: Claude (Sonnet 4.5)

## Current Status: Production-Ready ✅

**Phase 1**: Core Framework (Nov 1) ✅
**Phase 2A**: Strategic Optimization (Nov 3) ✅
**Phase 2B**: Governance & Validation (Nov 5) ✅
**LangGraph**: Premium Multi-Agent (Nov 6) ✅
**Google ADK**: Google AI Integration (Nov 6) ✅
**SkillSpec**: agentskills.io Standard (Dec 23) ✅
**Per-Tool Limits**: Fine-grained resource control (Dec 23) ✅
**Indeterminacy Evaluator**: NeurIPS 2025 LLM-as-Judge (Dec 23) ✅
**Evaluation Pipelines**: Research & Code Review experiments (Dec 23) ✅
**Claude Agent SDK**: Anthropic-native integration (Mar 27) ✅
**PyPI Release**: Published as `ai-agent-contracts` (Mar 26) ✅

**Causal Chamber pillar** (AAMAS 2027 / ECAI 2027 extension):
- **M1**: Adapter scaffolding (May 6) ✅
- **M2**: Adapter implementation + scoring (May 6) ✅
- **M3a/b/c**: 5 baseline agents (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner) (May 8) ✅
- **Per-tool conservation**: framework refactor for `ContractingCapability` (May 8) ✅
- **M4a/a.1**: Orchestrator + AgentSpec registry + CLI + analyzer (May 9) ✅
- **M4b smoke**: 45-cell LT smoke run (May 13) ✅ — exposed `llm_only` empty-graph bug
- **M4b post-smoke fix** (May 14, commit `ad96133`) ✅ — data-grounded `llm_only`
  via per-experiment per-node mean summary + `_ADJACENCY_MAX_TOKENS` 4096→32768.
  Verified k/M=1.00 single seed: SHD=27 (was 57), F1=0.76 (was 0), 54/57 edges
  recovered. **DeepSeek v4 Flash dominates the Pareto at high budget** when given
  the summary — rotates the §5.3 narrative (see plan).
- **M4b re-smoke + pilot**: Pending. Re-smoke = 45 cells (~2-3hr, ~$0.50) at
  3 seeds. Pilot = 450 cells (~24hr, ~$2) at 30 seeds. Both require
  `--cell-timeout-seconds 1800` (was 600) because the adjacency call now reasons
  for up to 10min at k/M=1.00.

**Metrics**:
- **Tests**: 1029 passing (chamber pillar adds ~250 tests on top of the framework's ~780)
- **Coverage**: 81%+
- **Integrations**: LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK, **Causal Chambers**

## Core Framework (Phase 1)

### Implementation

| Component | File | Purpose |
|-----------|------|---------|
| Contract | `core/contract.py` | Core data structures (C = I, O, S, R, T, Φ, Ψ) |
| Monitor | `core/monitor.py` | Real-time resource tracking |
| Tokens | `core/tokens.py` | Token counting & cost estimation |
| Enforcement | `core/enforcement.py` | Constraint enforcement & callbacks |
| LiteLLM | `integrations/litellm_wrapper.py` | 100+ LLM providers |

### Key Design Decisions

1. **Immutable Constraints**: Frozen dataclasses prevent accidental modification
2. **Event-Driven**: Callbacks for observability without tight coupling
3. **Strict vs Lenient**: Support both hard enforcement and soft monitoring
4. **Context Managers**: Pythonic `with` statement support
5. **Type Safety**: Strict mypy checking throughout

## Integrations

### LiteLLM (100+ Providers)
- Direct LLM calls with automatic tracking
- Streaming support
- Built-in token counting

### LangChain (Baseline)
**Value**: Governance & compliance for simple chains
- Multi-call budget protection
- Audit trails for compliance
- Policy enforcement
- **Limitation**: Cannot prevent single expensive call (tokens unknown until after API)

### LangGraph (Premium) ⭐
**Value**: Critical for complex multi-agent workflows
- Cycle/loop protection (prevents runaway costs)
- Multi-agent budget sharing
- Parallel execution governance
- **Use Case**: Validation loops, retry logic, multi-agent coordination

### Google ADK (Latest)
**Value**: Native Google AI integration
- Gemini model support
- Google AI Studio integration
- Vertex AI support

## Recent Additions (December 2025)

### SkillSpec (agentskills.io Standard)
**Value**: Industry-standard skill definitions for reusable agent behaviors
- Full compliance with agentskills.io open standard (Microsoft, OpenAI, Cursor, etc.)
- SKILL.md import/export (`to_skill_md()`, `from_skill_md()`)
- Progressive disclosure (metadata ~100 tokens, full instructions on activation)
- Name validation: 1-64 chars, lowercase alphanumeric + hyphens
- Backward compatible: `list[str | SkillSpec]` union type

**Files**:
- `core/contract.py` - `SkillSpec` dataclass (lines 395-612)
- `core/contract.py` - `Capabilities.skills` updated to accept union type
- Helper methods: `get_skill()`, `has_skill()`, `skill_names`, `skill_specs`

### Per-Tool Limits
**Value**: Fine-grained control over individual tool usage
- Individual limits per tool name: `per_tool_limits={"web_search": 5}`
- Aggregate limit still applies: `tool_invocations=20`
- Enforcement priority: per-tool checked first, then aggregate
- Helper methods: `can_use_tool()`, `get_remaining_tool_calls()`

**Files**:
- `core/contract.py` - `ResourceConstraints.per_tool_limits: dict[str, int]`
- `core/monitor.py` - `ResourceUsage.tool_usage_by_name: dict[str, int]`
- `core/monitor.py` - Per-tool limit checking in `check_constraints()`

### Indeterminacy-Aware LLM-as-Judge (Dec 23) ⭐
**Value**: Robust quality evaluation accounting for rating ambiguity

Implements the NeurIPS 2025 framework from "Validating LLM-as-a-Judge Systems
under Rating Indeterminacy" (Guerdan et al.). Standard LLM-as-judge approaches
can select suboptimal judges up to 31% worse than optimal when rating tasks
have inherent ambiguity.

**Key Concepts**:
- **Response Set Elicitation**: Ask judges "select ALL ranges that reasonably apply"
  instead of forcing single choice
- **Multi-label Vector (ω)**: P(option_k is reasonable) for each rating option
- **Indeterminacy Signal**: Judge disagreement indicates genuine ambiguity, not noise
- **MSE(srs/srs)**: Recommended metric (30% better than Hit Rate under ambiguity)

**Components**:
- `ResponseSet`: Set of options a judge deems reasonable
- `MultiLabelScore`: Probability vector + point estimate + indeterminacy level
- `IndeterminacyAwareScore`: Full score with response sets for all dimensions
- `IndeterminacyAwareEvaluator`: Main evaluator class

**Metrics**:
- `mse_srs_srs()`: MSE between soft response set vectors
- `decision_consistency()`: Agreement on downstream decisions at threshold τ
- `prevalence_bias()`: Systematic over/underestimation vs reference

**Files**:
- `benchmarks/research_agent/indeterminacy_evaluator.py` - Full implementation
- `tests/benchmarks/test_indeterminacy_evaluator.py` - 33 tests

**Reference**: https://github.com/lguerdan/indeterminacy

### Evaluation Pipelines (Dec 23)
**Value**: Systematic comparison of CONTRACTED vs UNCONTRACTED execution

Two complementary evaluation experiments demonstrating Agent Contracts' governance value:

**1. Research Pipeline** (`evaluation/research_pipeline/`)
- Multi-agent report generation (Researcher → Analyzer → Reporter)
- 25 curated research topics across 5 categories
- Conservation law enforcement for budget delegation
- Success criteria: sections complete, word count, citations

**2. Code Review Pipeline** (`evaluation/code_review_pipeline/`)
- Coder ↔ Reviewer iterative loop (Gemini 2.0 Flash)
- 175 LiveCodeBench problems (post-Feb 2025, contamination-free)
- Iteration limits prevent runaway agent loops
- Per-agent token and LLM call tracking

**Key Metrics Collected**:
- Total tokens consumed (contracted vs uncontracted)
- Iteration counts (runaway prevention)
- Success rates by difficulty
- Conservation law compliance

**Usage**:
```bash
# Research pipeline
python -m evaluation.research_pipeline.run_experiment --quick

# Code review pipeline
python -m evaluation.code_review_pipeline.run_experiment --n-problems 10
```

## Validation & Benchmarks

### Governance Validation (Nov 2)
**N=20 statistical validation**

✅ **What It Provides**:
- 100% budget enforcement (8/8 tests)
- 100% organizational policy compliance
- Quality improvement under constraints (77→86→95)
- High predictability (CV < 10%)

❌ **What It Doesn't**:
- Variance reduction (both agents already predictable at temp=0)
- Cost optimization (provides governance, not reduction)

**Value Proposition**: Organizational control over AI resources, not individual optimization

### Quality Framework (Nov 4, Enhanced Dec 23)
- **Original Evaluator**: Gemini 2.5 Flash, CV=5.2% (exceeds SOTA 10-15%)
- **Known Limitation**: Bimodal behavior at high quality (Q>90)
- **New**: `IndeterminacyAwareEvaluator` implements NeurIPS 2025 framework
  - Response set elicitation captures rating ambiguity
  - MSE(srs/srs) metric is 30% better than Hit Rate under indeterminacy
  - Judge disagreement treated as signal, not noise
- **Status**: Both evaluators production-ready

### Strategic Modes (Nov 3)
**H2 Hypothesis Validated**: Contract modes enable quality-cost-time tradeoffs

- **URGENT**: 87% quality, 50% faster
- **ECONOMICAL**: 81% quality, 32% fewer tokens
- **BALANCED**: 85% quality, balanced resources

**Pareto frontier confirmed** - no mode dominates another

## Key Learnings

### Technical
1. **Budget-awareness should be adaptive**: Only add cognitive overhead when budget tight (>70%)
2. **Complexity = Value**: More complex workflows → higher governance value (LangGraph > LangChain)
3. **Limitations can be strengths**: Focus on governance over single-call prevention

### Scientific Process
1. **Empirical validation critical**: N=3 showed 50% variance reduction, N=20 showed opposite
2. **Update beliefs based on evidence**: Changed positioning from "optimization" to "governance"
3. **User feedback invaluable**: "What's the real benefit?" forced honest assessment

### Integration Strategy
1. **LiteLLM**: Universal baseline
2. **LangChain**: Completeness (baseline feature)
3. **LangGraph**: Where real value is (premium feature)
4. **Google ADK**: Native Google integration

## Critical Bug Fixes

### LangChain Enforcement (Nov 6)
Four critical bugs fixed that made multi-call protection non-functional:

1. **Enforcer Never Used**: Created but never called
2. **Separate Monitors**: Enforcer tracked different monitor than wrapper
3. **Wrong Event Type**: Filtered for "violation", enforcer emitted "constraint_violated"
4. **State Management**: Contract marked FULFILLED after first call, blocked subsequent tracking

**Impact**: Multi-call protection now works correctly - Demo 3 stops after first violation

### Testing
- 325 tests passing (15 skipped - optional LangChain dependency)
- All enforcement tests updated to expect ACTIVE state (cumulative tracking)

## File Structure

```
agent-contracts/
├── src/agent_contracts/
│   ├── core/                    # Core framework
│   │   ├── contract.py          # Contract definitions
│   │   ├── monitor.py           # Resource monitoring
│   │   ├── tokens.py            # Token counting
│   │   ├── enforcement.py       # Enforcement
│   │   ├── wrapper.py           # Contract wrapper
│   │   ├── prompts.py           # Budget-aware prompts
│   │   └── planning.py          # Strategic planning
│   └── integrations/            # Framework integrations
│       ├── litellm_wrapper.py   # LiteLLM
│       ├── langchain.py         # LangChain
│       ├── langgraph.py         # LangGraph
│       └── google_adk.py        # Google ADK
├── tests/                       # 609+ tests, 91% coverage
├── benchmarks/                  # Live demonstrations
│   ├── langchain/              # LangChain demos
│   ├── langgraph/              # LangGraph demos
│   ├── google_adk/             # Google ADK demos
│   ├── governance/             # Governance validation
│   └── research_agent/         # Research agent demos
├── evaluation/                  # Experimental evaluations
│   ├── research_pipeline/      # Multi-agent research experiment
│   └── code_review_pipeline/   # Coder↔Reviewer experiment
└── docs/
    ├── whitepaper.md           # Theoretical foundation
    └── testing-strategy.md     # Test plan
```

## Development Infrastructure

### Build System
- **Package Manager**: uv
- **Python**: >=3.12
- **Build**: src-layout for isolation

### Code Quality
- **Pre-commit Hooks**: ruff (lint/format), mypy (type check), markdownlint
- **Testing**: pytest with coverage tracking
- **CI**: All checks must pass

## Next Steps

**Active: M4b chamber-pillar pilot sweep (post-fix)**
1. Re-smoke (~2-3hr, ~$0.50) — confirms post-fix Pareto monotone at 3 seeds:
   `uv run python -m evaluation.chamber_pipeline.run_experiment --chambers lt --budgets 0.10,0.50,1.00 --seeds 3 --cell-timeout-seconds 1800 --out runs/m4b-resmoke.parquet`
2. Full pilot (~24hr overnight, ~$2) — `--cell-timeout-seconds 1800` mandatory:
   `uv run python -m evaluation.chamber_pipeline.run_experiment --pilot --cell-timeout-seconds 1800 --out runs/m4-pilot.parquet`
3. Analyze: `uv run python -m evaluation.chamber_pipeline.analyze_results --input runs/m4-pilot.parquet --out-dir runs/m4-pilot-figs/ --check-m4-acceptance`

**M4c (deferred operational hardening, before M5)**
- Checkpointing / resume from partial Parquet
- Parallelism (`ThreadPoolExecutor` in `run_sweep`) — needs logger thread-safety + per-thread adapter
- MENU_SIZES vs `available_experiments()` consistency assert
- Integration test that hits one real LLM call (would have caught M4b root-cause bugs)
- Tighter `_SELECTION_MAX_TOKENS` if 200 → 50 maintains quality

**Other tracks (independent)**
- AutoGen integration
- CrewAI integration
- Audit dashboards / policy management UI / cost attribution reports

## Operational notes (chamber pillar)

- **OpenRouter rate limits**: `deepseek-v4-flash` is hosted by 8 providers; the orchestrator pins `Parasail` first because OpenRouter's auto-routing chose throttle-prone AtlasCloud. See `evaluation/chamber_pipeline/orchestrator.py:_CountingLLM.DEFAULT_PROVIDER_ORDER`.
- **Socket timeout**: `socket.setdefaulttimeout(30)` is set at `run_experiment.py` module load — without this, `litellm.completion(timeout=N)` doesn't propagate to the SSL socket and stuck calls hang the process forever.
- **Max tokens**: `_llm_select_loop` caps output at 200 tokens (selection step) and `llm_only_agent` at **32768** (adjacency emission, was 4096 pre-M4b-fix). DeepSeek v4 Flash is a *reasoning model* — `reasoning_tokens` typically 95% of `completion_tokens`. At 38-node adjacency prompts the 4096 cap was entirely consumed by hidden reasoning before any `content` was emitted (verified via `usage.completion_tokens_details.reasoning_tokens` on a 2-node diagnostic, 2026-05-14).
- **Cell timeout**: Pilot needs `--cell-timeout-seconds 1800` (was 600). LLM-only adjacency call at k=59 takes ~10min wall (612s measured 2026-05-14) since the model reasons over a ~22K-token data summary before emitting the 38-node graph.
- **DeepSeek v4 Flash + summary statistics is unreasonably good** at causal discovery on LT: SHD=27 / F1=0.76 / 54-of-57 edges at k/M=1.00, single seed. Beats every other variant including Planner+Reasoner. If 30-seed pilot confirms, plan §5.3 narrative rotates (LLM-only-with-summary becomes the strong-result, Planner+Reasoner becomes the "delegation has measurable cost" finding).

## References

- **Whitepaper**: `docs/whitepaper.md`
- **Testing Strategy**: `docs/testing-strategy.md`
- **Causal Chamber Plan**: `docs/causal_chamber_validation_plan.md` (full M4/M5/M6 spec for AAMAS 2027 / ECAI 2027)
- **Repository**: https://github.com/flyersworder/agent-contracts

---

*Last Updated: 2026-05-09 (M4a complete; M4b operationally ready, smoke + pilot pending)*
*Status: Production-ready, 609+ tests, 91%+ coverage*
*Integrations: LiteLLM, LangChain, LangGraph, Google ADK*
*Features: SkillSpec, Per-Tool Limits, Indeterminacy Evaluator, Evaluation Pipelines*
*Next: Run experiments, package for PyPI (v0.1.0)*
