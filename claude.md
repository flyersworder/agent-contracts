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
- **M4b cell-timeout root-cause fix** (May 15, commit `0d694cf`) ✅ — the
  pilot hung at 60/450 (and re-smoke hung at 30/45 yesterday) because
  `_invoke_with_timeout` used `with ThreadPoolExecutor(...) as exe:`. The
  context manager calls `shutdown(wait=True)` on exit, which blocks the main
  thread waiting for a worker stuck in a non-cancellable openssl SSL_read.
  `future.result(timeout=1800)` correctly raised TimeoutError *inside* the
  `with` block, but the exception couldn't escape because `__exit__` was
  wedged. Replaced with `threading.Thread(daemon=True)` + `worker.join(timeout)`;
  on timeout, raise and let the daemon thread leak. Now hangs are bounded
  at 1800s and the sweep advances. The earlier three fixes (socket timeout,
  max_tokens, provider rotation) were all real bugs but none would have
  fixed this hang on their own — the cell-timeout safety net was itself
  broken.
- **M4c checkpointing** (May 17, commit `856beb8`) ✅ — per-cell JSONL
  sidecar + resume-on-restart. Two pilot attempts (May 15 and 16-17) lost
  217+ ok-cells each when overnight stalls happened, because Parquet only
  flushes at sweep end. Added `evaluation/chamber_pipeline/checkpoint.py`
  (append_record_jsonl, read_records_jsonl, done_cell_keys, filter_done_cells,
  17 tests) and a new `--no-resume` CLI flag. `run_sweep` gains optional
  `skip_keys` for cell-level filtering. CLI writes one JSON line per cell to
  `<out>.jsonl` (POSIX-atomic small write), reads it on start to skip done
  cells, and consolidates JSONL → Parquet at sweep end. Validated via 4-phase
  smoke (fresh / resume-complete / resume-partial / collision); Option A
  collision policy refuses to clobber Parquet without sidecar.
- **M4b PILOT COMPLETE** (May 18, 20:44, commit base `4768945`) ✅
  - **450/450 cells, 442 ok, 8 timeouts** (1.8% error rate)
  - Wall time: **35h 33m** (Sun 09:18 → Mon 20:44)
  - All 8 errors are `planner_reasoner k=59` TimeoutErrors at 1800s — the
    cell-timeout safety net firing correctly on the most-LLM-call-intensive
    variant×budget combination (planner + reasoner + adjacency = 3 chained
    LLM calls). Sweep continued past each timeout instead of wedging.
  - **M4 acceptance criteria: ✓ PASS** (`analyze_results --check-m4-acceptance`):
    - All 5 variants' Pareto curves monotonic within 1.5σ noise
    - Random dominated by all 4 LLM variants at k/M ∈ {0.10, 0.51}
    - Random dominated by llm_only + llm_pc at k/M=1.00 (planner_reasoner
      falls out due to its 8 timeouts contaminating its mean)
  - **§5.3 narrative confirmed at 30 seeds** (was a single-seed hypothesis
    from May 14): **DeepSeek v4 Flash + data summary dominates the Pareto.**
    LLM-only at k/M=1.00: **SHD=26, F1=0.75**. Every other variant clusters
    at SHD≈53-57 / F1≈0.40-0.42. The gap is dramatic and consistent.
  - **"Delegation has measurable cost" finding**: planner_reasoner F1 stays
    essentially flat from k=0.51 (0.385) to k=1.00 (0.397) — unlike every
    other variant which gains substantially with budget. Combined with its
    8 timeouts at k=59, the most-delegated variant is both worse and less
    reliable at maximum budget. A paper-worthy operational observation.
  - Figures: `runs/m4-pilot-figs/pareto_shd.png`, `pareto_f1.png`.
  - Sidecar: `runs/m4-pilot.jsonl` (450 lines, kept for audit).

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

**M4b PILOT: COMPLETE (May 18, 20:44). M4 acceptance criteria PASS.**
- 450/450 cells, 442 ok / 8 timeouts (1.8% error rate, all planner_reasoner k=59)
- LLM-only dominates Pareto (SHD=26, F1=0.75 at k/M=1.00); 4× better than next variant
- Figures at `runs/m4-pilot-figs/`; sidecar `runs/m4-pilot.jsonl` kept for audit

**Active: M5 (trimmed scope — see plan §6.1 callout dated 2026-05-18)**

Scope revised post-M4b: skip 5-budget expansion (keep 3: 0.10, 0.50, 1.00).
M4b's dramatic effect at 3 budgets makes intermediate budgets curve-shape
refinement rather than headline material. Essentials below are non-negotiable.

1. **WT chamber sweep** — 360 CONTRACTED cells (1 chamber × 3 budgets × 4
   variants × 30 seeds; variant 2 / GreedyIG-lite is LT-only per plan §5.1).
   Establishes external validity beyond LT's 38-node graph.
   `uv run python -m evaluation.chamber_pipeline.run_experiment --chambers wt --budgets 0.10,0.50,1.00 --seeds 30 --cell-timeout-seconds 1800 --out runs/m5-wt.parquet`
2. **UNCONTRACTED baselines** — 270 cells across both chambers (essential for
   the framework paper's "contracting helps" claim; without these the chamber
   pillar is a causal-discovery benchmark, not a contracting validation).
3. **DeepSeek v4 Pro robustness sweep** — 270 cells (tests whether Flash
   dominance generalizes across model scale). Per plan §6.7.
4. Run on VPS (`173.212.217.40`, provisioned 2026-05-18). Total new work:
   ~900 cells vs original M5 plan's 1620. ~33% wall-time savings vs original.
5. Optional: re-run the 8 timed-out `planner_reasoner k=59` cells from M4b
   to recover full 30-seed coverage (checkpoint pattern: re-run the pilot
   command; the sidecar skips the 442 ok cells and only re-attempts the 8
   errors). Worth doing before declaring M5 complete.

**Now-optional: M6 (cross-pillar transfer study, plan §7)**

Post-M4b reassessment: M6 was load-bearing under the original "modest
chamber effect needs cross-pillar bridge" assumption. M4b's dramatic effect
makes chamber pillar standalone publishable. Two paths now both viable:

- (1) Defer §7 to a journal extension → submit single-pillar to AAMAS 2027,
  unblock M7 drafting ~3 weeks earlier.
- (2) Execute §7 as originally planned → two-pillar paper.

Defer the decision until M5 is underway and the WT result is in hand.

**M4c (mostly complete after May 17 work)**
- Checkpointing / resume from partial Parquet ✅ (commit `856beb8`)
- Parallelism (`ThreadPoolExecutor` in `run_sweep`) — pending, M5 priority
  (would cut WT+LT 5-budget sweep wall time from ~70h serial to ~18h on the
  4-vCPU VPS)
- MENU_SIZES vs `available_experiments()` consistency assert — still open
- Integration test that hits one real LLM call — still open; would have
  caught M4b root-cause bugs earlier
- Tighter `_SELECTION_MAX_TOKENS` if 200 → 50 maintains quality — open

**Other tracks (independent)**
- AutoGen integration
- CrewAI integration
- Audit dashboards / policy management UI / cost attribution reports
- **OR/optimization research ideas** (backlog, captured 2026-05-30) — stochastic
  budget allocation in delegation, OR-backed chamber experiment-selection
  baseline, heuristic-vs-optimum validation oracle. See
  `docs/or_optimization_research_ideas.md`. Thesis: upgrade our allocation
  heuristics to *stochastic/robust* optimization (exploits our indeterminacy
  modeling), NOT the deterministic LP the source article assumes.

## Operational notes (chamber pillar)

- **OpenRouter rate limits**: `deepseek-v4-flash` is hosted by 8 providers; the orchestrator pins providers in order `(Novita, AtlasCloud, Parasail, SiliconFlow)` because per-provider throughput drifts day-to-day (May 9: Parasail was fastest; May 15: Novita was 7× faster). See `evaluation/chamber_pipeline/orchestrator.py:_CountingLLM.DEFAULT_PROVIDER_ORDER` and re-probe before any multi-hour sweep.
- **Socket timeout**: `socket.setdefaulttimeout(30)` is set at `run_experiment.py` module load — without this, `litellm.completion(timeout=N)` doesn't propagate to the SSL socket and stuck calls hang the process forever.
- **Max tokens**: `_llm_select_loop` caps output at 200 tokens (selection step) and `llm_only_agent` at **32768** (adjacency emission, was 4096 pre-M4b-fix). DeepSeek v4 Flash is a *reasoning model* — `reasoning_tokens` typically 95% of `completion_tokens`. At 38-node adjacency prompts the 4096 cap was entirely consumed by hidden reasoning before any `content` was emitted (verified via `usage.completion_tokens_details.reasoning_tokens` on a 2-node diagnostic, 2026-05-14).
- **Cell timeout**: Pilot needs `--cell-timeout-seconds 1800` (was 600). LLM-only adjacency call at k=59 takes ~10min wall (612s measured 2026-05-14) since the model reasons over a ~22K-token data summary before emitting the 38-node graph.
- **DeepSeek v4 Flash + summary statistics is unreasonably good** at causal discovery on LT — **CONFIRMED AT 30 SEEDS** (May 18 M4b pilot): SHD=26 / F1=0.75 at k/M=1.00, every other variant clusters at SHD≈53-57 / F1≈0.40-0.42. The §5.3 narrative has rotated: LLM-only-with-summary is the strong result, Planner+Reasoner is the "delegation has measurable cost" finding (F1 stays flat 0.385→0.397 from k=0.51→1.00, plus 8/30 timeouts at k=59).
- **Checkpoint sidecar**: every pilot run writes one JSON line per cell to `<out>.jsonl` before the Parquet consolidates at sweep end. Resume-on-restart is automatic — re-running the same `--out` command after a kill skips already-done cells. The two May 15 and 16-17 overnight stalls lost ~217 cells each because this didn't exist; M4b May 18 pilot benefited from it (10 timeouts that would have wedged the older orchestrator instead just logged as errors and the sweep continued).
- **VPS provisioned**: `173.212.217.40` (Ubuntu 24.04, 4 vCPU, 8 GiB RAM, 145 GB disk). Repo cloned at `/root/agent-contracts`, uv synced with `--all-extras`, `.env` transferred (0600 perms). Ready to launch any pilot via `ssh root@173.212.217.40 'cd /root/agent-contracts && export PATH="$HOME/.local/bin:$PATH" && tmux new -d -s pilot "uv run python -m evaluation.chamber_pipeline.run_experiment --pilot --cell-timeout-seconds 1800 --out runs/m4-pilot.parquet > runs/m4-pilot.log 2>&1"'`. Pull results back via `rsync -av root@173.212.217.40:/root/agent-contracts/runs/ ./runs-vps/`.

## References

- **Whitepaper**: `docs/whitepaper.md`
- **Testing Strategy**: `docs/testing-strategy.md`
- **Causal Chamber Plan**: `docs/causal_chamber_validation_plan.md` (full M4/M5/M6 spec for AAMAS 2027 / ECAI 2027)
- **OR/Optimization Ideas**: `docs/or_optimization_research_ideas.md` (backlog of 3 OR-inspired directions)
- **Repository**: https://github.com/flyersworder/agent-contracts

---

*Last Updated: 2026-05-18 (M4b pilot COMPLETE, M4 acceptance PASS; M4c checkpointing landed)*
*Status: Production-ready, 1046+ tests, 81%+ coverage*
*Integrations: LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK, Causal Chambers*
*Features: SkillSpec, Per-Tool Limits, Indeterminacy Evaluator, Evaluation Pipelines, JSONL Checkpoint Sidecar*
*Pilot dataset: `runs/m4-pilot.parquet` (450 cells, 442 ok, 8 timeouts) — submission-ready for AAMAS 2027 / ECAI 2027*
*Next: M5 (WT chamber + 5 budget levels), VPS migration (already provisioned)*
