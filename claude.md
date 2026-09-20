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
  3 seeds. Pilot = 450 cells (~24hr, ~$2 est.) at 30 seeds. Both require
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
- **Tests**: 1235 passing / 1 skipped (chamber pillar ~250; delegation graphs ~150)
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

### Delegation Graphs / Flow Conservation (Jul 25, v0.4.0) ⭐
**Value**: Budget conservation for multi-parent delegation (fan-in), which the
tree law double-counts

`ContractingCapability` models a **tree** — every child has exactly one parent.
`DelegationGraph` generalizes it to a **DAG** where budget flows along edges.
Invariant at every node: `in-flow ≥ own consumption + out-flow`. Local checks
imply the global bound `Σ C(v) ≤ B(root)` by telescoping (internal allocations
appear once as in-flow at the head and once as out-flow at the tail, so they
cancel) — meaning **no global lock and no central accountant**.

**Key semantics worth remembering** (all learned the hard way in review):
- **Control flow may cycle; budget flow must not.** A budget cycle lets a node
  refund its own ancestor and collapses the proof. Cycle-creating edges rejected.
- **Refunds are computed against ORIGINAL allocations, not live ones** — that is
  what makes releasing sibling edges order-independent. Live values make each
  sibling's refund depend on release order, which would break reproducibility.
- **Verification certifies `B(root) + Σ refunds`, not `B(root)`,** once a node is
  abandoned. Frozen pre-refund in-flow keeps an over-spent node flagged; the cost
  is that an abandoned node can consume up to its refund before detection.
- **Per-tool is conserved on BOTH paths** — granting is *prevented* at allocate
  time, consuming an undeclared tool is *detected* at verify time. No
  deny-by-default, so an omitted key = unconstrained, NOT zero. Grant explicit
  zeros when you mean zero.
- **The monitor enforces `consumed ≤ in_flow` only.** The `+ out_flow` term has no
  node-local analogue and is verify()'s job alone.
- Not thread-safe. Reclaimed budget is not re-delegatable in v1.

**Files**:
- `core/delegation_graph.py` — `DelegationGraph`, `EdgeAllocation`, `GraphNode`,
  `FlowConservationError`, `CycleError`, `GraphLintError`
- `core/resource_vector.py` — `ResourceVector` (`None` = unbounded, never zero)
- `core/delegation.py` — **untouched**; equivalence verified by cross-validation test
- Whitepaper §4.6 for the proof and its scope

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

## Dependency Notes

### google-adk 1.x → 2.x silent major bump (Jun 20, 2026)
While merging 8 Dependabot PRs (#59–#66: pyjwt, python-multipart, langchain
1.2→1.3, cryptography 46→48, starlette, aiohttp, langsmith, pydantic-settings)
and running `uv sync --all-extras`, the lockfile shrank by ~1,240 lines and the
resolved dependency set dropped from **202 → 151 packages (51 removed, 0 added)**.

- **Root cause**: `google-adk` resolved from **1.28.1 → 2.2.0**. This rode in
  *transitively* — `pyproject.toml` pins `google-adk>=1.18.0` (a floor, no
  ceiling), so the full graph re-resolution forced by the langchain/cryptography
  bumps grabbed the newest satisfying release. **Dependabot opened no PR for it**
  because the requirement string never changed, only the resolved lock did.
- **Why 51 packages vanished**: ADK 2.x demoted its Google Cloud / Vertex AI
  stack from hard deps to optional extras — removed: the entire
  `google-cloud-*` set (aiplatform, bigquery, spanner, pubsub, logging,
  monitoring, …), `google-api-python-client`, the `opentelemetry-*-gcp`
  exporters, `sqlalchemy`/`alembic`/`mako`, `grpcio`, `proto-plus`, `protobuf`.
  None were on tested code paths — coverage held at 90%.
- **Verification**: `integrations/google_adk.py` still imports cleanly and the
  full suite passes **1073 / 1 skipped / 90% cov** under ADK 2.x.
- **Latent follow-up (not yet failing)**: ADK 2.x emits
  `BaseAgentConfig is deprecated and will be removed in future versions`. A
  future ADK major may remove the config API `google_adk.py` touches — watch
  for it on the next ADK bump.
- **VPS note**: `173.212.217.40` will re-resolve to ADK 2.x on its next
  `uv sync`. Chamber sweeps use the `chambers` extra + DeepSeek-via-LiteLLM
  (not ADK), so sweeps are unaffected — but the install footprint will shift.

### The httpx2 / MCP 2.0 split (Aug 21, 2026, PR #86)

A full `uv lock --upgrade` (151 → 153 packages). The HTTP client ecosystem
has **forked into two coexisting distributions**, and this repo now resolves
both at once.

- **`httpx2` is a separate PyPI package, not httpx version 2.0.** Classic
  `httpx` is still at 0.28.1 (its own next major sits unreleased in `1.0.dev*`
  prereleases). `httpx2` renames *both* the distribution and the import
  namespace — the wheel installs top-level `httpx2/`, so `import httpx` and
  `import httpx2` are different modules and the two **cannot collide**. That
  rename is what let each consumer migrate on its own schedule.
- **Who moved**: `openai` 3.x (`httpx2<3,>=2.7.0`), `anthropic` 1.x
  (`httpx2<3,>=2.0.0`), `mcp` 2.x (`httpx2>=2.5.0`). **Who did not**:
  `litellm` 1.97 still requires `httpx<1.0` **and** caps `openai<3.0.0`.
- **Net effect here**: litellm pins us to `openai` 2.54.0 (classic httpx;
  it offers httpx2 only via an opt-in `[httpx2]` extra — the bridge release),
  while `mcp` 2.0 pulls `httpx2` in through `claude-agent-sdk`. Both stacks
  are installed: **httpx 0.28.1 + httpx2 2.12.0**. Not a bug; expect the
  duplicated HTTP footprint until litellm crosses over.
- **MCP 2.0 reaches us transitively only — we never `import mcp`.**
  `Capabilities.mcp_servers` emits the *provider-side* remote-MCP tool schema
  for LiteLLM (`type: "mcp"`, `server_url`, `require_approval`), which is a
  wire format independent of the Python SDK; `claude_agent_sdk.py` passes
  servers through to the CLI. Every v2 breaking change (`FastMCP` →
  `MCPServer`, the consolidated `Client`) is in surface we do not touch.
  **mcp 1.x is now security-fixes-only**, so 2.0 is the maintained path, not
  an optional bump. Latent: **v2 enables OpenTelemetry tracing by default** —
  dormant here since we never instantiate a server or client.
- **Package churn**: added `httpx2`, `httpcore2`, `httpx2-jsfetch`,
  `mcp-types` (v2 splits every protocol type into its own lock-step package),
  `narwhals`, `ast-serialize`, `joserfc`, `python-discovery`, `truststore`.
  Removed `httpx-sse`, `pyopenssl`, and the `typer`/`rich`/`shellingham` CLI
  stack that mcp 1.x pulled in.
- **Verification**: 1235 passed / 1 skipped / 91% cov; `mypy 2.3.1 --strict`
  clean. Crucially, **passing tests are not sufficient evidence here** — every
  integration wraps its SDK import in `try/except ImportError` that stubs the
  symbols to `Any`, so the suite stays green with an SDK completely broken.
  All six `*_AVAILABLE` flags were confirmed `True` at runtime; that is what
  actually validates `claude-agent-sdk` 0.1.50 → 0.2.143 across the mcp 1→2
  boundary.

### Dependency floors raised to tested versions (Aug 21, 2026, PR #86)

Follow-through on the google-adk lesson above. The declared floors had drifted
far below what CI exercises — `langchain>=0.3.0` while testing 1.3.16,
`langgraph>=0.2.0` while testing 1.2.11, `google-adk>=1.18.0` while testing
2.7.1, `pandas>=2.0` while testing 3.0.5 — which made the published support
claim unverifiable. Floors now sit at the tested `major.minor`.

**Re-locking after the change produced a byte-identical resolution**, so this
corrected published metadata without moving a single installed version. No
ceilings were added (deliberate call — keeps users free to adopt new majors);
the pre-existing `numpy <2.6` bound is retained. Note this means a silent
major jump like google-adk's is still *possible*; the floor raise makes the
claim honest, it does not prevent recurrence.

### pre-commit / CI linter skew (Aug 21, 2026, PR #86)

`ruff-pre-commit` was pinned at v0.15.7 while the dev group had moved to ruff
0.16.4 — **hooks and CI were running different linter versions**, which is why
ruff 0.16's newly-promoted `UP042` had never surfaced. `pre-commit autoupdate`
realigned them (ruff v0.16.4, uv-pre-commit 0.12.5, markdownlint v0.49.1).
Worth re-checking whenever the ruff dev pin moves.

UP042 flagged two `class X(str, Enum)` definitions in `benchmarks/governance/`
(CI lints only `src/` and `tests/`, so these were reachable only via
pre-commit's all-files scan). Migrated to `enum.StrEnum`. **This is not a
cosmetic swap** — a `str`+`Enum` mixin renders as `"X.A"` under `str()` and
f-strings, while `StrEnum` renders the *value*, `"a"`. It was safe here only
because every stringification in those modules goes through an explicit
`.value`. Check that before applying UP042 anywhere else.

### claude-agent-sdk 0.2.144 dropped its Windows wheel (Aug 22, 2026, v0.5.0)

0.2.143 shipped five wheels (macOS arm64/x86_64, manylinux aarch64/x86_64,
**win_amd64**); 0.2.144 ships four — the Windows one is gone, and 0.2.144 is
the latest release, so nothing upstream has restored it yet. Windows installs
fall back to the 345 KB sdist, which **installs cleanly** (pure Python) but
without the ~100 MB bundled Claude Code CLI the platform wheels carry, so
`query()` then needs `claude` on `PATH`. Verified by installing the sdist with
`--no-binary :all:`: import succeeds, 1.0 MB on disk, no bundled binary.

Nothing here breaks: `CLAUDE_AGENT_SDK_AVAILABLE` only tests importability, the
suite stubs the SDK, and CI is ubuntu-only. Recorded because it is the mirror
image of the google-adk lesson above — there a *transitive* major rode in
unnoticed; here a *platform wheel silently disappeared* at the same version
floor. Neither shows up as a requirement-string change, so only inspecting the
resolved artifacts catches them. Re-check on the next SDK bump; pin
`claude-agent-sdk==0.2.143` if a Windows contributor needs the bundled CLI.

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

**Active: M7 Phase 3.** The plan is **§8 of
`docs/superpowers/specs/2026-08-29-m7-mechanism-and-missing-arms.md`**;
`docs/chamber-results.md` holds the results and
`docs/chamber-harness-validity-register.md` the defects.

Where the pillar stands: two chambers, four models, M6's ladder run on both, M7
Phases 1-3, the coverage oracle found, the WT `team_varsplit` prediction
confirmed, the corpus re-scored on one BLAS backend at two row caps with every
headline verdict holding, and the cross-vendor replication complete.

**Remaining work is the AAMAS 2027 write-up** (`paper/aamas2027/`, untracked
like the rest of `paper/`). Hard dates: **abstract 1 Oct 2026, paper 8 Oct
2026** (AoE), 8 pages + unlimited references, `aamas.cls` unmodified, double
blind, every author on OpenReview. Area: **GAAI**. The AI-use disclosure is
mandatory and this project qualifies. See `paper/aamas2027/README.md` for the
verified rule table and the open items (S5 tool/version confirmation, S6
anonymised mirror).

**Optional, post-deadline:** the LT budget ends; a distinct-design /
overlap guard in the analyzer (fires only on `one_shot`, which is already
re-analysed, so it changes nothing retroactively); a System-One judge inside
`evaluation/indeterminacy_evaluator.py`'s ensemble.

**Other tracks (independent):** AutoGen and CrewAI integrations; audit
dashboards, policy management UI, cost attribution; **OR/optimization research
ideas** — stochastic/robust budget allocation in delegation, an OR-backed
chamber selection baseline, a heuristic-vs-optimum validation oracle
(`docs/or_optimization_research_ideas.md`).

## Operational rules (chamber pillar)

Rules only. **Every measurement that produced them is in
`docs/chamber-results.md`; every harness defect is in
`docs/chamber-harness-validity-register.md`.** Read the register before
trusting any number.

### Running a sweep

- **`socket.setdefaulttimeout(30)` at `run_experiment.py` module load is
  load-bearing** — without it `litellm.completion(timeout=N)` never reaches the
  SSL socket and a stuck call hangs the process forever.
- **Never use `with ThreadPoolExecutor(...)` for cell timeouts.** `__exit__`
  calls `shutdown(wait=True)` and blocks on a worker stuck in a non-cancellable
  C call. Use a daemon `threading.Thread` + `join(timeout)` and let it leak.
- **Size `max_tokens` against the WORST call in a loop, not the first.**
  Reasoning grows with the prompt, and the prompt grows one spent-experiment
  line per step. Mis-sized twice (200, then 2048). Selection, reconcile and
  negotiate are all 32768; `llm_only`'s adjacency is 32768. A cap is a ceiling,
  not a reservation — generosity is free.
- **Pin `reasoning.effort` on every call.** Endpoint defaults diverge 130x;
  pinned they land at 0-163 (`low`) and 1,605-2,450 (`high`). **When probing,
  replicate the production call path** or the most dramatic number will be an
  artifact of what you omitted.
- **Re-probe provider order before any multi-hour sweep** — per-provider
  throughput and price drift day to day, and precision drifts per
  (provider, model). `PROVIDER_PRECISION_BY_MODEL` must be keyed by both.
- **Never schedule arms in blocks of time.** `iter_sweep_cells` interleaves
  (`2c7c598`); the old `for spec: for seed:` put each arm in its own window, so
  provider drift landed on one arm and no post-hoc analysis recovers it.
- **Re-measure cell cost before estimating any sweep's wall time**, and estimate
  from the ARM'S OWN call count. Never reuse yesterday's figure.
- **The JSONL sidecar is the only thing between you and a lost overnight run.**
  Parquet flushes at sweep end; re-running the same `--out` skips done cells.
- **Size workers by END-of-run memory, not the launch reading** — per-worker
  memory grows over a run. `rescore.py` has no checkpoint, so add a per-design
  sidecar before any pass longer than a few hours.
- Flags: `--max-workers` (process-parallel), `--model`, `--selection-effort`
  (env-carried, recorded per cell), `--cell-timeout-seconds`, `--no-resume`.
- **VPS** `173.212.217.40`: `ssh root@173.212.217.40 'cd /root/agent-contracts
  && export PATH="$HOME/.local/bin:$PATH" && tmux new -d -s pilot "uv run python
  -m evaluation.chamber_pipeline.run_experiment ... > runs/x.log 2>&1"'`;
  pull with `rsync -av root@173.212.217.40:/root/agent-contracts/runs/ ./runs-vps/`.

### Before trusting a contrast

- **Never pool rows whose `blas_backend` differs.** macOS/Accelerate and
  Linux/OpenBLAS produce different graphs from byte-identical inputs; PC turns a
  1e-10 divergence into a structural one. Run every arm of a comparison on one
  machine. (Gone at the DESIGN level: 2,202/2,207 nine-seed design means agree.)
- **Report every arm contrast at BOTH row caps and BOTH metrics** (directed and
  skeleton). 10 of 39 headline verdicts flip with the cap on directed F1.
- **Never quote an absolute LLM-arm F1 without the core-20 figure and the
  selection effort beside it.** Core-20 is LT-only.
- **Use the unequal-n MDE form** `2.8 * pooled_sd * sqrt(1/n_a + 1/n_b)`.
  Substituting `min(n_a, n_b)` inflates it 1.23x and flips verdicts. Clustering
  by design breaks equal-n, so coverage arms always hit this.
- **Cluster by distinct design before computing anything**, and put
  distinct-selection counts in every results table. A single-call arm re-picks:
  6 distinct designs across 30 cells at LT k=30.
- **Pre-flight the design pool with MEAN PAIRWISE OVERLAP against the pigeonhole
  floor `max(0, 2k-M)/k`, not the distinct-design count** — the count misses
  designs that differ by one entry. Costs no PC run and no LLM call; compute it
  when the designs exist, before any F1 does.
- **Report equivalences with their bound and their power, never as nulls.**
  Most of the MDE is PC noise, not arm variability: identical selections would
  still give MDE 0.031 (LT k=30, n=30).
- **Key a pre-registered decision rule on the INTERVAL** (contains prediction /
  excludes zero), never on a significance threshold — a threshold branch can
  fire while the estimate matches the prediction to 0.001.
- **Only an independent replication at pre-specified n buys power.** Neither
  bootstrapping nor more PC seeds does; both were measured.
- **Re-score offline before believing a contrast** (`rescore.py`, `$0`). Key the
  work by the ORDERED buy — pooling concatenates in sequence, so `[a,b]` and
  `[b,a]` score differently. A frame without `design_key` predates the fix.
- **Never read a shape off a partial sweep**, and **never pool across a regime
  change** without checking arm means agree. A pinned model id does not pin the
  computation: DeepSeek changed reasoning 2.4x mid-sweep under an unchanged
  model string.
- **A resolved n=30 verdict the paper rests on gets re-run on another day with
  its comparator** before it is written as resolved.
- **Check a cross-sweep contrast against a SECOND same-setting day** before
  buying a same-day replication.
- **Do not reuse one vendor's MDEs for another** — reasoning length is a
  variance multiplier. Measure each arm's sd in the replication.
- **Filter an archived probe on its parse-success column before averaging.**
  Prior-elicitation Parquets retain unparsed draws, and an empty response
  scores an identical value every time, so a naive mean pools non-measurements
  and can reverse the conclusion.

### Designing an arm or a probe

- **Verify the consumer before calling something a blocker for it.** The chamber
  pipeline calls `litellm.completion` directly via its own `_CountingLLM` and
  never touches a `ResourceMonitor`.
- **Reconstruct what a prompt fix renders on REAL cells offline before buying a
  sweep on it.** A unit test and one hand-picked cell are not evidence.
- **A prompt transplanted across API SHAPES is a defect, not a tuning
  difference** — leaving a required field null on a typed-decision API cost
  0.047 F1 at 300 rows and 0.076 at 1500, and it grows with rows. Chat-to-chat
  transplants have no analogue of this.
- **Tabulate per-entry buy rates BEFORE writing "composition shift."** Three
  cap-flips were each explained by ONE menu entry.
- **Before charging a menu entry against a budget, check that every arm's prompt
  can name it.** The observational baseline was charged while most prompts could
  not nominate it; granting it turned four resolved verdicts into ties.
- **Tabulate per-experiment means of uncontrolled sensors and timestamps before
  pooling any released dataset** — six WT entries were recorded on a different
  day 2.3 kPa apart.
- **Grant explicit per-tool zeros.** An omitted key means *unconstrained*, not
  zero, and `_require_per_tool_propagation` short-circuits on `granted == 0`, so
  a zero on an unknown key raises nothing. The keys are `"intervene"` and
  `"observe"`.
- **Classifier guards must test EXCLUSIVITY, not correctness** — assert exactly
  one rule matches each input and that no marker is a substring of another.
- **Where design diversity is needed, shuffle the menu order per seed.** Pinning
  temperature is a dead end: temperature 0.0 is itself nondeterministic here,
  and unset/1.0/0.0 are indistinguishable in diversity. Reproducibility exists
  at the level of ARM MEANS over n seeds, never at the cell.
- **Put both directions of a probability comparison in the SAME request** —
  across calls, per-call variation swamps the effect and manufactures a null.
- **A voided verdict may be recoverable — check before buying a re-run.**
  `verify()` is a pure function of recorded per-node spend, so `recertify.py`
  rebuilds the graph and calls the REAL `verify()`. Valid only when the
  correction ENLARGES the grant.
- **Before asking whether a model can help, ask what the ORACLE looks like.** A
  computable oracle leaves no room; a lexical oracle means any help is leakage.

### Open harness items

- `MENU_SIZES` vs `available_experiments()` consistency assert — open.
- An integration test that hits one real LLM call — open.
- Rung-4 negotiation parser reads restatement as claim (spec §11) — open;
  needs answer/restatement separation, not filtering.
- `overlap_frac` is structurally 0.0 for rung 4 — state as a scope limit.
- `ResourceConstraints.iterations` is honored only by Google ADK and Claude
  Agent SDK; LiteLLM, LangChain and LangGraph neither track nor enforce it.

## Harness lessons that changed a result

Every defect, with its measurement, is in
`docs/chamber-harness-validity-register.md` (forty entries). The transferable
shape of them:

- **A scaffold failure rate that varies with the experiment's x-axis makes the
  curve measure the harness.** `_SELECTION_MAX_TOKENS` truncated 0% of
  selections at k=6 and ~43% at k=30, because reasoning tracks history length —
  so "LLM selection stops helping as budget grows" was the cap. Instrument the
  degradation paths and record them per cell.
- **A "provisional until measured" gate is only as good as the measurement's
  FEASIBILITY.** WT `team` ran 300 cells reporting `conservation_certified =
  None` because the constant was never isolated there: the gate silently deleted
  a number rather than protecting one.
- **Passing tests are not sufficient evidence for an integration.** Every
  integration wraps its SDK import in `try/except ImportError` that stubs to
  `Any`, so the suite stays green with an SDK completely broken. Confirm the
  `*_AVAILABLE` flags at runtime.
- **Only inspecting the RESOLVED ARTIFACTS catches a silent dependency move** —
  neither `google-adk` 1.x→2.x nor a disappearing platform wheel changes a
  requirement string.
- **Test-integrity: never classify on a value that two cases can share.** Call
  kinds keyed on `max_tokens` broke the moment two caps matched; menu-size
  thresholds broke at zero margin. Key on prompt markers and assert exclusivity.
- **`UP042` (`class X(str, Enum)` → `StrEnum`) is not cosmetic** — a `str`+`Enum`
  mixin renders `"X.A"` under `str()`, `StrEnum` renders `"a"`. Safe only where
  every stringification goes through `.value`.
- **Keep pre-commit and CI on the same linter version**; skew hides new rules.

## Chamber results → `docs/chamber-results.md`

**All chamber-pillar results live in `docs/chamber-results.md`.** This file is
project memory loaded into every session: it holds rules, specs and status.
Results are a growing archive and belong in a document you open deliberately.
Harness defects live in `docs/chamber-harness-validity-register.md` — read it
before trusting a number.

Corpus as of 2026-09-20: **two chambers, four models, 19,384+ cells, ~$158.65,
zero errored cells.** The standing headlines, each with its detail, scope and
caveats in the results doc:

- **No LLM arm beats an LLM-free round-robin coverage rule**, on either chamber,
  at any budget, under four models across three vendors and two model classes.
- **Coordination is coverage**: `ΔF1 ≈ r · Δ(distinct variables)`, where `r` is
  an exchange rate measured with no model in the loop and depends on the chamber
  and the estimator's row cap. Eleven predictions, 8 close / 3 miss, two
  pre-registered out of sample.
- **Where the comparison resolves, no fan-in topology beats a single sequential
  loop** — and the topology effect is `rate(cap) × gap(model)`, so a cheaper
  model with a smaller coverage gap shows none at the cap of record.
- **The running record is not load-bearing.** `one_shot` ties the loop
  everywhere measured; its one loss did not replicate.
- **The contract is a floor on effort, not only a ceiling on spend.**
- **The aggregator is inert by measurement**, not by omission.
- **Coverage is a plateau, not the ceiling** — but the oracle above it is an
  oracle for the estimator at a fixed cap, and beatable only at the budget ends.
- **Most of the MDE is PC noise, not arm variability.** No agent design closes a
  floor set by the inference procedure.

## References

- **Chamber results**: `docs/chamber-results.md` (every experiment and what it showed)
- **Related work notes**: `docs/related-work/` — verified reading notes on external
  papers/posts. `2026-08-13-anthropic-multiagent-patterns.md` supplies M7 §1's
  motivating citation (a 12.7x multiagent 'win' on 4.2x the tokens that the authors
  themselves reduce to 'comparable' once scope is matched).
  `2026-08-27-google-antigravity-teamwork.md` supplies the **sharper second
  instance**: Google's Teamwork post attributes +3.3 points on TCSBench to
  orchestration *across a change of model* (3.6→3.7 Flash), and its agent count
  is decided at runtime ("Agent count and team structure can shift mid-run"), so
  **no per-condition budget can be quoted even in principle** — Anthropic's were
  unmatched but stated. It also supplies the **verifier foil** that scopes our
  top threat: every Antigravity headline sits on a cheap automatic verifier
  (Lean, lockstep Spike co-simulation, a benchmark) and ours has none, so
  "partitioning pays with headroom OR with a cheap verifier" is a stated scope
  condition rather than a concession. **Each file quarantines
  figures that could not be confirmed against source text** — a summarising fetch
  invents plausible numbers for quantities that exist only as chart axes. On the
  Antigravity page the summariser's numbers were all correct and it dropped a
  *scope qualifier* instead (attributing the seven problems to Flash when they
  are 3.1 Pro results of which three reproduce) — the failure is not
  deterministic, so being right once is not grounds for trusting it. That page
  is also gzip served under an `.html` name: `gunzip -c` before parsing.
  `2026-09-12-google-scaling-agent-systems.md` (Kim et al., published as
  "Capable language models can outgrow the benefits of collaboration",
  *Nat. Mach. Intell.* 8:1157–1172, 2026 — cite that, not arXiv
  2512.08296) is the **closest contemporaneous result**: under matched
  **reasoning-token** budgets and tool access (NOT total tokens — the
  arXiv wording, corrected 2026-09-17 against the published text) across
  six benchmarks, multi-agent coordination degrades sequential tasks by
  39–70 % — our sign, independently. Cite it; then
  say where ours differs: budgets contract-certified not assumed (H-C
  fails 35 % even when matched by design), the mechanism on our task is
  redundancy + a coverage plateau, NOT their information fragmentation /
  coordination tax (their words; "budget fragmentation" was our paraphrase)
  (`one_shot` ties the loop), and their 260 configurations are single
  passes with no repeat runs. The blog's 180 / four benchmarks / R² 0.513
  are v1 figures; quote the paper's 260 / six / 0.373.
  `2026-09-13-tran-kiela-equal-thinking-budgets.md` (arXiv 2604.02460) is
  the **second** independent matched-budget result with our sign — thinking
  tokens held constant, single agent matches or beats every team, and
  API-level budget controls carry artefacts (our §25/§32 from the other
  side). Two kinds of denominator (reasoning tokens in both prior studies,
  certified experiments here), one sign; three mechanisms offered (fragmentation, context
  utilisation, duplicated coverage) — `one_shot` tying the loop rules the
  first two out on OUR task, say no more than that.
  `2026-09-13-causalab-interactive-causal-discovery.md` (arXiv 2605.26029)
  is the closest published TASK: an LLM agent intervenes under a budget on
  a sampled SCM and is scored on the recovered mechanism. Cite it as the
  causal-discovery benchmark our coordination benchmark is not. Both notes
  are abstract-level only; read the bodies before quoting a number.
- **Harness validity register**: `docs/chamber-harness-validity-register.md`
- **Whitepaper**: `docs/whitepaper.md`
- **Testing Strategy**: `docs/testing-strategy.md`
- **Causal Chamber Plan**: `docs/causal_chamber_validation_plan.md` (full M4/M5/M6 spec for AAMAS 2027 / ECAI 2027)
- **M7 plan + paper positioning**: `docs/superpowers/specs/2026-08-29-m7-mechanism-and-missing-arms.md` — **§8 is the current plan (revised 2026-08-31)**: the framing that survived Phase 2, the four ranked accept/reject threats, and a Phase 3 led by **cross-vendor replication (~$20-30, the highest-value remaining work)**. §1 holds the loop-vs-graph positioning; §5's phase list is superseded by §8's sequencing
- **OR/Optimization Ideas**: `docs/or_optimization_research_ideas.md` (backlog of 3 OR-inspired directions)
- **Repository**: https://github.com/flyersworder/agent-contracts

---

*Last Updated: 2026-09-20 (claude.md restructured: results migrated to `docs/chamber-results.md`, 1,647 -> 797 lines, rules and status only. System-One probe: `typesafe/jev-1.13` scores AUC 0.879 on which variables are connected and rho -0.44 on which experiment to buy — it knows the chamber, not the estimator; no arm above `loop` is constructible from a model that emits no text; C4 holds in a fourth model. Three retractions recorded.)*
*Status: Production-ready, v0.5.0, 1718 tests passing (1 skipped), 91% coverage*
*Integrations: LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK, Causal Chambers*
*Features: SkillSpec, Per-Tool Limits, Indeterminacy Evaluator, Evaluation Pipelines, JSONL Checkpoint Sidecar, Delegation Graphs*
*Chamber corpus: two chambers, three arm models, 19,284 orchestrator cells / $158.58 — see `docs/chamber-results.md`*
*Next: the AAMAS 2027 draft. Abstract 1 Oct, paper 8 Oct 2026 (AoE); skeleton in `paper/aamas2027/`, untracked. All planned experiments are complete; the write-up is the remaining work.*
