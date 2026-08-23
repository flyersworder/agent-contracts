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

**M6 REVISED (2026-07-25): loop-vs-graph topology benchmark (plan §7.7)**

The deferred choice between "§7 cross-pillar" and "skip §7" was resolved by
an external trigger: **loop engineering** (named 2026-06-07, Addy Osmani) and
**graph engineering** (named ~2026-07-18) put multi-agent topology on the
field's agenda, and critical reviews of both note there is **no comparative
benchmark**. We already hold two of the three arms.

M6 is now a **five-rung coordination ladder** (respecified 2026-08-23; the
three-arm description below is superseded by
`docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md`, which is
authoritative): loop (`llm_pc`), ensemble, parallel-roles, chain
(`planner_reasoner`), and team (negotiation), at k ∈ {6, 30, 45}.
Arms 1-2 are already in `runs/m4-pilot.parquet`, so new compute is **90 cells
(~$1-2)**. Cross-pillar transfer (§7.1-7.6) becomes the journal extension.
**Both figures were wrong** (corrected 2026-08-23): M4b actually cost **$5.11**
for 450 cells, and M6 as respecified is **~390 cells, 65-95h, $9-13** — see
`docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md` §3. The
per-cell basis matters: M4b LLM cells averaged **7.2 min**, `planner_reasoner`
**8.7 min / $0.0196**; the 4.7 min/cell implied by the headline is contaminated
by 180 non-LLM cells averaging 0.15s. Estimate from LLM cells only.

Pre-registered: H-A chain underperforms loop (already supported: F1 0.397 vs
0.75); **H-B fan-in recovers delegation cost via exploration diversity — open,
this is the experiment**; H-C conservation compliance 100%.

Prerequisite shipped in **v0.4.0** (`core/delegation_graph.py`). **No open
blockers.** When building the fan-in arm, grant the aggregator an explicit
`per_tool={"intervene": 0, "observe": 0}` — an omitted key means *unconstrained*,
not zero, so
the matched-budget control would otherwise be unenforceable rather than merely
detectable after the fact. **The key is `"intervene"`** (the tool name
`ContractedChamberAgent.query_intervention` meters), not `"exp"`; an earlier
note here said `"exp"`, which fails *silently* — `DelegationGraph.
_require_per_tool_propagation` short-circuits on `granted == 0`, so a
zero-grant on an unknown key raises nothing while `"intervene"` stays
unconstrained. The adapter's aggregate monitor still caps total spend, so the
matched budget holds; what is lost is the per-role control. **`"observe"` needs
the same explicit zero**: `create_contracted_chamber_agent` inserts that key only
when `observation_budget > 0`, so without it the aggregator can call
`query_observation` without bound and acquire data outside the certified budget.

**Retracted (2026-07-25): the `add_iteration()` "gap" was a phantom.** An
earlier note here claimed M6 was blocked on wiring `ResourceUsage.add_iteration()`.
Wrong twice over. The chamber pipeline calls `litellm.completion` directly via
its own `_CountingLLM` (`orchestrator.py:239`) and never touches a
`ResourceMonitor`, so wiring the framework integrations would not reach chamber
cells at all. And the metric already exists as **`n_llm_calls`**, populated for
every LLM-variant ok-cell in `runs/m4-pilot.parquet` (6 / 30 / 59-60 across the
three budgets; null for `random` and `greedy_ig` because they issue no LLM
calls). Lesson: check whether the consumer actually goes through the framework
before calling something a blocker for it.

**Separate, real gap (not an M6 blocker)**: `ResourceConstraints.iterations` is
honored only by Google ADK (→ `max_llm_calls`) and Claude Agent SDK (→
`max_turns`). LiteLLM, LangChain, and LangGraph neither track nor enforce it,
and `contract.py`'s docstring wrongly claimed LangGraph mapped it to
`recursion_limit` (corrected 2026-07-25). Worth closing for library users.

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

*Last Updated: 2026-08-22 (release 0.5.0: dependency refresh, floors raised to tested versions, slimmed sdist)*
*Status: Production-ready, v0.5.0, 1235 tests passing (1 skipped), 91% coverage*
*Integrations: LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK, Causal Chambers*
*Features: SkillSpec, Per-Tool Limits, Indeterminacy Evaluator, Evaluation Pipelines, JSONL Checkpoint Sidecar, Delegation Graphs*
*Pilot dataset: `runs/m4-pilot.parquet` (450 cells, 442 ok, 8 timeouts) — submission-ready for AAMAS 2027 / ECAI 2027*
*Next: M5 (WT + UNCONTRACTED + Pro robustness), then M6 topology benchmark (§7.7)*
