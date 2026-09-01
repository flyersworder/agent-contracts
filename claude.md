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
for 450 cells, and M6 as respecified is **~417 cells, 65–90h, $9–14** — see
`docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md` §3. The
per-cell basis matters: M4b LLM cells averaged **7.2 min**, `planner_reasoner`
**8.7 min / $0.0196**; the 4.7 min/cell implied by the headline is contaminated
by 180 non-LLM cells averaging 0.15s. Estimate from LLM cells only.

Pre-registered: **H-A chain underperforms loop — OPEN, not supported**
(corrected 2026-08-23); **H-B fan-in recovers delegation cost via exploration
diversity — open, this is the experiment**; H-C conservation compliance 100%.

**H-A correction (2026-08-23).** This line previously read "already supported:
F1 0.397 vs 0.75". That compared the chain against **`llm_only`**, which is
**not a rung on the ladder** — spec §10.3 replaced it with `llm_pc` precisely
because it confounds topology with inference. Against the actual loop rung,
measured on `runs/m4-pilot.parquet` via `analyze_results --ladder`:

| k | chain (`planner_reasoner`) | loop (`llm_pc`) | delta | pooled MDE | verdict |
|---|---|---|---|---|---|
| 6 | 0.190 | 0.218 | −0.028 | 0.036 | below MDE |
| 30 | 0.386 | 0.379 | **+0.007** | 0.036 | below MDE |
| 59 | 0.398 | 0.425 | −0.027 | 0.036 | below MDE |

All three below the minimum detectable effect at n=30, under both the per-arm
and pooled-SD bounds, and k=30 points the wrong way. `llm_only` really does
reach F1 0.746 at k=59 — that M4b finding stands — but it is a statement about
inference, not about coordination topology. **At the observed SD, resolving a
~0.03 gap needs n≈55 per arm; M6 at n=30 will report "below MDE" for H-A
however it runs.** That is a reportable equivalence bound, not a null, and the
analyzer prints the MDE beside every delta so it cannot be read as one.

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

- **Temperature was NEVER pinned, and pinning it would not help** (measured
  2026-09-01, register §21). No DeepSeek run sent a `temperature` field —
  pre-Aug-30 files have no column, Phase 2 files have it null in all 960 rows.
  The arms are also inconsistent: scouts in `fan_in_*`/`team*` run at
  `_SCOUT_TEMPERATURE = 1.0` while the loop, `one_shot`, `critique`,
  `planner_reasoner` and `shared_blackboard` run unpinned. **But temperature
  0.0 is itself nondeterministic here** — six distinct picks in nine draws, and
  unset/1.0/0.0 are indistinguishable in diversity. So the mismatch is not a
  meaningful confound, and "pin temperature for reproducibility" is a dead end;
  where design diversity is needed, **shuffle the menu order per seed**.
  Reproducibility in this pillar exists at the level of ARM MEANS over n seeds,
  never at the cell — same conclusion as the BLAS finding.
- **ROOT CAUSE of that nondeterminism** (register §26, 2026-09-01): a chaotic
  branch point early in the reasoning trace. temperature=0 IS honoured — "17×3"
  returns `51` byte-identically 6/6 with the same reasoning-token count, and the
  long menu prompt with an unambiguous answer demanded returns `APPLE` 3/3 at
  reasoning=51. But on the real task, four draws share only **20-114 characters**
  of reasoning before forking at a semantically empty choice ("name" vs
  "experiment name"), then run 20k-54k chars to four different answers.
  Greedy decoding is deterministic given identical logits; fp8 MoE logits vary
  with batch composition, a near-tie flips the argmax, and a long trace
  amplifies it. **Isomorphic to the BLAS/PC finding** — a discrete decision on
  top of a continuous computation reproducible only to kernel noise.
  **Practical:** reasoning length is a variance multiplier, so **do not reuse
  DeepSeek MDEs for another vendor** — measure each arm's sd in the replication
  or a power difference will read as a failure to replicate.
- **Endpoint reasoning DEFAULTS diverge 130x — and we already pin past them**
  (probed 2026-08-31, register §25). With no `reasoning` parameter, one prompt
  gives Z.AI 6,889 tokens / GMICloud 2,600 / DeepInfra 52 on `glm-5.3-flash`.
  **The pipeline does not take that path**: every call sets `reasoning.effort`
  (`_SELECTION_REASONING_EFFORT="low"`, `_COORDINATION_REASONING_EFFORT="high"`),
  and with it pinned the endpoints land at 0-163 (`low`) and 1,605-2,450
  (`high`). **When probing, replicate the production call path or the most
  dramatic number will be an artifact of what you omitted.**
  Real findings from that probe: `Relace` is **fp4** on both models and among
  the cheapest, so price-first routing picks it — now declared ineligible; and
  `Reka` is fp4 for deepseek but fp8 for GLM, so `PROVIDER_PRECISION`'s
  provider-only keying is unsound in principle — fix before adding a third
  model.
- **OpenRouter rate limits**: `deepseek-v4-flash` is hosted by 8 providers; the orchestrator pins providers in order `(Novita, AtlasCloud, Parasail, SiliconFlow)` because per-provider throughput drifts day-to-day (May 9: Parasail was fastest; May 15: Novita was 7× faster). See `evaluation/chamber_pipeline/orchestrator.py:_CountingLLM.DEFAULT_PROVIDER_ORDER` and re-probe before any multi-hour sweep.
- **Socket timeout**: `socket.setdefaulttimeout(30)` is set at `run_experiment.py` module load — without this, `litellm.completion(timeout=N)` doesn't propagate to the SSL socket and stuck calls hang the process forever.
- **Max tokens**: `_llm_select_loop` caps output at 200 tokens (selection step) and `llm_only_agent` at **32768** (adjacency emission, was 4096 pre-M4b-fix). DeepSeek v4 Flash is a *reasoning model* — `reasoning_tokens` typically 95% of `completion_tokens`. At 38-node adjacency prompts the 4096 cap was entirely consumed by hidden reasoning before any `content` was emitted (verified via `usage.completion_tokens_details.reasoning_tokens` on a 2-node diagnostic, 2026-05-14).
- **Cell timeout**: Pilot needs `--cell-timeout-seconds 1800` (was 600). LLM-only adjacency call at k=59 takes ~10min wall (612s measured 2026-05-14) since the model reasons over a ~22K-token data summary before emitting the 38-node graph.
- **DeepSeek v4 Flash + summary statistics is unreasonably good** at causal discovery on LT — **CONFIRMED AT 30 SEEDS** (May 18 M4b pilot): SHD=26 / F1=0.75 at k/M=1.00, every other variant clusters at SHD≈53-57 / F1≈0.40-0.42. The §5.3 narrative has rotated: LLM-only-with-summary is the strong result, Planner+Reasoner is the "delegation has measurable cost" finding (F1 stays flat 0.385→0.397 from k=0.51→1.00, plus 8/30 timeouts at k=59).
- **Checkpoint sidecar**: every pilot run writes one JSON line per cell to `<out>.jsonl` before the Parquet consolidates at sweep end. Resume-on-restart is automatic — re-running the same `--out` command after a kill skips already-done cells. The two May 15 and 16-17 overnight stalls lost ~217 cells each because this didn't exist; M4b May 18 pilot benefited from it (10 timeouts that would have wedged the older orchestrator instead just logged as errors and the sweep continued).
- **VPS provisioned**: `173.212.217.40` (Ubuntu 24.04, 4 vCPU, 8 GiB RAM, 145 GB disk). Repo cloned at `/root/agent-contracts`, uv synced with `--all-extras`, `.env` transferred (0600 perms). Ready to launch any pilot via `ssh root@173.212.217.40 'cd /root/agent-contracts && export PATH="$HOME/.local/bin:$PATH" && tmux new -d -s pilot "uv run python -m evaluation.chamber_pipeline.run_experiment --pilot --cell-timeout-seconds 1800 --out runs/m4-pilot.parquet > runs/m4-pilot.log 2>&1"'`. Pull results back via `rsync -av root@173.212.217.40:/root/agent-contracts/runs/ ./runs-vps/`.

## Session 2026-08-24: the harness was moderating the result

Four findings, in descending order of consequence. All measured, not inferred.

### 1. `_SELECTION_MAX_TOKENS` made the harness a moderator of the IV

An instrumented k=30 cell (0731, providers pinned) attributed **every**
selection failure to truncation: `{'length': 13, 'empty': 0, 'offmenu': 0,
'ok': 17}`. 13 of 30 picks were `rng.choice`, because the 2048 cap was consumed
by reasoning before any content was emitted.

The cap has now been mis-sized **twice** (200, then 2048), both times
calibrated on the loop's **first** call, where reasoning is 415-976 tokens.
Reasoning scales with the prompt, and the prompt grows one spent-experiment
line per step. Late-loop (25 chosen): flash-0731 **2,175**, flash **11,690**.

Because the failure rate tracks history length, it was **0/36 at k=6 and ~43%
at k=30** — correlated with the experiment's independent variable. In M4b,
`llm_pc` beat `random` by **+0.034 F1 at k=6 (resolved)** and only **+0.018 at
k=30 (below MDE)**. "LLM selection stops helping as budget grows" was this cap.

**M6 exposure was worse.** The ladder's IV is how budget is *split across
agents*, and splitting shortens each agent's history: two scouts at k=15
truncate less than one loop at k=30. The fan-in rungs would have beaten the
loop for reasons unrelated to coordination — **H-B could have come out positive
as a pure `max_tokens` artifact.**

Fixed in `6ae85e5`: selection, reconcile, and negotiate all raised to 32768
(`max_tokens` is a ceiling, not a reservation — generosity is free).

**Consequence for calibration:** `_A95_RECONCILE` (8557) and `_C95_NEGOTIATE`
(4138) were measured against *truncated* calls. A truncated reconcile pinned
aggregator spend to exactly 8192 — inside P2's window (6418, 12836] — so
`tree_would_refuse` could read True *because the call truncated*. Both
constants MUST be re-derived from untruncated late-loop measurements at k=45
before any sweep reporting H-C or P2.

### 2. M4b rows can no longer be reused

Two independent reasons, either sufficient:

- **Provider-side change.** DeepSeek raised default reasoning under unchanged
  0423 weights (2026-08-13). `llm_pc` k=30 went from 244 s / 1,089 output
  tokens per call to ~1,300 s / ~3,600. Decomposed exactly: **4.35x more
  tokens x 1.55x lower throughput = 6.72x**, matching observed. (Throughput
  really did drop, 134 -> 87 tok/s, consistent with the old snapshot getting
  less compute — but it is the smaller factor.)
- **Selection semantics changed** (`7f284be`, `6ae85e5`).

The M6 plan's pilot-reuse for rungs 0 and 3 is therefore **void**; all five
rungs must run fresh. Parallelism makes that affordable.

### 2b. The BLAS backend is part of the configuration (2026-08-26)

Supersedes two earlier explanations of the same observation, both wrong:
"library drift breaks per-cell reproducibility" (`c77c610`) and, later the
same day, "an unrecoverable uncommitted working tree" (`bd46b0d`). The actual
cause is measured:

**macOS/Accelerate and Linux/OpenBLAS produce different causal graphs from
byte-identical inputs.** Verified directly, not by elimination: on the same
LT matrix, `np.corrcoef` differs between the machines, and `inv(C)[0,1]`
agrees to ~10 hex digits then diverges (`-0x1.a8471c316a312p+13` vs
`-0x1.a8471c315c71bp+13`, relative ~1e-10). Dataset md5s match; `numpy`
2.5.2, `scipy` 1.18.1, `causal-learn` 0.1.4.8 and `pandas` 3.0.5 are
identical on both.

PC converts that into *structural* noise. It is a sequence of accept/reject
tests at alpha, each conditioned on the previous ones, so a perturbation that
flips one borderline test forks the conditioning-set search rather than
nudging a number. Seeded `random`, LT, no LLM:

| k | seed | Accelerate (macOS) | OpenBLAS (VPS) |
|---|---|---|---|
| 15 | 0 | 0.3810 | 0.2857 |
| 15 | 1 | 0.2889 | 0.2759 |
| 59 | 0 | 0.3736 | 0.3864 |
| 59 | 2 | 0.3871 | 0.4615 |

**Which file came from which machine:**

- **macOS / Accelerate**: `m4-pilot.parquet`, `curve-lt-random.parquet`,
  `curve-wt-random.parquet`, `curve-wt-validate.parquet`.
- **Linux / OpenBLAS (VPS)**: `m6-ladder.parquet`, `m6-wt-ladder.parquet`,
  `m6-controls.parquet`.

Two consequences, one reassuring and one not:

1. **Both M6 ladders are OpenBLAS**, so the cross-chamber topology
   replication (LT vs WT, §"M6 WT LADDER COMPLETE") is platform-consistent
   and stands as reported.
2. **Every loop-vs-random contrast recorded so far is cross-platform** — the
   ladders are VPS, the random curves are local. The WT figures (+0.019 at
   k=14, +0.037 at k=21) are therefore *not* yet a clean contrast and must be
   recomputed against a VPS random baseline before use. Cheap: `random` runs
   without an LLM.

**Durable fix, `2e54aa1` + follow-up**: every `RunRecord` now carries
`pc_alpha`, `pc_max_rows`, `pc_collinearity_threshold`, `blas_backend` and
`platform_tag`. Never pool rows whose `blas_backend` differs.

For the paper's reproducibility statement: in a constraint-based discovery
algorithm, the seed does not determine the graph. Archive the resolved
environment *including the linear-algebra backend*, and run every arm of a
comparison on one machine.

### 3. `deepseek-v4-flash-0731` is the better snapshot

Same provider (Novita), late-loop call: **23.5 s / 2,175 tokens** against
flash's **105.0 s / 8,828**, for ~25% more cost per call. At cell level (n=3,
k=30) it reproduced M4b accuracy closely — F1 0.387 vs 0.379, SHD 57.3 vs 57.1
— with far tighter spread (wall 500/547/579 s vs flash 991/1642). Selectable
via the new `--model` flag; recorded per cell in `model_id`.

`~deepseek/deepseek-v4-flash-latest` is **not routable** (404); the `~` marks a
non-routable variant. `deepseek-v4-pro` truncates at 2048 with empty content
and costs 3x.

### 4. The seed does not control the LLM

`llm_pc_agent` calls `_llm_select_loop(...)` with **no temperature**, so the
provider default applies. Same seed, same config, two runs: **F1 0.330 and
0.482**. The seed governs only the fallback RNG and PC. Every cell is an
independent draw. Not yet changed — pinning temperature touches every LLM arm
and needs its own replication check.

### Infrastructure shipped

- **`--max-workers N`** (`116ac05`): process-parallel sweep, M4c's deferred
  item. Cells run at ~1.3% CPU (pure network wait). Measured **2.49x on 3
  workers** (83% efficiency); ~700 MB per worker caps the 8 GB VPS at ~8.
  Processes not threads: `_PcDegeneracyHandler` attaches to a *global* logger
  per cell, so concurrent cells in one process would cross-contaminate
  `n_pc_degeneracy`; and `_invoke_with_timeout` leaks a daemon thread per
  timeout.
- **`--model`** (`5f081ba`): applied after `static_kwargs` so an explicit flag
  outranks a spec default; guarded on `accepts_llm`.

### Known-open (recorded, deliberately not fixed blind)

- **Rung 4 negotiation parser** reads restatement as claim: the revise prompt
  shows the peer's proposals above the full menu and `_parse_name_list` scans
  the whole response, so a scout restating the peer inflates `n_contested` —
  rung 4's headline metric. Cannot be fixed by filtering (a genuine contest is
  the signal); needs answer/restatement separation. See spec §11.
- **The aggregator's reconcile output is discarded — RESOLVED as a threat
  (2026-08-27).** Measured by the `fan_in_agg` ablation rather than argued;
  see the section below. The paper still must not claim the aggregator
  *improves* the result, but the negative fan-in finding is not an artifact
  of a null aggregator.
- **`overlap_frac` is structurally 0.0 for rung 4** (pools disjoint by
  construction). State as a scope limit.

### Harness validity gate (k=45, all five rungs) + calibration — then LAUNCH

Ran a 15-cell gate at the largest budget, then 18 calibration cells at k=6/30,
before committing to the sweep. Both clean of errors (0/33 at a generous
timeout). What they established:

| check | result |
|---|---|
| Truncation at k=45 | **0.7%** (2/270 calls), no `length` failures — was 43% at k=30 |
| Timeouts | 0/15 at 7200 s; **max cell 1960 s** → sweep uses 5400 s, not M4b's 1800 |
| PC degeneracy | 0.00 |
| Conservation | 6/9 graph cells FAILED at k=45 → traced to `a95`, now fixed |
| P2 | demonstrable where spend lands in-window |

**`_A95_RECONCILE` → `_A95_RECONCILE_BY_K` at p75** (commit `fd7ee4c`).
Aggregator cost grows with k, so one constant could not work:

| k | p75 | spread | conserve | in P2 window |
|---|---|---|---|---|
| 6 | 7,646 | **48.8x** (500–24,415) | 7/9 | 2/9 |
| 30 | 11,427 | 5.2x (4,648–24,001) | 8/9 | 5/9 |
| 45 | 18,790 | 2.6x (9,783–25,168) | 9/9 | 6/9 |

p75 not median, for a design asymmetry: `_ROLE_C95` medians get multiplied by
`_PROVISION_MULTIPLE = 4`, but the aggregator gets `1.5 * a95` and **no
multiple** (a margin would destroy P2), so the median was imported without its
multiplier — and a median-sized budget overruns ~50% of executions by
construction. Uncalibrated budgets now **raise** rather than extrapolate.

**Two claims the paper must now qualify:**

1. **P2's window width equals the fan-in degree.** `max_i a_i < c <= sum_i a_i`
   gives `(f, n*f]` — 2x for two scouts. k=6's 48.8x spread cannot fit, so P2
   is demonstrable at k=30/k=45 and effectively not at k=6. The lever is more
   parents, not a better constant. Spec §12.
2. **H-C conflates mechanism with forecast.** A conservation failure means
   `verify()` correctly caught an overrun — the mechanism worked 100% of the
   time; our cost prediction did not. Report separately or a reader concludes
   the framework failed. Spec §6.

**M6 SWEEP LAUNCHED 2026-08-24 10:51 UTC** — `--m6 --model
openrouter/deepseek/deepseek-v4-flash-0731 --max-workers 6
--cell-timeout-seconds 5400 --out runs/m6-ladder.parquet`, 450 cells, ~20 h
projected (98 h serial), ~$10–15. Resume after any interruption by re-running
the identical command; the JSONL sidecar skips completed cells.

**Early science from the gate (n=3, directional only):** role differentiation
halves scout overlap (0.79 → 0.32) and recovers distinct coverage (27.6 → 38.0
of 45) — H-B's mechanism, visible. `team` reaches full 45/45 coverage yet the
worst F1 (0.319 vs loop 0.463), separating topology cost from accounting:
splitting selection between blind scouts is worse than one sequential loop
even at equal coverage.

**Still open, recorded not fixed:** scout `c95` is unverified at k=45 (per-scout
tokens now recorded so the next calibration can check); temperature unpinned on
`llm_pc` so the seed does not control the LLM (variance, not bias); rung-4
negotiation restatement (gate showed contested 4/45, negfail 0 — no inflation
visible yet).

### Test-integrity lesson (third occurrence today)

Raising the caps broke **six tests** that classified call types by
`max_tokens`, which became ambiguous the moment two caps shared a value — the
same fragility as the `len(names) > 30` menu-size threshold with zero margin
that was replaced earlier the same day. Replaced with a prompt-marker
classifier in `conftest.call_kind`, guarded by
`test_call_kind_markers_are_unambiguous`. A bare `"designer"` marker is
insufficient: the reconcile system prompt says "one of two designers".

## Session 2026-08-25: the provider and the WT dataset were both moderators

Full detail, with measurements: **`docs/chamber-harness-validity-register.md`**
(the running register of every harness defect that changed a result). Summary:

1. **Provider order was costing 4.7x and mixing precisions.** One OpenRouter
   model id is served by many endpoints at different prices: identical
   `deepseek-v4-flash-0731` is $0.280/M out on Parasail/SiliconFlow/Baidu and
   $1.320/M on Novita/AtlasCloud/DeepSeek/Cloudflare. M6 ran 422/450 cells on
   Novita, billing **$54.53** total ($49.68 of it Novita's); the same token
   counts at Parasail's prices predict **$12.10**. Re-probed: Parasail
   17-34s vs Novita 21s, so there is no throughput reason to pay it. Order is
   now `(Parasail, SiliconFlow, Baidu, Novita)`, commit `6fe16e5`.
   - `Together` excluded despite the low price: it spends the whole 32768 cap
     on reasoning and returns EMPTY content, which degrades to `rng.choice`.
   - The comment claiming Novita and AtlasCloud were "both fp8" was wrong --
     **AtlasCloud is fp4** and served 27 of the 450 M6 cells. Measured, not
     distorting (residualised on arm x budget: -0.004 vs +0.000, p=0.61).
     Now a `PROVIDER_PRECISION` table plus a mutation-verified test; the old
     test asserted `order == DEFAULT_PROVIDER_ORDER`, true by construction.

2. **WT switched `wt_walks_v1` -> `wt_validate_v1`** (commit `05a811c`). The
   walks release is a random-walk time series, median lag-1 autocorrelation
   **0.9999**, so 320,000 rows carry ~19 independent observations. Fisher-Z
   assumes i.i.d., and on that input the budget response INVERTS. Same menu,
   30 seeds/point: walks slope **-0.0007 (p=0.06)**, validate **+0.0042
   (p=1.4e-13)**, LT reference **+0.0041**. Dynamic range 0.022 -> 0.107.
   **An earlier write-up called the flat walks curve an external-validity
   finding ("the wind tunnel is insensitive to selection"). That is retracted
   -- the pipeline was insensitive, not the chamber.**

3. **PC now drops collinear columns locally instead of aborting globally.**
   WT's four barometers all read ambient pressure in `standard` (all six pairs
   r>0.9998, none a true edge); `cond(R)` ~ 1e7 made Fisher-Z raise and return
   all-zeros for **all 32 nodes**, F1=0. 15/60 runs; now 0/150. Cost: the four
   are pure sinks and 13 of 42 true edges point into them; `pressure_upwind`
   survives, so the three dropped sinks forfeit **9 of 42** -- a recall ceiling
   of 0.786, stated as a WT scope limit.
   Counted as `n_collinear_dropped` and flagged contaminating, because which
   columns are duplicate depends on which experiments were bought (on WT the
   rate moves 0.90 -> 1.00 with k).

**Free results already extracted from the existing 450+180 cells** (no new
compute; see the register and the analysis in-session):
- **Cost-accuracy Pareto**: `planner_reasoner` is the ONLY Pareto-optimal arm
  at k=30 AND k=45 -- cheapest and most accurate. Every fan-in topology is
  strictly dominated. **This retires the M4b "delegation has measurable cost"
  narrative.**
- **Redundancy decomposition**: `fan_in_homog`'s residual against the loop's
  own accuracy-vs-distinct-experiments curve is +0.006 at both k=30 and k=45 --
  its entire -0.079 deficit is duplicated work, not worse selection. `team`
  reaches identical 30/30 coverage and is still -0.047 (paired p=0.0005,
  Holm-adjusted 0.0016 across the 6 within-budget contrasts; unpaired Welch
  gives 0.0001 -- quote the Holm figure), so its cost is genuine coordination,
  not redundancy. **WITHDRAWN 2026-08-30 (M7 Phase 1 + coverage sweeps).**
  That coverage is identical at the EXPERIMENT level and not at the VARIABLE
  level (team 23.4 vs the loop's 27.9): 5.6 variables are bought by both
  scouts while `overlap_frac` reads 0.0 by construction. A direct LLM-free
  manipulation of variable coverage (15 vs 30 variables, weak levels excluded,
  n=30 each) gives **+0.0073 F1 per distinct variable**, which predicts -0.033
  of the measured -0.048 -- **about two-thirds of team's deficit IS
  redundancy**, and the -0.015 residual is below the contrast's own MDE.
  An earlier edit the same day said the conclusion survived; that rested on a
  flat-slope reading over the loop's narrow 25-30 range at n=10, now
  withdrawn. See `docs/chamber-results.md` §"M7 PHASE 1" and register entry 20
  (the first manipulation was confounded with intervention strength).
- **Budget matching verified by identity**: solving `distinct = |A|+|B|-shared`
  against `overlap_frac` gives implied scout budgets of exactly 3/15/22 with
  zero non-integer cells across all fan-in cells.
- **Noise floor**: at k=M selection freedom is zero, so spread there is pure PC
  noise. LT k=59: sd 0.038, max-mean 0.069. WT `wt_validate_v1` k=28: sd 0.076,
  max-mean 0.134. The 0.065 measured on `wt_walks_v1` is STALE.
- **Seed pairing carries no information** (cross-arm r = -0.03), confirming the
  unpinned-temperature note; unpaired MDEs are valid.

## Chamber results → `docs/chamber-results.md`

**All chamber-pillar results live in `docs/chamber-results.md`**, not here.
This file is project memory loaded into every session; it holds instructions,
operational lessons and status. Results are a growing archive and belong in a
document you open deliberately.

As of 2026-08-31: **3,441 cells, $108.39, zero errored cells**, two chambers,
two models. Headlines, with the detail and the caveats in the results doc:

- **Where the comparison resolves, no fan-in topology beats a single
  sequential loop**, and where it resolves in the loop's favour the margin is
  0.040-0.079 F1, on both chambers and under a 3.9x pricier model. Scoped
  2026-08-29, WT `team` re-run 2026-08-30: of 24 topology-vs-loop contrasts,
  10 resolve, 9 favouring the loop. The
  exceptions travel with the claim — `team` beats the loop at WT k=7
  (+0.040, resolved), nothing resolves at LT k=45, and the chain resolves in
  neither direction anywhere.
- **The gap to random closes because random catches up**, not because the
  loop degrades — the loop saturates at F1 ≈ 0.42 by k=30.
- **The contract is a floor on effort, not only a ceiling on spend.**
- **Topology is at least as large a lever as model choice, and cheaper.**
- The aggregator is **inert by measurement**, not by omission (30/30 cells).
- **The running record is not load-bearing** (M7 Phase 2, 2026-08-31, 960
  cells). `one_shot` — ONE call picking all k experiments, no record at all —
  ties the loop at LT k=30/45 and at all three WT budgets, losing only at
  LT k=6 (−0.059). The M6 ordering replicates, but **the record-survival axis
  we built the ladder on does not explain it**; do not draft from that axis.
- **What the axis DOES buy, on both chambers, at the middle budget only**:
  sharing a record beats *splitting* one. `shared_blackboard` vs
  `fan_in_spec` — same two role prompts — gives +0.053 (LT k=30) and +0.046
  (WT k=14), both resolved, nothing at the small or large budget. Sharing a
  record with yourself (the loop) is worth nothing. Cross-run; WT sits on the
  MDE boundary after drift adjustment.
- **Report equivalences with their bound and their power**, never as nulls.
  `one_shot`'s cells are NOT independent draws (register §24): a single call
  re-picks the same design, 6 distinct across 30 cells at LT k=30. Every Phase
  2 verdict survives selection-level re-analysis, but that bound widens to
  ±0.051. **Any single-call arm must be analysed at the selection level, and
  distinct-selection counts belong in every results table.**
- **`critique` TIES the loop** (corrected 2026-09-01 by design-level
  re-scoring): a reviewer pass costs 3 extra flat calls and moves accuracy by
  |Δ| < 0.022 on either chamber at any budget. The earlier "resolved worse at
  LT k=30/45" rested on a single favourable PC draw and is **retracted**.
- **AN LLM-FREE COVERAGE RULE MATCHES EVERY LLM ARM** (2026-09-01, see the
  results doc's "THE COVERAGE ORACLE"). `coverage_max_ms` — round-robin over
  distinct variables, no model — ties the best LLM arm at LT k=6/30/45; none
  resolves above it. **But it is only near-optimal where coverage binds**: at
  k=6 the rule beats random by just +0.007 while the loop beats random by
  +0.056 and the rule by +0.034. So the LLM's contribution is confined to the
  tight-budget regime; above k/M≈0.5 every arm converges on the coverage
  optimum. Treat the rule as a **computable near-oracle** — rare in agent
  benchmarks — and report every arm as distance-from-optimum. **BOTH CHAMBERS**
  as of 2026-09-01: `wt_coverage_max` (`wt_menu_taxonomy.py`, 28 entries / 21
  variables) ties the best LLM at WT k=7/14/21 too, and at k=21 sits 0.021
  ABOVE it. Six budgets, two chambers, no LLM arm resolves above the rule.
  Same small-budget escape on both (rule − random is +0.002 at WT k=7, +0.007
  at LT k=6, rising to +0.045 / +0.073 at the large budgets).
- **Breadth beats depth even where the fat menu entries ARE the real drivers.**
  `wt_coverage_min` was pre-registered to WIN (its variables `hatch`/`load_in`/
  `load_out` have out-degree 6/8/8) and lost badly (0.124/0.165/0.229 vs
  0.188/0.232/0.282). Buying a driver's several entries makes ONE variable vary
  repeatedly; breadth activates a new source each time. **Out-degree is not
  what the budget buys — a distinct varying variable is.**
- **All 9 case studies read** (2026-09-01). Only two are causal discovery:
  `causal_discovery_iid` (LT, GES/UT-IGSP, 20 vars) and `causal_discovery_time`
  (WT, PCMCI+ on `wt_walks_v1`, 16 vars). The others are ICA, changepoints,
  symbolic regression, mechanistic models and three OOD tasks — different
  problems. `lt_interventions_standard_v1` (ours) is also used by `ood_sensors`.
- **The CONTEMPORANEOUS ground truth is bipartite, depth 1, ZERO mediators**
  (register §29). lt/standard 29 sources + 9 sinks; wt/standard 21 + 11. But
  it is **not degenerate** — 172 unshielded colliders on LT, 66 on WT, so
  skeleton recovery and collider orientation are real work; what is absent is
  mediation and high-order conditioning. **The chamber's depth is TEMPORAL**
  (`load_in(t)→rpm_in(t+1)→pressure(t+2)`) and our pooled-i.i.d. reduction
  discards the dimension it lives in.
- **`wt/pressure-control` has real depth but NO MENU — checked and closed
  2026-09-01.** 24 length-2 paths, 45% of its 44 edges on one, against 0% in
  standard. But both its releases ship exactly ONE experiment
  (`wt_pc_validate_v1`: `validate_pressure_downwind_loads`;
  `wt_pressure_control_v1`: `hatch_0`), so a budgeted selection task is
  impossible there. The mediators exist; the interventions revealing them are
  not purchasable.
- **We only ever ran `standard`** — 10,104 LT and 11,086 WT cells, zero on any
  other configuration. It was the CLI default, never a considered choice — but
  it is also the only WT config with a menu, so the oversight was in not
  checking, not in the outcome.
- **`team_varsplit` DOES NOT REPLICATE ON WT** (2026-09-02, 300 cells, n=50,
  both budgets, re-scored at 9 PC seeds). LT k=30 strengthens to **+0.043
  RESOLVED**; WT k=14 is **−0.000** and k=21 **+0.017 below MDE (0.024)**.
  The mechanism fires on both — varsplit buys +0.9 / +1.3 more distinct
  variables, `overlap_frac` 0.000 — but WT's menu (28 entries / 21 variables,
  18 of them singletons) leaves almost no cross-scout duplication to remove,
  against LT's 59/30. **The paper's positive result is one chamber at one
  budget; say so.** Also: the arm is **infeasible at k/M = 0.75** — 2 of 50
  k=21 cells raise because a variable partition cannot leave both scouts a
  pool above budget. Partition-granularity needs menu slack.
- **The cell-level version of that contrast said the opposite** (+0.0155 at
  k=14, +0.0160 at k=21, "stable across budgets"). Nine-seed re-scoring took
  k=14 to zero. §27's failure mode, recurring: **never read a WT contrast off
  single-draw scores.**
- **The authors' own WT case study uses PCMCI+ on `wt_walks_v1`**
  (`causal_discovery_time.ipynb`, tau_max=10, alpha=1e-2, 16 variables). We
  rejected walks for autocorrelation — correct GIVEN PC, but their answer to
  the same autocorrelation is a different METHOD, not a different dataset.
  State our switch as an estimator-forced deviation. **Read every case study a
  testbed ships with, not just the one matching your method.**
- **WT is the worse chamber, not the safer one**: 17 trivial sources carrying
  **40%** of its 42 edges (LT: 18 / 32%), core 15 nodes / 25 edges, plus 9
  in-edges on the collinear-dropped barometers (6 from real drivers).
- **Our node set is 38; the chambers' own case study uses 20** (register §28).
  The 18 extra are ALL pure sources (out-degree 1, in-degree 0) — apparatus
  settings (`t_*`, `osr_*`, `v_*`, `diode_*`) each driving one sensor — and
  they carry 18 of 57 true edges. **78% of the LT loop's budget response sits
  on those edges** (full F1 0.206→0.421 vs core-20 0.176→0.223). Comparisons
  are unaffected (all arms share the node set; Phase 2 verdicts hold under
  directed, skeleton AND core-20 scoring) but **never quote an absolute F1
  without the core-20 figure beside it**. The authors also use GES/UT-IGSP not
  PC, grid-search alpha, score the whole CPDAG, and do not subsample — state
  each as a deviation.
- **Re-score offline before believing a contrast** (`rescore.py`, register
  §27, §30). **Key the work by the ORDERED buy** — pooling concatenates in
  sequence and PC subsamples 300 rows, so `[a,b]` and `[b,a]` score
  differently (0.133 vs 0.105 on LT). `design_key` joins; `selection_key`
  clusters. A frame without `design_key` predates the fix and is refused.
  `--max-workers N` is process-parallel (**3.13x on 4 vCPU**, output
  byte-identical at every setting; BLAS thread count verified not to change
  PC's output — re-probe on a new machine). `chosen_experiments` lets any M7 cell be rebuilt and scored at m PC
  seeds for **$0**; 9 seeds cut MDEs ~35% (LT k=30 0.031→0.019) because the
  averaged-away half is inference noise, not arm. **Cluster by distinct design
  first** (§24). Validated 191/191 exact against production. Only M7-era files
  have the column — M6 ladders and the axis test stay at cell-level MDEs and
  must not be quoted beside the tighter ones.
- **Why the middle budget** (variance probe, 2026-08-31, 3,150 LLM-free PC
  runs). Untying the two things `seed` controls — the buy and PC's 300-row
  subsample — decomposes the spread: **selection variance falls 8x with budget
  (sd 0.036 -> 0.005) while PC noise rises (0.032 -> 0.041)**, so the flat
  total sd hides an inversion. Meanwhile the loop captures 2.1 selection-sd at
  k=30 against 1.3 at k=6. **Room to differ falls with budget; skill at
  exploiting it rises; the payoff peaks where they cross.** That accounts for
  the inverted-U on both chambers, the axis test resolving only mid-range, and
  `one_shot` sitting exactly on random at LT k=6 (0.160 vs 0.163) yet matching
  the loop at k=30.
- **WT half-replicates it** (same day, 3,150 more runs). Skill peaks mid-range
  on both (loop captures +1.4 selection-sd at WT k=14 vs −0.3 at k=7), but
  **WT's room does NOT fall monotonically** (0.046 → 0.027 → 0.040 → 0.007).
  Use the chamber-general sentence — "the payoff peaks mid-range because that
  is where agents exploit the room best" — not the LT-only "room falls while
  skill rises".
- **Most of our MDE is PC noise, not arm variability.** If two arms selected
  identically, noise alone would give MDE 0.031 (LT k=30, n=30) and 0.029
  (WT k=21, n=50) — at or above several observed MDEs. **WT's nine Phase 2
  ties are partly a power result**: WT noise doubles across the budget range
  (0.033 → 0.067). Resolving a 0.02 gap needs **n≈110 (WT k=21)** or **n≈75
  (LT k=30)**. Quote this beside every "below MDE" so it reads as power, not
  as a null. No agent design closes a floor set by the inference procedure.

Paper readiness, the ranked open threats, and the per-dataset index are in
that document's final section. Harness defects stay in
`docs/chamber-harness-validity-register.md` — read it before trusting a number.

## References

- **Chamber results**: `docs/chamber-results.md` (every experiment and what it showed)
- **Related work notes**: `docs/related-work/` — verified reading notes on external
  papers/posts. `2026-08-13-anthropic-multiagent-patterns.md` supplies M7 §1's
  motivating citation (a 12.7x multiagent 'win' on 4.2x the tokens that the authors
  themselves reduce to 'comparable' once scope is matched). **Each file quarantines
  figures that could not be confirmed against source text** — a summarising fetch
  invents plausible numbers for quantities that exist only as chart axes.
- **Harness validity register**: `docs/chamber-harness-validity-register.md`
- **Whitepaper**: `docs/whitepaper.md`
- **Testing Strategy**: `docs/testing-strategy.md`
- **Causal Chamber Plan**: `docs/causal_chamber_validation_plan.md` (full M4/M5/M6 spec for AAMAS 2027 / ECAI 2027)
- **M7 plan + paper positioning**: `docs/superpowers/specs/2026-08-29-m7-mechanism-and-missing-arms.md` — **§8 is the current plan (revised 2026-08-31)**: the framing that survived Phase 2, the four ranked accept/reject threats, and a Phase 3 led by **cross-vendor replication (~$20-30, the highest-value remaining work)**. §1 holds the loop-vs-graph positioning; §5's phase list is superseded by §8's sequencing
- **OR/Optimization Ideas**: `docs/or_optimization_research_ideas.md` (backlog of 3 OR-inspired directions)
- **Repository**: https://github.com/flyersworder/agent-contracts

---

*Last Updated: 2026-08-22 (release 0.5.0: dependency refresh, floors raised to tested versions, slimmed sdist)*
*Status: Production-ready, v0.5.0, 1235 tests passing (1 skipped), 91% coverage*
*Integrations: LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK, Causal Chambers*
*Features: SkillSpec, Per-Tool Limits, Indeterminacy Evaluator, Evaluation Pipelines, JSONL Checkpoint Sidecar, Delegation Graphs*
*Pilot dataset: `runs/m4-pilot.parquet` (450 cells, 442 ok, 8 timeouts) — submission-ready for AAMAS 2027 / ECAI 2027*
*Next: M5 (WT + UNCONTRACTED + Pro robustness), then M6 topology benchmark (§7.7)*
