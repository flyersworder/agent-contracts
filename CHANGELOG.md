# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The dependency graph moved into the `httpx2` / MCP 2.0 era** (151 → 153 packages). The HTTP client ecosystem has forked, and this project now resolves both halves of it at once. `httpx2` is a *separate distribution*, not httpx version 2.0 — classic `httpx` remains at 0.28.1, and `httpx2` renames both the package and the import namespace (`import httpx2`), so the two are distinct modules that cannot collide. `openai` 3.x, `anthropic` 1.x, and `mcp` 2.x have crossed over; `litellm` 1.97 has not, and additionally caps `openai<3.0.0`. The result is `httpx` 0.28.1 and `httpx2` 2.12.0 installed side by side — expected, not a resolution fault, and it persists until litellm migrates.

  Headline moves: `mcp` 1.28.1 → 2.0.0 (protocol revision 2026-07-28), `claude-agent-sdk` 0.1.50 → 0.2.143, `openai` 2.24.0 → 2.54.0, `litellm` 1.85.0 → 1.97.0, `google-adk` 2.2.0 → 2.7.1, plus `mypy` 1.19.1 → 2.3.1 and `ruff` 0.14.10 → 0.16.4 in the dev group.

  **MCP 2.0 does not change this project's public API.** `mcp` is never imported directly — `Capabilities.mcp_servers` emits the provider-side remote-MCP tool schema consumed by LiteLLM, which is a wire format independent of the Python SDK, and the Claude Agent SDK integration passes servers through unchanged. Every v2 breaking change lands in SDK surface this project does not touch. Users pinning `mcp` themselves should note that the 1.x line is now security-fixes-only.

- **Dependency floors raised to the versions actually under test.** The declared floors had drifted well below what CI exercises — `langchain>=0.3.0` while testing 1.3.16, `langgraph>=0.2.0` while testing 1.2.11, `google-adk>=1.18.0` while testing 2.7.1, `pandas>=2.0` while testing 3.0.5 — so the package advertised support for combinations that were never verified. Floors now sit at the tested `major.minor`.

  This narrows what the extras will install: `pip install ai-agent-contracts[langchain]` no longer resolves LangChain 0.3.x, and the equivalent applies to the `langgraph`, `google-adk`, `claude-agent-sdk`, `eval`, and `chambers` extras. **The base package is effectively unaffected** — its only runtime dependency is `python-dotenv`, whose floor moved 1.0.0 → 1.2. No upper bounds were introduced, so adopting a new major of any integration remains possible without waiting on a release here; the pre-existing `numpy<2.6` bound is retained. Re-locking after the change produced a byte-identical resolution, so no installed version moved.

- **The sdist now ships only the library, its test suite, and metadata** — 2.9 MB to 196 KB, a 93% reduction. Every previous release published the whole repository, so `ai_agent_contracts-0.4.0.tar.gz` carried 2.6 MB of experiment pipelines and figures from `evaluation/`, 1.7 MB of run output from `results/` (a single 984 KB JSON file among it), plus `benchmarks/`, `docs/`, `uv.lock`, and the project's AI memory file. None of it is needed to build, install, or verify the package. `pyproject.toml` gains an explicit `[tool.hatch.build.targets.sdist]` allowlist, so new top-level directories cannot start shipping unnoticed.

  `tests/evaluation` is excluded along with the `evaluation/` package it covers — those tests failed collection from an sdist in any environment, since their subject was never present. The remaining suite runs from the extracted archive: 876 passed, 24 skipped.

  **The wheel is unaffected** — rebuilt after this change it is bit-for-bit identical (sha256 `104b99ee…`) to the wheel published for 0.4.0, so `pip install ai-agent-contracts` is unchanged. Only source installs, mirrors, and downstream packagers see the difference.

## [0.4.0] - 2026-07-25

### Added

**Delegation graphs (flow conservation)**

- `DelegationGraph` generalizes budget conservation from a tree to a DAG, so a node can be funded by more than one parent. `ContractingCapability` modelled a strict hierarchy — every child had exactly one parent — which cannot express an aggregator merging two independent workers, or any shared downstream budget. Under the tree law such a node was double-counted.
- The invariant checked at every node is `in-flow ≥ own consumption + out-flow`. Local checks imply the global bound `Σ consumption ≤ root budget` by a telescoping argument: every internal allocation appears once as in-flow at its head and once as out-flow at its tail, so internal terms cancel. This means node-level checks need no global lock and no central accountant. See whitepaper §4.6.
- `ResourceVector`: resource arithmetic (`+`, `-`, `<=`) across tokens, cost, tool invocations, iterations, and per-tool counts, with `None` meaning *unbounded* rather than zero on every axis.
- `seal()` validates a graph once and freezes its topology, reporting every problem it finds rather than one per attempt: budget cycles, unfunded nodes, out-flow already exceeding in-flow, and per-tool grants the funding does not support.
- `release()` reclaims an edge's proportional share of a target's unused budget, computed against *original* allocations so that releasing sibling edges in any order yields the same result. Computing against live values would make each sibling's refund depend on release order.
- `abandon()` handles a node that timed out or crashed: it refunds the node's unconsumed budget to its parents and freezes its flow state, so an over-spent node stays flagged instead of having its violation erased by the refund.
- Per-tool budget is conserved on both paths — a node may neither grant nor consume a tool its in-flow does not constrain. Granting is *prevented* at allocation time; consuming an undeclared tool is *detected* at verification, since per-tool limits have no deny-by-default semantics. Granting an explicit zero converts detection into prevention.
- `FlowConservationError` subclasses the existing `ConservationViolationError`, so code already catching conservation failures also catches flow failures. It carries the node, dimension, in-flow, consumption, out-flow, deficit, and the funding edges as an audit trail.
- Exported from `agent_contracts.core`: `DelegationGraph`, `ResourceVector`, `EdgeAllocation`, `GraphNode`, `FlowConservationError`, `CycleError`, `GraphLintError`.

**Iteration tracking**

- `ResourceUsage.iterations` and `add_iteration()`. `ResourceConstraints.iterations` had been a declared budget since 0.1.0 but was never tracked, so the limit could not be enforced. `ResourceMonitor.check_constraints()` now reports an `iterations` violation. Note that `add_iteration()` is not yet called from any integration — wiring it into the LLM call paths, and adding `iterations` to `ResourceUsage.to_dict()`, remains open.

### Changed

- `core/delegation.py` is untouched. The tree law remains the single-parent special case of the flow law, and the equivalence is verified by a test that runs both implementations over identical scenarios and compares their accounting. That test also documents the two places they legitimately differ: the tree law's optional coordination reserve has no graph analogue, and a tree's remaining budget clamps at zero where a graph residual goes negative so an overrun stays visible.

### Notes

- **What verification certifies under abandonment.** The telescoping proof holds for the live graph. Because an abandoned node is checked against its frozen pre-refund in-flow, verification certifies `Σ consumption ≤ root budget + Σ refunds`, with the two coinciding when nothing is abandoned. The bound is tight — an abandoned node can consume up to its refunded amount before detection. Checking against *live* in-flow instead would put honest nodes on a knife-edge equality, since a refund drives post-refund in-flow to approximately `consumption + out-flow`.
- `ResourceVector.per_tool` is a read-only mapping (`MappingProxyType` over a defensive copy), matching the existing `ResourceConstraints.per_tool_limits` convention. `ResourceVector.ZERO` is a module-level singleton returned verbatim by several `DelegationGraph` queries, so a plain mutable dict here would have let a caller poison it — and every other graph's zero reads — through one returned reference.
- **Where each half of the invariant is enforced.** Materializing a node's contract from its summed in-flow means the existing `ResourceMonitor` enforces `consumption ≤ in-flow` node-locally, with strict/lenient modes and callbacks applying unchanged. The `+ out-flow` term has no node-local analogue — a contract does not know what it has delegated — and is enforced by graph-level verification alone.
- `DelegationGraph` is not thread-safe. `allocate`, `release`, and `abandon` each read then write shared residuals.
- Reclaimed budget is not re-delegatable in v1: allocation is a build-phase operation and reclamation a run-phase one, so refunds change accounting and reporting but nothing can re-spend them. `abandon()` likewise refunds only a node's in-edges; budget it had already delegated downstream stays with the child.

## [0.3.2] - 2026-05-20

### Fixed

- **`ResourceConstraints` immutability.** Although `ResourceConstraints` is a frozen dataclass, its `per_tool_limits` field was a plain `dict` — mutable after construction and aliased to the dict the caller passed in, so a frozen contract could be silently modified. `per_tool_limits` is now stored as a read-only `MappingProxyType` over a defensive copy.
- **`get_model_pricing` versioned-model resolution.** The prefix match returned the *first* matching key, so versioned model ids resolved to the wrong model — `gpt-4o-2024-08-06` was priced as `gpt-4`, an order-of-magnitude error. The longest matching prefix now wins.
- **`SkillSpec` name validation.** The name regex was off by one and rejected valid 64-character names (the agentskills.io limit is 1–64), and it accepted consecutive hyphens that its own error message forbids. Validation now uses a structural regex (no consecutive, leading, or trailing hyphens) plus an explicit 1–64 length check.
- **`ResourceConstraints` rejects boolean budgets.** Because `bool` is a subclass of `int`, `tokens=True` and boolean `per_tool_limits` values silently passed validation. Booleans are now rejected on every numeric axis.
- **Cost-axis conservation tolerance.** Conservation checks on the USD cost axis used an unguarded floating-point comparison, so an exact-budget split such as `0.1 + 0.1 + 0.1` against a `0.3` budget spuriously raised `ConservationViolationError`. Cost comparisons now apply a `1e-9` tolerance; the integer-valued token and per-tool axes are unaffected.
- **`ContractAgent` strict mode.** `strict_mode=True` never raised `ContractViolationError` despite the documented behavior, making strict and lenient modes indistinguishable at the wrapper API. Strict mode now raises on a constraint violation; lenient mode still returns an `ExecutionResult`.

## [0.3.1] - 2026-04-24

### Changed

- **`litellm` moved from required to optional dependency.** `litellm` is used only by `ContractedLLM` in `integrations/litellm_wrapper.py`. Treating it as a required dependency pulled in ~70 hard-pinned transitive dependencies (including `aiohttp==3.13.3` with several active CVEs) for every install — including users who only use the `Contract` / `ContractExecutor` surface with a different LLM integration (LangChain, LangGraph, Google ADK, or Claude Agent SDK). `litellm` now joins the existing pattern of optional integration extras (`langchain`, `langgraph`, `google-adk`, `claude-agent-sdk`).
- **`ContractedLLM` is a conditional import** in both `agent_contracts` and `agent_contracts.integrations`, matching the pattern used for the other integrations. A new `LITELLM_AVAILABLE` flag is exported for runtime capability checks.

### Migration

- If you were installing `ai-agent-contracts` (no extras) and using `ContractedLLM` or `ContractExecutor`, switch to `pip install ai-agent-contracts[litellm]`. Importing `ContractedLLM` when `litellm` is not installed now yields `None` at import time rather than a hard `ImportError` (mirroring the other optional integrations).

## [0.3.0] - 2026-03-28

### Added

**Pre-Execution Hooks**
- `CheckContext` frozen dataclass: contract, monitor, phase, and integration metadata
- `HookResult` frozen dataclass: allow/block with configurable action severity
- `CheckHook` type alias for hook callables
- `pre_check_hooks` and `post_check_hooks` on `ContractEnforcer.__init__`
- `metadata` parameter on `ContractEnforcer.check_constraints()` (backward-compatible)
- `add_pre_check_hook()`, `remove_pre_check_hook()`, `add_post_check_hook()`, `remove_post_check_hook()` methods
- Hook actions: WARN/THROTTLE (informational, non-blocking) and SOFT_STOP/HARD_STOP (blocking)
- Post-check hooks are observational only (cannot block execution)
- Integration metadata pass-through from all 5 integrations (LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK)
- Claude Agent SDK `_pre_tool_use_hook` refactored to route through enforcer for hook consistency
- Defensive copy of metadata dict to prevent cross-hook mutation
- Exception safety: hook errors caught and logged, never crash enforcement
- 23 new tests, 646+ total tests passing
- Documentation: `docs/pre-execution-hooks.md` with usage guide and behavioral monitor design

### Changed
- `ContractEnforcer.check_constraints()` signature: added optional `metadata` parameter (fully backward-compatible)
- Claude Agent SDK `aexecute()` now routes constraint checks through enforcer instead of directly calling monitor

## [0.2.0] - 2026-03-27

### Added

**Claude Agent SDK Integration**
- `ContractedClaudeAgent` for governing Claude Agent SDK agents
- Hook-based enforcement via `PreToolUse` (blocks) and `PostToolUse` (audit)
- Exact token tracking from `AssistantMessage.usage`
- SDK-native limit mapping: `iterations` → `max_turns`, `cost_usd` → `max_budget_usd`
- Per-tool limits, web search limits, and temporal enforcement via hooks
- Dual API: `aexecute()` (async) and `execute()` (sync)
- Full passthrough of SDK features (tools, MCP servers, subagents, skills, permissions)
- 33 tests with mocked SDK

## [0.1.0] - 2026-03-26

Initial public release. A formal framework for governing autonomous AI agents through
explicit resource constraints and temporal boundaries.

### Added

**Core Framework**
- Contract data structures (C = I, O, S, R, T, Phi, Psi)
- ResourceConstraints: tokens, API calls, cost, iterations, per-tool limits
- TemporalConstraints: deadlines, max duration, soft/hard deadline types
- Resource monitoring with real-time tracking (thread-safe)
- Constraint enforcement with strict/lenient modes and callbacks
- Contract delegation with conservation laws
- Strategic contract modes (URGENT, ECONOMICAL, BALANCED)
- Budget-aware prompt generation

**Integrations**
- LiteLLM integration (100+ LLM providers)
- LangChain integration with multi-call budget protection
- LangGraph integration with cycle/loop protection
- Google ADK integration

**Extensions**
- SkillSpec: agentskills.io standard for reusable agent behaviors
- Per-tool resource limits (fine-grained control per tool name)
- Indeterminacy-aware LLM-as-Judge evaluator (NeurIPS 2025 framework)
- Research and Code Review evaluation pipelines

**Infrastructure**
- CI: linting (ruff), type checking (mypy), Python 3.12/3.13 matrix
- 609+ tests, 91%+ coverage
- Shared test fixtures in `tests/conftest.py`
- Shared token extraction utilities for integrations (`_token_utils.py`)
- LICENSE file (Apache-2.0)

### Changed
- License changed from CC-BY-4.0 (paper) to Apache-2.0 (software)
- PyPI package name: `ai-agent-contracts` (the name `agent-contracts` was already taken)

[Unreleased]: https://github.com/flyersworder/agent-contracts/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/flyersworder/agent-contracts/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/flyersworder/agent-contracts/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/flyersworder/agent-contracts/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/flyersworder/agent-contracts/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flyersworder/agent-contracts/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flyersworder/agent-contracts/releases/tag/v0.1.0
