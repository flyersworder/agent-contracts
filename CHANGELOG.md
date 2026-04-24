# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/flyersworder/agent-contracts/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/flyersworder/agent-contracts/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/flyersworder/agent-contracts/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/flyersworder/agent-contracts/releases/tag/v0.1.0
