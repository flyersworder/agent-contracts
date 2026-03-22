# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Thread safety for ResourceUsage and ResourceMonitor
- Shared token extraction utilities for integrations (`_token_utils.py`)
- Gemini model pricing in MODEL_PRICING database
- Shared test fixtures in `tests/conftest.py`
- CI: linting (ruff), type checking (mypy), Python 3.12/3.13 matrix
- LICENSE file (CC-BY-4.0)

### Changed
- Moved `datasets` and `matplotlib` to optional `[eval]` dependencies
- Extracted SkillSpec to dedicated `core/skillspec.py` module
- Extracted Capabilities, AgentSpec, ExecutionConfig to `core/capabilities.py`
- Renamed executor's `ExecutionResult` to `ContractExecutionResult`
- Renamed LangChain's `ContractedLLM` to `ContractedChainLLM`
- Trimmed public API surface from ~58 to ~28 exports
- ResourceMonitor ownership via dependency injection in ContractEnforcer

### Fixed
- State machine bypass: wrapper now uses `contract.violate()` instead of direct state assignment
- Unified `ContractViolationError` to single canonical type across all integrations
- Duplicate violation recording in ResourceMonitor
- Silent exception swallowing replaced with proper logging
- `raise e` changed to bare `raise` to preserve tracebacks
- Removed dead code: no-op `_setup_node_tracking`, simplified `ContractedAdkMultiAgent`

## [0.1.0] - Unreleased

### Added
- Core framework: Contract, ResourceConstraints, TemporalConstraints
- Resource monitoring with real-time tracking
- Constraint enforcement with strict/lenient modes
- LiteLLM integration (100+ providers)
- LangChain integration
- LangGraph integration with cycle protection
- Google ADK integration
- SkillSpec (agentskills.io standard)
- Per-tool resource limits
- Indeterminacy-aware LLM-as-Judge evaluator
- Research and Code Review evaluation pipelines
- Strategic contract modes (URGENT, ECONOMICAL, BALANCED)
- Budget-aware prompt generation
- Contract delegation with conservation laws
