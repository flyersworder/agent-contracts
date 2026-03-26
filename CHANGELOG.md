# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/flyersworder/agent-contracts/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/flyersworder/agent-contracts/releases/tag/v0.1.0
