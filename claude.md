# Agent Contracts Project Memory

This document tracks the development progress and key decisions for the Agent Contracts framework.

## Project Overview

**Agent Contracts** is a formal framework for governing autonomous AI agents through explicit resource constraints and temporal boundaries. The framework implements the theoretical foundations described in `docs/whitepaper.md`.

- **Repository**: https://github.com/flyersworder/agent-contracts
- **Started**: November 1, 2025
- **Primary Developer**: qingye
- **AI Assistant**: Claude (Sonnet 4.5)

## Current Status: Phase 1 Complete ✅

**Date Completed**: November 1, 2025
**Total Tests**: 145 passing
**Code Coverage**: 96%
**Commits**: 6 major implementation commits

## Phase 1: Core Framework Implementation

### 1. Core Contract Data Structures ✅
**Commit**: `07aa66c` - "Implement Phase 1: Core contract data structures"
**File**: `src/agent_contracts/core/contract.py`

Implemented the formal contract definition: C = (I, O, S, R, T, Φ, Ψ)

**Components**:
- `Contract`: Main contract class with lifecycle management
- `ResourceConstraints`: Multi-dimensional resource budget (frozen dataclass)
- `TemporalConstraints`: Time-related boundaries (frozen dataclass)
- `InputSpecification`: Input schema and constraints
- `OutputSpecification`: Output schema and quality criteria
- `SuccessCriterion`: Measurable success conditions
- `TerminationCondition`: Contract termination events
- `ContractState`: Lifecycle states (DRAFTED, ACTIVE, FULFILLED, VIOLATED, EXPIRED, TERMINATED)
- `DeadlineType`: Hard vs soft deadlines

**Tests**: 28 tests, 98% coverage

### 2. Resource Monitoring System ✅
**Commit**: `8412122` - "Implement Phase 1: Resource monitoring system"
**File**: `src/agent_contracts/core/monitor.py`

Real-time resource tracking and constraint validation.

**Components**:
- `ResourceUsage`: Tracks actual consumption across all dimensions
  - Tokens, API calls, web searches, tool invocations
  - Memory (peak tracking), compute time, cost
  - Elapsed time calculation, dictionary export
- `ViolationInfo`: Records constraint violation details
- `ResourceMonitor`: Validates usage against constraints
  - Real-time violation detection
  - Usage percentage calculation
  - Violation history tracking

**Features**:
- Peak memory tracking
- Multi-dimensional constraint validation
- Usage percentage calculation
- Dictionary serialization for logging

**Tests**: 39 tests, 97% coverage

### 3. Token Counting and Cost Tracking ✅
**Commit**: `9032e79` - "Implement Phase 1: Token counting and cost tracking"
**File**: `src/agent_contracts/core/tokens.py`

Token estimation and cost calculation for LLM interactions.

**Components**:
- `TokenCount`: Input/output/total token breakdown
- `CostEstimate`: Input/output/total cost breakdown
- `TokenCounter`: Utility class for estimation and cost calculation
- `MODEL_PRICING`: Pricing database (updated November 2025)

**Pricing Database**:
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini, GPT-3.5 Turbo
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku), Claude 3.5 (Sonnet, Haiku)

**Features**:
- Simple heuristic token counting (~4 chars per token)
- Message array token counting with overhead
- Multimodal content support (images)
- Custom pricing override support
- Case-insensitive model name matching
- Versioned model support (e.g., "gpt-4-0613")

**Tests**: 32 tests, 100% coverage

### 4. Basic Enforcement Mechanisms ✅
**Commit**: `ab1c1f3` - "Implement Phase 1: Basic enforcement mechanisms"
**File**: `src/agent_contracts/core/enforcement.py`

Active constraint monitoring and enforcement during agent execution.

**Components**:
- `EnforcementAction`: Enum for enforcement actions (warn, soft_stop, hard_stop, throttle)
- `EnforcementEvent`: Event class for notifications
- `EnforcementCallback`: Type alias for event callbacks
- `ContractEnforcer`: Main enforcement class

**Features**:
- Start/stop lifecycle management
- Resource constraint checking
- Temporal constraint enforcement (deadlines, duration limits)
- Strict mode: immediate termination on violation
- Lenient mode: warnings only, execution continues
- Event-driven architecture with callbacks
- Usage summary generation
- Graceful error handling in callbacks
- Multiple callback support

**Tests**: 29 tests, 100% coverage

### 5. LiteLLM Integration Wrapper ✅
**Commit**: `0fc531f` - "Implement Phase 1: LiteLLM integration wrapper"
**File**: `src/agent_contracts/integrations/litellm_wrapper.py`
**Dependency**: `litellm>=1.60.0`

Seamless wrapper for LLM API calls with automatic contract enforcement.

**Components**:
- `ContractedLLM`: Main wrapper class
- `ContractViolationError`: Custom exception for violations

**Features**:
- Wraps litellm.completion() for 100+ LLM providers
- Automatic token counting using litellm's built-in tracking
- Real-time cost tracking with fallback to TokenCounter
- Streaming support with periodic constraint checks
- Context manager protocol (`with` statement support)
- Event callbacks for all LLM interactions
- Auto-start functionality for convenience
- Graceful error handling for failed API calls

**Usage Example**:
```python
from agent_contracts import Contract, ResourceConstraints, ContractedLLM

contract = Contract(
    id="chatbot",
    name="Chatbot Agent",
    resources=ResourceConstraints(tokens=10000, api_calls=50, cost_usd=1.0)
)

with ContractedLLM(contract) as llm:
    response = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

**Tests**: 17 tests, 87% coverage

## Development Infrastructure

### Build System
- **Package Manager**: uv (modern Python package manager)
- **Build Backend**: uv_build
- **Layout**: src-layout for proper isolation
- **Python Version**: >=3.12

### Code Quality Tools

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
- `uv-lock`: Lock file synchronization
- `ruff`: Fast Python linter and formatter (replaces black, isort, flake8)
- `mypy`: Static type checking (strict mode)
- `markdownlint`: Markdown file validation
- Standard file checks (trailing whitespace, EOF, YAML, etc.)

**Configuration** (`pyproject.toml`):
- Ruff: Line length 100, comprehensive rule set
- Mypy: Strict mode with full type coverage
- Pytest: Coverage tracking, HTML reports
- All tools configured for Python 3.12+

### Testing Strategy

**Test Organization**:
- `tests/core/`: Core module tests
- `tests/integrations/`: Integration tests
- Mirror structure of `src/agent_contracts/`

**Coverage Metrics**:
- Overall: 96%
- Core modules: 97-100%
- Integration: 87%

**Test Approach**:
- Unit tests for all components
- Integration tests for realistic scenarios
- Mocking for external dependencies (litellm)
- Edge case coverage
- Error condition testing

## Key Design Decisions

### 1. Immutable Constraints
**Decision**: Use frozen dataclasses for `ResourceConstraints` and `TemporalConstraints`
**Rationale**: Prevent accidental modification after contract creation, enforce immutability

### 2. Event-Driven Architecture
**Decision**: Use callback-based events for enforcement notifications
**Rationale**: Enables observability, logging, and custom handling without tight coupling

### 3. Strict vs Lenient Modes
**Decision**: Support both strict (raise errors) and lenient (log warnings) enforcement
**Rationale**: Different use cases require different violation handling strategies

### 4. LiteLLM Integration
**Decision**: Use litellm instead of direct provider SDKs
**Rationale**:
- Universal provider support (100+ models)
- Built-in token counting and cost tracking
- Active maintenance and wide adoption
- Lightweight compared to LangChain

### 5. Heuristic Token Counting
**Decision**: Implement simple character-based estimation (~4 chars/token) as fallback
**Rationale**: No external dependencies, fast approximation for cost estimation

### 6. Context Manager Support
**Decision**: Implement `__enter__` and `__exit__` for ContractedLLM
**Rationale**: Pythonic API, automatic cleanup, clear lifecycle management

## Documentation

### Core Documentation
- `docs/whitepaper.md`: Theoretical framework (1609 lines)
- `docs/testing-strategy.md`: Comprehensive testing plan (850 lines)
- `docs/README.md`: Documentation navigation
- `README.md`: Project overview and citation info

### Code Documentation
- Full type hints on all functions (Python 3.12+ syntax)
- Comprehensive docstrings with Args/Returns/Raises
- References to whitepaper sections in docstrings
- Examples in integration tests

## File Structure

```
agent-contracts/
├── src/
│   └── agent_contracts/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── contract.py        # Core data structures
│       │   ├── monitor.py         # Resource monitoring
│       │   ├── tokens.py          # Token counting
│       │   └── enforcement.py     # Enforcement mechanisms
│       └── integrations/
│           ├── __init__.py
│           └── litellm_wrapper.py # LiteLLM integration
├── tests/
│   ├── core/
│   │   ├── test_contract.py
│   │   ├── test_monitor.py
│   │   ├── test_tokens.py
│   │   └── test_enforcement.py
│   └── integrations/
│       └── test_litellm_wrapper.py
├── docs/
│   ├── README.md
│   ├── whitepaper.md
│   └── testing-strategy.md
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## Next Steps (Phase 2)

According to `docs/testing-strategy.md`, the roadmap includes:

### Phase 2: Advanced Features (Weeks 3-4)
- [ ] Quality metrics and monitoring
- [ ] Skill verification system
- [ ] Success criteria evaluation engine
- [ ] Contract composition and delegation
- [ ] Advanced temporal constraints (soft deadlines with quality decay)

### Phase 3: Integration & Validation (Weeks 5-6)
- [ ] Framework adapters (LangChain, AutoGen, CrewAI)
- [ ] Benchmark suite implementation (15 tasks from testing strategy)
- [ ] Performance optimization
- [ ] Real-world validation studies

### Phase 4: Production Readiness (Weeks 7-8)
- [ ] API documentation
- [ ] Example gallery
- [ ] Performance benchmarks
- [ ] Security audit
- [ ] Release preparation

## Lessons Learned

1. **Start with solid foundations**: The formal contract definition (whitepaper) guided all implementation decisions
2. **Test-driven development**: 96% coverage caught many edge cases early
3. **Modern tooling**: uv, ruff, and mypy dramatically improved development speed
4. **Clear separation**: Core framework independent of integrations enables flexibility
5. **Type safety**: Strict mypy checking prevented many runtime errors

## Open Questions / Future Considerations

1. **Async Support**: Should we add async/await support for ContractedLLM?
2. **Persistent Storage**: How to serialize/deserialize contracts for long-running agents?
3. **Distributed Enforcement**: How to enforce contracts across multiple processes/machines?
4. **Cost Attribution**: How to attribute shared costs in multi-agent scenarios?
5. **Contract Negotiation**: Should contracts be negotiable between agents?

## References

- **Whitepaper**: `docs/whitepaper.md`
- **Testing Strategy**: `docs/testing-strategy.md`
- **Repository**: https://github.com/flyersworder/agent-contracts
- **LiteLLM Documentation**: https://docs.litellm.ai/

---

*Last Updated: November 1, 2025*
*Phase: 1 (Complete)*
*Next Milestone: Begin Phase 2 Advanced Features*
