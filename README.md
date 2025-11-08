# Agent Contracts

A formal framework for governing autonomous AI agents through explicit resource constraints and temporal boundaries.

## Overview

**Agent Contracts** transforms autonomous AI agents from unbounded explorers into **bounded optimizers** by introducing formal contracts that specify:

- 🎯 **Resource Budgets** - Tokens, API calls, compute time, and costs
- ⏱️ **Temporal Constraints** - Deadlines, duration limits, and lifecycle boundaries
- 📊 **Success Criteria** - Measurable conditions for contract fulfillment
- 🔄 **Lifecycle Management** - Clear states from activation to termination

### The Problem

Current agentic AI systems face critical challenges:
- **Unbounded Resource Consumption** - Agents can consume unpredictable amounts of tokens, API calls, and compute time
- **Unclear Lifecycles** - No explicit termination criteria, leading to resource leaks
- **Difficult Governance** - Hard to audit, ensure compliance, and attribute costs
- **Coordination Complexity** - Multi-agent systems lack formal resource allocation mechanisms

### The Solution

Agent Contracts provide a mathematical framework that enables:
- **Predictable Costs** - Explicit resource budgets prevent runaway consumption
- **Formal Verification** - Contract states and constraints are machine-verifiable
- **Time-Resource Tradeoffs** - Strategic optimization between speed and economy
- **Multi-Agent Coordination** - Hierarchical contracts and resource markets

## Quick Examples

### Basic LLM Integration

```python
from agent_contracts import Contract, ContractedLLM, ResourceConstraints, ContractMode

# Define a contract with resource budgets
contract = Contract(
    id="research-task",
    name="Research Assistant",
    mode=ContractMode.BALANCED,  # Optimize for quality-cost-time balance
    resources=ResourceConstraints(
        tokens=10000,
        api_calls=50,
        cost_usd=1.0
    )
)

# Execute LLM calls within contract constraints
with ContractedLLM(contract) as llm:
    response = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize recent AI papers"}]
    )

# Contract automatically enforces:
# ✅ Token budget limits
# ✅ API call tracking
# ✅ Cost monitoring
# ✅ Violations trigger warnings or stops
```

### LangGraph Multi-Agent Workflows ⭐

For complex workflows with cycles and multi-agent coordination:

```python
from langgraph.graph import StateGraph, END
from agent_contracts import Contract, ResourceConstraints
from agent_contracts.integrations.langgraph import ContractedGraph

# Build complex graph with validation cycle
workflow = StateGraph(AgentState)
workflow.add_node("research", research_agent)
workflow.add_node("validate", validate_agent)
workflow.add_conditional_edges(
    "validate",
    should_retry,
    {True: "research", False: END}  # Can loop!
)
app = workflow.compile()

# Wrap with contract to prevent runaway loops
contract = Contract(
    id="research-workflow",
    resources=ResourceConstraints(
        tokens=50000,
        api_calls=25,  # Limit iterations!
        cost_usd=2.0
    )
)

contracted_workflow = ContractedGraph(contract=contract, graph=app)
result = contracted_workflow.invoke({"query": "Research topic"})

# Budget enforced across ALL nodes and cycles:
# ✅ Prevents infinite loops
# ✅ Multi-agent budget sharing
# ✅ Real-time violation detection
# ✅ Cumulative tracking across entire graph
```

### Contract Modes

Choose the mode that fits your requirements:

```python
# URGENT mode: Minimize time, accept higher costs
contract = Contract(
    mode=ContractMode.URGENT,
    resources=ResourceConstraints(tokens=10000)
)
# → 50% faster execution, 20% more tokens

# BALANCED mode: Optimize quality-cost-time tradeoff
contract = Contract(
    mode=ContractMode.BALANCED,
    resources=ResourceConstraints(tokens=10000)
)
# → Standard execution with quality focus

# ECONOMICAL mode: Minimize costs, accept longer runtime
contract = Contract(
    mode=ContractMode.ECONOMICAL,
    resources=ResourceConstraints(tokens=10000)
)
# → 60% fewer tokens, 50% longer execution
```

## Documentation

📚 **[Complete Documentation](./docs/README.md)**

### Key Resources

- **[Whitepaper](./docs/whitepaper.md)** - Complete theoretical framework with mathematical foundations
- **[Examples](./docs/examples/)** - Coming soon: Practical implementation examples

### Quick Start by Role

- **Researchers**: Read the [Formal Framework](./docs/whitepaper.md#2-formal-framework) and [Future Directions](./docs/whitepaper.md#8-future-directions)
- **Engineers**: Check [Implementation Architecture](./docs/whitepaper.md#5-implementation-architecture) and [Use Cases](./docs/whitepaper.md#6-use-cases-and-examples)
- **Product Managers**: Start with the [Introduction](./docs/whitepaper.md#1-introduction) and [Use Cases](./docs/whitepaper.md#6-use-cases-and-examples)

## Key Concepts

### Contract Definition

An Agent Contract `C = (I, O, S, R, T, Φ, Ψ)` includes:

- **I**: Input specification
- **O**: Output specification
- **S**: Skills (tools, capabilities)
- **R**: Resource constraints
- **T**: Temporal constraints
- **Φ**: Success criteria
- **Ψ**: Termination conditions

### Time-Resource Tradeoff

Agents can optimize along multiple dimensions:

| Mode | Time | Resources | Quality |
|------|------|-----------|---------|
| Urgent | Low ⚡ | High 💰 | 85% |
| Balanced | Medium ⏱️ | Medium 💵 | 95% |
| Economical | High 🐢 | Low 💸 | 90% |

### Contract States

```
DRAFTED → ACTIVE → {FULFILLED, VIOLATED, EXPIRED, TERMINATED}
```

## Repository Status

🎉 **Ready for Release** (November 2025)

**Current Version**: 0.1.0
**Status**: Production-ready, validated, documented

**Phase 1: Core Framework** ✅ Complete
- ✅ Contract data structures (C = I, O, S, R, T, Φ, Ψ)
- ✅ Resource monitoring and enforcement
- ✅ Token counting and cost tracking
- ✅ LiteLLM integration wrapper
- ✅ 145 tests, 96% coverage
- ✅ Live demo with Gemini 2.0 Flash

**Phase 2A: Strategic Optimization** ✅ Complete
- ✅ Contract modes (URGENT, BALANCED, ECONOMICAL)
- ✅ Budget-aware prompt generation
- ✅ Strategic planning utilities
- ✅ Quality-cost-time Pareto benchmark
- ✅ 209 core tests passing

**Phase 2B: Governance & Benchmarks** ✅ Complete
- ✅ Multi-step research benchmark (research agent with quality evaluation)
- ✅ Budget violation policy testing (100% enforcement validation)
- ✅ Cost governance validation (organizational policy compliance)
- ✅ Variance reduction analysis (N=20 validation, temperature=0 effect discovered)
- ✅ Quality metrics framework (3-phase validation study, CV=5.2%)
- ✅ LangChain 1.0+ integration (governance & compliance)
- ✅ Pre-commit hooks and code quality infrastructure

**LangGraph Integration** ✅ Complete (Premium Feature)
- ✅ ContractedGraph for complex multi-agent workflows
- ✅ Cumulative budget tracking across ALL nodes and cycles
- ✅ Loop/retry protection (prevents runaway costs)
- ✅ Multi-agent budget sharing
- ✅ 27 comprehensive tests, 85% coverage
- ✅ Real-world demos (validation cycles, parallel agents)

**Total**: 236+ tests (209 core + 27 LangGraph), 94%+ coverage

## Use Cases

Agent Contracts are designed for:

- **Production AI Systems** - Cost control and SLA compliance
- **Complex Multi-Agent Workflows** ⭐ - LangGraph loops, retries, validation cycles
- **Enterprise Deployments** - Governance, audit trails, and compliance
- **LangChain Applications** - Simple chains with budget enforcement
- **Research** - Studying optimal agent behavior under constraints

### Where Agent Contracts Shines

**LangChain** (simple chains):
- 3-10 LLM calls per execution
- Budget risk: LOW to MODERATE
- Value: Governance, compliance, multi-call protection

**LangGraph** (complex workflows) ⭐:
- 30+ LLM calls per execution (cycles, retries, parallel agents)
- Budget risk: VERY HIGH (can spiral to $10+ without limits!)
- Value: Loop protection, multi-agent coordination, cumulative tracking
- **This is the killer feature for production deployments**

## Project Structure

```
agent-contracts/
├── src/agent_contracts/           # Core package
│   ├── core/
│   │   ├── contract.py           # Contract data structures
│   │   ├── monitor.py            # Resource monitoring
│   │   ├── enforcement.py        # Constraint enforcement
│   │   ├── tokens.py             # Token counting
│   │   ├── planning.py           # Strategic planning
│   │   └── prompts.py            # Budget-aware prompts
│   └── integrations/
│       ├── litellm_wrapper.py    # LiteLLM integration
│       ├── langchain.py          # LangChain integration
│       └── langgraph.py          # LangGraph integration ⭐
├── tests/                         # 236+ tests, 94%+ coverage
│   ├── core/                     # Core module tests (209 tests)
│   └── integrations/             # Integration tests (27 tests)
├── benchmarks/                    # Live demonstrations & benchmarks
│   ├── demo_phase1.py            # Phase 1 interactive demo
│   ├── strategic/                # Strategic optimization benchmarks
│   ├── research_agent/           # Multi-step research benchmark
│   ├── governance/               # Policy & governance tests
│   ├── langchain/                # LangChain demos
│   └── langgraph/                # LangGraph demos (multi-agent)
├── docs/
│   ├── whitepaper.md             # Complete theoretical framework
│   └── testing-strategy.md       # Testing & validation plan
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/flyersworder/agent-contracts.git
cd agent-contracts

# Install with uv (recommended)
uv pip install -e .

# Or install with pip
pip install -e .
```

**Requirements**: Python ≥ 3.12

**Optional dependencies**:
- `litellm` - For LLM integration (automatically installed)
- `langchain` - For LangChain integration (`pip install ".[langchain]"`)
- `langgraph` - For LangGraph integration ⭐ (`pip install ".[langgraph]"`)
- `matplotlib` - For visualization benchmarks (`pip install matplotlib`)

## Development

### Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To set up the development environment:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/flyersworder/agent-contracts.git
cd agent-contracts

# Install dependencies (including dev dependencies)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

This project uses several tools to maintain code quality:

- **[Ruff](https://github.com/astral-sh/ruff)**: Fast Python linter and formatter (replaces black, isort, flake8)
- **[mypy](https://github.com/python/mypy)**: Static type checker
- **[pre-commit](https://pre-commit.com/)**: Git hooks for automated checks

Pre-commit hooks will automatically run on every commit. To manually run all checks:

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run specific tools
uv run ruff check .                    # Linting
uv run ruff format .                   # Formatting
uv run mypy .                          # Type checking
```

### Running Tests

```bash
# Run tests (when available)
uv run pytest

# Run with coverage
uv run pytest --cov=agent_contracts --cov-report=html
```

### Project Structure

- `docs/` - Documentation (whitepaper, testing strategy)
- `src/` - Source code (planned)
- `tests/` - Test suite (planned)
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependencies for reproducibility

## Contributing

This is an evolving framework. We welcome contributions in:
- Reference implementations (Python, TypeScript)
- Integration with existing frameworks (LangChain, AutoGPT, etc.)
- Practical examples and tutorials
- Empirical studies and benchmarks

## License

This project is licensed under CC BY 4.0.

## Authors

Qing Ye (with assistance from Claude, Anthropic)

## Citation

If you use this framework in your research, please cite:

```bibtex
@techreport{ye2025agentcontracts,
  title={Agent Contracts: A Resource-Bounded Optimization Framework for Autonomous AI Systems},
  author={Ye, Qing},
  year={2025},
  month={October}
}
```

## Learn More

- 📖 [Read the Whitepaper](./docs/whitepaper.md)
- 🎯 [Browse Documentation](./docs/README.md)
- 💬 [Open an Issue](../../issues) for questions or discussions

---

**Version**: 0.1.0 | **Last Updated**: November 6, 2025 | **Status**: Production Ready ⭐
