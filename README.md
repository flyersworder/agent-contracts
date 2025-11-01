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

## Quick Example

```python
# Define a contract for a code review agent
contract = Contract(
    name="PR Review Agent",
    resources={
        "tokens": 50000,
        "api_calls": 30,
        "cost": 2.50  # USD
    },
    temporal={
        "deadline": "5 minutes",
        "urgency": "high"
    },
    success_criteria=[
        {"name": "completion", "weight": 0.4},
        {"name": "accuracy", "weight": 0.3},
        {"name": "timeliness", "weight": 0.3}
    ]
)

# Execute within contract constraints
agent = ContractAgent(contract)
result = agent.execute(pull_request)

# Contract automatically enforces:
# - Resource consumption limits
# - Deadline compliance
# - Quality-speed tradeoffs
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

🚧 **Early Development**

- ✅ Theoretical framework complete
- ⏳ Reference implementation in progress
- ⏳ Integration examples coming soon

## Use Cases

Agent Contracts are designed for:

- **Production AI Systems** - Cost control and SLA compliance
- **Multi-Agent Systems** - Resource coordination and task distribution
- **Enterprise Deployments** - Governance, audit trails, and compliance
- **Research** - Studying optimal agent behavior under constraints

## Project Structure

```
agent-contracts/
├── docs/
│   ├── README.md          # Documentation index
│   ├── whitepaper.md      # Complete theoretical framework
│   └── examples/          # Code examples (coming soon)
├── src/                   # Reference implementation (planned)
├── tests/                 # Test suite (planned)
└── README.md              # This file
```

## Installation

*Installation instructions will be added when the reference implementation is available.*

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
@techreport{ye2024agentcontracts,
  title={Agent Contracts: A Resource-Bounded Optimization Framework for Autonomous AI Systems},
  author={Ye, Qing},
  year={2024},
  month={October}
}
```

## Learn More

- 📖 [Read the Whitepaper](./docs/whitepaper.md)
- 🎯 [Browse Documentation](./docs/README.md)
- 💬 [Open an Issue](../../issues) for questions or discussions

---

**Version**: 1.0 | **Last Updated**: October 29, 2024
