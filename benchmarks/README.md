# Agent Contracts Benchmarks

This directory contains comprehensive benchmarks and demonstrations for the Agent Contracts framework.

## Quick Reference

| Benchmark | Purpose | Command |
|-----------|---------|---------|
| Research Agent | Multi-step agent comparison | `uv run python -m benchmarks.research_agent.benchmark` |
| Governance | Budget enforcement validation | `uv run python -m benchmarks.governance.budget_violation_test` |
| Strategic Modes | Pareto frontier testing | `uv run python -m benchmarks.strategic.strategic_optimization_test` |
| LangChain | LangChain integration demo | `uv run python benchmarks/langchain/demo_integration.py` |
| LangGraph | LangGraph integration demo | `uv run python benchmarks/langgraph/demo_integration.py` |
| Google ADK | Google ADK integration demo | `uv run python benchmarks/google_adk/demo_integration.py` |

## Benchmark Categories

### 1. Research Agent Benchmark (`research_agent/`)

The core benchmark comparing contracted vs uncontracted multi-step research agents:

```bash
# Quick test (1 question)
uv run python -m benchmarks.research_agent.benchmark --max-questions 1

# Full benchmark (5 questions)
uv run python -m benchmarks.research_agent.benchmark
```

**What It Tests:**
- Decompose-research-synthesize workflow
- Strategic resource allocation per step
- Quality vs cost efficiency tradeoffs

**Key Result:** Contracted agents achieve **+7.3% better quality** with similar cost by strategically allocating reasoning effort.

See `research_agent/README.md` for detailed documentation.

### 2. Governance Validation (`governance/`)

Tests organizational governance capabilities:

```bash
# Budget violation test
uv run python -m benchmarks.governance.budget_violation_test

# Cost governance test
uv run python -m benchmarks.governance.cost_governance_test

# Variance reduction test
uv run python -m benchmarks.governance.variance_reduction_test
```

**What It Tests:**
- **Budget Violation**: Hard budget enforcement with graceful failure
- **Cost Governance**: Organization-wide policy compliance
- **Variance Reduction**: Predictability improvement

**Key Result:** 100% budget enforcement compliance, preventing runaway costs.

### 3. Strategic Optimization (`strategic/`)

Tests the Pareto frontier of quality-cost-time tradeoffs:

```bash
uv run python -m benchmarks.strategic.strategic_optimization_test
```

**Contract Modes:**
- **URGENT**: Minimize time (fast execution)
- **BALANCED**: Balance quality, cost, and time
- **ECONOMICAL**: Minimize cost

**Key Result:** Pareto-optimal frontier validated - no mode strictly dominates another.

### 4. Quality Validation (`quality_validation/`)

Tests the LLM-as-judge evaluator reliability:

```bash
# Test-retest reliability
uv run python benchmarks/quality_validation/test_harness.py

# Reasoning effort impact
uv run python benchmarks/quality_validation/test_reasoning_effort.py
```

**Key Result:** CV=5.2% reliability (exceeds SOTA 10-15%).

See `quality_validation/study_design.md` for methodology.

## Integration Demos

### LangChain (`langchain/`)

Demonstrates governance for LangChain chains and agents:

```bash
uv run python benchmarks/langchain/demo_integration.py
```

**What It Provides:**
- ✅ Token tracking & cost monitoring
- ✅ Multi-call budget protection
- ✅ Complete audit trails
- ✅ Organizational policy enforcement

### LangGraph (`langgraph/`)

Demonstrates governance for complex LangGraph workflows:

```bash
uv run python benchmarks/langgraph/demo_integration.py
```

**What It Provides:**
- ✅ Per-node token tracking
- ✅ Cycle/loop protection (prevents runaway costs)
- ✅ Multi-agent coordination governance
- ✅ State-aware budget enforcement

### Google ADK (`google_adk/`)

Demonstrates governance for Google Agent Development Kit:

```bash
uv run python benchmarks/google_adk/demo_integration.py
```

**What It Provides:**
- ✅ Detailed token tracking (prompt/response/thinking)
- ✅ Multi-turn conversation protection
- ✅ Multi-agent system governance
- ✅ Per-tool usage tracking & limits
- ✅ Hierarchical delegation with budget conservation

## Requirements

- Python 3.12+
- `GOOGLE_API_KEY` environment variable (for Gemini models)
- Dependencies installed via `uv sync`

For optional integrations:
```bash
# LangChain
uv sync --extra langchain

# LangGraph
uv sync --extra langgraph

# Google ADK
uv sync --extra google-adk
```

## Configuration

All benchmarks use the Gemini 3 Flash model by default:
- Model: `gemini/gemini-3-flash-preview` (LiteLLM format)
- API Key: Set `GOOGLE_API_KEY` in `.env` or environment

## Output

Results are saved to `results/` directories within each benchmark folder:
- JSON files with detailed metrics
- Console output with summaries
- Visualization scripts where applicable

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── research_agent/              # Multi-step research benchmark
│   ├── README.md
│   ├── benchmark.py
│   ├── agent.py
│   ├── contracted_agent.py
│   ├── uncontracted_agent.py
│   ├── evaluator.py
│   └── questions.py
├── governance/                  # Governance validation
│   ├── README.md
│   ├── budget_violation_test.py
│   ├── cost_governance_test.py
│   └── variance_reduction_test.py
├── strategic/                   # Strategic mode testing
│   ├── strategic_optimization_test.py
│   └── pareto_visualization.py
├── quality_validation/          # Evaluator reliability
│   ├── study_design.md
│   ├── test_harness.py
│   └── test_reasoning_effort.py
├── langchain/                   # LangChain integration
│   ├── README.md
│   └── demo_integration.py
├── langgraph/                   # LangGraph integration
│   ├── README.md
│   └── demo_integration.py
└── google_adk/                  # Google ADK integration
    ├── README.md
    ├── demo_integration.py
    └── demo_delegation.py
```
