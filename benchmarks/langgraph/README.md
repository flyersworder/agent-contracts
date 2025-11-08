# LangGraph Integration Benchmarks

This directory contains benchmarks demonstrating Agent Contracts with LangGraph for complex multi-agent workflows.

## Why LangGraph is a Better Fit for Agent Contracts

### LangChain vs LangGraph

**LangChain** (basic integration):
- Simple sequential/branching chains
- Typically 3-10 LLM calls per execution
- Budget risk: LOW to MODERATE

**LangGraph** (premium integration - THIS!):
- Complex stateful workflows with cycles, retries, and parallel execution
- Multi-agent systems with message passing
- Can easily make 30+ LLM calls in a single workflow
- Budget risk: **VERY HIGH** - can spiral out of control!

### Where Agent Contracts Shines

LangGraph workflows have:
- **Cycles and loops** - Research → Validate → Research (repeat until valid)
- **Conditional branching** - Different paths based on intermediate results
- **Parallel execution** - Multiple agents working simultaneously
- **Retries and error recovery** - Failed nodes trigger re-execution

Without governance, a single LangGraph workflow can:
- Loop indefinitely (validation never passes)
- Make hundreds of API calls trying to solve a hard problem
- Cost $10+ for what should be a $0.50 task

**This is where Agent Contracts provides CRITICAL value!**

## Agent Contracts vs LangSmith

Important distinction:

### LangSmith (Observability)
- "Here's what happened and how much it cost" (retrospective)
- Tells you AFTER the damage is done
- Great for debugging and analysis

### Agent Contracts (Governance)
- "Stop before you exceed the budget" (proactive)
- Prevents runaway costs DURING execution
- Stops loops before they spiral

**They're complementary, not competitive:**
- LangSmith tells you what agents did after the fact
- Agent Contracts controls what agents CAN do in the first place
- Use both for complete observability + governance stack!

## What Agent Contracts Provides for LangGraph

### ✅ Core Value

1. **Cumulative Budget Tracking Across Entire Graph**
   - Track tokens/cost across ALL nodes and cycles
   - Budget shared by entire workflow, not per-node
   - Critical for multi-agent coordination

2. **Cycle & Loop Protection**
   - Prevent infinite loops in validation cycles
   - Limit retry attempts in error recovery
   - Stop execution when budget exhausted (mid-workflow!)

3. **Multi-Agent Budget Sharing**
   - Multiple agents share single budget pool
   - No single agent can monopolize resources
   - Fair allocation across parallel branches

4. **Real-Time Violation Detection**
   - Check constraints at every node execution
   - Stop workflow immediately on violation (strict mode)
   - Log violations and continue (lenient mode)

5. **Complete Audit Trail**
   - Track which nodes consumed how many resources
   - Debug why workflows exceeded budget
   - Compliance documentation for complex workflows

### 🎯 Key Use Cases

Perfect for:
- **Research workflows** with validation loops
- **Multi-agent systems** with coordination
- **Error-recovery patterns** with retries
- **Exploration tasks** that might never converge
- **Production deployments** where cost control is critical

## Benchmarks

### `demo_integration.py`

Demonstrates LangGraph integration with realistic multi-agent scenarios:

1. **Simple Sequential Graph** - Basic workflow with budget tracking
2. **Graph with Cycles** - Validation loop with retry limits
3. **Multi-Agent Coordination** - Parallel agents with shared budget
4. **Complex Research Workflow** - Full research → validate → refine cycle

**Run it**:
```bash
uv run python benchmarks/langgraph/demo_integration.py
```

## Key Insight

LangGraph represents where AI development is heading:
- Not simple chains, but complex agentic workflows
- Not single agents, but multi-agent systems
- Not fixed paths, but dynamic exploration with cycles

**This is where governance is MOST needed!**

A simple LangChain might make 3 calls. A LangGraph workflow can make 300.

Agent Contracts ensures those 300 calls don't become 3000 - and don't bankrupt your API budget.

## Strategic Positioning

- **LangChain integration** = Baseline feature (for completeness)
- **LangGraph integration** = Premium feature (where the value is!)
- **LangGraph + LangSmith + Agent Contracts** = Complete production stack

## Real-World Impact

Without Agent Contracts:
```
Research loop iteration 1: $0.20
Research loop iteration 2: $0.20
Research loop iteration 3: $0.20
... (loops indefinitely, validation never passes)
Research loop iteration 47: $0.20
Total: $9.40 (yikes!)
```

With Agent Contracts:
```
Research loop iteration 1: $0.20 ✓
Research loop iteration 2: $0.20 ✓
Research loop iteration 3: $0.20 ✓
Budget limit ($2.00) reached - stopping execution
Total: $0.60 (saved $8.80!)
```

**That's the value proposition.**
