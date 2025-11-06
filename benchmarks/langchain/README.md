# LangChain Integration Benchmarks

This directory contains benchmarks demonstrating Agent Contracts with LangChain.

## What Agent Contracts Provides

Agent Contracts adds **governance and compliance** capabilities on top of LangChain's token tracking:

### ✅ What Works

1. **Token Tracking & Cost Monitoring**
   - Automatic extraction from LangChain responses
   - Multi-resource tracking (tokens, API calls, cost, time)
   - Complete audit trails for compliance

2. **Organizational Policy Enforcement**
   - Company-wide cost policies
   - Budget violation detection and logging
   - Compliance documentation

3. **Multi-Call Protection**
   - Cumulative budget tracking across multiple operations
   - Prevent second expensive call after budget exceeded
   - Strict vs lenient enforcement modes

### ⚠️ Current Limitations

**Single-Call Prevention**: Cannot prevent a SINGLE LLM call from exceeding budget because:
- Token count unknown until AFTER API call completes
- Money already spent by the time we detect violation
- Can only detect and log violation, not prevent it

**Why This Matters**:
- First expensive call will always complete
- Enforcement works for SUBSEQUENT calls
- Focus: Detection + multi-call protection, not single-call prevention

## Value Proposition

**LangChain Provides**:
- ✓ Token usage metadata (via `usage_metadata`)
- ✓ Model response tracking

**Agent Contracts Adds**:
- ✓✓✓ **Governance**: Organization-wide policy enforcement
- ✓✓✓ **Compliance**: Complete audit trails for regulatory requirements
- ✓✓✓ **Protection**: Multi-call budget enforcement
- ✓✓✓ **Detection**: Budget violation logging and alerting

## Benchmarks

### `demo_integration.py`

Demonstrates the LangChain integration with realistic scenarios:

1. **Token Tracking** - Accurate tracking from LangChain responses
2. **Audit Trails** - Complete execution logs for compliance
3. **Multi-Call Protection** - Budget enforcement across multiple calls
4. **Policy Enforcement** - Organization-wide cost governance

**Run it**:
```bash
uv run python benchmarks/langchain/demo_integration.py
```

## Key Insight

Agent Contracts complements LangChain by adding **organizational governance** that individual developers need but LangChain doesn't provide:

- LangChain: "Here's your token usage" (tracking)
- Agent Contracts: "This violates company policy" (governance)

Perfect for:
- Enterprises with AI cost policies
- Teams needing budget accountability
- Compliance and audit requirements
- Multi-agent systems with shared budgets
