# Google ADK Integration Benchmarks

This directory contains benchmarks demonstrating Agent Contracts with Google Agent Development Kit (ADK).

## Why Google ADK + Agent Contracts?

Google ADK is a code-first Python framework for building sophisticated AI agents with:
- Tool integration
- Multi-agent systems
- Hierarchical coordination

Agent Contracts adds **governance and resource control**:

### ✅ What Agent Contracts Provides

1. **Detailed Token Tracking**
   - Automatic extraction with breakdown (prompt/response/thinking tokens)
   - Cached content tracking
   - Per-API-call cost estimation

2. **Multi-Turn Conversation Protection**
   - Cumulative budget tracking across conversation turns
   - Session-aware budgeting
   - Prevents runaway costs in chatbots

3. **Multi-Agent System Governance**
   - Single budget for entire agent hierarchies
   - Prevents budget explosion from agent coordination
   - Tracks usage across all sub-agents

4. **Per-Tool Usage Tracking & Limits**
   - Track which tools are used and how often
   - Set limits on expensive/risky tools
   - Prevent tool abuse in multi-turn conversations

5. **Complete Audit Trails**
   - Execution logs for compliance (SOC2, GDPR, etc.)
   - Cost attribution by contract/user/session
   - Debugging and performance optimization

6. **Hierarchical Delegation**
   - Budget delegation to sub-agents
   - Conservation law enforcement (children can't exceed parent budget)
   - Dynamic team formation with budget control

### ⚠️ Current Limitations

**Single-Turn Prevention**: Cannot prevent a SINGLE API call from exceeding budget because:
- Token count unknown until AFTER API call completes
- Money already spent by the time we detect violation
- Can only detect and log violation, not prevent it

## Benchmarks

### `demo_integration.py`

Demonstrates core integration features:

1. **Token Tracking** - Detailed breakdown of all token types
2. **Multi-Turn Protection** - Budget enforcement across conversation
3. **Multi-Agent Governance** - Shared budget for agent hierarchies
4. **Audit Trails** - Complete execution logs
5. **Per-Tool Tracking** - Track and limit tool usage
6. **Convenience API** - Simplified agent creation

**Run it**:
```bash
uv run python benchmarks/google_adk/demo_integration.py
```

### `demo_delegation.py`

Demonstrates hierarchical delegation features:

1. **Budget Delegation** - Allocate budget to sub-agents
2. **Conservation Laws** - Ensure children don't exceed parent budget
3. **Dynamic Teams** - Create teams with automatic budget allocation
4. **Budget Release** - Reclaim unused budget from completed tasks

**Run it**:
```bash
uv run python benchmarks/google_adk/demo_delegation.py
```

## Requirements

```bash
# Install Google ADK dependencies
uv sync --extra google-adk

# Set up Google API key
export GOOGLE_API_KEY="your-api-key"
# Or add to .env file
```

## Key Insight

Google ADK enables sophisticated multi-agent systems, but these systems can:
- Loop indefinitely in validation cycles
- Make hundreds of coordinated API calls
- Spiral costs out of control

**Agent Contracts ensures governance over these complex workflows:**

```
Without Agent Contracts:
  Coordinator → Researcher → Analyzer → Reviewer
  Each agent makes 5+ API calls
  Retry loop runs 10 times
  Total: 150+ API calls, $15+ cost

With Agent Contracts:
  Total budget: $2.00
  Execution stops at budget limit
  Total: 20 API calls, $2.00 cost (saved $13!)
```

## Strategic Positioning

- **LangChain** = Baseline governance feature
- **LangGraph** = Premium governance feature (cycles/loops)
- **Google ADK** = Native Google ecosystem integration
- All integrations work together for complete governance stack

## Model Support

Default model: `gemini-3-flash-preview`

Compatible with all Gemini models available through Google ADK.
