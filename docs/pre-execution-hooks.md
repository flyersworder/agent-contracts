# Pre-Execution Hooks & Behavioral Monitor

**Status**: Pre-execution hooks implemented (v0.2.0) | Behavioral monitor designed (future)
**Inspiration**: [Plano](https://github.com/katanemo/plano) filter chain pattern and agentic signals

## Overview

Agent Contracts governs **resources** (budget) and **time** (deadlines). Pre-execution hooks add a third dimension: **user-defined policy governance** — custom logic that runs before and after constraint checks across all integrations.

```
ContractEnforcer
  ├── ResourceMonitor    → "Are we within budget?"
  ├── TemporalMonitor    → "Are we within time?"
  ├── Pre/Post Hooks     → "Does this pass custom policy?" (IMPLEMENTED)
  └── BehavioralMonitor  → "Are we making progress?" (FUTURE)
```

## Pre-Execution Hooks

### Quick Start

```python
from agent_contracts import (
    Contract, ContractEnforcer, ContractedLLM,
    CheckContext, HookResult, EnforcementAction, ResourceConstraints,
)

# Define a custom hook
def topic_guard(ctx: CheckContext) -> HookResult:
    messages = ctx.metadata.get("messages", [])
    if any("forbidden" in str(m) for m in messages):
        return HookResult(
            allow=False,
            reason="Off-topic request",
            action=EnforcementAction.HARD_STOP,
        )
    return HookResult()

# Use with ContractedLLM
contract = Contract(
    id="guarded-task",
    resources=ResourceConstraints(tokens=10000),
)
llm = ContractedLLM(contract)
llm.enforcer.add_pre_check_hook(topic_guard)

# Or pass hooks at construction time
enforcer = ContractEnforcer(
    contract,
    pre_check_hooks=[topic_guard],
)
```

### Core Types

```python
@dataclass(frozen=True)
class CheckContext:
    """Context passed to hooks."""
    contract: Contract
    monitor: ResourceMonitor
    phase: Literal["pre_check", "post_check"]
    metadata: dict[str, Any]  # integration-specific data

@dataclass(frozen=True)
class HookResult:
    """Result from a hook."""
    allow: bool = True
    reason: str = ""
    action: EnforcementAction = EnforcementAction.WARN  # only consulted when allow=False

CheckHook = Callable[[CheckContext], HookResult]
```

### Hook Behavior by Action

| Action | Emits Event | Blocks Execution |
|--------|-------------|-----------------|
| `WARN` | Yes | No |
| `THROTTLE` | Yes | No |
| `SOFT_STOP` | Yes | Yes |
| `HARD_STOP` | Yes | Yes |

Post-check hooks are **observational only** — they run after constraint checking but cannot block execution regardless of the action specified.

### Integration Metadata

Each integration passes context through `metadata` so hooks can make informed decisions:

| Integration | `metadata` contents |
|---|---|
| **LiteLLM** | `{"integration": "litellm", "model": ..., "messages": ...}` |
| **LangChain** | `{"integration": "langchain"}` (via base ContractAgent) |
| **LangGraph** | `{"integration": "langgraph"}` |
| **Google ADK** | `{"integration": "google_adk"}` |
| **Claude Agent SDK** | `{"integration": "claude_agent_sdk", "tool_name": ..., "phase": ...}` |

### API Reference

**ContractEnforcer methods:**

```python
# Construction
enforcer = ContractEnforcer(
    contract,
    pre_check_hooks=[hook1, hook2],   # run before constraint checks
    post_check_hooks=[hook3],          # run after (observational)
)

# Dynamic management
enforcer.add_pre_check_hook(hook)
enforcer.remove_pre_check_hook(hook)
enforcer.add_post_check_hook(hook)
enforcer.remove_post_check_hook(hook)

# Pass metadata from integrations
enforcer.check_constraints(metadata={"integration": "litellm", "model": "gpt-4"})
```

### Design Decisions

- **Frozen dataclasses** — consistent with `ResourceConstraints`, `ViolationInfo`, etc.
- **Metadata is `dict[str, Any]`** — integrations populate it; core framework doesn't depend on contents. Defensively copied to prevent cross-hook mutation.
- **Exception safety** — hook exceptions are caught and logged (like callbacks), never crash enforcement.
- **Backward compatible** — `check_constraints()` defaults `metadata` to `None`; existing code works unchanged.

---

## Behavioral Monitor (Future Design)

A planned third monitor that answers: *"Is the agent making progress?"*

### Motivation

A long-running agent can exhaust its budget by repeating the same failing action. The current system can only tell you *after* the budget is gone. A behavioral monitor would detect loops, repetition, and efficiency decay in real-time.

### Proposed Data Model

```python
@dataclass
class CallRecord:
    """Single LLM call record for behavioral analysis."""
    timestamp: datetime
    input_hash: str           # SHA-256 of input for repetition detection
    output_tokens: int        # for efficiency tracking
    tool_calls: list[str]     # tools invoked in this call
    node_name: str | None     # for LangGraph: which node

class BehavioralMonitor:
    """Detects behavioral anti-patterns in agent execution."""

    def __init__(self, window_size: int = 20, max_history: int = 100):
        self.history: deque[CallRecord]  # bounded rolling window

    def record_call(self, record: CallRecord) -> None: ...

    def detect_loops(self, threshold: float = 0.8) -> bool:
        """Are recent input hashes repeating? (stuck agent)"""

    def repetition_score(self) -> float:
        """0.0 = all unique, 1.0 = all identical (within window)"""

    def efficiency_trend(self) -> float:
        """Ratio of output tokens to input tokens over time. Declining = problem."""

    def tool_diversity(self) -> float:
        """Are we using the same tool over and over? Low diversity = potential loop."""
```

### Design Decisions

- **Rolling window** (`deque(maxlen=N)`) prevents unbounded memory growth
- **Hash-based repetition** avoids storing full message content (privacy + memory)
- **Configurable thresholds** — what counts as "stuck" varies by use case
- **Signals, not hard blocks** — behavioral anomalies default to `WARN`, not `HARD_STOP`
- **Optional on ContractEnforcer** — `behavioral_monitor: BehavioralMonitor | None = None`
- **Bridge to hooks** — a behavioral monitor can be registered as a `pre_check_hook` even before it becomes a first-class component

### When to Implement

When there's a concrete user pain point — e.g., a LangGraph agent stuck in retry loops, or a research agent repeating the same web searches.
