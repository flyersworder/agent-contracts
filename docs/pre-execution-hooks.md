# Pre-Execution Hooks & Behavioral Monitor

**Status**: Pre-execution hooks implemented (v0.3.0) | Behavioral feedback system designed (future)
**Inspiration**: [Plano](https://github.com/katanemo/plano) filter chain pattern and agentic signals

## Overview

Agent Contracts governs **resources** (budget) and **time** (deadlines). Pre-execution hooks add a third dimension: **user-defined policy governance** — custom logic that runs before and after constraint checks across all integrations.

```
ContractEnforcer
  ├── ResourceMonitor     → "Are we within budget?"
  ├── TemporalMonitor     → "Are we within time?"
  ├── Pre/Post Hooks      → "Does this pass custom policy?"  (v0.3.0)
  └── Behavioral Feedback → "Are we making progress?"        (FUTURE)
        ├── Layer 1: Signals  — detect loops, repetition, efficiency decay
        ├── Layer 2: Feedback — translate signals into prompt guidance
        └── Layer 3: Memory   — learn patterns across sessions
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

## Behavioral Feedback System (Future Design)

A planned closed-loop system that detects behavioral anti-patterns and feeds actionable guidance back to agents so they can self-correct in real-time and improve over time.

### From Monitor to Feedback Loop

The key insight: **monitoring without feedback is just logging**. The real value comes when behavioral signals drive behavior change.

A passive monitor does this:

```
Agent acts → Monitor observes → Emit warning event
```

A feedback system does this:

```
Agent acts → Monitor observes → Advisor generates guidance →
  Inject into next prompt → Agent adapts → Better outcomes
```

This is the difference between a smoke detector (alerts you) and a thermostat (self-corrects).

### Three-Layer Architecture

Each layer is independently buildable and useful:

```
┌─────────────────────────────────────────────────┐
│  Layer 3: Memory (cross-session learning)       │
│  BehavioralProfile — persisted patterns         │
├─────────────────────────────────────────────────┤
│  Layer 2: Feedback (signal → guidance)          │
│  BehavioralAdvisor — prompt injection           │
├─────────────────────────────────────────────────┤
│  Layer 1: Signals (pattern detection)           │
│  BehavioralMonitor — loops, trends, diversity   │
├─────────────────────────────────────────────────┤
│  Foundation: Pre-Execution Hooks (v0.3.0)       │
│  CheckContext + metadata — injection point      │
└─────────────────────────────────────────────────┘
```

### Layer 1: Signals (Detect Patterns)

Real-time detection of behavioral anti-patterns within a single contract execution.

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

    # Detection methods
    def detect_loops(self, threshold: float = 0.8) -> bool:
        """Are recent input hashes repeating? (stuck agent)"""

    def repetition_score(self) -> float:
        """0.0 = all unique, 1.0 = all identical (within window)"""

    def efficiency_trend(self) -> float:
        """Ratio of output tokens to input tokens over time. Declining = problem."""

    def tool_diversity(self) -> float:
        """Are we using the same tool over and over? Low diversity = potential loop."""

    def most_used_tool(self) -> str:
        """Which tool dominates recent usage?"""

    def progress_score(self) -> float:
        """0.0 = no progress, 1.0 = steady progress. Composite of all signals."""
```

**Design decisions:**

- **Rolling window** (`deque(maxlen=N)`) prevents unbounded memory growth
- **Hash-based repetition** avoids storing full message content (privacy + memory)
- **Configurable thresholds** — what counts as "stuck" varies by use case
- **Signals, not hard blocks** — behavioral anomalies default to `WARN`, not `HARD_STOP`
- **Optional on ContractEnforcer** — `behavioral_monitor: BehavioralMonitor | None = None`

**How it integrates:** Each integration calls `behavioral_monitor.record_call()` after each LLM call, similar to how they currently call `add_tokens()` / `add_api_call()`.

### Layer 2: Feedback (Translate Signals to Guidance)

Converts raw signals into actionable natural language feedback that gets injected into the agent's next prompt. This extends the existing `generate_adaptive_instruction()` pattern from `prompts.py` — where budget utilization drives prompt guidance — to behavioral signals.

```python
class BehavioralAdvisor:
    """Translates behavioral signals into agent guidance."""

    def __init__(self, monitor: BehavioralMonitor):
        self.monitor = monitor

    def generate_feedback(self) -> str:
        """Generate natural language feedback for the agent.

        Returns empty string when behavior is healthy (no noise).
        """
        feedback = []

        if self.monitor.repetition_score() > 0.6:
            feedback.append(
                "You appear to be repeating similar actions. "
                "Consider synthesizing your current results or trying "
                "a different approach."
            )

        trend = self.monitor.efficiency_trend()
        if trend < -0.3:  # declining efficiency
            feedback.append(
                "Your recent outputs are declining in substance. "
                "Take a step back and refocus on the core objective."
            )

        if self.monitor.tool_diversity() < 0.2:
            most_used = self.monitor.most_used_tool()
            feedback.append(
                f"You're relying heavily on {most_used}. "
                "Consider whether a different tool might be more effective."
            )

        return "\n".join(feedback) if feedback else ""
```

**Injection point — pre-execution hooks (already built in v0.3.0):**

```python
def behavioral_feedback_hook(ctx: CheckContext) -> HookResult:
    """Inject behavioral feedback into the next LLM call."""
    advisor = ctx.metadata.get("behavioral_advisor")
    if advisor:
        feedback = advisor.generate_feedback()
        if feedback:
            # Attach feedback to metadata for integration to pick up
            # and prepend to the system prompt
            ctx.metadata["behavioral_feedback"] = feedback
    return HookResult()  # never blocks, just advises
```

**Existing patterns this builds on:**

| Existing (implemented) | Extension (future) |
|------------------------|-------------------|
| `generate_adaptive_instruction(budget_utilization)` | `advisor.generate_feedback()` based on behavioral signals |
| `get_time_pressure()` → 0.0-1.0 | `monitor.progress_score()` → 0.0-1.0 |
| `pre_check_hooks` inspect/block | Behavioral hook injects feedback into next prompt |
| `ResourceUsage` tracks aggregates | `BehavioralMonitor` tracks patterns over time |

### Layer 3: Memory (Learn Across Sessions)

Persist behavioral profiles so future contracts benefit from historical patterns. This is the most ambitious layer — it turns the system from a single-execution observer into a **strategy advisor**.

```python
@dataclass
class BehavioralProfile:
    """Persisted behavioral patterns across contract executions."""

    # Per-task-type statistics (learned over time)
    avg_iterations_to_success: dict[str, float]
    tool_effectiveness: dict[str, float]     # tool → success correlation
    optimal_search_depth: dict[str, int]     # task_type → recommended max searches
    common_failure_patterns: list[str]       # recurring anti-patterns

    def recommend_constraints(self, task_type: str) -> ResourceConstraints:
        """Suggest resource constraints based on historical behavior.

        Example: if research tasks historically succeed in ~8 iterations
        but degrade after 12, recommend iterations=10 as a safe bound.
        """
        ...

    def generate_pre_briefing(self, task_type: str) -> str:
        """Generate a pre-execution briefing based on past patterns.

        Example: 'Research tasks work best with 3-4 focused web searches.
        Avoid broad queries — they tend to lead to repetitive results.'
        """
        ...
```

**Cross-session learning examples:**

- "Research tasks with >5 web searches tend to loop — suggest capping at 4 and synthesizing"
- "This agent type is most efficient in the first 3 iterations, then quality degrades"
- "Tool X followed by tool Y has 80% success rate; tool X alone, only 30%"
- "Code review tasks succeed faster when the agent reads tests first"

**Persistence options:**

- **Simple**: JSON/SQLite file per project or agent type
- **Structured**: Integration with the existing `ExecutionLog` audit trail
- **Collaborative**: Shared profiles across a team (e.g., "our research agents work best with these constraints")

### How the Three Layers Work Together

**Example: A research agent stuck in a loop**

```
Iteration 1: Agent searches "quantum computing advances 2026"
Iteration 2: Agent searches "recent quantum computing papers 2026"
Iteration 3: Agent searches "quantum computing breakthroughs 2026"
```

**Layer 1 (Signals)** detects: `repetition_score = 0.85`, `tool_diversity = 0.1`

**Layer 2 (Feedback)** generates and injects into the next prompt:
> "You've made 3 similar web searches with overlapping results. Synthesize what you have rather than searching again. Consider focusing on a specific subtopic."

**Layer 3 (Memory)** records: "Research tasks on broad topics tend to loop after 3 web searches. For future research contracts, recommend `per_tool_limits={'web_search': 4}` and include guidance to narrow the topic early."

**Next time** a research contract is created, the profile suggests tighter constraints and a pre-briefing — the agent starts smarter.

### Implementation Roadmap

| Layer | Complexity | Prerequisite | When to Build |
|-------|-----------|--------------|---------------|
| **Layer 1: Signals** | Medium | Pre-execution hooks (done) | When a user reports a looping agent |
| **Layer 2: Feedback** | Medium | Layer 1 | When signal detection is validated |
| **Layer 3: Memory** | High | Layers 1 + 2, persistence design | When feedback proves valuable in practice |

Each layer delivers standalone value. Layer 1 alone catches stuck agents. Layer 2 alone helps agents self-correct. Layer 3 is the long-term multiplier.
