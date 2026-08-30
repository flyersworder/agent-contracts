# Agent Contracts

[![PyPI version](https://img.shields.io/pypi/v/ai-agent-contracts.svg)](https://pypi.org/project/ai-agent-contracts/)
[![Tests](https://github.com/flyersworder/agent-contracts/actions/workflows/ci.yml/badge.svg)](https://github.com/flyersworder/agent-contracts/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A formal framework for governing autonomous AI agents through explicit resource
constraints and temporal boundaries.

Agent Contracts turn autonomous agents from unbounded explorers into **bounded
optimizers**: declare a budget, and the framework tracks consumption as the
agent runs, enforces the limit, and leaves an audit trail. It wraps six agent
frameworks without changing how you write agents.

```bash
pip install ai-agent-contracts
```

## Contents

- [Why](#why)
- [Quick start](#quick-start)
- [Integrations](#integrations)
- [Key concepts](#key-concepts)
- [Feature guide](#feature-guide)
- [Research: the causal-chamber pillar](#research-the-causal-chamber-pillar)
- [Documentation](#documentation)
- [Installation](#installation)
- [Development](#development)
- [Project structure](#project-structure)
- [Status](#status)
- [Contributing](#contributing)

## Why

Agentic systems fail in ways ordinary software does not:

| Problem | What a contract adds |
|---|---|
| **Unbounded consumption** — an agent can burn unpredictable tokens, calls and time | An explicit budget, enforced *during* execution rather than reported after |
| **Unclear lifecycles** — no explicit termination criterion | Machine-verifiable states from activation to fulfilment |
| **Hard governance** — difficult to audit, attribute cost, or prove compliance | An event stream and audit trail per contract |
| **Multi-agent coordination** — no formal allocation between agents | Hierarchical delegation and DAG flow conservation, with a provable global bound |

A contract is `C = (I, O, S, R, T, Φ, Ψ)` — inputs, outputs, state, resources,
time, preconditions, postconditions. The [whitepaper](./docs/whitepaper.md) has
the formal treatment.

## Quick start

```python
from agent_contracts import Contract, ContractedLLM, ContractMode, ResourceConstraints

contract = Contract(
    id="research-task",
    name="Research Assistant",
    mode=ContractMode.BALANCED,          # quality-cost-time balance
    resources=ResourceConstraints(
        tokens=10_000,
        api_calls=50,
        cost_usd=1.0,
    ),
)

with ContractedLLM(contract) as llm:
    response = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize recent AI papers"}],
    )
    usage = llm.get_usage_summary()["usage"]
    print(f"{usage['tokens']} tokens · ${usage['cost_usd']:.4f}")
```

Token budgets, call counts and cost are tracked as the agent runs; crossing a
limit raises or warns depending on whether you run strict or lenient.

## Integrations

Six integrations share one contract object. The value scales with how many
calls a framework can make on your behalf.

| Integration | Calls per run | Budget risk | What the contract buys |
|---|---|---|---|
| **LiteLLM** | 1 per call | Low | Universal baseline — 100+ providers, token counting, cost |
| **LangChain** | 3–10 | Low–moderate | Multi-call protection, audit trail, policy compliance |
| **LangGraph** ⭐ | 30+ (cycles, retries, parallel agents) | **Very high** | Loop protection, multi-agent budget sharing, cumulative tracking across every node |
| **Claude Agent SDK** | 10–100+ tool calls | High | Per-tool limits and temporal enforcement via `PreToolUse` hooks; audit via `PostToolUse` |
| **Google ADK** | 10–50+ | High | Multi-turn protection, hierarchical governance, prompt/response/thinking/cached token split |
| **Causal Chambers** | research harness | — | Budget-matched agent experiments on a physical testbed |

**LangGraph is where the framework earns its keep.** A graph with a validation
cycle can spiral past $10 unbounded; the contract enforces one budget across
every node and every iteration.

<details>
<summary><b>LangGraph</b> — one budget across a whole graph, cycles included</summary>

```python
from langgraph.graph import END, StateGraph

from agent_contracts import Contract, ResourceConstraints
from agent_contracts.integrations import ContractedGraph

graph = StateGraph(AgentState)
graph.add_node("researcher", research_node)
graph.add_node("validator", validate_node)
graph.add_conditional_edges(
    "validator", should_retry, {"retry": "researcher", "done": END}
)

contract = Contract(
    id="research-workflow",
    name="Research Workflow",
    resources=ResourceConstraints(tokens=50_000, api_calls=20),
)

contracted = ContractedGraph(contract, graph.compile())
result = contracted.invoke({"query": "..."})
print(contracted.get_remaining_budget())
```

The budget is cumulative across all nodes and cycles, so a retry loop that
never terminates hits the contract instead of your bill.

</details>

<details>
<summary><b>Google ADK</b> — one budget for a multi-agent hierarchy</summary>

```python
from google.adk.agents import Agent

from agent_contracts import Contract, ResourceConstraints
from agent_contracts.integrations import ContractedAdkAgent

coordinator = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    sub_agents=[researcher, analyst],
)

contract = Contract(
    id="multi-agent-research",
    name="Multi-Agent Research",
    resources=ResourceConstraints(tokens=30_000, api_calls=15),
)

contracted = ContractedAdkAgent(contract, coordinator)
result = contracted.run(
    user_id="u1", session_id="s1", message="Analyze the EV market"
)
```

Token tracking is broken out by prompt, response, thinking and cached.

</details>

## Key concepts

### Contract definition

An Agent Contract `C = (I, O, S, R, T, Φ, Ψ)`:

| Symbol | Field | Meaning |
|---|---|---|
| **I** | `inputs` | Input specification |
| **O** | `outputs` | Output specification |
| **S** | `state` | Current lifecycle state |
| **R** | `resources` | Resource constraints — tokens, calls, cost, tools, memory |
| **T** | `temporal` | Deadlines and duration limits |
| **Φ** | `success_criteria` | Preconditions and success conditions |
| **Ψ** | `termination_conditions` | Postconditions and termination rules |

### Contract states

```text
DRAFT → ACTIVE → {FULFILLED, VIOLATED, EXPIRED, TERMINATED}
```

### Time-resource tradeoff

Contract modes let you trade quality, cost and time explicitly rather than
implicitly. Validated on an N=20 benchmark — no mode dominates another, so the
three sit on a Pareto frontier:

| Mode | Quality | Effect |
|---|---|---|
| `URGENT` | 87% | ~50% faster, ~20% more tokens |
| `BALANCED` | 85% | Balanced resources |
| `ECONOMICAL` | 81% | ~32% fewer tokens, longer runtime |

```python
from agent_contracts import Contract, ContractMode, ResourceConstraints

contract = Contract(
    id="urgent-task",
    name="Urgent Task",
    mode=ContractMode.URGENT,
    resources=ResourceConstraints(tokens=10_000),
)
```

## Feature guide

<details>
<summary><b>Per-tool resource limits</b></summary>

Per-tool limits are checked before the aggregate limit. An **omitted key means
unconstrained, not zero** — grant an explicit `0` when you mean zero.

```python
from agent_contracts import Contract, ResourceConstraints

contract = Contract(
    id="research-agent",
    name="Research Agent",
    resources=ResourceConstraints(
        tokens=10_000,
        tool_invocations=20,        # aggregate limit across all tools
        per_tool_limits={
            "web_search": 5,
            "code_exec": 3,
            # tools not listed are bounded only by the aggregate
        },
    ),
)
```

</details>

<details>
<summary><b>Delegation graphs (multi-parent budgets)</b></summary>

When one agent is funded by several others — an aggregator merging two workers,
a shared reviewer — a strict hierarchy cannot express it without
double-counting. `DelegationGraph` models delegation as a DAG where budget
flows along edges:

```python
from agent_contracts import Contract, ResourceConstraints
from agent_contracts.core import DelegationGraph

root = Contract(
    id="research",
    name="Research",
    resources=ResourceConstraints(tokens=100_000, per_tool_limits={"web_search": 20}),
)

graph = DelegationGraph(root)
for name in ("scout_a", "scout_b", "aggregator"):
    graph.add_node(name)

graph.allocate("root", "scout_a", tokens=40_000, per_tool={"web_search": 10})
graph.allocate("root", "scout_b", tokens=40_000, per_tool={"web_search": 10})
graph.allocate("scout_a", "aggregator", tokens=15_000, per_tool={"web_search": 0})
graph.allocate("scout_b", "aggregator", tokens=15_000, per_tool={"web_search": 0})

graph.seal()  # validate the whole graph, then freeze its topology

graph.contract_for("aggregator").resources.tokens  # 30_000 — the sum of its in-edges
graph.verify()                                     # raises if any node breaks the invariant
```

The invariant at every node is `in-flow ≥ own consumption + out-flow`. Because
that is a purely local check, and internal allocations cancel when summed
across the graph, satisfying it everywhere guarantees total consumption never
exceeds the root budget — **with no global lock and no central accountant**.
See [whitepaper §4.6](docs/whitepaper.md) for the proof and its scope.

Two semantics worth knowing: control flow may cycle but **budget flow must
not** (a cycle would let a node refund its own ancestor), and refunds are
computed against original allocations rather than live ones, which is what
makes releasing sibling edges order-independent.

</details>

<details>
<summary><b>Pre-execution hooks (custom policy)</b></summary>

Custom governance logic that runs before every constraint check, on all
integrations:

```python
from agent_contracts import (
    CheckContext,
    Contract,
    ContractedLLM,
    EnforcementAction,
    HookResult,
    ResourceConstraints,
)

def topic_guard(ctx: CheckContext) -> HookResult:
    messages = ctx.metadata.get("messages", [])
    if any("off-topic" in str(m) for m in messages):
        return HookResult(
            allow=False,
            reason="Request outside allowed domain",
            action=EnforcementAction.HARD_STOP,
        )
    return HookResult()  # allow by default

contract = Contract(
    id="guarded-agent",
    name="Guarded Agent",
    resources=ResourceConstraints(tokens=10_000, cost_usd=1.0),
)

with ContractedLLM(contract) as llm:
    llm.enforcer.add_pre_check_hook(topic_guard)
```

`WARN` and `THROTTLE` are informational; `SOFT_STOP` and `HARD_STOP` block.
Post-check hooks are observational and cannot block. See
[docs/pre-execution-hooks.md](./docs/pre-execution-hooks.md).

</details>

<details>
<summary><b>Agent skills (agentskills.io standard)</b></summary>

```python
from agent_contracts import Capabilities, Contract, SkillSpec

code_review = SkillSpec(
    name="code-reviewer",
    description="Review code for best practices, security issues, and test coverage.",
    instructions="""
    ## Instructions
    1. Read the target files
    2. Check error handling, security, and test coverage
    3. Provide detailed feedback
    """,
    allowed_tools=["Read", "Grep", "Glob"],
    version="1.0.0",
)

contract = Contract(
    id="review-task",
    name="Code Review",
    capabilities=Capabilities(
        skills=[code_review, "simple-skill"],  # SkillSpec and plain strings both work
        tools=["web_search"],
    ),
)

skill = contract.capabilities.get_skill("code-reviewer")
```

SKILL.md import/export via `to_skill_md()` / `from_skill_md()`, progressive
disclosure (metadata ≈ 100 tokens, instructions loaded on activation), and
backward compatibility with plain string skills.

</details>

## Research: the causal-chamber pillar

Beyond the library, this repository hosts an empirical research pillar that
uses contracts as a measurement instrument. Because budgets can be split across
agents *and verified*, agent topologies can be compared at **exactly matched
spend** — which is what makes the comparison meaningful.

The M6 experiment runs a five-rung coordination ladder (single loop, ensemble,
parallel roles, relay, negotiating team) on two
[Causal Chambers](https://causalchamber.org) — physical devices whose true
causal graph is known by construction, so a recovered graph can be graded
objectively rather than by an LLM judge.

**2,221 cells, $94.05, zero errored cells, two chambers, two models.** Of 24
topology-vs-loop contrasts, 10 resolve and 9 favour a single sequential loop.

- [`docs/chamber-results.md`](./docs/chamber-results.md) — every experiment and
  what it showed
- [`docs/chamber-harness-validity-register.md`](./docs/chamber-harness-validity-register.md)
  — 19 harness defects that each changed, or could have changed, a number.
  **Read this before trusting any figure**; six of them looked like findings first.
- [`docs/causal_chamber_validation_plan.md`](./docs/causal_chamber_validation_plan.md)
  — the full experiment plan

The harness lives in [`evaluation/chamber_pipeline/`](./evaluation/chamber_pipeline/)
and needs the `chambers` extra.

## Documentation

📚 **[Complete documentation index](./docs/README.md)**

| Document | What it covers |
|---|---|
| [Whitepaper](./docs/whitepaper.md) | Formal framework, the conservation proof, use cases |
| [Pre-execution hooks](./docs/pre-execution-hooks.md) | Custom governance hooks and monitor design |
| [Testing strategy](./docs/testing-strategy.md) | Test plan and validation approach |
| [Chamber results](./docs/chamber-results.md) | Empirical results of record |
| [Validity register](./docs/chamber-harness-validity-register.md) | Every harness defect found, and its effect |
| [Quality measurement](./docs/quality_measurement_research.md) | LLM-as-judge under rating indeterminacy |

**By role**: researchers start with the
[formal framework](./docs/whitepaper.md#2-formal-framework); engineers with
[implementation architecture](./docs/whitepaper.md#5-implementation-architecture);
product managers with the [introduction](./docs/whitepaper.md#1-introduction).

## Installation

```bash
pip install ai-agent-contracts     # or: uv add ai-agent-contracts
```

The package imports as `agent_contracts`:

```python
from agent_contracts import Contract, ResourceConstraints
```

**Requirements**: Python ≥ 3.12.

**Optional extras** — install only what you use:

| Extra | Install | For |
|---|---|---|
| `litellm` | `uv sync --extra litellm` | 100+ LLM providers |
| `langchain` | `uv sync --extra langchain` | LangChain chains |
| `langgraph` ⭐ | `uv sync --extra langgraph` | LangGraph graphs |
| `google-adk` | `uv sync --extra google-adk` | Google ADK agents |
| `claude-agent-sdk` | `uv sync --extra claude-agent-sdk` | Claude Agent SDK |
| `chambers` | `uv sync --extra chambers` | Causal-chamber research harness |
| `eval` | `uv sync --extra eval` | Benchmark datasets and plots |

## Development

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/flyersworder/agent-contracts.git
cd agent-contracts

uv sync --dev              # or --all-extras for every integration
uv run pre-commit install
```

```bash
uv run pytest                                        # run the suite
uv run pytest --cov=agent_contracts --cov-report=html  # with coverage

uv run pre-commit run --all-files                    # all checks
uv run ruff check .                                  # lint
uv run ruff format .                                 # format
uv run mypy .                                        # type check
```

Quality gates: [Ruff](https://github.com/astral-sh/ruff) (lint + format),
[mypy](https://github.com/python/mypy) in strict mode, and
[pre-commit](https://pre-commit.com/) hooks that run on every commit. CI must
be green on Python 3.12 and 3.13 before merge.

## Project structure

```text
agent-contracts/
├── src/agent_contracts/
│   ├── core/
│   │   ├── contract.py           # Contract, ResourceConstraints, Capabilities
│   │   ├── monitor.py            # Real-time resource tracking
│   │   ├── enforcement.py        # Constraint enforcement and hooks
│   │   ├── delegation.py         # Hierarchical delegation (tree)
│   │   ├── delegation_graph.py   # DAG delegation with flow conservation
│   │   ├── resource_vector.py    # Resource algebra
│   │   ├── skillspec.py          # agentskills.io SkillSpec
│   │   ├── tokens.py             # Token counting and cost
│   │   ├── planning.py           # Strategic planning
│   │   ├── executor.py           # Contract execution
│   │   └── prompts.py            # Budget-aware prompts
│   └── integrations/
│       ├── litellm_wrapper.py    # LiteLLM
│       ├── langchain.py          # LangChain
│       ├── langgraph.py          # LangGraph ⭐
│       ├── google_adk.py         # Google ADK
│       ├── claude_agent_sdk.py   # Claude Agent SDK
│       └── causalchamber.py      # Causal Chambers (research)
├── tests/                        # 1,507 tests
├── benchmarks/                   # Live demos: governance, strategic, per-framework
├── evaluation/                   # Experiments
│   ├── chamber_pipeline/         # Causal-chamber coordination ladder
│   ├── research_pipeline/        # Multi-agent research
│   └── code_review_pipeline/     # Coder ↔ Reviewer loop
└── docs/                         # Whitepaper, results, validity register
```

## Status

**v0.5.0 — production ready.** 1,507 tests, 1 skipped, **91% coverage**,
`mypy --strict` clean, CI green on Python 3.12 and 3.13.

| Area | State |
|---|---|
| Core framework — contracts, monitoring, enforcement, tokens | ✅ |
| Integrations — LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK | ✅ |
| Contract modes and strategic planning | ✅ Validated at N=20 |
| Per-tool limits and SkillSpec (agentskills.io) | ✅ |
| Pre-execution hooks (custom policy) | ✅ All integrations |
| Delegation: hierarchical tree and DAG flow conservation | ✅ v0.4.0 |
| Evaluation pipelines — research, code review | ✅ |
| Causal-chamber research pillar | ✅ M6 complete, M7 planned |
| AutoGen / CrewAI integrations | 🔜 Planned |
| Audit dashboards and cost-attribution reporting | 🔜 Planned |

**Known limitation**: `ResourceConstraints.iterations` is honoured only by
Google ADK (`max_llm_calls`) and the Claude Agent SDK (`max_turns`). LiteLLM,
LangChain and LangGraph neither track nor enforce it.

## Contributing

Contributions welcome — reference implementations, new framework integrations,
practical examples, and empirical studies. Please open an issue to discuss
substantial changes first.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).

## Authors

Qing Ye (with assistance from Claude, Anthropic).

## Citation

```bibtex
@techreport{ye2025agentcontracts,
  title={Agent Contracts: A Resource-Bounded Optimization Framework
         for Autonomous AI Systems},
  author={Ye, Qing},
  year={2025},
  month={October}
}
```

## Learn more

- 📖 [Whitepaper](./docs/whitepaper.md)
- 🔬 [Chamber results](./docs/chamber-results.md)
- 🎯 [Documentation index](./docs/README.md)
- 💬 [Open an issue](../../issues)

---

**v0.5.0** · Last updated 30 August 2026
