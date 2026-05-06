# Causal Chamber M1 — Convention decisions for §12 Q1+Q2

**Status**: Draft (M1 deliverable)
**Created**: 2026-05-06
**Owner**: qingye
**Parent doc**: [`causal_chamber_validation_plan.md`](causal_chamber_validation_plan.md)
**Acceptance criterion (M1)**: "§12 Q1+Q2 decisions documented based on a read
of existing integrations."

This file records the conventions the chamber adapter will follow, derived
from a read of every file in `src/agent_contracts/integrations/`, the
`SuccessCriterion` definition in `core/contract.py`, and the two sister
evaluation pipelines (`evaluation/research_pipeline/`,
`evaluation/code_review_pipeline/`).

The plan itself is unchanged; the §4.2 / §4.3 sketches are illustrative and
the plan explicitly defers final shape to M1–M2. Where this doc disagrees
with those sketches, this doc supersedes them for implementation.

---

## 1. What the existing integrations look like

Surveyed five integration modules:

| File | Public surface | Class style | Construction |
|---|---|---|---|
| `litellm_wrapper.py` | `ContractedLLM` | regular class, `__init__` | class only |
| `langchain.py` | `ContractedChain`, `ContractedChainLLM`, `create_contracted_chain` | regular class | class + factory |
| `langgraph.py` | `ContractedGraph`, `create_contracted_graph` | regular class | class + factory |
| `google_adk.py` | `ContractedAdkAgent`, `ContractedAdkMultiAgent`, `DelegatingAdkAgent`, `create_contracted_adk_agent` | regular class | class + factory |
| `claude_agent_sdk.py` | `ContractedClaudeAgent` | regular class | class only |

Patterns that hold across all five:

1. **Class names are `Contracted<Framework>`** — `ContractedLLM`,
   `ContractedChain`, `ContractedGraph`, `ContractedAdkAgent`,
   `ContractedClaudeAgent`. None invent a new noun like "Bridge", "Session",
   or "Wrapper".
2. **No `@dataclass` decorator anywhere.** Every integration is a regular
   class with explicit `__init__`. The §4.2 sketch's `@dataclass class
   ChamberContract` deviates from convention.
3. **First constructor parameter is `contract: Contract`.** The wrapper
   wraps a contract that the *caller* has constructed. None of the
   integrations construct a Contract internally; that responsibility lives
   with the caller (typically a pipeline or benchmark script). The §4.2
   sketch's `ChamberContract.__post_init__` building a Contract internally
   inverts this convention.
4. **Optional-import block at module top is uniform**:
   ```python
   try:
       from <pkg> import ...
       <PKG>_AVAILABLE = True
   except ImportError:
       <PKG>_AVAILABLE = False
       <Symbol> = Any  # type: ignore
   ```
   Followed by a runtime check inside `__init__` raising `ImportError` with
   an install hint if unavailable.
5. **Factory function is the dominant secondary surface.** 3 of 5
   integrations expose both a `Contracted<X>` class and a
   `create_contracted_<x>(...)` convenience factory. The factory exists for
   the common case where the caller wants sensible defaults; the class
   exists for the power-user case where every option needs to be set
   explicitly.
6. **`integrations/__init__.py` follows a parallel pattern.** Each
   integration's public symbols are imported in their own `try/except
   ImportError` block, with the symbols set to `None` (and `<PKG>_AVAILABLE
   = False`) on import failure. The chamber adapter slots in identically.

What does *not* hold across all five:

- **Subclassing `ContractAgent[InputT, OutputT]`** is used by `langchain.py`,
  `langgraph.py`, and `google_adk.py`, but `litellm_wrapper.py` and
  `claude_agent_sdk.py` manually wire their own `ResourceMonitor` /
  `TemporalMonitor` / `ContractEnforcer`. Both styles are live convention —
  see §3 below for which one the chamber adapter uses and why.

---

## 2. Decision Q1 — Adapter API signature

### 2.1 The decision

Two surfaces, mirroring `langchain.py` / `langgraph.py` / `google_adk.py`:

```python
# src/agent_contracts/integrations/causalchamber.py

class ContractedChamberAgent:
    """Contract-governed agent operating on a Causal Chamber dataset.

    Wraps a caller-constructed Contract with chamber-specific tooling
    (intervention queries, observation queries, ground-truth scoring),
    emitting tool events under per_tool_limits the same way litellm_wrapper
    emits per-call token tracking.
    """

    def __init__(
        self,
        contract: Contract,
        chamber: ChamberId,                 # "lt" | "wt"
        configuration: ConfigId = "standard",
        agent: Callable[..., Any] | None = None,  # the policy under test
        data_root: str | os.PathLike = "./data/causalchamber",
        strict_mode: bool = True,
    ) -> None:
        ...


def create_contracted_chamber_agent(
    chamber: ChamberId,
    intervention_budget: int,
    observation_budget: int = 0,
    configuration: ConfigId = "standard",
    *,
    agent: Callable[..., Any] | None = None,
    contract_id: str | None = None,
    extra_resources: ResourceConstraints | None = None,
    strict_mode: bool = True,
) -> ContractedChamberAgent:
    """Convenience factory that builds the Contract and wraps it in one call.

    For benchmark-style usage where the caller doesn't need full Contract
    customization. Construction-grade callers should use ContractedChamberAgent
    directly with their own Contract.
    """
    ...
```

### 2.2 Why this shape and not the §4.2 sketch

The §4.2 sketch did three things that the rest of the codebase does not:

| §4.2 sketch | Convention | Resolution |
|---|---|---|
| `@dataclass class ChamberContract` | Regular class, explicit `__init__` everywhere | Drop `@dataclass`; use regular class |
| `ChamberContract.__post_init__` builds the Contract | All integrations take `contract: Contract` as input | Caller constructs Contract; the factory function (not the class) is the convenience that combines them |
| Class noun is `ChamberContract` | Class noun is `Contracted<X>` everywhere | Rename to `ContractedChamberAgent` |

The rename matters more than it looks. `ChamberContract` reads as "a kind of
contract specialized for chambers"; `ContractedChamberAgent` reads as "an
agent that operates on chambers and is wrapped by a contract." The latter is
what the object actually is — and matches the mental model of every other
integration.

The factory function `create_contracted_chamber_agent` recovers the §4.2
sketch's ergonomics: callers who just want "give me a chamber agent with a
budget of k interventions" don't need to construct a `ResourceConstraints`
and a `Contract` by hand.

### 2.3 Wiring style: subclass `ContractAgent` or hand-wire monitors?

The chamber adapter **hand-wires** `ResourceMonitor` / `TemporalMonitor` /
`ContractEnforcer`, following `litellm_wrapper.py` and the
most-recently-added `claude_agent_sdk.py`. Reasons:

- The chamber agent's "step" is not a single LLM call (LiteLLM model) and
  not a graph invocation (LangGraph / LangChain model). It is a
  decide-intervene-observe loop with per-tool events emitted at every
  iteration. Hand-wiring makes the per-tool emission point explicit and
  obvious; subclassing `ContractAgent` would obscure it behind framework
  inheritance.
- `claude_agent_sdk.py` is the recently-added integration with a
  comparable mental model (agent loop with hook-based per-tool tracking)
  and it hand-wires. That precedent applies here.
- Hand-wiring also means the chamber adapter has zero hidden dependencies
  on the `ContractAgent` base class's input/output type-parameter system
  (`ContractAgent[InputT, OutputT]`), which is inconvenient for an agent
  whose "input" is "the chamber" and whose "output" is "an adjacency
  matrix + edge confidences" — neither fits the dict-shaped I/O the base
  class is designed around.

### 2.4 What the adapter does *not* own

To keep the integration thin and the responsibilities clean, the chamber
adapter is explicitly **not** responsible for:

- Choosing experiments (that's the *agent's* job — the agent calls
  `query_intervention(name)` and the adapter just spends one unit of
  budget)
- Inferring the graph (the agent or its downstream classical step does
  that)
- Computing SHD / F1 / CI coverage (that lives in the *pipeline*, not the
  integration — see Q2 below)
- Storing or comparing across runs (that's the orchestrator's job)

The adapter owns: dataset loading, ground-truth retrieval, two tools
(`query_intervention`, `query_observation`), and the per-tool-event wiring
into the framework's monitor.

### 2.5 Files affected by Q1

- New: `src/agent_contracts/integrations/causalchamber.py`
- Edit: `src/agent_contracts/integrations/__init__.py` — add the
  try/except block exporting `ContractedChamberAgent`,
  `create_contracted_chamber_agent`, and `CAUSAL_CHAMBER_AVAILABLE`,
  matching the existing five blocks.
- Edit: `pyproject.toml` — add `chambers` extra per §4.4 of the plan.

---

## 3. Decision Q2 — Where the validators live

### 3.1 The decision

**No new top-level `validators/` submodule.** SHD, F1, and CI-coverage
scoring functions live in **`evaluation/chamber_pipeline/scoring.py`** —
the same location §8 of the plan already specifies for the pipeline.

The §4.3 sketch's two validators are *scoring functions* (callables that
take an agent's reported adjacency matrix + the ground-truth reference and
return a numeric score), not `SuccessCriterion` instances. This is a real
distinction that the §4.3 framing partially obscured.

### 3.2 Why "no `validators/`"

Three converging signals:

1. **No `validators/` directory exists in `src/agent_contracts/`** — the
   pattern doesn't exist anywhere in the codebase. Adding one for a single
   pillar would establish a new architectural layer for one consumer.
2. **`SuccessCriterion.condition` is a string expression**, not a callable
   (verified in `core/contract.py:507-525`):
   ```python
   @dataclass
   class SuccessCriterion:
       name: str
       condition: str | Any   # e.g., "quality_score >= 0.80"
       weight: float = 1.0
       required: bool = False
   ```
   Real example in `evaluation/good_enough/adk_agents.py:451`:
   ```python
   SuccessCriterion(
       name="quality_threshold",
       condition=f"quality_score >= {quality_threshold}",
       weight=1.0,
       required=True,
   )
   ```
   The string is evaluated against a namespace populated by the pipeline.
   What looked like "validators slot into success_criteria" is really
   "scoring functions populate variables; success_criteria are string
   conditions on those variables."
3. **Sister pipelines (research, code_review) don't use `success_criteria`
   at all.** They keep a `success: bool` field on a `PipelineResult`
   dataclass and compute it inline in their orchestrators. Only the
   `good_enough/` pipeline uses `success_criteria`, because it specifically
   needs a quality threshold the agent can self-monitor against. The
   chamber pillar's primary metric (SHD) is *not* something the agent
   self-monitors — it's computed post-hoc by the orchestrator against
   ground truth the agent never sees. So `success_criteria` is the wrong
   home for it on structural grounds, not just style grounds.

### 3.3 What scoring.py looks like

```python
# evaluation/chamber_pipeline/scoring.py

import numpy as np
import pandas as pd

def shd(predicted: pd.DataFrame, reference: pd.DataFrame) -> int:
    """Structural Hamming Distance between predicted and reference adjacency.

    Standard causal-discovery metric: number of edge insertions, deletions,
    and reversals required to transform `predicted` into `reference`.
    Bounded above by n²; lower is better.
    """
    ...

def f1_edges(predicted: pd.DataFrame, reference: pd.DataFrame) -> float:
    """F1 score on edge presence (treating each (i,j) cell as a binary classification)."""
    ...

def ci_coverage(
    edge_intervals: dict[tuple[str, str], tuple[float, float]],
    reference: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calibration-coverage scoring on edge-presence CIs.

    Returns:
        (coverage, mean_interval_width) — coverage is the fraction of
        ground-truth indicators that fell inside the agent's reported
        (1-α) interval; mean_interval_width is the average width across
        all reported intervals (lower is better at fixed coverage).
    """
    ...
```

These are pipeline-local pure functions. The orchestrator calls them at
the end of each run, packages the metrics into the run record, and writes
the row to Parquet. No framework modification needed.

### 3.4 If we want SuccessCriterion-style pass/fail

Optional and additive: the orchestrator can also build a
`SuccessCriterion` with an SHD threshold per cell, e.g.:

```python
contract = Contract(
    id=f"chamber-{chamber}-k{k}",
    resources=ResourceConstraints(per_tool_limits={"intervene": k}),
    success_criteria=[
        SuccessCriterion(
            name="shd_threshold",
            condition=f"shd <= {shd_pass_threshold(chamber, k)}",
            required=True,
        ),
        SuccessCriterion(
            name="ci_coverage_target",
            condition="ci_coverage >= 0.95",
            required=False,
        ),
    ],
)
```

This is purely additive to the scoring-function-in-pipeline approach: the
scoring functions populate the namespace (`shd`, `ci_coverage`, etc.) that
`SuccessCriterion.condition` is evaluated against. No `validators/` module
required to support it.

This is the same pattern `good_enough/` already uses, so we have a working
precedent if the AAMAS reviewers ever ask "but can your framework
*formally* express a chamber success criterion?"

### 3.5 Files affected by Q2

- New: `evaluation/chamber_pipeline/scoring.py` (per §8 of the plan,
  unchanged)
- Not created: `src/agent_contracts/validators/causal.py` (the §4.3 plan
  explicitly listed this as one of two options; this decision picks the
  other)

---

## 4. Open items NOT covered by Q1+Q2

These remain deferred to M2 and beyond per the plan's §12:

- **Q3** — Whether to add the simulator path (Path B from §3.2). Decision
  deferred to M5 timeline check.
- **Q4** — Paper source organization (branch vs new file). Deferred to M7.
- **Q5** — Standalone `agent-contracts-bench` extraction. Deferred
  post-submission.

Q3–Q5 do not block M1 or M2.

---

## 5. M1 sign-off checklist

Per §9 of the plan, M1's acceptance criteria are:

- [ ] `integrations/causalchamber.py` stub committed (carries the API
  shape decided in §2.1, even if every method just `raise
  NotImplementedError`)
- [ ] `chambers` extra in `pyproject.toml` (per §4.4 of the plan)
- [ ] Failing smoke test exists (will pass once M2 lands)
- [x] §12 Q1+Q2 decisions documented based on a read of existing
  integrations (this file)

The first three items are next; this file completes the fourth.

---

## 6. References

- Plan: [`causal_chamber_validation_plan.md`](causal_chamber_validation_plan.md), especially §4.2, §4.3, §8, §9, §12
- Source files read for this decision:
  - `src/agent_contracts/integrations/litellm_wrapper.py`
  - `src/agent_contracts/integrations/langchain.py`
  - `src/agent_contracts/integrations/langgraph.py`
  - `src/agent_contracts/integrations/google_adk.py`
  - `src/agent_contracts/integrations/claude_agent_sdk.py`
  - `src/agent_contracts/integrations/__init__.py`
  - `src/agent_contracts/core/contract.py` (lines 506–540 for `SuccessCriterion`)
  - `evaluation/research_pipeline/orchestrator.py`
  - `evaluation/code_review_pipeline/orchestrator.py`
  - `evaluation/good_enough/adk_agents.py` (lines 440–486 for the only `success_criteria` usage in the codebase)
  - `evaluation/README.md`
