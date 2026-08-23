# M6 Coordination Ladder: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three coordination arms (`fan_in_homog`, `fan_in_spec`, `team`) to the chamber pipeline, backed by per-node budget enforcement through `DelegationGraph`, plus the tree-accounting scorer that turns P2 into a measurement.

**Architecture:** The three new arms share one agent function parameterised by role differentiation and negotiation. Budget flows through a `DelegationGraph` whose scout and aggregator nodes carry real `ResourceMonitor`s, and the adapter routes each tool call to the monitor of whichever node is currently acting. All new behaviour is opt-in so the two reused M4b arms stay byte-identical.

**Tech Stack:** Python 3.12+, pytest, pandas, LiteLLM via `_CountingLLM`, `agent_contracts.core.delegation_graph`, `causalchamber`. Run with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-08-22-m6-coordination-ladder-design.md`

## Global Constraints

- **Reuse safety is non-negotiable.** Rungs 0 (`llm_pc`) and 3 (`planner_reasoner`) must behave identically before and after every task. Every new parameter defaults to the current behaviour. Task 1 adds the regression test that pins this and it must stay green through Task 8.
- **The metered tool key is `"intervene"`**, never `"exp"`. `_require_per_tool_propagation` short-circuits on `granted == 0`, so a zero-grant on an unknown key raises nothing.
- **Grant `"observe"` an explicit zero too.** `create_contracted_chamber_agent` inserts that key only when `observation_budget > 0` (`causalchamber.py:496-498`) and `can_use_tool` treats an absent key as unlimited (`monitor.py:605-609`).
- **Never reuse `_SELECTION_MAX_TOKENS = 200` for a reasoning call.** DeepSeek v4 Flash spends the cap on reasoning tokens and returns empty `content`. New calls get `_RECONCILE_MAX_TOKENS` / `_NEGOTIATE_MAX_TOKENS`, sized 4–8× expected content.
- **Scout seeds are `2*seed` and `2*seed + 1`**, never `seed` and `seed + 1` — M4b seeds are contiguous `0..29`, so `seed + 1` collides with the next cell's scout_a. **But the seed alone does not decorrelate the scouts**: `_llm_select_loop` uses it only for the fallback RNG reached on an off-menu or duplicate response (`agents.py:435-441`). On the happy path rung 1's two scouts receive byte-identical messages. Diversity there must come from an **explicit `temperature`**, passed through to the completion call and recorded per cell — no `temperature` appears anywhere in `orchestrator.py` today, so leaving it to the provider default makes H-B uncontrolled and unreproducible.
- Token budgets are **non-binding at execution**: node monitors record tokens for certification arithmetic and must not halt on the token dimension. Interventions are live-gated.
- `mypy 2.3.1 --strict` clean; `uv run pytest -q` green after every task.

---

### Task 1: Additive per-node metering and token attribution

Two things must be true at once, and getting either wrong silently guts a headline claim:

- Routing must be **additive**, not a replacement. Every chamber call in the new arms runs inside an `as_node` block, so if the node monitor *replaces* the aggregate one, the adapter's `intervention_budget=k` cap is never consulted and the matched-budget guarantee evaporates.
- Somebody must write LLM token spend **into the node monitors**. `_CountingLLM` accumulates `total_input_tokens` / `total_output_tokens` per cell and writes them to the `RunRecord`; `DelegationGraph._consumed()` reads `node.monitor.usage`. Nothing connects the two, so without this every node's token consumption is 0 and `verify()` is trivially true — H-2 would be unfalsifiable.

**Files:**
- Modify: `src/agent_contracts/integrations/causalchamber.py`
- Test: `tests/integrations/test_causalchamber.py`

**Interfaces:**
- Produces:
  - `ContractedChamberAgent.__init__(contract, chamber, configuration="standard", agent=None, data_root=..., strict_mode=True, node_monitors: Mapping[str, ResourceMonitor] | None = None, token_meter: Callable[[], int] | None = None)`
  - `create_contracted_chamber_agent(..., node_monitors=None, token_meter=None)` — passes both through
  - `ContractedChamberAgent.as_node(name: str) -> ContextManager[None]` — routes metering **and** attributes the token delta measured across the block
  - `ContractedChamberAgent._charged_monitors() -> list[ResourceMonitor]` — the aggregate monitor, plus the active node's if one is set

- [ ] **Step 1: Write the failing tests**

```python
# tests/integrations/test_causalchamber.py
import pytest

from agent_contracts.core.contract import ResourceConstraints
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from agent_contracts.integrations.causalchamber import (
    create_contracted_chamber_agent,
)

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE, reason="causalchamber not installed"
)


@requires_causalchamber
def test_as_node_blocks_on_the_node_limit():
    agg = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=10, node_monitors={"aggregator": agg}
    )
    name = adapter.available_experiments()[0]
    with adapter.as_node("aggregator"), pytest.raises(Exception):
        adapter.query_intervention(name)


@requires_causalchamber
def test_aggregate_cap_still_binds_inside_as_node():
    """Routing is additive: the k cap must stay live, not be bypassed."""
    scout = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 99}))
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=2, node_monitors={"scout_a": scout}
    )
    menu = adapter.available_experiments()
    with adapter.as_node("scout_a"):
        adapter.query_intervention(menu[0])
        adapter.query_intervention(menu[1])
        with pytest.raises(Exception):      # aggregate k=2 exhausted
            adapter.query_intervention(menu[2])


@requires_causalchamber
def test_as_node_attributes_token_delta_to_the_node():
    counter = {"n": 0}
    mon = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 5}))
    adapter = create_contracted_chamber_agent(
        chamber="lt", intervention_budget=5,
        node_monitors={"scout_a": mon}, token_meter=lambda: counter["n"],
    )
    with adapter.as_node("scout_a"):
        counter["n"] += 1234           # stands in for _CountingLLM's totals
    assert mon.usage.tokens == 1234


@requires_causalchamber
def test_default_none_preserves_aggregate_behaviour():
    adapter = create_contracted_chamber_agent(chamber="lt", intervention_budget=2)
    menu = adapter.available_experiments()
    adapter.query_intervention(menu[0])
    adapter.query_intervention(menu[1])
    with pytest.raises(Exception):
        adapter.query_intervention(menu[2])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/integrations/test_causalchamber.py -k "as_node or aggregate_cap" -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'node_monitors'`.

- [ ] **Step 3: Implement additive routing and token attribution**

```python
# causalchamber.py
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

    def __init__(
        self,
        contract: Contract,
        chamber: ChamberId,
        configuration: ConfigId = "standard",
        agent: Callable[..., Any] | None = None,
        data_root: str | os.PathLike[str] = "./data/causalchamber",
        strict_mode: bool = True,
        node_monitors: Mapping[str, ResourceMonitor] | None = None,
        token_meter: Callable[[], int] | None = None,
    ) -> None:
        ...
        self._node_monitors = dict(node_monitors or {})
        self._token_meter = token_meter
        self._active_node: str | None = None

    @contextmanager
    def as_node(self, name: str) -> Iterator[None]:
        """Meter tool calls in this block against `name` as well as the aggregate.

        On exit, the token delta measured by `token_meter` across the block is
        attributed to the node's monitor. That is the only thing connecting
        `_CountingLLM`'s totals to `DelegationGraph._consumed()`.
        """
        if name not in self._node_monitors:
            raise KeyError(f"no monitor registered for node {name!r}")
        if self._active_node is not None:
            # Nesting would charge the inner block's tokens to both nodes and
            # silently inflate the certification arithmetic H-2 depends on.
            raise RuntimeError(
                f"as_node({name!r}) nested inside as_node({self._active_node!r}); "
                "close the outer block first"
            )
        start = self._token_meter() if self._token_meter is not None else None
        self._active_node = name
        try:
            yield
        finally:
            self._active_node = None
            if start is not None:
                self._node_monitors[name].usage.add_tokens(self._token_meter() - start)

    def _charged_monitors(self) -> list[ResourceMonitor]:
        """Every monitor that must approve, and be charged for, a tool call."""
        monitors = [self._resource_monitor]
        if self._active_node is not None:
            monitors.append(self._node_monitors[self._active_node])
        return monitors
```

- [ ] **Step 4: Route both chamber tools through every charged monitor**

In `query_intervention` (and identically in `query_observation` with `"observe"`), replace the single pre-check and the single charge:

```python
        for monitor in self._charged_monitors():
            if not monitor.can_use_tool("intervene"):
                self._enforcer._emit_event(
                    EnforcementEvent(
                        event_type="tool_blocked",
                        contract=self.contract,
                        message="Tool 'intervene' blocked: per-tool budget exhausted",
                        data={
                            "tool_name": "intervene",
                            "experiment_name": experiment_name,
                            # Report the limit that actually fired, not the
                            # aggregate one — otherwise a node blocked at 0
                            # is reported as blocked at k.
                            "limit": monitor.constraints.per_tool_limits.get("intervene"),
                            "actual": monitor.usage.get_tool_usage("intervene"),
                        },
                    )
                )
                if self.strict_mode:
                    raise ContractViolationError(
                        self.contract,
                        "per_tool_limit",
                        f"intervention budget exhausted (limit="
                        f"{monitor.constraints.per_tool_limits.get('intervene')})",
                    )

        df = self._dataset.get_experiment(experiment_name).as_pandas_dataframe()

        for monitor in self._charged_monitors():
            monitor.usage.add_tool_invocation("intervene")
```

- [ ] **Step 5: Thread both parameters through the factory**

`create_contracted_chamber_agent` (`causalchamber.py:444-455`) has neither parameter. Add `node_monitors=None` and `token_meter=None` as keyword-only arguments and forward them to the constructor. Task 7 Step 4 depends on this; without it every new arm raises `TypeError`.

- [ ] **Step 6: Run to verify all four pass**

Run: `uv run pytest tests/integrations/test_causalchamber.py -v`
Expected: PASS, including every pre-existing test unchanged.

- [ ] **Step 7: Add the two zero-grant regression tests**

```python
def test_wrong_key_exp_leaves_intervene_unconstrained():
    """Pins the trap: a zero-grant on an unknown key is silent."""
    m = ResourceMonitor(ResourceConstraints(per_tool_limits={"exp": 0}))
    assert m.can_use_tool("intervene") is True      # NOT blocked
    m2 = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    assert m2.can_use_tool("intervene") is False


def test_observe_absent_means_unlimited():
    m = ResourceMonitor(ResourceConstraints(per_tool_limits={"intervene": 0}))
    assert m.can_use_tool("observe") is True        # the side channel
    m2 = ResourceMonitor(
        ResourceConstraints(per_tool_limits={"intervene": 0, "observe": 0})
    )
    assert m2.can_use_tool("observe") is False
```

- [ ] **Step 8: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all green, count increased by 6.

```bash
git add src/agent_contracts/integrations/causalchamber.py tests/integrations/test_causalchamber.py
git commit -m "feat(chamber): additive per-node metering, token attribution, zero-grant regressions"
```

---

### Task 2: Blind role prompts and output caps

`build_reasoner_select_prompt` frames the task as refining "the Planner's picks (which appear in the `already_chosen` block)" (`llm_planner.py:162-177`). With `starting_chosen=None` that block is empty and the system message references nothing, so it cannot be reused blind.

**Files:**
- Modify: `evaluation/chamber_pipeline/llm_planner.py`
- Modify: `evaluation/chamber_pipeline/agents.py` (constants only)
- Test: `tests/evaluation/test_chamber_llm_agents.py`

**Interfaces:**
- Produces:
  - `build_scout_broad_prompt(menu: list[str], remaining_budget: int, already_chosen: list[str] | None = None) -> list[dict[str, str]]`
  - `build_scout_targeted_prompt(menu: list[str], remaining_budget: int, already_chosen: list[str] | None = None) -> list[dict[str, str]]`

  Both must match `PromptBuilder = Callable[[list[str], int, list[str] | None], list[dict[str, str]]]` (`agents.py:351`). `_llm_select_loop` calls `prompt_builder(menu, remaining, all_chosen)` with three positional arguments (`agents.py:424`); a two-parameter builder raises `TypeError` on every `differentiate=True` run. `already_chosen` here is the scout's **own** prior picks within its loop, never another agent's.
  - `build_reconcile_prompt(chosen_a: list[str], chosen_b: list[str]) -> list[dict[str, str]]`
  - `agents._RECONCILE_MAX_TOKENS: int`, `agents._NEGOTIATE_MAX_TOKENS: int`

- [ ] **Step 1: Write the failing test**

```python
def test_blind_scout_prompts_never_reference_already_chosen():
    from evaluation.chamber_pipeline.llm_planner import (
        build_reconcile_prompt,
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )
    menu = ["uniform_t_ir_1_mid", "uniform_l_12_mid", "uniform_diode_ir_3_mid"]
    for build in (build_scout_broad_prompt, build_scout_targeted_prompt):
        msgs = build(menu, 3, None)
        text = " ".join(m["content"] for m in msgs).lower()
        assert "already_chosen" not in text
        assert "planner" not in text
        assert "uniform_t_ir_1_mid" in text


def test_scout_roles_differ():
    from evaluation.chamber_pipeline.llm_planner import (
        build_reconcile_prompt,
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )
    a = build_scout_broad_prompt(["x"], 1, None)[0]["content"]
    b = build_scout_targeted_prompt(["x"], 1, None)[0]["content"]
    assert a != b


def test_scout_prompts_match_the_PromptBuilder_arity():
    import inspect
    from evaluation.chamber_pipeline.llm_planner import (
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
        build_select_prompt,
    )
    for build in (build_scout_broad_prompt, build_scout_targeted_prompt):
        assert len(inspect.signature(build).parameters) == len(
            inspect.signature(build_select_prompt).parameters
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_llm_agents.py -k blind_scout -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the prompts, modelled on `build_select_prompt`**

Write both with the same user-message body as `build_select_prompt` (menu plus remaining budget, one experiment name on its own line) and differing system messages: broad framing asks for maximum coverage of distinct intervention targets; targeted framing asks for experiments that disambiguate variables whose relationships are least constrained. Neither may mention another agent — they are blind by definition.

Add `build_reconcile_prompt`, which receives both scouts' selections and asks for a deduplicated, conflict-resolved ordering.

- [ ] **Step 3b: Add an explicit temperature parameter**

Give `_llm_select_loop` a `temperature: float | None = None` argument forwarded to the completion call, and a module constant `_SCOUT_TEMPERATURE = 1.0`. Record the value used in the `RunRecord` so a cell is reproducible. Rung 1's entire diversity mechanism is this number; leaving it to the provider default means a low-temperature default silently makes `overlap_frac` 1.0 and degenerates rung 1 into rung 0 at double the budget.

- [ ] **Step 4: Add the output-cap constants**

```python
# agents.py, beside _SELECTION_MAX_TOKENS = 200 and _ADJACENCY_MAX_TOKENS = 32768
_RECONCILE_MAX_TOKENS = 8192   # uncapped reasoning call; 4-8x expected content
_NEGOTIATE_MAX_TOKENS = 4096   # short proposals; still far above the 200 cap
```

- [ ] **Step 5: Run and commit**

Run: `uv run pytest tests/evaluation/test_chamber_llm_agents.py -v`
Expected: PASS.

```bash
git add evaluation/chamber_pipeline/llm_planner.py evaluation/chamber_pipeline/agents.py tests/evaluation/test_chamber_llm_agents.py
git commit -m "feat(chamber): blind scout role prompts and reconcile/negotiate token caps"
```

---

### Task 3: Overlap metric

**Files:**
- Create: `evaluation/chamber_pipeline/coordination.py`
- Test: `tests/evaluation/test_chamber_coordination.py`

**Interfaces:**
- Produces: `overlap_fraction(chosen_a: list[str], chosen_b: list[str]) -> float | None` — `|A ∩ B| / min(|A|, |B|)`, and **`None`** when either side is empty. Returning `0.0` there would make a cell where a scout got no picks indistinguishable from perfect disjointness, the H-B success case, and the analyzer would average the artifact in.

- [ ] **Step 1: Write the failing test**

```python
from evaluation.chamber_pipeline.coordination import overlap_fraction


def test_overlap_fraction_disjoint_is_zero():
    assert overlap_fraction(["a", "b"], ["c", "d"]) == 0.0


def test_overlap_fraction_identical_is_one():
    assert overlap_fraction(["a", "b"], ["a", "b"]) == 1.0


def test_overlap_fraction_uses_min_denominator():
    assert overlap_fraction(["a", "b", "c"], ["a"]) == 1.0


def test_overlap_fraction_empty_is_none_not_zero():
    assert overlap_fraction([], ["a"]) is None
    assert overlap_fraction(["a"], []) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_coordination.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

```python
def overlap_fraction(chosen_a: list[str], chosen_b: list[str]) -> float | None:
    """Fraction of the smaller selection that also appears in the larger.

    None when either side is empty - undefined, not zero.
    """
    if not chosen_a or not chosen_b:
        return None
    shared = len(set(chosen_a) & set(chosen_b))
    return shared / min(len(chosen_a), len(chosen_b))
```

- [ ] **Step 4: Run and commit**

Run: `uv run pytest tests/evaluation/test_chamber_coordination.py -v`
Expected: 4 PASS.

```bash
git add evaluation/chamber_pipeline/coordination.py tests/evaluation/test_chamber_coordination.py
git commit -m "feat(chamber): overlap_fraction metric"
```

---

### Task 4: The fan-in graph builder

**Files:**
- Modify: `evaluation/chamber_pipeline/coordination.py`
- Test: `tests/evaluation/test_chamber_coordination.py`

**Interfaces:**
- Produces: `build_fan_in_graph(k: int, c95: int, a95: int) -> DelegationGraph` with nodes `scout_a`, `scout_b`, `aggregator`, sealed, using the §4 formulas:
  `F = ceil(1.5*a95)`, `S = ceil(2*c95*ceil(k/2)) + F`, root `tokens = 2*S`.

- [ ] **Step 1: Write the failing test**

```python
import math
from agent_contracts.core.delegation_graph import DelegationGraph
from evaluation.chamber_pipeline.coordination import build_fan_in_graph


def test_fan_in_graph_seals_and_funds_the_aggregator():
    graph = build_fan_in_graph(k=30, c95=2303, a95=38752)
    assert graph.is_sealed
    F = math.ceil(1.5 * 38752)
    assert graph.in_flow("aggregator").tokens == 2 * F
    assert graph.in_flow("scout_a").per_tool["intervene"] == 15
    assert graph.in_flow("scout_b").per_tool["intervene"] == 15


def test_aggregator_is_zeroed_on_both_chamber_tools():
    graph = build_fan_in_graph(k=30, c95=2303, a95=38752)
    per_tool = graph.in_flow("aggregator").per_tool
    assert per_tool["intervene"] == 0
    assert per_tool["observe"] == 0


def test_scout_monitors_permit_their_first_intervention():
    """Regression: allocate() defaults tool_invocations to 0, which blocks
    every tool call before per-tool budgets are ever consulted."""
    graph = build_fan_in_graph(k=6, c95=1350, a95=21163)
    for scout in ("scout_a", "scout_b"):
        assert graph.monitor_for(scout).can_use_tool("intervene") is True
    assert graph.monitor_for("aggregator").can_use_tool("intervene") is False


def test_odd_budget_gives_the_remainder_to_scout_a():
    graph = build_fan_in_graph(k=45, c95=2778, a95=39191)
    assert graph.in_flow("scout_a").per_tool["intervene"] == 23
    assert graph.in_flow("scout_b").per_tool["intervene"] == 22
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_coordination.py -k fan_in_graph -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

```python
import math
from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.delegation_graph import DelegationGraph


def build_fan_in_graph(k: int, c95: int, a95: int) -> DelegationGraph:
    forward = math.ceil(1.5 * a95)
    scout_tokens = math.ceil(2 * c95 * math.ceil(k / 2)) + forward
    root = Contract(
        id=f"m6-root-k{k}",
        name="M6 root",
        resources=ResourceConstraints(
            tokens=2 * scout_tokens, per_tool_limits={"intervene": k, "observe": 0}
        ),
    )
    graph = DelegationGraph(root)
    for name in ("scout_a", "scout_b", "aggregator"):
        graph.add_node(name)
    # tool_invocations MUST be explicit. `allocate()` defaults every
    # unspecified dimension to 0 (not None), and `can_use_tool` checks the
    # aggregate branch first: `tool_invocations is not None and usage >= 0`
    # is True at zero usage, so an omitted grant blocks the node's very first
    # tool call before any per-tool budget is consulted. Verified: without
    # this, `monitor_for("scout_a").can_use_tool("intervene")` is False on a
    # freshly sealed graph.
    graph.allocate(
        DelegationGraph.ROOT, "scout_a",
        tokens=scout_tokens,
        tool_invocations=math.ceil(k / 2),
        per_tool={"intervene": math.ceil(k / 2), "observe": 0},
    )
    graph.allocate(
        DelegationGraph.ROOT, "scout_b",
        tokens=scout_tokens,
        tool_invocations=k // 2,
        per_tool={"intervene": k // 2, "observe": 0},
    )
    for scout in ("scout_a", "scout_b"):
        graph.allocate(
            scout, "aggregator",
            tokens=forward,
            tool_invocations=0,          # the aggregator makes no tool calls
            per_tool={"intervene": 0, "observe": 0},
        )
    graph.seal()
    return graph
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evaluation/test_chamber_coordination.py -k fan_in_graph -v`
Expected: 3 PASS. If `seal()` raises `GraphLintError: funded with nothing`, the token forward is zero — check `a95` reached the function.

- [ ] **Step 5: Commit**

```bash
git add evaluation/chamber_pipeline/coordination.py tests/evaluation/test_chamber_coordination.py
git commit -m "feat(chamber): fan-in DelegationGraph builder with per-role budgets"
```

---

### Task 5: `fan_in_agents` — rungs 1 and 2

**Files:**
- Modify: `evaluation/chamber_pipeline/agents.py`
- Test: `tests/evaluation/test_chamber_fan_in.py` (create)

**Interfaces:**
- Consumes: `build_fan_in_graph`, `overlap_fraction`, `_llm_select_loop`, `pool_experiment_data`, `run_pc`, `adapter.as_node`.
- Produces:
  `fan_in_agents(adapter, model=..., seed=0, pc_alpha=0.05, *, scout_a_budget: int, scout_b_budget: int, differentiate: bool = False, llm=None) -> pd.DataFrame`
  and a side-channel dict on the adapter, `adapter.coordination_stats`, carrying `overlap_frac` and `n_experiments_distinct`.

- [ ] **Step 0: Create the shared fixtures these tests need**

None of these fixtures exists. Three constraints the first draft got wrong:

- **There is no stub adapter and no two-node chamber.** `tests/evaluation/test_chamber_agents.py` builds *real* adapters via `create_contracted_chamber_agent(chamber="lt", ...)` behind a `requires_causalchamber` skipif. LT has 38 variables and 59 experiments.
- **Real experiment names look like `uniform_t_ir_1_mid`**, not `exp_0`. A responder that returns an off-menu name sends `_llm_select_loop` into its seeded random fallback (`agents.py:435-441`), so the test measures the fallback rather than the agent.
- **The adapter must be built with `node_monitors` registered**, or `as_node` raises `KeyError` before any agent logic runs.

```python
# tests/evaluation/conftest.py
from typing import Any

import pytest

from agent_contracts.core.contract import ResourceConstraints
from agent_contracts.core.monitor import ResourceMonitor
from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE, reason="causalchamber not installed"
)


class RecordingLLM:
    """FakeLLM plus max_tokens, so tests can classify calls by their cap."""

    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = 0

    def __call__(
        self, *, model: str, messages: list[dict[str, str]],
        max_tokens: int | None = None, **_: Any,
    ) -> dict:
        idx = len(self.calls)
        self.calls.append(
            {"model": model, "messages": messages, "idx": idx, "max_tokens": max_tokens}
        )
        self.total_tokens += 100      # stands in for _CountingLLM's accumulation
        return {"choices": [{"message": {"content": self._responder(idx, messages)}}]}


def _menu_from(messages: list[dict[str, str]]) -> list[str]:
    """Recover the menu from the user message.

    Parse only the text after the `Menu:` marker. `build_select_prompt` renders
    an "Already spent (do not repeat...)" block BEFORE the menu in the same
    message (`llm_planner.py:87-104`), and those names also start with
    `uniform_`. Scraping the whole body returns the scout's own prior pick from
    round 2 onward, which `_llm_select_loop` rejects as a duplicate and replaces
    via its seeded random fallback (`agents.py:435-441`) - so the test would
    silently measure the fallback instead of the responder.
    """
    body = messages[-1]["content"]
    _, _, after = body.partition("Menu:\n")
    return [
        tok
        for line in after.splitlines()
        for tok in [line.strip("- ").strip()]
        if tok.startswith("uniform_")
    ]


@pytest.fixture
def fake_llm():
    """Cycles through the menu so the two scouts do not trivially collide."""
    return RecordingLLM(
        lambda idx, msgs: (_menu_from(msgs) or [""])[idx % max(1, len(_menu_from(msgs)))]
    )


@pytest.fixture
def counting_llm():
    return RecordingLLM(lambda _i, msgs: (_menu_from(msgs) or [""])[0])


@pytest.fixture
def conflict_llm():
    """Always names the FIRST menu item, so both scouts collide every round."""
    return RecordingLLM(lambda _i, msgs: (_menu_from(msgs) or [""])[0])


@pytest.fixture
def make_ladder_adapter():
    """Factory: build an LT adapter whose token_meter tracks THIS test's LLM.

    A fixture that captured one specific LLM would charge add_tokens(0) on
    every as_node exit for tests driving the agent with a different fixture,
    silently zeroing the attribution H-2 depends on — and passing, because
    those tests do not assert on tokens.
    """
    from evaluation.chamber_pipeline.coordination import build_fan_in_graph

    def _make(llm, k: int = 4):
        graph = build_fan_in_graph(k=k, c95=1350, a95=21163)
        adapter = create_contracted_chamber_agent(
            chamber="lt",
            intervention_budget=k,
            node_monitors={
                n: graph.monitor_for(n) for n in ("scout_a", "scout_b", "aggregator")
            },
            token_meter=lambda: llm.total_tokens,
        )
        adapter.delegation_graph = graph
        return adapter

    return _make
```

Every test in Tasks 5 and 6 builds its adapter with `make_ladder_adapter(<its own llm fixture>)` and carries `@requires_causalchamber`. For example:

```python
@requires_causalchamber
def test_overlap_recorded(make_ladder_adapter, fake_llm):
    adapter = make_ladder_adapter(fake_llm)
    fan_in_agents(adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm)
    ...
```

- [ ] **Step 1: Write the failing test with a FakeLLM**

```python
# tests/evaluation/test_chamber_fan_in.py
import pandas as pd
from evaluation.chamber_pipeline.agents import fan_in_agents


@requires_causalchamber
def test_fan_in_returns_square_adjacency_over_node_names(make_ladder_adapter, fake_llm):
    out = fan_in_agents(
        adapter, seed=0, scout_a_budget=2, scout_b_budget=2, llm=fake_llm
    )
    nodes = adapter.ground_truth().columns.tolist()
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == nodes and list(out.columns) == nodes


@requires_causalchamber
def test_scouts_use_decorrelated_seeds(make_ladder_adapter, fake_llm, monkeypatch):
    seen = []
    import evaluation.chamber_pipeline.agents as A
    real = A._llm_select_loop
    monkeypatch.setattr(
        A, "_llm_select_loop",
        lambda *a, **k: (seen.append(a[3]), real(*a, **k))[1],
    )
    fan_in_agents(
        adapter, seed=7, scout_a_budget=1, scout_b_budget=1, llm=fake_llm
    )
    assert seen[:2] == [14, 15]        # 2*seed, 2*seed+1 — never 7, 8


@requires_causalchamber
def test_aggregator_consumes_tokens_via_reconciliation(make_ladder_adapter, counting_llm):
    """Without this call the fan-in edges carry budget nothing spends."""
    fan_in_agents(
        adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=counting_llm
    )
    from evaluation.chamber_pipeline.agents import _RECONCILE_MAX_TOKENS
    reconcile = [
        c for c in counting_llm.calls if c["max_tokens"] == _RECONCILE_MAX_TOKENS
    ]
    assert len(reconcile) == 1
    assert reconcile[0]["max_tokens"] != 200      # never the selection cap
    # The call being made is not enough: its tokens must reach the node monitor,
    # or DelegationGraph._consumed() reads zero and verify() is vacuous.
    agg = adapter.delegation_graph.monitor_for("aggregator")
    assert agg.usage.tokens > 0


@requires_causalchamber
def test_overlap_recorded(make_ladder_adapter, fake_llm):
    fan_in_agents(
        adapter, seed=0, scout_a_budget=1, scout_b_budget=1, llm=fake_llm
    )
    stats = adapter.coordination_stats
    assert 0.0 <= stats["overlap_frac"] <= 1.0
    assert stats["n_experiments_distinct"] >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_fan_in.py -v`
Expected: FAIL with ImportError on `fan_in_agents`.

- [ ] **Step 3: Implement**

```python
def fan_in_agents(
    adapter,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
    seed: int = 0,
    pc_alpha: float = 0.05,
    *,
    scout_a_budget: int,
    scout_b_budget: int,
    differentiate: bool = False,
    llm: LLMCallable | None = None,
) -> pd.DataFrame:
    """Two blind scouts fund one aggregator (rungs 1 and 2)."""
    from evaluation.chamber_pipeline.coordination import overlap_fraction
    from evaluation.chamber_pipeline.llm_planner import (
        build_reconcile_prompt,
        build_scout_broad_prompt,
        build_scout_targeted_prompt,
    )

    nodes = _node_names(adapter)
    # Set on EVERY path, including the early returns. Task 8 reads this in
    # run_cell; an attribute that only exists on the happy path raises
    # AttributeError on empty-menu cells.
    adapter.coordination_stats = {"overlap_frac": None, "n_experiments_distinct": 0}
    if _intervention_budget(adapter) <= 0 or not adapter.available_experiments():
        return _empty_adjacency(nodes)
    llm = llm or _default_llm()

    prompt_a = build_scout_broad_prompt if differentiate else build_select_prompt
    prompt_b = build_scout_targeted_prompt if differentiate else build_select_prompt

    with adapter.as_node("scout_a"):
        chosen_a, dfs_a = _llm_select_loop(
            adapter, llm, model, 2 * seed,
            spend=scout_a_budget, starting_chosen=None, prompt_builder=prompt_a,
        )
    with adapter.as_node("scout_b"):
        chosen_b, dfs_b = _llm_select_loop(
            adapter, llm, model, 2 * seed + 1,
            spend=scout_b_budget, starting_chosen=None, prompt_builder=prompt_b,
        )

    # The aggregator's reconciliation call. This is REQUIRED, not decorative:
    # PC is not an LLM call, so without it the aggregator consumes nothing, the
    # fan-in edges carry budget nobody spends, and P2 has no empirical form.
    with adapter.as_node("aggregator"):
        llm(
            model=model,
            messages=build_reconcile_prompt(chosen_a, chosen_b),
            max_tokens=_RECONCILE_MAX_TOKENS,
        )

    seen, dfs = set(), []
    for name, frame in zip(chosen_a + chosen_b, dfs_a + dfs_b, strict=True):
        if name not in seen:
            seen.add(name)
            dfs.append(frame)

    adapter.coordination_stats = {
        "overlap_frac": overlap_fraction(chosen_a, chosen_b),
        "n_experiments_distinct": len(seen),
    }
    if not dfs:
        return _empty_adjacency(nodes)
    return run_pc(pool_experiment_data(dfs, nodes), nodes, alpha=pc_alpha, seed=seed)
```

Note the deduplication: duplicates still *cost* budget (each `query_intervention` was metered) but are dropped before pooling so PC does not see inflated *n*.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evaluation/test_chamber_fan_in.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Verify reuse safety**

Run: `uv run pytest tests/evaluation/test_chamber_llm_agents.py tests/evaluation/test_chamber_planner_reasoner.py -q`
Expected: unchanged pass count — rungs 0 and 3 are untouched.

- [ ] **Step 6: Commit**

```bash
git add evaluation/chamber_pipeline/agents.py tests/evaluation/test_chamber_fan_in.py
git commit -m "feat(chamber): fan_in_agents for ensemble and parallel-roles rungs"
```

---

### Task 6: `team_agents` — rung 4 negotiation

**Files:**
- Modify: `evaluation/chamber_pipeline/agents.py`, `evaluation/chamber_pipeline/llm_planner.py`
- Test: `tests/evaluation/test_chamber_team.py` (create)

**Interfaces:**
- Consumes: everything from Task 5.
- Produces:
  - `build_negotiate_propose_prompt(menu, budget, role) -> list[dict[str, str]]`
  - `build_negotiate_revise_prompt(menu, budget, own: list[str], other: list[str]) -> list[dict[str, str]]`
  - `team_agents(adapter, model=..., seed=0, pc_alpha=0.05, *, scout_a_budget, scout_b_budget, llm=None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
@requires_causalchamber
def test_team_makes_exactly_four_negotiation_calls(make_ladder_adapter, counting_llm):
    from evaluation.chamber_pipeline.agents import team_agents
    team_agents(
        adapter, seed=0,
        scout_a_budget=1, scout_b_budget=1, llm=counting_llm,
    )
    from evaluation.chamber_pipeline.agents import _NEGOTIATE_MAX_TOKENS
    negotiation = [
        c for c in counting_llm.calls if c["max_tokens"] == _NEGOTIATE_MAX_TOKENS
    ]
    assert len(negotiation) == 4          # propose + revise, per scout


@requires_causalchamber
def test_team_backstop_removes_contested_picks_from_scout_b(make_ladder_adapter, conflict_llm):
    """Both scouts name the same experiment; scout_a keeps it, scout_b re-picks.

    The backstop must apply to the EXECUTED selections, not just the proposals
    — scouts execute blind after negotiating, so filtering only the proposals
    leaves them free to re-collide. Implement it by removing contested names
    from scout_b's selectable menu before its select loop runs.
    """
    from evaluation.chamber_pipeline.agents import team_agents
    team_agents(
        adapter, seed=0,
        scout_a_budget=2, scout_b_budget=2, llm=conflict_llm,
    )
    assert adapter.coordination_stats["overlap_frac"] == 0.0


def test_team_channel_cannot_be_a_bidirectional_graph_edge():
    """Only the SECOND edge of the pair is a cycle, and only on an unsealed graph.

    `build_fan_in_graph` returns a sealed graph and `allocate` calls
    `_require_unsealed()` before any cycle check, so a sealed graph raises the
    sealed error, not CycleError. And scout_a -> scout_b alone is not a cycle:
    scout_b's only out-edge feeds the childless aggregator. It is the return
    edge that closes the loop. Hence: message passing, not graph edges.
    """
    import pytest
    from agent_contracts.core.contract import Contract, ResourceConstraints
    from agent_contracts.core.delegation_graph import CycleError, DelegationGraph

    root = Contract(
        id="t", name="t", resources=ResourceConstraints(tokens=1000)
    )
    graph = DelegationGraph(root)
    for n in ("scout_a", "scout_b"):
        graph.add_node(n)
    graph.allocate(DelegationGraph.ROOT, "scout_a", tokens=100)
    graph.allocate(DelegationGraph.ROOT, "scout_b", tokens=100)
    graph.allocate("scout_a", "scout_b", tokens=0)      # not yet a cycle
    with pytest.raises(CycleError):
        graph.allocate("scout_b", "scout_a", tokens=0)  # closes it
```

Note the third test seals the graph in `build_fan_in_graph`; if `allocate` raises a "sealed" error before the cycle check, build an unsealed graph inline instead and assert `CycleError` there.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_team.py -v`
Expected: FAIL with ImportError on `team_agents`.

- [ ] **Step 3: Implement the protocol**

One upfront round, O(1) in k:

1. Each scout proposes its intended selections (`build_negotiate_propose_prompt`, capped at `_NEGOTIATE_MAX_TOKENS`).
2. Each scout sees the other's proposal and revises once (`build_negotiate_revise_prompt`).
3. Deterministic backstop: any experiment still claimed by both is assigned to `scout_a` and removed from scout_b's selectable menu. Filtering only the proposals is not enough — the scouts execute blind afterwards and would simply re-collide.
4. Both execute via `_llm_select_loop`, scout_b over the reduced menu, then the aggregator reconciles and PC runs.

**This requires a new parameter on `_llm_select_loop`.** It currently takes no menu argument — it reads `adapter.available_experiments()` itself (`agents.py:354`, `causalchamber.py:203-210`), and that list is static. The only existing lever is `starting_chosen`, which is wrong here twice over: it renders the excluded names into scout_b's prompt as an "Already spent" block (`llm_planner.py:87-91`), destroying the blindness Task 2 exists to establish; and it feeds `actual_spend = min(spend, len(available))`, so a large contested set silently under-spends scout_b and breaks matched-budget comparability.

Add `exclude: set[str] | None = None`, applied to the menu *without* touching the prompt:

```python
    menu = [m for m in adapter.available_experiments() if m not in (exclude or set())]
```

Assert in the team arm that `len(menu) >= scout_b_budget` after exclusion; if the contested set is large enough to violate that, fall back to assigning contested picks alternately rather than all to scout_a.

The scout-to-scout channel is a Python variable passed between the two calls. It is **not** a `DelegationGraph` edge: the return edge of a bidirectional pair raises `CycleError` regardless of carrying zero, because `allocate()` runs `_reaches()` before inspecting the amount. Control flow cycles; budget flow does not — P3 made concrete.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evaluation/test_chamber_team.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/chamber_pipeline/agents.py evaluation/chamber_pipeline/llm_planner.py tests/evaluation/test_chamber_team.py
git commit -m "feat(chamber): team_agents with one-round allocation negotiation"
```

---

### Task 7: Registry, kwargs, and the opt-in token cap

**Files:**
- Modify: `evaluation/chamber_pipeline/orchestrator.py`
- Test: `tests/evaluation/test_chamber_orchestrator.py`

**Interfaces:**
- Consumes: `fan_in_agents`, `team_agents`.
- Produces: three `AgentSpec` entries named `fan_in_homog`, `fan_in_spec`, `team`, each `chambers=("lt", "wt")`, `accepts_llm=True`, `kind="llm_multi"`, `extra_kwargs=("scout_a_budget", "scout_b_budget")`.

- [ ] **Step 1: Write the failing test**

```python
def test_new_arms_registered_with_scout_budgets():
    from evaluation.chamber_pipeline.orchestrator import get_spec
    for name in ("fan_in_homog", "fan_in_spec", "team"):
        spec = get_spec(name)
        assert spec.extra_kwargs == ("scout_a_budget", "scout_b_budget")
        assert spec.kind == "llm_multi"


def test_scout_budgets_split_with_remainder_to_a():
    from evaluation.chamber_pipeline.orchestrator import _build_agent_kwargs, get_spec
    kwargs = _build_agent_kwargs(get_spec("team"), budget_k=45, seed=0, pc_alpha=0.05, llm=None)
    assert kwargs["scout_a_budget"] == 23
    assert kwargs["scout_b_budget"] == 22
    assert kwargs["scout_a_budget"] + kwargs["scout_b_budget"] == 45


def test_existing_arms_kwargs_unchanged():
    from evaluation.chamber_pipeline.orchestrator import _build_agent_kwargs, get_spec
    kwargs = _build_agent_kwargs(get_spec("planner_reasoner"), budget_k=30, seed=0, pc_alpha=0.05, llm=None)
    assert kwargs == {"seed": 0, "pc_alpha": 0.05, "planner_budget": 15, "reasoner_budget": 15}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_orchestrator.py -k scout -v`
Expected: FAIL with `Unknown agent name: 'fan_in_homog'`.

- [ ] **Step 3: Implement the registry entries and the kwargs branch**

```python
    if "scout_a_budget" in spec.extra_kwargs:
        kwargs["scout_b_budget"] = budget_k // 2
        kwargs["scout_a_budget"] = budget_k - budget_k // 2
    if spec.name == "fan_in_spec":
        kwargs["differentiate"] = True
```

`fan_in_homog` and `fan_in_spec` both dispatch to `fan_in_agents`; only `differentiate` differs.

- [ ] **Step 4: Wire the graph, its monitors, and the opt-in token cap**

This is the join between Tasks 4, 5 and 6 — without it `adapter.as_node(...)` raises `KeyError`. In `run_cell`, for the three new arms only:

```python
from evaluation.chamber_pipeline.coordination import build_fan_in_graph

graph = build_fan_in_graph(k=budget_k, c95=C95[budget_k], a95=A95[budget_k])
adapter = create_contracted_chamber_agent(
    chamber, intervention_budget=budget_k, ...,
    # No aggregate token cap: `as_node` charges node monitors only, so the
    # adapter's own `usage.tokens` stays 0 for the life of the cell and any
    # constraint on it is unreachable. Token budgets live on the graph, where
    # verify() can actually see them.
    node_monitors={
        n: graph.monitor_for(n) for n in ("scout_a", "scout_b", "aggregator")
    },
    # Without token_meter, as_node attributes nothing, every node's token
    # consumption stays 0, and verify() is trivially true - H-2 unfalsifiable.
    token_meter=lambda: counting_llm.total_input_tokens + counting_llm.total_output_tokens,
)
adapter.delegation_graph = graph      # read back by the Task 8 scorer
```

The adapter must be constructed **after** `counting_llm`, since `token_meter` closes over it. `C95` and `A95` are module-level dicts keyed by budget, holding the §4 table values. Rungs 0 and 3 must continue to receive neither `extra_resources` nor `node_monitors`.

- [ ] **Step 4b: Assert token spend never gates a tool call**

```python
@requires_causalchamber
def test_tokens_do_not_gate_tool_calls():
    """Tokens are certified post-hoc, not enforced. A binding cap would
    truncate new-arm cells while reused rungs 0 and 3 ran uncapped."""
    from evaluation.chamber_pipeline.coordination import build_fan_in_graph
    graph = build_fan_in_graph(k=6, c95=1350, a95=21163)
    m = graph.monitor_for("scout_a")
    assert m.can_use_tool("intervene") is True    # baseline: not already blocked
    m.usage.add_tokens(10 ** 9)
    assert m.can_use_tool("intervene") is True    # tokens still do not gate tools
```

The baseline assertion matters: without it this test passes vacuously if the node is blocked for an unrelated reason, which is exactly what the missing `tool_invocations` grant caused.

- [ ] **Step 5: Run and confirm reuse safety**

Run: `uv run pytest tests/evaluation/ -q`
Expected: all green, `test_existing_arms_kwargs_unchanged` passing.

- [ ] **Step 6: Commit**

```bash
git add evaluation/chamber_pipeline/orchestrator.py tests/evaluation/test_chamber_orchestrator.py
git commit -m "feat(chamber): register ensemble/parallel-roles/team arms"
```

---

### Task 8: Tree-accounting scorer and result schema

**Files:**
- Create: `evaluation/chamber_pipeline/tree_accounting.py`
- Modify: `evaluation/chamber_pipeline/results.py`
- Test: `tests/evaluation/test_chamber_tree_accounting.py` (create)

**Interfaces:**
- Produces:
  - `tree_certified_bound(graph: DelegationGraph) -> int` — the total-consumption bound a drop-policy tree accountant would certify
  - `dag_certified_bound(graph: DelegationGraph) -> int` — `B(root)` plus refunds
  - `RunRecord` gains `overlap_frac: float | None`, `n_experiments_distinct: int | None`, `conservation_certified: bool | None`, `tree_accounting_bound: int | None`

- [ ] **Step 1: Write the failing test**

```python
from evaluation.chamber_pipeline.coordination import build_fan_in_graph
from evaluation.chamber_pipeline.tree_accounting import (
    dag_certified_bound,
    tree_certified_bound,
)


def test_tree_bound_exceeds_dag_bound_on_a_fan_in_graph():
    graph = build_fan_in_graph(k=30, c95=2303, a95=38752)
    assert tree_certified_bound(graph) > dag_certified_bound(graph)


def test_the_gap_equals_the_second_parents_forward():
    graph = build_fan_in_graph(k=30, c95=2303, a95=38752)
    gap = tree_certified_bound(graph) - dag_certified_bound(graph)
    assert gap == graph.in_flow("aggregator").tokens // 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_tree_accounting.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

The asymmetry is the whole point, and getting it wrong collapses the gap to zero. A dropped edge is **invisible to its source but real at its target**: the accountant does not know `scout_b` forwarded `F`, yet the aggregator genuinely holds `2F`.

So for each node, permitted spend = *real* in-flow minus *accountant-visible* out-flow, clamped at zero:

```
root          2S - 2S = 0
scout_a        S -  F
scout_b        S -  0 = S      <- its forward to the aggregator is invisible
aggregator    2F -  0 = 2F     <- but the aggregator really holds both forwards
                        -----
tree_certified_bound  = 2S + F
```

`dag_certified_bound` returns `B(root)` plus any refunds recorded on the graph, which for a freshly sealed graph is `2S`. The gap is therefore exactly `F`, the double-counted forward from the dropped second parent — the empirical form of P2.

Summing "the budget the forest makes available" instead gives `(S−F) + S + F = 2S`, no gap at all, and both tests fail. Note also that `build_fan_in_graph` goes through `DelegationGraph.allocate`, which *refuses* the over-commitment P2 is about — so the scorer must model the counterfactual explicitly rather than re-running anything.

- [ ] **Step 4: Add the four result columns**

Add them to `RunRecord` as optional fields defaulting to `None`, following the existing pattern at `results.py:120-145`. Populate them in `run_cell` with `getattr(adapter, "coordination_stats", {})` and `getattr(adapter, "delegation_graph", None)` — rungs 0 and 3 set neither, and a bare attribute access raises `AttributeError` on every reused-arm cell.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all green. Confirm existing Parquet round-trip tests still pass with the new optional columns.

- [ ] **Step 6: Commit**

```bash
git add evaluation/chamber_pipeline/tree_accounting.py evaluation/chamber_pipeline/results.py tests/evaluation/test_chamber_tree_accounting.py
git commit -m "feat(chamber): tree-accounting counterfactual scorer and coordination columns"
```

---

### Task 9: Ladder figures and the power table

**Files:**
- Modify: `evaluation/chamber_pipeline/analyze_results.py`
- Test: `tests/evaluation/test_chamber_analyze_results.py`

**Interfaces:**
- Consumes: the four columns added in Task 8.
- Produces:
  - `ladder_frame(df: pd.DataFrame) -> pd.DataFrame` — one row per (rung, budget) with mean F1, mean SHD, mean tokens, mean wall time, failure rate, mean `overlap_frac`
  - `minimum_detectable_effect(df: pd.DataFrame, agent: str, budget_k: int) -> float` — `2.8 * sd * sqrt(2/n)`
  - `plot_ladder(df, out_dir: Path) -> list[Path]` — accuracy, cost, and failure-rate panels

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from evaluation.chamber_pipeline.analyze_results import (
    ladder_frame,
    minimum_detectable_effect,
)

RUNGS = ["llm_pc", "fan_in_homog", "fan_in_spec", "planner_reasoner", "team"]


def _synthetic(n=30):
    rng = np.random.default_rng(0)
    rows = []
    for agent in RUNGS:
        for k in (6, 30, 45):
            for seed in range(n):
                rows.append({
                    "agent_name": agent, "budget_k": k, "seed": seed,
                    "status": "ok", "f1": float(rng.normal(0.4, 0.04)),
                    "shd": 55.0, "tokens_in": 1000, "tokens_out": 500,
                    "wall_time_seconds": 300.0, "overlap_frac": 0.3,
                })
    return pd.DataFrame(rows)


def test_ladder_frame_has_one_row_per_rung_and_budget():
    out = ladder_frame(_synthetic())
    assert len(out) == len(RUNGS) * 3
    assert {"f1_mean", "failure_rate", "overlap_frac_mean"} <= set(out.columns)


def test_mde_matches_the_closed_form():
    df = _synthetic()
    got = minimum_detectable_effect(df, "llm_pc", 30)
    sd = df[(df.agent_name == "llm_pc") & (df.budget_k == 30)].f1.std()
    assert abs(got - 2.8 * sd * np.sqrt(2 / 30)) < 1e-9


def test_failure_rate_counts_non_ok_cells():
    df = _synthetic()
    df.loc[df.index[:3], "status"] = "error"
    out = ladder_frame(df)
    assert out.failure_rate.max() > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evaluation/test_chamber_analyze_results.py -k ladder -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the three functions**

`ladder_frame` groups by `(agent_name, budget_k)` and aggregates; `failure_rate` is the share of cells whose `status != "ok"`, so it must be computed **before** filtering to ok-cells. `minimum_detectable_effect` uses the within-agent per-cell SD of `f1` over ok-cells, **with `ddof=1`** (pandas' `Series.std()` default). A NumPy implementation defaulting to `ddof=0` differs by `sqrt(30/29)` ≈ 1.7 % and fails the closed-form test with no hint why. `plot_ladder` writes three PNGs with rungs on the x-axis in ladder order, one line per budget.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evaluation/test_chamber_analyze_results.py -v`
Expected: PASS, existing analyzer tests unchanged.

- [ ] **Step 5: Print the MDE alongside every accuracy comparison**

Add the MDE column to the analyzer's text output. Per spec §6, the paper reports an equivalence bound, not a null — an accuracy difference below the MDE must never be printed without the MDE next to it.

- [ ] **Step 6: Commit**

```bash
git add evaluation/chamber_pipeline/analyze_results.py tests/evaluation/test_chamber_analyze_results.py
git commit -m "feat(chamber): ladder frame, MDE, and coordination-cost figures"
```

---

## Post-implementation: pre-flight before the sweep

Not tasks — operational steps gated on both tracks landing.

1. Run the 5-cell `llm_pc` k=45 calibration; recompute `c95(45)` and replace the interpolated `~2778`.
2. Run the 24-cell overlap probe at k=30. Abort and redesign if `fan_in_spec` **and** `team` both exceed `overlap_frac` 0.8.
3. Run the 20-cell reuse-validity guard; `llm_pc`'s 15-cell mean must sit within 0.02 F1 of M4b.
4. Confirm the theory track's Aug 29 gate passed before committing VPS time.
