"""Causal Chamber integration for Agent Contracts.

This module provides contract-aware tooling for agents operating on the
Causal Chamber datasets (Gamella et al., *Nature Machine Intelligence* 2025;
<https://causalchamber.ai/>). It wraps a caller-constructed Contract with
chamber-specific tools (intervention queries, observation queries) and
emits per-tool events under the framework's `per_tool_limits` machinery.

This is the **AAMAS / ECAI 2027 mainstream-venue extension** pillar — see
`docs/causal_chamber_validation_plan.md` for the full design and
`docs/causal_chamber_M1_decisions.md` for the conventions this stub follows.

Example (post-M2; today this raises NotImplementedError):
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.causalchamber import (
    ...     ContractedChamberAgent,
    ...     create_contracted_chamber_agent,
    ... )
    >>>
    >>> # Power-user form: caller constructs Contract
    >>> contract = Contract(
    ...     id="chamber-lt-k15",
    ...     resources=ResourceConstraints(per_tool_limits={"intervene": 15}),
    ... )
    >>> agent = ContractedChamberAgent(
    ...     contract=contract,
    ...     chamber="lt",
    ...     configuration="standard",
    ... )
    >>>
    >>> # Convenience form: factory builds the Contract for you
    >>> agent = create_contracted_chamber_agent(
    ...     chamber="lt",
    ...     intervention_budget=15,
    ... )

Status: M1 stub — API shape only. M2 lands the real implementation; M3
plugs in the five baseline agents from §5 of the validation plan.
"""

import dataclasses
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.enforcement import ContractEnforcer, EnforcementEvent
from agent_contracts.core.monitor import ResourceMonitor, TemporalMonitor
from agent_contracts.core.wrapper import ContractViolationError

# Optional dependency: causalchamber. Pattern matches the other integrations
# (litellm_wrapper, langchain, langgraph, google_adk, claude_agent_sdk).
try:
    from causalchamber.datasets import Dataset
    from causalchamber.ground_truth import graph as _gt_graph

    CAUSAL_CHAMBER_AVAILABLE = True
except ImportError:
    CAUSAL_CHAMBER_AVAILABLE = False
    Dataset = Any  # type: ignore[assignment, misc]
    _gt_graph = Any  # type: ignore[assignment]


ChamberId = Literal["lt", "wt"]
ConfigId = Literal["standard", "pressure-control"]


# Per-chamber dataset selection. LT and WT use different dataset names
# because their interventional designs differ:
#   - LT: 59-experiment uniform menu (lt_interventions_standard_v1)
#   - WT: 28-experiment menu (wt_validate_v1)
# See §3.2 of docs/causal_chamber_validation_plan.md for menu sizes.
#
# WT was `wt_walks_v1` until 2026-08-25. That release is a random-walk
# TIME SERIES: median lag-1 autocorrelation 0.9999, so its 320,000 rows per
# experiment carry roughly 19 independent observations. Feeding it to PC's
# Fisher-Z test -- which assumes i.i.d. samples -- inverted the budget
# response. Measured over the same 28-experiment menu, 12 seeds per point:
#
#   k/M               0.11   0.25   0.50   0.75   1.00
#   wt_walks_v1       0.181  0.178  0.157  0.144  0.155   F1 DECLINES
#   wt_validate_v1    0.070  0.120  0.163  0.164  0.257   F1 rises
#
# Under wt_walks_v1, SHD worsened 55 -> 67 and predicted edges grew 25 -> 38
# as more data arrived: spurious density from a violated test assumption,
# not a property of the wind tunnel. `wt_validate_v1` covers the same menu
# with lag-1 autocorrelation 0.14 and reproduces LT's qualitative shape,
# which is what makes the two chambers comparable at all.
#
# This is a scientific choice and must be stated in the paper, not buried
# here: we use the near-i.i.d. WT release because PC's independence test is
# invalid on the random-walk release.
# Keyed by (chamber, configuration), NOT by chamber alone. `configuration`
# selects the ground-truth graph via `causalchamber.ground_truth.graph(...)`,
# and the chamber is physically different in each mode -- in
# `pressure-control` the hatch is servo-driven, which introduces the mediators
# `standard` does not have (wt/standard: 21 sources, 0 mediators, 11 sinks;
# wt/pressure-control: 19 / 3 / 10). Keying the dataset by chamber alone meant
# asking for `pressure-control` returned the `standard` DATA scored against the
# `pressure-control` GRAPH -- a silent mismatch that no column would reveal.
# Not previously reached: every recorded run used `standard`. Unpaired
# combinations raise rather than guess; `wt_pc_validate_v1` is the release to
# wire when the pressure-control arm is actually run.
DATASET_FOR_CHAMBER_CONFIGURATION: dict[tuple[str, str], str] = {
    ("lt", "standard"): "lt_interventions_standard_v1",
    ("wt", "standard"): "wt_validate_v1",
}

#: Backwards-compatible view for callers that only know the chamber. Retained
#: because it is part of the module's published surface; prefer the pair-keyed
#: table above.
DATASET_FOR_CHAMBER: dict[str, str] = {
    chamber: name
    for (chamber, configuration), name in DATASET_FOR_CHAMBER_CONFIGURATION.items()
    if configuration == "standard"
}


def dataset_for(chamber: str, configuration: str) -> str:
    """Dataset release backing one (chamber, configuration) pair.

    Raises:
        ValueError: If the pair has no wired dataset. Raising is the point:
            the alternative is scoring one configuration's data against
            another's ground-truth graph, which produces plausible numbers
            and no error.
    """
    try:
        return DATASET_FOR_CHAMBER_CONFIGURATION[(chamber, configuration)]
    except KeyError:
        raise ValueError(
            f"no dataset is wired for chamber={chamber!r} "
            f"configuration={configuration!r}; wired pairs are "
            f"{sorted(DATASET_FOR_CHAMBER_CONFIGURATION)}. The configuration "
            "selects the ground-truth GRAPH, so pairing it with another "
            "configuration's data scores against the wrong truth silently."
        ) from None


class ContractedChamberAgent:
    """Contract-governed agent operating on a Causal Chamber dataset.

    Wraps a caller-constructed Contract with chamber-specific tools and
    emits tool events under `per_tool_limits["intervene"]` and
    `per_tool_limits["observe"]`. The agent (passed via `agent=...`) is the
    policy under test — Random, GreedyIG-lite, LLM-only, LLM+PC, or
    Planner+Reasoner per §5 of the validation plan.

    This class hand-wires `ResourceMonitor` / `TemporalMonitor` /
    `ContractEnforcer` rather than subclassing `ContractAgent`, matching
    the convention used by `litellm_wrapper.py` and `claude_agent_sdk.py`.
    See §2.3 of `docs/causal_chamber_M1_decisions.md` for the rationale.

    Responsibilities (intentionally narrow per §2.4 of M1 decisions):
        - Load the chamber dataset on construction
        - Retrieve the ground-truth graph for scoring (held internally; the
          agent does not see it)
        - Expose `query_intervention()` and `query_observation()` as tools
          that spend per-tool budget
        - Emit enforcement events on each tool call

    Explicit non-responsibilities:
        - Choosing which experiment to query (the agent does that)
        - Inferring the graph from query results (the agent / classical step)
        - Computing SHD / F1 / CI coverage (lives in the pipeline, not the
          integration — see `evaluation/chamber_pipeline/scoring.py`)
        - Multi-run aggregation (the orchestrator's job)

    Attributes:
        contract: The contract governing this agent's execution
        chamber: Chamber identifier ("lt" or "wt")
        configuration: Chamber configuration ("standard" or "pressure-control")
        agent: Optional callable representing the policy under test
        strict_mode: If True, violations halt execution immediately
    """

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
        """Initialize the contracted chamber agent.

        Args:
            contract: Contract defining resource and temporal constraints. The
                caller is responsible for setting `per_tool_limits` for the
                "intervene" and "observe" tools.
            chamber: Which physical chamber's dataset to load.
            configuration: Chamber configuration variant. Defaults to "standard".
            agent: Optional callable implementing the policy under test. If
                None, the integration only exposes tools; the agent loop is
                external (useful for unit testing).
            data_root: Local directory for cached chamber datasets. Created
                on first use.
            strict_mode: If True, constraint violations raise immediately;
                if False, violations are logged but execution continues.

        Raises:
            ImportError: If the `causalchamber` package is not installed.
        """
        if not CAUSAL_CHAMBER_AVAILABLE:
            raise ImportError(
                "causalchamber is required for the Causal Chamber integration. "
                "Install with: pip install 'ai-agent-contracts[chambers]'"
            )

        self.contract = contract
        self.chamber: ChamberId = chamber
        self.configuration: ConfigId = configuration
        self.agent = agent
        self.data_root = os.fspath(data_root)
        self.strict_mode = strict_mode

        # Hand-wire monitors and enforcer (pattern from claude_agent_sdk.py /
        # litellm_wrapper.py — see §2.3 of M1 decisions doc).
        self._resource_monitor = ResourceMonitor(contract.resources)
        # Per-node metering is ADDITIVE: a node monitor is consulted and charged
        # *in addition to* the aggregate one, never instead of it. Replacing it
        # would bypass the adapter's intervention_budget=k cap and silently
        # dissolve the matched-budget guarantee the ladder depends on.
        self._node_monitors: dict[str, ResourceMonitor] = dict(node_monitors or {})
        self._token_meter = token_meter
        self._active_node: str | None = None
        self._temporal_monitor = TemporalMonitor(contract)
        self._events: list[dict[str, Any]] = []
        self._enforcer = ContractEnforcer(
            contract,
            strict_mode=strict_mode,
            callbacks=[self._on_enforcement_event],
            monitor=self._resource_monitor,
        )

        # Dataset and ground-truth handles populated lazily on first access.
        # The package's Dataset(...) call needs the parent dir to already exist
        # before it tries to write the downloaded zip — create it eagerly so
        # any subsequent load() / ground_truth() / query_*() call just works.
        os.makedirs(self.data_root, exist_ok=True)
        self._dataset: Any = None
        # Every experiment this adapter actually served, in spending order.
        # Recorded HERE rather than in each agent because the adapter is the
        # single choke point every arm passes through -- seven agents can each
        # forget to report their picks; `query_intervention` cannot. Until
        # 2026-08-29 only the COUNT of distinct picks was kept, which made
        # "why does an arm with identical coverage score worse?" unanswerable
        # after the fact: the answer is in WHICH experiments were bought, and
        # nothing wrote them down.
        self.purchased: list[str] = []
        self._ground_truth: Any = None

    # ------------------------------------------------------------ data loading

    def load(self) -> None:
        """Download (if needed) the chamber dataset and load ground truth.

        Idempotent — subsequent calls are no-ops. Called automatically on
        first tool use; can also be called eagerly to surface download
        errors at construction time rather than mid-run.
        """
        if self._dataset is None:
            self._dataset = Dataset(
                name=dataset_for(self.chamber, self.configuration),
                root=self.data_root,
                download=True,
            )
        if self._ground_truth is None:
            self._ground_truth = _gt_graph(
                chamber=self.chamber,
                configuration=self.configuration,
            )

    def _ensure_loaded(self) -> None:
        """Trigger lazy load on first access."""
        if self._dataset is None or self._ground_truth is None:
            self.load()

    def available_experiments(self) -> list[str]:
        """Return the list of experiment names (the menu, size M).

        Available without spending any budget — this is the catalog the
        agent consults when planning which interventions to query.
        """
        self._ensure_loaded()
        return list(self._dataset.available_experiments())

    # ------------------------------------------------------------------ tools

    @contextmanager
    def as_node(self, name: str) -> Iterator[None]:
        """Meter tool calls in this block against `name` as well as the aggregate.

        On exit, the token delta measured by `token_meter` across the block is
        attributed to the node's monitor. That is the only thing connecting
        `_CountingLLM`'s totals to `DelegationGraph._consumed()`; without it
        every node's token consumption is zero and `verify()` is trivially true.
        """
        if name not in self._node_monitors:
            raise KeyError(f"no monitor registered for node {name!r}")
        if self._active_node is not None:
            # Nesting would charge the inner block's tokens to both nodes and
            # silently inflate the certification arithmetic.
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

    def query_intervention(self, experiment_name: str) -> Any:
        """Spend one unit of `per_tool_limits["intervene"]` and return data.

        Flow follows the convention used by `claude_agent_sdk.py`'s pre/post
        tool hooks:

            1. Pre-check: gate via `ResourceMonitor.can_use_tool("intervene")`.
               If exhausted in strict_mode, raise `ContractViolationError`
               immediately without running the tool or charging the budget.
            2. Run the tool (load the experiment from the dataset).
            3. Post-check: increment `tool_usage_by_name["intervene"]` and
               emit a `tool_use` enforcement event for the audit trail.

        "Charge on success" means a failed query (e.g., bad name) does not
        consume budget — the agent gets to retry without penalty.

        Args:
            experiment_name: Name of the pre-recorded experiment (one of the
                M names returned by `available_experiments()`).

        Returns:
            DataFrame of measurements for the requested experiment.

        Raises:
            ContractViolationError: When per-tool budget is exhausted in
                strict_mode.
            KeyError / ValueError: From the underlying `Dataset` when the
                experiment name is unknown — propagated as-is. Budget is
                NOT charged on this path.
        """
        self._ensure_loaded()

        # Pre-check every charged monitor, not just the aggregate one.
        for monitor in self._charged_monitors():
            if monitor.can_use_tool("intervene"):
                continue
            # Report the limit that actually fired, not the aggregate one --
            # otherwise a node blocked at 0 is reported as blocked at k.
            limit = monitor.constraints.per_tool_limits.get("intervene")
            self._enforcer._emit_event(
                EnforcementEvent(
                    event_type="tool_blocked",
                    contract=self.contract,
                    message="Tool 'intervene' blocked: per-tool budget exhausted",
                    data={
                        "tool_name": "intervene",
                        "experiment_name": experiment_name,
                        "limit": limit,
                        "actual": monitor.usage.get_tool_usage("intervene"),
                        "node": self._active_node,
                    },
                )
            )
            if self.strict_mode:
                raise ContractViolationError(
                    self.contract,
                    "per_tool_limit",
                    f"intervention budget exhausted (limit={limit})",
                )

        # Run
        df = self._dataset.get_experiment(experiment_name).as_pandas_dataframe()

        # Post: charge every monitor that approved it + emit audit event.
        # Appended on the SUCCESS path only, so the roster matches what the
        # budget was actually charged for -- a failed lookup costs nothing and
        # must not appear as a purchase.
        self.purchased.append(experiment_name)
        for monitor in self._charged_monitors():
            monitor.usage.add_tool_invocation("intervene")
        self._enforcer._emit_event(
            EnforcementEvent(
                event_type="tool_use",
                contract=self.contract,
                message="Tool 'intervene' executed",
                data={
                    "tool_name": "intervene",
                    "experiment_name": experiment_name,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                },
            )
        )
        return df

    def query_observation(self, n_samples: int = 1) -> Any:
        """Spend one unit of `per_tool_limits["observe"]` and return passive data.

        Returns `n_samples` rows from a designated observational source.

        **M2 semantic note:** the LT `lt_interventions_standard_v1` and WT
        `wt_walks_v1` datasets do not ship a separate "purely observational"
        experiment. As a stand-in, this method returns the first `n_samples`
        rows of the *first* listed experiment, treating those rows as a
        passive baseline view of the chamber. This is a deliberate
        placeholder — M3 may refine the semantic once concrete agents
        surface what they actually need from `query_observation()`.

        The budget tracking is not a placeholder: per-tool enforcement,
        violation events, and audit emissions all behave correctly today.

        Args:
            n_samples: Number of passive samples to draw. Counts as ONE
                unit of `per_tool_limits["observe"]` — the *call* is the
                budgeted resource, not the row count, matching the pattern
                used by `per_tool_limits["intervene"]`.

        Returns:
            DataFrame of `n_samples` passive observations.

        Raises:
            ContractViolationError: When per-tool budget is exhausted in
                strict_mode.
            ValueError: If `n_samples <= 0`.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        self._ensure_loaded()

        for monitor in self._charged_monitors():
            if monitor.can_use_tool("observe"):
                continue
            limit = monitor.constraints.per_tool_limits.get("observe")
            self._enforcer._emit_event(
                EnforcementEvent(
                    event_type="tool_blocked",
                    contract=self.contract,
                    message="Tool 'observe' blocked: per-tool budget exhausted",
                    data={
                        "tool_name": "observe",
                        "n_samples": n_samples,
                        "limit": limit,
                        "actual": monitor.usage.get_tool_usage("observe"),
                        "node": self._active_node,
                    },
                )
            )
            if self.strict_mode:
                raise ContractViolationError(
                    self.contract,
                    "per_tool_limit",
                    f"observation budget exhausted (limit={limit})",
                )

        # Stand-in passive source: first n_samples rows of first experiment.
        # See M2 semantic note in the docstring.
        first_name = self._dataset.available_experiments()[0]
        df = self._dataset.get_experiment(first_name).as_pandas_dataframe().head(n_samples)

        for monitor in self._charged_monitors():
            monitor.usage.add_tool_invocation("observe")
        self._enforcer._emit_event(
            EnforcementEvent(
                event_type="tool_use",
                contract=self.contract,
                message="Tool 'observe' executed",
                data={
                    "tool_name": "observe",
                    "n_samples": n_samples,
                    "rows_returned": int(df.shape[0]),
                },
            )
        )
        return df

    # ------------------------------------------------------------- ground-truth

    def ground_truth(self) -> Any:
        """Return the ground-truth adjacency matrix for this chamber/config.

        Held by the integration but **not** exposed to the agent during a
        run — only the orchestrator should call this for post-hoc scoring
        (SHD, F1, CI coverage). Exposed here as a method (not a property)
        to make the "this is for scoring, not for the agent" intent visible
        in call sites.

        Calls `causalchamber.ground_truth.graph(chamber, configuration)`
        on first invocation; subsequent calls return the cached DataFrame.

        Returns:
            Square adjacency-matrix DataFrame with rows/columns indexed by
            node names. Nonzero entries denote edges.
        """
        self._ensure_loaded()
        return self._ground_truth

    # ------------------------------------------------------------- run loop

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound agent under contract enforcement.

        Thin wrapper: starts the enforcer, dispatches to
        `self.agent(self, *args, **kwargs)`, and stops the enforcer in a
        `try/finally` so the contract state transitions even on exception.

        For tests that only exercise the tools (and drive the loop
        themselves), call `query_intervention()` / `query_observation()`
        directly without going through `run()`.

        Args:
            *args: Forwarded to `self.agent`.
            **kwargs: Forwarded to `self.agent`.

        Returns:
            Whatever `self.agent` returns.

        Raises:
            RuntimeError: If `agent` was not provided at construction.
            Exception: Any exception from the agent or from contract
                enforcement (e.g., `ContractViolationError`) propagates;
                the enforcer is stopped in `finally` regardless.
        """
        if self.agent is None:
            raise RuntimeError(
                "ContractedChamberAgent.run() requires an `agent` callable "
                "passed at construction time."
            )

        self._ensure_loaded()
        self._enforcer.start()
        try:
            return self.agent(self, *args, **kwargs)
        finally:
            self._enforcer.stop(reason="run() complete")

    # ------------------------------------------------------------- internals

    def _on_enforcement_event(self, event: EnforcementEvent) -> None:
        """Append enforcement events to the audit log."""
        self._events.append(
            {
                "type": event.event_type,
                "message": event.message,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        """Read-only view of the enforcement event log."""
        return list(self._events)


def create_contracted_chamber_agent(
    chamber: ChamberId,
    intervention_budget: int,
    observation_budget: int = 0,
    configuration: ConfigId = "standard",
    *,
    agent: Callable[..., Any] | None = None,
    contract_id: str | None = None,
    extra_resources: ResourceConstraints | None = None,
    data_root: str | os.PathLike[str] = "./data/causalchamber",
    strict_mode: bool = True,
    node_monitors: Mapping[str, ResourceMonitor] | None = None,
    token_meter: Callable[[], int] | None = None,
) -> ContractedChamberAgent:
    """Build a ContractedChamberAgent with sensible defaults.

    Convenience factory for benchmark-style usage where the caller doesn't
    need full Contract customization. Constructs a `Contract` whose
    `ResourceConstraints.per_tool_limits` enforces the supplied budgets,
    then wraps it.

    Power-user callers (multi-agent setups, custom termination conditions,
    success criteria with SHD thresholds, etc.) should construct a
    `Contract` directly and pass it to `ContractedChamberAgent(...)`.

    Args:
        chamber: Which physical chamber's dataset to load.
        intervention_budget: Max number of interventional queries.
        observation_budget: Max number of passive observations. Defaults to 0,
            which is enforced as a hard zero -- the limit is always emitted,
            so a caller who wants unbounded observations must say so through
            `extra_resources`, which wins on key conflicts.
        configuration: Chamber configuration variant. Defaults to "standard".
        agent: Optional callable implementing the policy under test.
        contract_id: Optional explicit contract id. Defaults to
            f"chamber-{chamber}-{configuration}-k{intervention_budget}".
        extra_resources: Optional additional `ResourceConstraints` to merge
            (e.g., a token cap for LLM-bearing variants). The caller is
            responsible for passing a constraints object whose per-tool
            limits include the chamber tools, or this function will overwrite
            them.
        data_root: Local directory for cached chamber datasets.
        strict_mode: Forwarded to the constructed agent.

    Returns:
        A ContractedChamberAgent ready to call.

    Raises:
        ImportError: If the `causalchamber` package is not installed
            (raised by the `ContractedChamberAgent` constructor).
    """
    if contract_id is None:
        contract_id = f"chamber-{chamber}-{configuration}-k{intervention_budget}"

    # Build per-tool limits, merging any extra resource constraints the caller
    # supplied. Caller-provided per_tool_limits are merged with the chamber
    # tools (caller wins on key conflicts so they can override budgets).
    # `observe` is emitted unconditionally, zero included. Omitting the key
    # does not mean "no observations" -- `can_use_tool` treats an absent key as
    # *unconstrained*, so the pre-fix `if observation_budget > 0` guard made
    # every default-constructed agent able to call `query_observation` without
    # limit, the opposite of what the parameter documents.
    per_tool_limits: dict[str, int] = {
        "intervene": intervention_budget,
        "observe": observation_budget,
    }

    if extra_resources is not None:
        # Caller-provided per_tool_limits win on key conflicts.
        merged_per_tool = {**per_tool_limits, **extra_resources.per_tool_limits}
        resources = dataclasses.replace(extra_resources, per_tool_limits=merged_per_tool)
    else:
        resources = ResourceConstraints(per_tool_limits=per_tool_limits)

    contract = Contract(
        id=contract_id,
        name=f"Causal Chamber: {chamber}/{configuration}",
        description=(
            f"Causal-discovery contract for {chamber}/{configuration} chamber, "
            f"intervention budget k={intervention_budget}"
            + (f", observation budget={observation_budget}" if observation_budget > 0 else "")
        ),
        resources=resources,
    )

    return ContractedChamberAgent(
        contract=contract,
        chamber=chamber,
        configuration=configuration,
        agent=agent,
        data_root=data_root,
        strict_mode=strict_mode,
        node_monitors=node_monitors,
        token_meter=token_meter,
    )


__all__ = [
    "CAUSAL_CHAMBER_AVAILABLE",
    "DATASET_FOR_CHAMBER",
    "ChamberId",
    "ConfigId",
    "ContractedChamberAgent",
    "create_contracted_chamber_agent",
]


# Suppress "imported but unused" warnings for the lazy-imported handles —
# they're held here so M2 can call them without re-importing.
_ = (Dataset, _gt_graph)
