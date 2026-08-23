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
from collections.abc import Callable
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


# Per-chamber dataset selection. LT and WT use different dataset names because
# their interventional designs differ:
#   - LT: 59-experiment uniform menu (lt_interventions_standard_v1)
#   - WT: 28-experiment random-walk menu (wt_walks_v1)
# See §3.2 of docs/causal_chamber_validation_plan.md for menu sizes.
DATASET_FOR_CHAMBER: dict[str, str] = {
    "lt": "lt_interventions_standard_v1",
    "wt": "wt_walks_v1",
}


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
                name=DATASET_FOR_CHAMBER[self.chamber],
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

        # Pre-check
        if not self._resource_monitor.can_use_tool("intervene"):
            self._enforcer._emit_event(
                EnforcementEvent(
                    event_type="tool_blocked",
                    contract=self.contract,
                    message="Tool 'intervene' blocked: per-tool budget exhausted",
                    data={
                        "tool_name": "intervene",
                        "experiment_name": experiment_name,
                        "limit": self.contract.resources.per_tool_limits.get("intervene"),
                        "actual": self._resource_monitor.usage.get_tool_usage("intervene"),
                    },
                )
            )
            if self.strict_mode:
                raise ContractViolationError(
                    self.contract,
                    "per_tool_limit",
                    f"intervention budget exhausted "
                    f"(limit={self.contract.resources.per_tool_limits.get('intervene')})",
                )

        # Run
        df = self._dataset.get_experiment(experiment_name).as_pandas_dataframe()

        # Post: charge budget + emit audit event
        self._resource_monitor.usage.add_tool_invocation("intervene")
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

        if not self._resource_monitor.can_use_tool("observe"):
            self._enforcer._emit_event(
                EnforcementEvent(
                    event_type="tool_blocked",
                    contract=self.contract,
                    message="Tool 'observe' blocked: per-tool budget exhausted",
                    data={
                        "tool_name": "observe",
                        "n_samples": n_samples,
                        "limit": self.contract.resources.per_tool_limits.get("observe"),
                        "actual": self._resource_monitor.usage.get_tool_usage("observe"),
                    },
                )
            )
            if self.strict_mode:
                raise ContractViolationError(
                    self.contract,
                    "per_tool_limit",
                    f"observation budget exhausted "
                    f"(limit={self.contract.resources.per_tool_limits.get('observe')})",
                )

        # Stand-in passive source: first n_samples rows of first experiment.
        # See M2 semantic note in the docstring.
        first_name = self._dataset.available_experiments()[0]
        df = self._dataset.get_experiment(first_name).as_pandas_dataframe().head(n_samples)

        self._resource_monitor.usage.add_tool_invocation("observe")
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
