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

import os
from collections.abc import Callable
from typing import Any, Literal

from agent_contracts.core.contract import Contract, ResourceConstraints
from agent_contracts.core.enforcement import ContractEnforcer, EnforcementEvent
from agent_contracts.core.monitor import ResourceMonitor, TemporalMonitor

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

        # Dataset and ground-truth handles populated lazily in M2.
        self._dataset: Any = None
        self._ground_truth: Any = None

    # ------------------------------------------------------------------ tools

    def query_intervention(self, experiment_name: str) -> Any:
        """Spend one unit of `per_tool_limits["intervene"]` and return data.

        The agent calls this to "spend" intervention budget. The integration:
            1. Checks whether the per-tool budget allows the call
            2. Loads the named experiment from the dataset
            3. Increments `tool_usage_by_name["intervene"]`
            4. Emits an enforcement event
            5. Returns the experiment's measurements as a DataFrame

        Args:
            experiment_name: Name of the pre-recorded experiment (one of the
                M experiments listed by `Dataset.available_experiments()`).

        Returns:
            DataFrame of measurements for the requested experiment.

        Raises:
            NotImplementedError: M1 stub. Implementation lands in M2.
            ContractViolationError: (post-M2) when the per-tool budget is
                exhausted in strict_mode.
        """
        raise NotImplementedError(
            "M1 stub. M2 will wire dataset access + per-tool event emission. "
            "See docs/causal_chamber_validation_plan.md §9 milestone M2."
        )

    def query_observation(self, n_samples: int = 1) -> Any:
        """Spend `n_samples` units of `per_tool_limits["observe"]` and return data.

        The agent calls this to draw passive (non-interventional) samples
        from the chamber. Used by §5 baselines that need observational data
        in addition to interventional data.

        Args:
            n_samples: Number of passive samples to draw.

        Returns:
            DataFrame of n_samples passive observations.

        Raises:
            NotImplementedError: M1 stub. Implementation lands in M2.
        """
        raise NotImplementedError(
            "M1 stub. M2 will wire passive-sample access + per-tool tracking."
        )

    # ------------------------------------------------------------- ground-truth

    def ground_truth(self) -> Any:
        """Return the ground-truth adjacency matrix for this chamber/config.

        Held by the integration but **not** exposed to the agent during a
        run — only the orchestrator should call this for post-hoc scoring
        (SHD, F1, CI coverage). Exposed here as a method (not a property)
        to make the "this is for scoring, not for the agent" intent visible
        in call sites.

        Returns:
            DataFrame adjacency matrix with rows/columns indexed by node names.

        Raises:
            NotImplementedError: M1 stub. M2 will call
                `causalchamber.ground_truth.graph(chamber, configuration)`.
        """
        raise NotImplementedError("M1 stub. M2 will call causalchamber.ground_truth.graph(...).")

    # ------------------------------------------------------------- run loop

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound agent under contract enforcement.

        Convenience wrapper for the common case `agent(self, *args, **kwargs)`
        executed under the contract's monitor + enforcer. For test cases that
        only need the tools (and drive the loop themselves), call
        `query_intervention()` / `query_observation()` directly.

        Raises:
            NotImplementedError: M1 stub. M3 implements the agent loop once
                the five baselines exist.
            RuntimeError: If `agent` was not provided at construction.
        """
        if self.agent is None:
            raise RuntimeError(
                "ContractedChamberAgent.run() requires an `agent` callable "
                "passed at construction time."
            )
        raise NotImplementedError(
            "M1 stub. M3 wires the agent loop with start/stop monitoring "
            "around `self.agent(self, *args, **kwargs)`."
        )

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
        observation_budget: Max number of passive observations. Defaults to 0.
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
        NotImplementedError: M1 stub. M2 lands the real Contract assembly.
    """
    raise NotImplementedError(
        "M1 stub. M2 will assemble Contract(per_tool_limits={...}) and call "
        "ContractedChamberAgent(...). See docs/causal_chamber_M1_decisions.md "
        "§2.1 for the intended API."
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
