"""Sweep orchestrator for the chamber pillar.

Owns the AgentSpec registry (the single source of truth for which
variants exist, which chambers each is compatible with, and what kwargs
each accepts), the per-cell runner `run_cell`, and the full-sweep
runner `run_sweep`. M4's `run_experiment.py` CLI is a thin wrapper
around `run_sweep`; the orchestrator itself has no CLI dependencies
and is fully testable with mocked LLM via the `llm` parameter.

Design (resolves the four open questions from M3 final review):

1. **Agent dispatch** (M3-review #5): AgentSpec registry. Each agent
   declares its name, callable, chamber compatibility, kwargs schema,
   and whether it accepts an injectable LLM. The orchestrator iterates
   the registry; per-cell dispatch is data-driven, not if/elif.

2. **Compatibility API** (M3-review #6): `AgentSpec.chambers` is a
   tuple of chamber-id strings the agent supports. The orchestrator
   filters BEFORE invoking — incompatible cells produce a "skipped"
   RunRecord, never a NotImplementedError mid-sweep. (The
   NotImplementedError raise inside the agent remains as a defensive
   double-check, but the registry is the contractual API.)

3. **Metadata aggregation** (M3-review #4): each cell installs a
   logging handler scoped to `evaluation.chamber_pipeline.inference`
   that captures PC-degeneracy warnings into a per-cell counter.
   Aggregated into the RunRecord's `n_pc_degeneracies` field — no
   log-scraping needed at analysis time.

4. **Sweep harness**: serial nested for-loop (chamber → budget →
   agent → seed). Parallelism deferred to M5 if M4 wall-time is a
   problem; the simpler design is easier to debug at pilot scale
   (450 cells) and the LLM API is the bottleneck anyway.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agent_contracts.integrations.causalchamber import (
    ChamberId,
    ConfigId,
    create_contracted_chamber_agent,
)

from .agents import (
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    planner_reasoner_agents,
    random_agent,
)
from .results import RunRecord, now_iso
from .scoring import f1_edges, shd

# Type alias for the LLM callable threaded through agents that accept it.
# Mirrors agents.py's LLMCallable. Production callers pass
# `litellm.completion`; tests pass FakeLLM.
LLMCallable = Callable[..., Any]


@dataclass(frozen=True)
class AgentSpec:
    """Descriptor for one chamber-pillar agent variant.

    The single source of truth for what a variant is, where it can
    run, and what kwargs it accepts. The orchestrator dispatches
    agents through this descriptor — callers don't import individual
    agent functions.

    Attributes:
        name: Short variant name (registry key). Matches the names
            used in `RunRecord.agent_name` and the §6.5 figure legend.
        run: The agent callable. Must match
            `agent(adapter, **kwargs) -> pd.DataFrame`.
        chambers: Tuple of chamber IDs this agent is compatible with.
            The orchestrator filters cells using this — incompatible
            chambers produce skipped RunRecords without ever invoking
            the agent. Tracks plan §5.1 footnotes (e.g.,
            GreedyIG-lite is LT-only: chambers=("lt",)).
        accepts_llm: True iff the agent accepts an `llm=` keyword.
            Lets the orchestrator decide whether to pass the
            injectable LLM callable (real `litellm.completion` or
            FakeLLM in tests).
        kind: Coarse classification — "non_llm" / "llm_single" /
            "llm_multi". Used by the orchestrator to decide whether
            to track LLM-specific metadata (`n_llm_calls`).
        extra_kwargs: Per-variant required kwargs that aren't
            seed/pc_alpha/llm. E.g., `planner_reasoner_agents`
            requires `planner_budget` and `reasoner_budget`. The
            orchestrator computes these per-cell from the cell's
            total budget.
    """

    name: str
    run: Callable[..., Any]
    chambers: tuple[ChamberId, ...]
    accepts_llm: bool = False
    kind: str = "non_llm"  # "non_llm" | "llm_single" | "llm_multi"
    extra_kwargs: tuple[str, ...] = ()

    def is_compatible(self, chamber: ChamberId) -> bool:
        """True iff this agent runs on the given chamber.

        The orchestrator gates dispatch on this — never invokes the
        agent for an incompatible chamber. Defensive double-check
        inside the agent (NotImplementedError) remains, but this is
        the contractual API.
        """
        return chamber in self.chambers


# The five plan §5.1 variants. Edit this list to add/remove variants
# from sweeps; the orchestrator picks them up automatically.
#
# Compatibility per plan §5.1:
#   - Random, LLM-only, LLM+PC, Planner+Reasoner: both chambers (LT, WT)
#   - GreedyIG-lite: LT only (WT's experimental design has no discrete
#     intervention targets — see plan §5.1 row 2 footnote and the
#     NotImplementedError in `agents.greedy_ig_lite_agent`)
AGENT_REGISTRY: list[AgentSpec] = [
    AgentSpec(
        name="random",
        run=random_agent,
        chambers=("lt", "wt"),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="greedy_ig_lite",
        run=greedy_ig_lite_agent,
        chambers=("lt",),  # WT skipped per plan §5.1
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="llm_only",
        run=llm_only_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
    ),
    AgentSpec(
        name="llm_pc",
        run=llm_pc_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
    ),
    AgentSpec(
        name="planner_reasoner",
        run=planner_reasoner_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("planner_budget", "reasoner_budget"),
    ),
]


def get_spec(name: str) -> AgentSpec:
    """Look up an AgentSpec by name. KeyError on unknown name."""
    for spec in AGENT_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(
        f"Unknown agent name: {name!r}. Available: {sorted(s.name for s in AGENT_REGISTRY)}"
    )


# ---------------------------------------------------------------------------
# PC-degeneracy capture
# ---------------------------------------------------------------------------


class _PcDegeneracyHandler(logging.Handler):
    """Logging handler that counts PC singular-matrix fallback warnings.

    Installed on `evaluation.chamber_pipeline.inference` for the
    duration of one cell. The warning text is matched on a stable
    substring (`"fell back"`) — same wording the inference module
    uses, kept loose enough to survive minor message tweaks but tight
    enough to ignore unrelated future warnings.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # The inference module's fallback warning starts with "PC
        # inference fell back to all-zeros adjacency". Match loosely
        # on "fell back" so wording tweaks don't silently break this.
        if "fell back" in record.getMessage().lower():
            self.count += 1


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------


def _budget_fraction(budget_k: int, menu_size: int) -> float:
    """Compute k/M, clamped to [0, 1]. Returns 0.0 on empty menus."""
    if menu_size <= 0:
        return 0.0
    return min(1.0, budget_k / menu_size)


def _build_agent_kwargs(
    spec: AgentSpec,
    budget_k: int,
    seed: int,
    pc_alpha: float,
    llm: LLMCallable | None,
) -> dict[str, Any]:
    """Construct the kwargs dict to pass to `spec.run`.

    Centralizes the per-variant kwargs assembly:
      - All variants get `seed`.
      - PC-using variants get `pc_alpha` (random, greedy_ig_lite,
        llm_pc, planner_reasoner — i.e., everyone except llm_only).
      - LLM-bearing variants get `llm` if provided.
      - Planner+Reasoner additionally needs `planner_budget` and
        `reasoner_budget` — split evenly with the remainder going to
        the planner (the more-defensive default per plan §5.3).
    """
    kwargs: dict[str, Any] = {"seed": seed}

    # llm_only doesn't take pc_alpha (no PC inference step).
    if spec.name != "llm_only":
        kwargs["pc_alpha"] = pc_alpha

    if spec.accepts_llm and llm is not None:
        kwargs["llm"] = llm

    if "planner_budget" in spec.extra_kwargs:
        # Even split with remainder to planner: floor(k/2) for reasoner,
        # k - reasoner for planner. Total exactly equals budget_k so
        # conservation is satisfied at the framework level too.
        reasoner_budget = budget_k // 2
        planner_budget = budget_k - reasoner_budget
        kwargs["planner_budget"] = planner_budget
        kwargs["reasoner_budget"] = reasoner_budget

    return kwargs


def run_cell(
    spec: AgentSpec,
    chamber: ChamberId,
    configuration: ConfigId,
    budget_k: int,
    seed: int,
    pc_alpha: float = 0.05,
    llm: LLMCallable | None = None,
) -> RunRecord:
    """Run one cell of the sweep grid and return a RunRecord.

    Catches all exceptions per cell — this method NEVER raises
    (the orchestrator depends on per-cell isolation: one bad cell
    must not lose the surrounding sweep). Skipped, ok, and error
    cells all produce well-formed RunRecords with `status` set
    accordingly.

    Args:
        spec: The agent variant to run.
        chamber: Which chamber to load.
        configuration: Chamber configuration.
        budget_k: Total intervention budget for this cell.
        seed: RNG seed.
        pc_alpha: PC independence-test significance level (ignored for
            llm_only; passed through for all others).
        llm: Injectable LLM callable. None means the agent's own
            default is used (which lazy-imports `litellm.completion`).

    Returns:
        A RunRecord with status "ok" / "skipped" / "error".
    """
    started_at = now_iso()

    # Pre-flight: is this agent compatible with this chamber?
    if not spec.is_compatible(chamber):
        return RunRecord(
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=0.0,  # filled below if we can load the chamber
            seed=seed,
            status="skipped",
            started_at=started_at,
            finished_at=started_at,
            skip_reason=(
                f"agent '{spec.name}' is not compatible with chamber '{chamber}' "
                f"(spec.chambers = {spec.chambers})"
            ),
        )

    # Build the chamber adapter. Failure here is an "error" cell, not
    # a "skipped" — agent compatibility was satisfied but adapter
    # construction itself broke (e.g., disk full, network down for
    # first-time dataset download).
    try:
        adapter = create_contracted_chamber_agent(
            chamber=chamber,
            configuration=configuration,
            intervention_budget=budget_k,
        )
        menu_size = len(adapter.available_experiments())
        budget_fraction = _budget_fraction(budget_k, menu_size)
    except Exception as exc:
        return RunRecord(
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=0.0,
            seed=seed,
            status="error",
            started_at=started_at,
            finished_at=now_iso(),
            error_type=type(exc).__name__,
            error_message=_truncate(str(exc), 500),
        )

    # Wrap the agent invocation in a logging handler scoped to inference,
    # so we count PC-degeneracy fallbacks per cell.
    inference_logger = logging.getLogger("evaluation.chamber_pipeline.inference")
    handler = _PcDegeneracyHandler()
    inference_logger.addHandler(handler)
    # Don't let the handler-level filter override the logger level.
    prev_level = inference_logger.level
    if prev_level > logging.WARNING:
        inference_logger.setLevel(logging.WARNING)

    kwargs = _build_agent_kwargs(spec, budget_k, seed, pc_alpha, llm)
    n_llm_calls: int | None = _maybe_count_llm_calls(llm) if spec.accepts_llm else None

    t0 = time.perf_counter()
    try:
        predicted = spec.run(adapter, **kwargs)
        wall = time.perf_counter() - t0

        # Score against ground truth.
        truth = adapter.ground_truth()
        cell_shd = float(shd(predicted, truth))
        cell_f1 = float(f1_edges(predicted, truth))

        # Edge counts (excluding diagonal). These are useful for
        # spotting degenerate "all-zeros" outputs without re-loading
        # the Parquet later.
        n_edges_pred = int(predicted.values.sum() - predicted.values.trace())
        n_edges_truth = int(truth.values.sum() - truth.values.trace())

        finished_at = now_iso()
        n_llm_after: int | None = _maybe_count_llm_calls(llm) if spec.accepts_llm else None
        n_llm_calls_for_cell: int | None = None
        if spec.accepts_llm:
            n_llm_calls_for_cell = (
                (n_llm_after - n_llm_calls)
                if (n_llm_after is not None and n_llm_calls is not None)
                else None
            )

        # PC variants populate degeneracy count; llm_only doesn't run PC.
        n_pc_degen: int | None = None if spec.name == "llm_only" else handler.count

        return RunRecord(
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            shd=cell_shd,
            f1=cell_f1,
            n_edges_predicted=n_edges_pred,
            n_edges_truth=n_edges_truth,
            wall_time_seconds=wall,
            n_llm_calls=n_llm_calls_for_cell,
            n_pc_degeneracies=n_pc_degen,
        )
    except NotImplementedError as exc:
        # Defensive: the registry already filtered this, but the agent
        # may have its own compatibility check (e.g., greedy_ig_lite's
        # menu-parse guard). Treat as a skip rather than an error so
        # the M5 figure doesn't show this as a failure.
        return RunRecord(
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="skipped",
            started_at=started_at,
            finished_at=now_iso(),
            skip_reason=_truncate(str(exc), 500),
        )
    except Exception as exc:
        return RunRecord(
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="error",
            started_at=started_at,
            finished_at=now_iso(),
            error_type=type(exc).__name__,
            error_message=_truncate(str(exc), 500),
            extra={"traceback": _truncate(traceback.format_exc(), 2000)},
        )
    finally:
        inference_logger.removeHandler(handler)
        inference_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


@dataclass
class SweepSpec:
    """Parameters defining a sweep — the §6.1 cell grid for one experiment.

    The orchestrator iterates the Cartesian product of `chambers x
    budget_fractions x agent_names x seeds`, dispatching each cell
    via `run_cell`. AGENT_REGISTRY is the source of truth for which
    agents exist; `agent_names` is a filter, defaulting to "all".

    Attributes:
        chambers: Chambers to sweep. Default both LT and WT.
        budget_fractions: k/M values to test. Plan §6.1 default
            five-level: (0.10, 0.25, 0.50, 0.75, 1.00). M4 pilot
            uses three: (0.10, 0.50, 1.00).
        agent_names: Variants to include. None = all from registry.
        seeds: Range of RNG seeds. Default 30 per plan §6.1.
        configuration: Chamber configuration. "standard" per §6.1.
        pc_alpha: PC independence-test significance level.
    """

    chambers: tuple[ChamberId, ...] = ("lt", "wt")
    budget_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    agent_names: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = tuple(range(30))
    configuration: ConfigId = "standard"
    pc_alpha: float = 0.05

    def selected_specs(self) -> list[AgentSpec]:
        """The AgentSpec list this sweep will dispatch (filtered by agent_names)."""
        if self.agent_names is None:
            return list(AGENT_REGISTRY)
        names_set = set(self.agent_names)
        return [s for s in AGENT_REGISTRY if s.name in names_set]


# Plan §3.2 menu sizes (intervention catalogs per chamber). Used to convert
# budget fraction → integer k. Centralized here so the orchestrator doesn't
# need to load the chamber dataset just to compute k.
MENU_SIZES: dict[ChamberId, int] = {"lt": 59, "wt": 28}


def _budget_k_for(chamber: ChamberId, fraction: float) -> int:
    """Convert a fractional budget to integer k for a chamber.

    Round-half-to-even for stability (Python's default int(x + 0.5) is
    actually banker's rounding via round()). Clamp to [1, menu_size].
    """
    menu = MENU_SIZES[chamber]
    k = round(fraction * menu)
    return max(1, min(menu, k))


def iter_sweep_cells(
    sweep: SweepSpec,
) -> Iterator[tuple[AgentSpec, ChamberId, int, float, int]]:
    """Iterate cells of a sweep as (spec, chamber, budget_k, fraction, seed).

    Pure: doesn't load chambers or invoke agents. Useful for sizing
    a sweep ("how many cells will I run?") and for the CLI's dry-run
    mode.
    """
    specs = sweep.selected_specs()
    for chamber in sweep.chambers:
        for fraction in sweep.budget_fractions:
            budget_k = _budget_k_for(chamber, fraction)
            for spec in specs:
                for seed in sweep.seeds:
                    yield spec, chamber, budget_k, fraction, seed


def count_cells(sweep: SweepSpec, *, exclude_skipped: bool = False) -> int:
    """How many cells `iter_sweep_cells` will yield.

    Args:
        sweep: The sweep spec.
        exclude_skipped: If True, exclude cells the registry would
            skip due to chamber-incompatibility. Useful for sizing
            "how many real runs" vs "how many cells the orchestrator
            will iterate through."
    """
    if not exclude_skipped:
        return sum(1 for _ in iter_sweep_cells(sweep))
    return sum(1 for spec, chamber, *_ in iter_sweep_cells(sweep) if spec.is_compatible(chamber))


def run_sweep(
    sweep: SweepSpec,
    llm: LLMCallable | None = None,
    on_cell: Callable[[RunRecord, int, int], None] | None = None,
) -> list[RunRecord]:
    """Run a full sweep and return all RunRecords.

    Serial nested loop. Per-cell exceptions are captured into the
    RunRecord's `error_type` / `error_message` — the sweep itself
    never raises mid-flight.

    Args:
        sweep: The cell-grid spec.
        llm: LLM callable threaded into LLM-bearing agents. Pass
            `litellm.completion` for production sweeps; FakeLLM for
            tests; None to let agents use their own default
            (which lazy-imports litellm).
        on_cell: Optional progress callback invoked after each cell
            completes, with `(record, idx, total)`. The CLI uses
            this for tqdm-style progress bars.

    Returns:
        One RunRecord per cell, in iteration order.
    """
    cells = list(iter_sweep_cells(sweep))
    total = len(cells)
    records: list[RunRecord] = []
    for idx, (spec, chamber, budget_k, _fraction, seed) in enumerate(cells):
        record = run_cell(
            spec=spec,
            chamber=chamber,
            configuration=sweep.configuration,
            budget_k=budget_k,
            seed=seed,
            pc_alpha=sweep.pc_alpha,
            llm=llm,
        )
        records.append(record)
        if on_cell is not None:
            on_cell(record, idx, total)
    return records


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    """Cap a string at n characters, appending an ellipsis suffix."""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _maybe_count_llm_calls(llm: LLMCallable | None) -> int | None:
    """Pull the call count off a FakeLLM-shaped object, or return None.

    Real `litellm.completion` is a function with no `.calls` attr, so
    this returns None for it — leaving `n_llm_calls` as None for
    production runs (M4b will plumb token-spend tracking via a
    different mechanism, likely the existing ContractedLLM monitor).
    For tests using FakeLLM, the call count is read off `llm.calls`.
    """
    calls = getattr(llm, "calls", None)
    if calls is None:
        return None
    try:
        return len(calls)
    except TypeError:
        return None


__all__ = [
    "AGENT_REGISTRY",
    "MENU_SIZES",
    "AgentSpec",
    "SweepSpec",
    "count_cells",
    "get_spec",
    "iter_sweep_cells",
    "run_cell",
    "run_sweep",
]


# Re-export iterables / suppress unused-import diagnostics for the
# variables we keep available to callers via __all__.
_ = (Iterable, field)
