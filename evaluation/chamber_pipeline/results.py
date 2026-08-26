"""Run-result records for the chamber pillar's sweep experiments.

Each cell of the §6.1 cell grid (chamber x budget x variant x seed)
produces exactly one `RunRecord` describing what happened. Records are
collected by the orchestrator, serialized to Parquet/CSV at sweep
end, and consumed by the analysis layer (M5+) to produce the §5.3
Pareto figure.

Design choices:

- **Frozen dataclass.** Sweep results are append-only by construction —
  what gets written to disk is what was computed. Mutability would
  invite "I tweaked this number after the run" bugs that are hard to
  catch in a 1620-row Parquet.
- **Explicit success vs skip vs error.** Three terminal states a cell
  can land in: `ok` (agent ran and produced an adjacency), `skipped`
  (agent isn't compatible with this chamber — e.g., GreedyIG-lite on
  WT per plan §5.1), `error` (agent raised an unexpected exception).
  The orchestrator never crashes on per-cell failures; one bad cell
  doesn't lose the surrounding 449.
- **Metadata flat, not nested.** Pandas / Parquet read flat schemas
  cleanly; nested dicts force JSON-blob columns that are awkward to
  query. Keep all fields scalar.
- **Optional LLM/PC fields default to None, not 0.** "0 LLM calls"
  is a meaningful claim about a non-LLM variant; "no measurement"
  is a different thing for a variant that doesn't track that axis.
  None → nullable column in Parquet; 0 → counted-as-zero.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

# The terminal status of a single cell. The orchestrator guarantees
# exactly one of these is set on every RunRecord it produces.
RunStatus = Literal["ok", "skipped", "error"]


@dataclass(frozen=True)
class RunRecord:
    """One cell of a chamber sweep — a single (chamber, budget, agent, seed) run.

    Required fields are immutable identity (which cell this is) plus
    status. Optional fields carry per-cell measurements that may be
    None when not applicable (e.g., scoring fields are None for
    skipped cells; LLM-call fields are None for non-LLM variants).

    Attributes:
        chamber: Chamber identifier ("lt" or "wt").
        configuration: Chamber configuration ("standard" or
            "pressure-control"). Plan §6.1 holds this constant per
            chamber, but the field is kept for forward-compat with
            future ablation sweeps.
        agent_name: Variant name as registered in `AGENT_REGISTRY`
            (e.g., "random", "greedy_ig_lite", "llm_pc").
        budget_k: Intervention-budget cell value (`per_tool_limits["intervene"]`).
        budget_fraction: `budget_k / menu_size` — the §6.1 x-axis value.
            Carried for analysis convenience so plotters don't need
            to recompute menu sizes per chamber.
        seed: RNG seed for this cell.
        status: One of "ok" / "skipped" / "error".
        shd: Structural Hamming Distance vs ground truth. None for
            non-"ok" cells.
        f1: F1 score on edge presence. None for non-"ok" cells.
        n_edges_predicted: Count of 1-entries in the predicted
            adjacency (excluding diagonal). None for non-"ok" cells.
        n_edges_truth: Count of 1-entries in the ground-truth
            adjacency. Constant per chamber but recorded per row for
            flat-schema convenience.
        wall_time_seconds: How long the agent took (excluding adapter
            load + scoring). None for skipped/error cells.
        n_llm_calls: How many times the agent invoked the LLM
            callable. None for non-LLM variants. 0 is a real
            measurement (e.g., budget=0 short-circuit).
        n_selection_fallbacks: How many selection calls returned an
            unparseable response and fell back to a random unspent
            experiment. A nonzero value means the cell's "LLM selection"
            was partly random; equal to `n_llm_calls` means it was entirely
            random. None for non-LLM variants and for cells recorded before
            this field existed (the M4b pilot).
        overlap_frac: Fraction of the smaller scout selection also chosen by
            the other scout. None when either scout selected nothing --
            undefined, not zero, since zero is the perfectly-disjoint success
            case. Only set for the fan-in and team rungs.
        n_experiments_distinct: Distinct experiments the cell queried in total.
        n_contested: For the team rung, how many experiment claims the
            negotiation failed to resolve. Without it the rung's defining
            mechanism is unmeasurable -- a team whose scouts never agree on a
            split looks identical to one whose negotiation worked. None for
            every other arm.
        n_negotiation_failures: For the team rung, how many of the four
            negotiation rounds (two proposals, two revisions) returned
            nothing parseable. Non-zero does NOT imply the split degenerated:
            a single unparseable revision still leaves that scout its own
            proposal. Only a scout losing BOTH its rounds drops it to the
            seeded fallback partition -- which still varies across seeds, so
            even total failure does not produce identical queries. A single
            integer cannot say WHICH rounds failed; 2 covers both one dead
            scout and the more damaging both-revisions-lost case.
        conservation_certified: Whether `DelegationGraph.verify()` passed at
            the end of the cell (hypothesis H-C).
        scout_a_tokens: Tokens scout_a spent, from its node monitor. Recorded
            from 2026-08-24: the k=45 gate found the aggregator overrunning
            its grant on 6 of 9 graph cells, and could NOT say whether the
            scouts overran too, because only aggregator spend was stored. An
            unrecorded node reads as a clean node.
        scout_b_tokens: Same for scout_b.
        aggregator_tokens: Tokens the aggregator spent on its single
            indivisible reconciliation call.
        max_tree_fragment: The largest single parent grant into the
            aggregator -- the biggest indivisible call any tree encoding of
            the same grants could fund.
        tree_would_refuse: Whether `aggregator_tokens` landed in whitepaper
            §4.6 P2's incompleteness window, `max_i a_i < c <= sum_i a_i`,
            where the DAG law admits the call and no tree encoding does. None
            when the comparison carries no evidence: fewer than two parents,
            or a call neither encoding could fund.
        model_id: The model identifier requested for this cell.
        reasoning_effort: The reasoning effort requested, or "unset" when the
            call relied on the provider default. Recorded because a pinned
            model snapshot does not pin behaviour: DeepSeek raised the default
            effort under unchanged 0423 weights on 2026-08-13.
        providers_used: Comma-separated upstream providers that served this
            cell, sorted. More than one means provider rotation fired.
        n_collinear_dropped: How many columns PC dropped as numerically
            duplicate of an earlier column (see
            `inference.DEFAULT_COLLINEARITY_THRESHOLD`). A LOCAL loss --
            the dropped node makes no claim while the rest of the graph is
            still inferred -- unlike `n_pc_degeneracies`, which is total.
            None for variants that never run PC.
        n_pc_degeneracies: How many times PC's singular-matrix
            fallback fired during this cell. Captured by a logging
            handler the orchestrator installs around each cell. None
            if the cell didn't run PC at all (LLM-only variant).
        tokens_in: Cumulative input/prompt tokens spent on this cell's
            LLM calls. None for non-LLM variants and for cells whose
            LLM target doesn't report a usage block (e.g., FakeLLM).
        tokens_out: Same but for output/completion tokens.
        cost_usd: Cumulative cost in USD for this cell's LLM calls,
            read from `_hidden_params.response_cost` when LiteLLM
            populates it. None when the LLM target doesn't report.
        error_type: Exception class name for "error" cells; None
            otherwise.
        error_message: First 500 chars of the exception message for
            "error" cells; None otherwise. (Truncated to keep
            Parquet rows compact.)
        skip_reason: Free-text reason the cell was skipped. None
            unless status == "skipped". Typical: "agent
            incompatible with chamber" with the underlying
            NotImplementedError message.
        started_at: ISO 8601 timestamp at agent-call start.
        finished_at: ISO 8601 timestamp at agent-call end (whether
            ok / error). Useful for sweep-runtime accounting.
    """

    # --- identity (always set) ---
    chamber: str
    configuration: str
    agent_name: str
    budget_k: int
    budget_fraction: float
    seed: int

    # --- terminal status ---
    status: RunStatus

    # --- timestamps (always set; finished_at == started_at for skipped) ---
    started_at: str
    finished_at: str

    # --- scoring (None unless status == "ok") ---
    shd: float | None = None
    f1: float | None = None
    n_edges_predicted: int | None = None
    n_edges_truth: int | None = None

    # --- runtime / instrumentation ---
    wall_time_seconds: float | None = None
    n_llm_calls: int | None = None
    n_selection_fallbacks: int | None = None
    overlap_frac: float | None = None
    n_experiments_distinct: int | None = None
    n_contested: int | None = None
    n_negotiation_failures: int | None = None
    conservation_certified: bool | None = None
    aggregator_tokens: int | None = None
    scout_a_tokens: int | None = None
    scout_b_tokens: int | None = None
    max_tree_fragment: int | None = None
    tree_would_refuse: bool | None = None
    model_id: str | None = None
    reasoning_effort: str | None = None
    providers_used: str | None = None
    n_pc_degeneracies: int | None = None
    n_collinear_dropped: int | None = None

    # --- PC provenance: the three parameters that silently determine the
    #     graph. Recorded per cell because `runs/m6-controls.parquet`
    #     disagreed with a later `random` run on identical seed, code and
    #     dependency versions, and the disagreement was undiagnosable --
    #     none of these travelled with the row. A cell that cannot state
    #     the configuration that produced it cannot be pooled with another
    #     cell, which is the only thing a sweep is for. ---
    pc_alpha: float | None = None
    pc_max_rows: int | None = None
    pc_collinearity_threshold: float | None = None

    # --- LLM cost / token tracking ---
    # Populated by the orchestrator's _CountingLLM wrapper for cells whose
    # agent accepts an LLM. Stay None for non-LLM variants and for cells
    # where the LLM target didn't report a usage block (e.g., FakeLLM in
    # tests). Real production runs against litellm's OpenAI-shape
    # responses populate all three. Carrying them in the schema from M4
    # forward means M4b and M5 Parquets are mergeable without a schema
    # migration.
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None

    # --- failure / skip context ---
    error_type: str | None = None
    error_message: str | None = None
    skip_reason: str | None = None

    # --- arbitrary per-variant metadata (kept as a JSON-stringifiable dict
    #     because Parquet's column type for arbitrary mixed maps is fragile;
    #     stringifying preserves data without forcing a schema) ---
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict representation, suitable for `pd.DataFrame.from_records`.

        The `extra` dict is JSON-stringified into the `extra_json` column
        to avoid Parquet's mixed-type-column headaches. Empty extras
        produce `extra_json=None` rather than `"{}"` to keep null
        semantics clear. `default=str` is used as a defensive fallback
        for non-JSON-serializable values (e.g., a numpy array
        accidentally stuffed into `extra`) so write_records_parquet
        doesn't raise mid-sweep on a single rogue value.
        """
        d = asdict(self)
        extra = d.pop("extra")
        d["extra_json"] = json.dumps(extra, default=str) if extra else None
        return d


def write_records_parquet(records: list[RunRecord], path: str | os.PathLike[str]) -> None:
    """Write a list of RunRecord to a Parquet file.

    Uses pandas' Parquet writer (which depends on pyarrow). Creates
    parent directories as needed. Empty record lists produce an empty
    Parquet file with the schema preserved (so downstream `pd.read_parquet`
    doesn't need to special-case "no runs yet").

    Args:
        records: All RunRecords from a sweep, in any order.
        path: Output Parquet path. Parent directory is created.
    """
    import pandas as pd

    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    if not records:
        # Build an empty DataFrame with the schema by serializing one
        # placeholder record then dropping it — preserves columns so
        # readers don't need a special case.
        placeholder = RunRecord(
            chamber="",
            configuration="",
            agent_name="",
            budget_k=0,
            budget_fraction=0.0,
            seed=0,
            status="ok",
            started_at="",
            finished_at="",
        )
        df = pd.DataFrame.from_records([placeholder.to_dict()]).iloc[0:0]
    else:
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
    df.to_parquet(path, index=False)


def write_records_csv(records: list[RunRecord], path: str | os.PathLike[str]) -> None:
    """Write a list of RunRecord to a CSV file.

    Convenience for human inspection; Parquet remains the
    machine-readable format. Same empty-list semantics as
    `write_records_parquet`.
    """
    import pandas as pd

    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    if not records:
        df = pd.DataFrame(columns=list(_record_field_names_with_extra()))
    else:
        df = pd.DataFrame.from_records([r.to_dict() for r in records])
    df.to_csv(path, index=False)


def _record_field_names_with_extra() -> tuple[str, ...]:
    """Field names produced by RunRecord.to_dict() (incl. extra_json)."""
    placeholder = RunRecord(
        chamber="",
        configuration="",
        agent_name="",
        budget_k=0,
        budget_fraction=0.0,
        seed=0,
        status="ok",
        started_at="",
        finished_at="",
    )
    return tuple(placeholder.to_dict().keys())


def now_iso() -> str:
    """Current time as ISO 8601 string. Centralized so tests can monkey-patch."""
    return datetime.now().isoformat(timespec="seconds")


__all__ = [
    "RunRecord",
    "RunStatus",
    "now_iso",
    "write_records_csv",
    "write_records_parquet",
]
