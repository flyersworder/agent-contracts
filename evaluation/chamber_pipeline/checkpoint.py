"""Per-cell JSONL checkpoint sidecar for chamber sweeps.

Guards against the M4b pilot stall mode (2026-05-17) where hours of
completed work were lost because Parquet only flushes at sweep end.
Each cell appends one JSON line to a sidecar atomically; on restart,
the sweep skips cells already in the sidecar and consolidates the
sidecar into the requested Parquet at sweep completion.

Design:

- **JSONL.** One JSON object per line, newline-terminated.
  Append-atomic at line granularity on POSIX (one `write()` syscall
  per cell). A kill mid-write loses at most the current line. JSON
  was chosen over binary serialization formats so the sidecar is
  human-inspectable and safe to read from untrusted sources.
- **Read tolerates partial trailing lines.** A crash mid-write leaves
  a malformed final line; `read_records_jsonl` skips it without
  raising so resume can proceed.
- **Schema parity with Parquet.** Uses `RunRecord.to_dict()` directly,
  so the consolidated Parquet at sweep end is byte-identical to
  today's `write_records_parquet` output. No migration needed.
- **Resume key is the 5-tuple** `(chamber, configuration, agent_name,
  budget_k, seed)` — unique per cell by `iter_sweep_cells` construction.
  Set-difference filtering is O(1) per cell lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .results import RunRecord

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

# (chamber, configuration, agent_name, budget_k, seed) — the cell-identity
# tuple. Configuration is per-sweep-invariant but included for safety so
# accidental cross-config sidecar sharing doesn't silently skip work.
CellKey = tuple[str, str, str, int, int]


def append_record_jsonl(record: RunRecord, path: str | os.PathLike[str]) -> None:
    """Append one RunRecord to the JSONL sidecar.

    Atomic at line granularity: a single `open("a")` + `write()` per
    cell. POSIX guarantees small writes (well under PIPE_BUF) are
    atomic, so a kill mid-write loses at most the current line. Parent
    directories are created if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict()) + "\n"
    with p.open("a") as f:
        f.write(line)


def read_records_jsonl(path: str | os.PathLike[str]) -> list[RunRecord]:
    """Read all complete RunRecord lines from the sidecar.

    Returns an empty list if the file doesn't exist. Lines that fail
    to parse as JSON (e.g., a truncated final line from a crash) are
    skipped silently — only complete JSON objects become records.
    """
    p = Path(path)
    if not p.exists():
        return []
    records: list[RunRecord] = []
    with p.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(_record_from_dict(d))
    return records


def _record_from_dict(d: dict[str, Any]) -> RunRecord:
    """Inverse of `RunRecord.to_dict()` — re-inflates `extra_json` to `extra`."""
    d = dict(d)
    extra_json = d.pop("extra_json", None)
    d["extra"] = json.loads(extra_json) if extra_json else {}
    return RunRecord(**d)


def done_cell_keys(records: Iterable[RunRecord]) -> set[CellKey]:
    """Build the resume keyset from records — 5-tuples per cell."""
    return {(r.chamber, r.configuration, r.agent_name, r.budget_k, r.seed) for r in records}


def filter_done_cells(
    cells: Iterable[tuple[Any, str, int, float, int]],
    done: set[CellKey],
    configuration: str,
) -> Iterator[tuple[Any, str, int, float, int]]:
    """Yield only cells whose 5-tuple key is NOT in `done`.

    `cells` matches `iter_sweep_cells`'s shape: `(spec, chamber,
    budget_k, fraction, seed)`. `configuration` is the sweep-level
    config string; the cell tuple doesn't carry it because it's
    sweep-invariant.

    If `done` is empty, yields every cell unchanged.
    """
    if not done:
        yield from cells
        return
    for cell in cells:
        spec, chamber, budget_k, _fraction, seed = cell
        key: CellKey = (chamber, configuration, spec.name, budget_k, seed)
        if key not in done:
            yield cell


__all__ = [
    "CellKey",
    "append_record_jsonl",
    "done_cell_keys",
    "filter_done_cells",
    "read_records_jsonl",
]
