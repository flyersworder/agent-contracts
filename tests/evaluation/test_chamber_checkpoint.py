"""Tests for the chamber pipeline's per-cell JSONL checkpoint sidecar.

The checkpoint module guards against the M4b pilot stall mode where
hours of completed work were lost because Parquet only flushes at sweep
end. Each cell is appended to a JSONL sidecar atomically; on restart,
the sweep skips cells already in the sidecar.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from evaluation.chamber_pipeline.checkpoint import (
    append_record_jsonl,
    done_cell_keys,
    filter_done_cells,
    read_records_jsonl,
)
from evaluation.chamber_pipeline.results import RunRecord


def _make_record(**overrides: Any) -> RunRecord:
    """Build a RunRecord with sensible defaults; override per test."""
    defaults: dict[str, Any] = {
        "chamber": "lt",
        "configuration": "standard",
        "agent_name": "random",
        "budget_k": 5,
        "budget_fraction": 0.085,
        "seed": 0,
        "status": "ok",
        "started_at": "2026-05-17T09:00:00",
        "finished_at": "2026-05-17T09:00:01",
        "shd": 12.0,
        "f1": 0.5,
    }
    return RunRecord(**{**defaults, **overrides})


class TestAppendReadRoundTrip:
    """Per-cell append → read returns RunRecord with all fields preserved."""

    def test_single_record_roundtrip(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        rec = _make_record()
        append_record_jsonl(rec, sidecar)
        read_back = read_records_jsonl(sidecar)
        assert read_back == [rec]

    def test_multiple_records_preserve_order(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        recs = [_make_record(seed=i, started_at=f"2026-05-17T09:00:0{i}") for i in range(3)]
        for r in recs:
            append_record_jsonl(r, sidecar)
        read_back = read_records_jsonl(sidecar)
        assert read_back == recs

    def test_extra_dict_roundtrips(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        rec = _make_record(extra={"provider": "novita", "retries": 2})
        append_record_jsonl(rec, sidecar)
        read_back = read_records_jsonl(sidecar)
        assert read_back == [rec]
        assert read_back[0].extra == {"provider": "novita", "retries": 2}

    def test_empty_extra_roundtrips_as_empty_dict(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        rec = _make_record()
        append_record_jsonl(rec, sidecar)
        assert read_records_jsonl(sidecar)[0].extra == {}

    def test_error_record_with_none_fields_roundtrips(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        rec = _make_record(
            status="error",
            shd=None,
            f1=None,
            error_type="TimeoutError",
            error_message="cell exceeded 1800.0s wall-clock timeout",
        )
        append_record_jsonl(rec, sidecar)
        assert read_records_jsonl(sidecar) == [rec]

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "does-not-exist.jsonl"
        assert read_records_jsonl(sidecar) == []

    def test_read_tolerates_partial_trailing_line(self, tmp_path: Path) -> None:
        """A kill mid-write leaves a partial JSON line; read skips it without raising."""
        sidecar = tmp_path / "partial.jsonl"
        rec = _make_record()
        append_record_jsonl(rec, sidecar)
        # Simulate a crash: append a partial JSON object (no newline, malformed)
        with sidecar.open("a") as f:
            f.write('{"chamber": "lt", "agent_nam')  # truncated
        read_back = read_records_jsonl(sidecar)
        assert read_back == [rec]

    def test_append_creates_parent_dirs(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "deep" / "nested" / "smoke.jsonl"
        rec = _make_record()
        append_record_jsonl(rec, sidecar)
        assert sidecar.exists()
        assert read_records_jsonl(sidecar) == [rec]

    def test_one_record_per_line(self, tmp_path: Path) -> None:
        """JSONL invariant: each record is exactly one line ending in \\n."""
        sidecar = tmp_path / "smoke.jsonl"
        recs = [_make_record(seed=i) for i in range(3)]
        for r in recs:
            append_record_jsonl(r, sidecar)
        text = sidecar.read_text()
        lines = text.splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line parses as standalone JSON


class TestCellKeyExtraction:
    """done_cell_keys produces the 5-tuple set used for resume filtering."""

    def test_extracts_identity_tuple(self) -> None:
        rec = _make_record(
            chamber="lt",
            configuration="standard",
            agent_name="random",
            budget_k=5,
            seed=7,
        )
        keys = done_cell_keys([rec])
        assert keys == {("lt", "standard", "random", 5, 7)}

    def test_multiple_records_yield_set(self) -> None:
        recs = [
            _make_record(seed=0),
            _make_record(seed=1),
            _make_record(seed=2),
        ]
        keys = done_cell_keys(recs)
        assert len(keys) == 3
        assert ("lt", "standard", "random", 5, 0) in keys
        assert ("lt", "standard", "random", 5, 1) in keys
        assert ("lt", "standard", "random", 5, 2) in keys

    def test_empty_records_yield_empty_set(self) -> None:
        assert done_cell_keys([]) == set()


class TestFilterDoneCells:
    """filter_done_cells skips already-completed cells from a sweep iterator."""

    def _cell(self, agent_name: str, chamber: str, k: int, seed: int) -> tuple:
        """Mimic the iter_sweep_cells tuple shape: (spec, chamber, k, fraction, seed).

        `chamber` is a string ChamberId in the real orchestrator; `spec` is an
        AgentSpec with a `.name` attribute.
        """

        class _StubSpec:
            def __init__(self, name: str) -> None:
                self.name = name

        return (_StubSpec(agent_name), chamber, k, k / 60.0, seed)

    def test_skips_done_cells(self) -> None:
        all_cells = [
            self._cell("random", "lt", 5, 0),
            self._cell("random", "lt", 5, 1),
            self._cell("random", "lt", 5, 2),
        ]
        done = {("lt", "standard", "random", 5, 1)}
        remaining = list(filter_done_cells(all_cells, done, configuration="standard"))
        assert len(remaining) == 2
        seeds = [cell[4] for cell in remaining]
        assert seeds == [0, 2]

    def test_empty_done_returns_all(self) -> None:
        cells = [self._cell("random", "lt", 5, i) for i in range(3)]
        remaining = list(filter_done_cells(cells, set(), configuration="standard"))
        assert remaining == cells

    def test_all_done_returns_empty(self) -> None:
        cells = [self._cell("random", "lt", 5, i) for i in range(3)]
        done = {("lt", "standard", "random", 5, i) for i in range(3)}
        assert list(filter_done_cells(cells, done, configuration="standard")) == []

    def test_different_configuration_does_not_match(self) -> None:
        """A done-record from a different configuration must not skip cells of the current sweep."""
        cells = [self._cell("random", "lt", 5, 0)]
        done = {("lt", "pressure-control", "random", 5, 0)}  # different config
        remaining = list(filter_done_cells(cells, done, configuration="standard"))
        assert len(remaining) == 1


class TestAppendIsAppendNotOverwrite:
    """Repeated appends accumulate; they don't truncate the file."""

    def test_second_append_preserves_first(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "smoke.jsonl"
        first = _make_record(seed=0)
        second = _make_record(seed=1)
        append_record_jsonl(first, sidecar)
        append_record_jsonl(second, sidecar)
        assert read_records_jsonl(sidecar) == [first, second]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


class TestErroredCellsAreRetriedOnResume:
    """An error is not a result.

    The M6 sweep lost 296 cells to an OpenRouter weekly-limit 403. Every one
    was written to the sidecar with status="error", and `done_cell_keys`
    counted them as done -- so resuming would have skipped them and frozen a
    154-cell dataset as if it were complete. CLAUDE.md claimed a re-run
    "only re-attempts the errors"; it never did.
    """

    def _rec(self, agent: str, seed: int, status: str) -> RunRecord:
        return RunRecord(
            chamber="lt",
            configuration="standard",
            agent_name=agent,
            budget_k=30,
            budget_fraction=0.5,
            seed=seed,
            status=status,  # type: ignore[arg-type]
            started_at="2026-08-24T00:00:00Z",
            finished_at="2026-08-24T00:00:01Z",
        )

    def test_an_errored_cell_is_not_done(self) -> None:
        from evaluation.chamber_pipeline.checkpoint import done_cell_keys

        keys = done_cell_keys([self._rec("llm_pc", 0, "error")])
        assert keys == set(), "an errored cell must be retried, not skipped"

    def test_ok_and_skipped_cells_are_done(self) -> None:
        from evaluation.chamber_pipeline.checkpoint import done_cell_keys

        keys = done_cell_keys([self._rec("llm_pc", 0, "ok"), self._rec("llm_pc", 1, "skipped")])
        assert len(keys) == 2

    def test_a_mixed_sidecar_retries_only_the_failures(self) -> None:
        from evaluation.chamber_pipeline.checkpoint import done_cell_keys

        recs = [self._rec("llm_pc", i, "ok") for i in range(3)]
        recs += [self._rec("llm_pc", i, "error") for i in range(3, 8)]
        keys = done_cell_keys(recs)
        assert len(keys) == 3
        assert all(k[4] < 3 for k in keys)

    def test_a_retried_cell_does_not_duplicate_in_the_consolidation(self) -> None:
        """The retry writes a SECOND line for the same key.

        Consolidation reads the whole sidecar, so without dedup the Parquet
        would carry both the stale error and the good result for one cell --
        inflating n and poisoning every per-(arm, budget) mean.
        """
        from evaluation.chamber_pipeline.checkpoint import latest_per_cell

        stale = self._rec("llm_pc", 0, "error")
        good = self._rec("llm_pc", 0, "ok")
        out = latest_per_cell([stale, good])
        assert len(out) == 1
        assert out[0].status == "ok", "the later record must win"
