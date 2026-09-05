"""Tests for the provider-drift audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.analyze_drift import (
    audit,
    block_r_cutoff,
    reasoning_frame,
    residual_trend,
    window_overlap,
    within_block_trends,
)

BASE = pd.Timestamp("2026-09-01 12:00:00")


def _cells(arm, budget_k, tokens_per_call, *, n_calls=26, start_min=0, dur_min=5, status="ok"):
    rows = []
    for i, tpc in enumerate(tokens_per_call):
        started = BASE + pd.Timedelta(minutes=start_min + i)
        rows.append(
            {
                "agent_name": arm,
                "budget_k": budget_k,
                "status": status,
                "started_at": started.isoformat(),
                "finished_at": (started + pd.Timedelta(minutes=dur_min)).isoformat(),
                "n_llm_calls": n_calls,
                "tokens_out": None if n_calls == 0 else tpc * n_calls,
                "f1": 0.3,
            }
        )
    return rows


def test_reasoning_frame_drops_cells_that_issue_no_llm_call():
    """`random` and the coverage rules carry no provider signal.

    Real LLM-free rows null BOTH `n_llm_calls` and `tokens_out`, so on archived
    data either guard alone would drop them. The `n_llm_calls > 0` guard earns
    its place on the case reality does not currently produce: a zero call count
    beside a non-null token count, which divides to infinity and would poison
    every statistic downstream of it rather than failing loudly.
    """
    rows = _cells("llm_pc", 30, [5000.0] * 5) + _cells("random", 30, [0.0] * 5, n_calls=0)
    got = reasoning_frame(pd.DataFrame(rows))
    assert set(got["agent_name"]) == {"llm_pc"}
    assert len(got) == 5

    poisoned = pd.DataFrame(rows)
    poisoned.loc[poisoned["agent_name"] == "random", "tokens_out"] = 5.0
    survivors = reasoning_frame(poisoned)
    assert set(survivors["agent_name"]) == {"llm_pc"}
    assert np.isfinite(survivors["tokens_per_call"]).all()


def test_reasoning_frame_drops_errored_cells():
    rows = _cells("llm_pc", 30, [5000.0] * 4) + _cells("llm_pc", 30, [9e9], status="error")
    assert len(reasoning_frame(pd.DataFrame(rows))) == 4


def test_reasoning_frame_orders_by_launch_not_completion():
    """Ranking by completion time manufactures a trend under parallel workers.

    A slow cell finishes later BY DEFINITION, so completion order correlates
    with duration and therefore with anything duration depends on. Launch
    order is set by the scheduler and carries no such artefact. Here the cells
    launch in order but finish in reverse.
    """
    rows = []
    for i, tpc in enumerate([1000.0, 2000.0, 3000.0, 4000.0]):
        started = BASE + pd.Timedelta(minutes=i)
        finished = BASE + pd.Timedelta(minutes=100 - i)  # reverse completion
        rows.append(
            {
                "agent_name": "llm_pc",
                "budget_k": 30,
                "status": "ok",
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "n_llm_calls": 26,
                "tokens_out": tpc * 26,
                "f1": 0.3,
            }
        )
    got = reasoning_frame(pd.DataFrame(rows))
    assert list(got["tokens_per_call"]) == [1000.0, 2000.0, 3000.0, 4000.0]


def test_reasoning_frame_requires_the_columns_it_needs():
    rows = _cells("llm_pc", 30, [5000.0] * 3)
    frame = pd.DataFrame(rows).drop(columns=["n_llm_calls"])
    with pytest.raises(ValueError, match="unchecked"):
        reasoning_frame(frame)


def test_block_r_cutoff_loosens_for_small_blocks():
    """A fixed cutoff over-flags: r's null sd grows as blocks shrink."""
    assert block_r_cutoff(50) == pytest.approx(3.0 / np.sqrt(47))
    assert block_r_cutoff(30) > block_r_cutoff(50)
    assert np.isnan(block_r_cutoff(4))


def test_within_block_trends_finds_a_planted_trend_and_ignores_a_flat_block():
    rising = list(np.linspace(3000, 9000, 40))
    flat = [5000.0 + (-1) ** i * 20 for i in range(40)]
    frame = pd.DataFrame(_cells("a", 30, rising) + _cells("b", 30, flat, start_min=100))
    blocks = within_block_trends(reasoning_frame(frame)).set_index("agent_name")
    assert blocks.loc["a", "r_within"] > 0.99
    assert abs(blocks.loc["b", "r_within"]) < 0.3


def test_residual_trend_is_zero_when_only_arm_composition_differs():
    """The defect this guards: a cheap arm running first looks like drift.

    Each arm is internally flat; they differ only in level, and the cheap one
    runs first. A raw correlation reports a strong upward trend. Residualising
    on arm x budget must return it to zero.
    """
    # Jitter, so the residuals are not identically zero -- a perfectly flat
    # block leaves nothing to correlate and `residual_trend` rightly says nan.
    rng = np.random.default_rng(7)
    cheap = _cells("cheap", 30, list(2000.0 + rng.normal(0, 50, 40)))
    dear = _cells("dear", 30, list(8000.0 + rng.normal(0, 50, 40)), start_min=100)
    cells = reasoning_frame(pd.DataFrame(cheap + dear))
    raw = np.corrcoef(range(len(cells)), cells["tokens_per_call"])[0, 1]
    assert raw > 0.85, "the fixture must actually look like drift before residualising"
    assert abs(residual_trend(cells)) < 0.25


def test_window_overlap_separates_concurrent_from_sequential_arms():
    concurrent = pd.DataFrame(
        _cells("a", 30, [5000.0] * 10, start_min=0, dur_min=30)
        + _cells("b", 30, [5000.0] * 10, start_min=0, dur_min=30)
    )
    sequential = pd.DataFrame(
        _cells("a", 30, [5000.0] * 10, start_min=0, dur_min=1)
        + _cells("b", 30, [5000.0] * 10, start_min=500, dur_min=1)
    )
    assert window_overlap(within_block_trends(reasoning_frame(concurrent))) > 0.8
    assert window_overlap(within_block_trends(reasoning_frame(sequential))) < 0.2


def test_audit_flags_a_real_trend_and_clears_noise():
    rising = list(np.linspace(3000, 9000, 50))
    noisy = list(5000 + np.random.default_rng(0).normal(0, 300, 50))
    flagged = audit(pd.DataFrame(_cells("a", 30, rising)), label="planted")
    clean = audit(pd.DataFrame(_cells("a", 30, noisy)), label="noise")
    assert flagged["verdict"].startswith("FLAG")
    assert flagged["blocks_trending"] == 1
    assert clean["verdict"].startswith("CLEAN")
    assert clean["blocks_trending"] == 0


def test_audit_reports_no_llm_cells_rather_than_claiming_clean():
    """A file of LLM-free arms is UNCHECKED, not verified drift-free."""
    result = audit(pd.DataFrame(_cells("random", 30, [0.0] * 10, n_calls=0)), label="llmfree")
    assert result["verdict"] == "N/A (no LLM cells)"
    assert result["n_ok"] == 0
