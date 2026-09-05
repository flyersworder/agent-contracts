"""Three defects found reviewing 702444a, each reproduced before being fixed.

None was reachable through the current prompt builders, and the shipped WT
constant (6102) is unaffected -- verified on `runs/calib-wt-negotiate.parquet`:
zero cells with zero negotiate tokens, node sums equal to cell totals in all
27. These guard the paths a future calibration would take.
"""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.chamber_pipeline import orchestrator as orch
from evaluation.chamber_pipeline.llm_planner import build_negotiate_propose_prompt, call_kind
from evaluation.chamber_pipeline.orchestrator import _CountingLLM

MENU = ["uniform_a", "uniform_b"]


# --- 1. the "derived" calibrated set was a one-time snapshot ----------------


def test_calibrated_set_tracks_the_dict_at_runtime(monkeypatch):
    """Deleting a measurement must make that chamber provisional immediately.

    `_NEGOTIATE_CALIBRATED_CHAMBERS = frozenset(_C95_NEGOTIATE_BY_CHAMBER)`
    binds ONCE at import. `_ladder_calibration` then read the live dict while
    `is_provisional_calibration` read the frozen snapshot, so after any
    runtime change the two disagreed: the first raised "not calibrated" while
    the second reported the chamber calibrated. That is precisely the
    two-sources-of-truth hazard the derivation was introduced to remove, and
    the review found the review's own test creating it.
    """
    monkeypatch.delitem(orch._C95_NEGOTIATE_BY_CHAMBER, "wt")
    assert orch.is_provisional_calibration("wt", 14, negotiates=True) is True


def test_adding_a_measurement_makes_the_chamber_non_provisional(monkeypatch):
    """The other direction, so the test cannot pass by always returning True."""
    monkeypatch.setitem(orch._C95_NEGOTIATE_BY_CHAMBER, "zz", 1234)
    assert orch.is_provisional_calibration("zz", 14, negotiates=True) is False


# --- 2. call_kind raised on the production hot path ------------------------


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": None}],
        [{"role": "user"}],
        [{"role": "user", "content": ["a", "b"]}],
        [],
    ],
    ids=["none-content", "missing-content", "list-content", "empty"],
)
def test_call_kind_degrades_to_unknown_instead_of_raising(messages):
    """A malformed prompt must not turn a PAID call into a cell error.

    `_attribute` runs after `_accumulate_usage`, so a raise here discards a
    response the sweep has already been billed for and fails a cell that
    otherwise succeeded. "unknown" already exists as the auditable bucket for
    prompts no marker matches; routing malformed content there is strictly
    better than raising, and keeps the failure visible in `extra_json`
    rather than as a traceback.
    """
    assert call_kind(messages) == "unknown"


def test_attribute_survives_a_malformed_prompt_and_still_bills_the_tokens():
    def target(**_: Any) -> dict:
        return {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    meter = _CountingLLM(target=target)
    meter(model="deepseek-v4-flash-0731", messages=[{"role": "user", "content": None}])
    assert meter.tokens_by_kind["unknown"] == 15
    assert meter.total_input_tokens + meter.total_output_tokens == 15


# --- 3. a usage-less response silently contributes a 0-token call ----------


def test_calls_missing_usage_are_counted_separately():
    """A provider that omits `usage` would drag a calibration median DOWN.

    `_accumulate_usage` is best-effort by design and leaves totals at 0 for a
    response whose usage it cannot parse. That call still increments the kind's
    call count, so `negotiate_tokens / n_negotiate_calls` reads lower than the
    truth and the constant under-provisions -- and under-provisioning surfaces
    as conservation failures that read as MECHANISM failures. Counting them
    makes a contaminated calibration visible instead of plausible.
    """

    def no_usage(**_: Any) -> dict:
        return {"choices": [{"message": {"content": "uniform_a"}}]}

    meter = _CountingLLM(target=no_usage)
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_negotiate_propose_prompt(MENU, 2, "A"),
    )
    assert meter.n_unmetered_calls == 1
    assert meter.n_negotiate_calls == 1
    assert meter.negotiate_tokens == 0


def test_metered_calls_are_not_counted_as_unmetered():
    def target(**_: Any) -> dict:
        return {
            "choices": [{"message": {"content": "uniform_a"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    meter = _CountingLLM(target=target)
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_negotiate_propose_prompt(MENU, 2, "A"),
    )
    assert meter.n_unmetered_calls == 0


def test_per_kind_attribution_reaches_the_record_for_audit():
    """The docstring calls a non-empty "unknown" bucket a defect.

    It was not checkable: `tokens_by_kind` lived only on the meter and never
    reached a record, so nobody could confirm after a sweep that every call
    was attributed. It rides in `extra` -> `extra_json`, which costs no new
    column and is null for arms that make no LLM call.
    """
    from evaluation.chamber_pipeline.results import RunRecord

    rec = RunRecord(
        chamber="wt",
        configuration="standard",
        agent_name="team",
        budget_k=14,
        budget_fraction=0.5,
        seed=0,
        status="ok",
        started_at="2026-09-05T00:00:00Z",
        finished_at="2026-09-05T00:01:00Z",
        extra={"tokens_by_kind": {"select": 100, "negotiate_propose": 50}},
    )
    assert "tokens_by_kind" in rec.to_dict()["extra_json"]
