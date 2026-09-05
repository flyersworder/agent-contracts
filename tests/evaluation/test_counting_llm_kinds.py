"""Per-kind token attribution on the cell's LLM meter.

A cell records node totals (`scout_a_tokens`, `aggregator_tokens`), but a
scout's total is selection PLUS negotiation. `_C95_NEGOTIATE` provisions only
the second, so without splitting the total by call kind the constant cannot be
measured after the fact -- which is why it was never isolated on WT and why
all 300 WT `team` cells carry `conservation_certified = None`.
"""

from __future__ import annotations

from typing import Any

from evaluation.chamber_pipeline.llm_planner import (
    build_negotiate_propose_prompt,
    build_negotiate_revise_prompt,
    build_reconcile_prompt,
    build_select_prompt,
)
from evaluation.chamber_pipeline.orchestrator import _CountingLLM

MENU = ["uniform_a", "uniform_b", "uniform_c"]


def _responder(usage: dict[str, int]) -> Any:
    def target(**_: Any) -> dict:
        return {
            "choices": [{"message": {"content": "uniform_a"}}],
            "usage": dict(usage),
        }

    return target


def _meter(prompt_tokens: int = 10, completion_tokens: int = 5) -> _CountingLLM:
    return _CountingLLM(
        target=_responder({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
    )


def test_tokens_are_attributed_to_the_calling_prompts_kind():
    meter = _meter()
    meter(model="deepseek-v4-flash-0731", messages=build_select_prompt(MENU, 3, None))
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_negotiate_propose_prompt(MENU, 2, "A"),
    )
    assert meter.tokens_by_kind["select"] == 15
    assert meter.tokens_by_kind["negotiate_propose"] == 15
    assert meter.calls_by_kind == {"select": 1, "negotiate_propose": 1}


def test_negotiate_totals_pool_both_rounds_and_exclude_reconcile():
    """The exact quantity `_C95_NEGOTIATE` is the per-call median of.

    Reconcile is the AGGREGATOR's call, provisioned by `a95`. Folding it in
    would charge aggregator cost to the scouts' constant -- the same
    confusion that made the test-only classifier count reconcile as
    negotiation.
    """
    meter = _meter()
    meter(model="deepseek-v4-flash-0731", messages=build_negotiate_propose_prompt(MENU, 2, "A"))
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_negotiate_revise_prompt(MENU, 2, ["uniform_a"], ["uniform_b"]),
    )
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_reconcile_prompt(["uniform_a"], ["uniform_b"]),
    )
    meter(model="deepseek-v4-flash-0731", messages=build_select_prompt(MENU, 3, None))

    assert meter.negotiate_tokens == 30
    assert meter.n_negotiate_calls == 2


def test_per_kind_totals_sum_to_the_cell_total():
    """No call may be dropped or double-counted.

    A kind bucket that misses a call reads as a cheaper call than it was, and
    every constant derived from it is provisioned low.
    """
    meter = _meter(prompt_tokens=7, completion_tokens=3)
    for msgs in (
        build_select_prompt(MENU, 3, None),
        build_negotiate_propose_prompt(MENU, 2, "A"),
        build_reconcile_prompt(["uniform_a"], []),
    ):
        meter(model="deepseek-v4-flash-0731", messages=msgs)
    total = meter.total_input_tokens + meter.total_output_tokens
    assert sum(meter.tokens_by_kind.values()) == total == 30


def test_unclassifiable_prompt_lands_in_unknown_rather_than_vanishing():
    """Silent loss is the failure mode; an explicit bucket is auditable."""
    meter = _meter()
    meter(model="deepseek-v4-flash-0731", messages=[{"role": "user", "content": "hi"}])
    assert meter.tokens_by_kind["unknown"] == 15
    assert sum(meter.tokens_by_kind.values()) == (
        meter.total_input_tokens + meter.total_output_tokens
    )


def test_provider_rotation_attributes_every_attempt_to_one_kind():
    """Rotation bills real tokens; the calibration must see all of them.

    `calls` already records one entry per ATTEMPT so rotation shows up in
    cost attribution. Per-kind totals must agree, or a rotated negotiation
    call would be provisioned at its successful attempt's cost alone.
    """
    attempts: list[int] = []

    def flaky(**_: Any) -> dict:
        attempts.append(1)
        finish = "error" if len(attempts) == 1 else "stop"
        return {
            "choices": [{"message": {"content": "uniform_a"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    meter = _CountingLLM(target=flaky)
    meter(
        model="deepseek-v4-flash-0731",
        messages=build_negotiate_propose_prompt(MENU, 2, "A"),
    )
    assert len(attempts) == 2
    assert meter.n_negotiate_calls == 2
    assert meter.negotiate_tokens == 30
    assert meter.negotiate_tokens == meter.total_input_tokens + meter.total_output_tokens
