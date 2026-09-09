"""Smoke tests for the oracle probe; the science is in `docs/chamber-results.md`."""

from __future__ import annotations

import pytest

from evaluation.chamber_pipeline import oracle_probe

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("data/causalchamber/wt_validate_v1").exists(),
    reason="needs the cached wt_validate_v1 dataset",
)


def test_score_is_a_bounded_mean_over_seeds() -> None:
    menu, _, _, _ = oracle_probe._load("wt")
    f1 = oracle_probe.score("wt", menu[:3], seeds=[0, 1])
    assert 0.0 <= f1 <= 1.0


def test_greedy_prefixes_nest_and_never_repeat() -> None:
    rows = oracle_probe.greedy_oracle("wt", 2, seeds=[0], workers=2, log=lambda *_: None)
    assert [r["k"] for r in rows] == [1, 2]
    import json

    a, b = (json.loads(r["chosen"]) for r in rows)
    assert a == b[:1] and len(set(b)) == 2
