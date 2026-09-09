"""Smoke tests for the oracle probe; the science is in `docs/chamber-results.md`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.chamber_pipeline import oracle_probe
from evaluation.chamber_pipeline.rescore import parse_selection

pytestmark = pytest.mark.skipif(
    not Path("data/causalchamber/wt_validate_v1").exists(),
    reason="needs the cached wt_validate_v1 dataset",
)


def test_score_is_bounded_and_core_is_nan_off_lt() -> None:
    menu, _, _, _ = oracle_probe._load("wt")
    f1, core = oracle_probe.score("wt", menu[:3], seeds=[0, 1])
    assert 0.0 <= f1 <= 1.0
    assert core != core  # NaN: core-20 is LT-only


def test_greedy_prefixes_nest_carry_provenance_and_never_repeat() -> None:
    scorer = oracle_probe.Scorer("wt", 2)
    try:
        rows = oracle_probe.greedy_oracle(scorer, 2, seeds=[0], log=lambda *_: None)
    finally:
        scorer.close()
    assert [r["k"] for r in rows] == [1, 2]
    a, b = (parse_selection(r["chosen_experiments"]) for r in rows)
    assert a == b[:1] and len(set(b)) == 2
    assert {"design_key", "selection_key", "blas_backend", "pc_alpha"} <= rows[1].keys()


def test_main_refuses_overlapping_seed_sets(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="overlap"):
        oracle_probe.main(
            ["--chamber", "wt", "--seeds", "2", "--report-seeds", "2", "--report-seed-offset", "1"]
        )


def test_main_refuses_budgets_beyond_the_menu() -> None:
    with pytest.raises(SystemExit, match="exceed"):
        oracle_probe.main(["--chamber", "wt", "--budgets", "40"])


def test_main_emits_every_file_the_docs_cite(tmp_path: Path) -> None:
    out = tmp_path / "probe"
    oracle_probe.main(
        [
            "--chamber",
            "wt",
            "--budgets",
            "2",
            "--k-max",
            "2",
            "--seeds",
            "1",
            "--report-seeds",
            "1",
            "--contexts",
            "1",
            "--rule-seeds",
            "1",
            "--workers",
            "2",
            "--corpus",
            str(tmp_path / "missing.parquet"),
            "--out",
            str(out),
        ]
    )
    for suffix in ("greedy", "context", "policies"):
        assert (tmp_path / f"probe-wt-{suffix}.parquet").exists()
    policies = pd.read_parquet(tmp_path / "probe-wt-policies.parquet")
    assert {
        "greedy_oracle",
        "greedy+swap_oracle",
        "rank_at_k",
        "rank_pooled",
        "coverage_rule",
        "random",
    } <= set(policies.policy)
    assert json.loads(policies.report_seeds.iloc[0]) == [100]
    assert (policies.blas_backend == policies.blas_backend.iloc[0]).all()


def test_mde_reduces_to_the_equal_n_form() -> None:
    assert oracle_probe.mde(0.1, 30, 0.1, 30) == pytest.approx(2.8 * 0.1 * (2 / 30) ** 0.5)
