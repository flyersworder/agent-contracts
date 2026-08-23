"""Tests for the chamber-pillar CLI entry point.

Covers `evaluation.chamber_pipeline.run_experiment.main` and its
argument-parsing helpers. Tests exercise the CLI through `main(argv)`
rather than invoking the subprocess — same path, faster, no shell
quoting traps.

Most tests use `--mock-llm --dry-run` or tiny mocked-LLM sweeps to
avoid touching the real OpenRouter API. The full M4 pilot runs from
the same CLI surface but is invoked manually (M4b).
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from agent_contracts.integrations import CAUSAL_CHAMBER_AVAILABLE
from evaluation.chamber_pipeline.run_experiment import (
    M5_SPEC,
    PILOT_SPEC,
    CliMockLLM,
    _build_mock_llm,
    _build_sweep_from_args,
    build_arg_parser,
    main,
)

requires_causalchamber = pytest.mark.skipif(
    not CAUSAL_CHAMBER_AVAILABLE,
    reason="causalchamber not installed — install with pip install 'ai-agent-contracts[chambers]'",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgParser:
    """Pure parser tests — no chamber data needed."""

    def test_pilot_flag(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--pilot", "--out", "ignored.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep == PILOT_SPEC

    def test_m5_flag(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--m5", "--out", "ignored.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep == M5_SPEC

    def test_pilot_and_m5_mutually_exclusive(self) -> None:
        """argparse should error on both flags."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--pilot", "--m5", "--out", "x.parquet"])

    def test_custom_chambers(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--chambers", "lt,wt", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.chambers == ("lt", "wt")

    def test_custom_budgets(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--budgets", "0.1,0.5,1.0", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.budget_fractions == (0.1, 0.5, 1.0)

    def test_custom_variants(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--variants", "random,llm_pc", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.agent_names == ("random", "llm_pc")

    def test_empty_variants_means_all(self) -> None:
        """Empty --variants → agent_names=None → all from registry."""
        parser = build_arg_parser()
        args = parser.parse_args(["--variants", "", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.agent_names is None

    def test_custom_seeds(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--seeds", "5", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.seeds == (0, 1, 2, 3, 4)

    def test_pc_alpha_propagates(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--pc-alpha", "0.01", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.pc_alpha == 0.01


# ---------------------------------------------------------------------------
# Pre-baked specs match plan §9
# ---------------------------------------------------------------------------


class TestPilotSpec:
    """The M4 pilot spec matches plan §9 milestone M4."""

    def test_pilot_is_lt_only(self) -> None:
        assert PILOT_SPEC.chambers == ("lt",)

    def test_pilot_three_budgets(self) -> None:
        assert PILOT_SPEC.budget_fractions == (0.10, 0.50, 1.00)

    def test_pilot_thirty_seeds(self) -> None:
        assert len(PILOT_SPEC.seeds) == 30

    def test_pilot_pins_the_five_m4b_variants(self) -> None:
        """Named explicitly, never `None`.

        `None` means "every registered agent", so the preset would follow the
        registry -- and `--pilot` would stop reproducing the published pilot
        the moment any arm was added. It did: the M6 ladder took it from 450
        to 720 cells before this was pinned.
        """
        assert PILOT_SPEC.agent_names == (
            "random",
            "greedy_ig_lite",
            "llm_only",
            "llm_pc",
            "planner_reasoner",
        )

    def test_pilot_total_cells_is_450(self) -> None:
        from evaluation.chamber_pipeline.orchestrator import count_cells

        # Plan §9 M4 acceptance criterion: "= 450 runs"
        assert count_cells(PILOT_SPEC) == 450
        assert count_cells(PILOT_SPEC, exclude_skipped=True) == 450


class TestM5Spec:
    """The M5 spec matches plan §6.1 (post-review reconciliation)."""

    def test_m5_both_chambers(self) -> None:
        assert M5_SPEC.chambers == ("lt", "wt")

    def test_m5_five_budgets(self) -> None:
        assert M5_SPEC.budget_fractions == (0.10, 0.25, 0.50, 0.75, 1.00)

    def test_m5_total_post_compat_filter(self) -> None:
        """Plan §6.1: 1350 runs after compat filter (LT 750 + WT 600)."""
        from evaluation.chamber_pipeline.orchestrator import count_cells

        assert count_cells(M5_SPEC, exclude_skipped=True) == 1350


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run prints cell counts and exits 0 without invoking agents."""

    def test_dry_run_returns_zero(self, capsys) -> None:
        rc = main(["--pilot", "--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "450 cells" in captured.out

    def test_dry_run_does_not_require_out(self) -> None:
        # No --out, no error.
        rc = main(["--pilot", "--dry-run"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Mock-LLM end-to-end
# ---------------------------------------------------------------------------


class TestMockLlmConstruction:
    """The CLI's _build_mock_llm produces a working LLM-shaped callable."""

    def test_mock_llm_is_callable(self) -> None:
        llm = _build_mock_llm()
        assert callable(llm)

    def test_mock_llm_returns_dict_response(self) -> None:
        llm = _build_mock_llm()
        result = llm(
            model="x",
            messages=[
                {"role": "user", "content": "Menu:\nuniform_a_mid\nuniform_b_mid"},
            ],
        )
        assert "choices" in result
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]

    def test_mock_llm_records_calls(self) -> None:
        llm = _build_mock_llm()
        llm(model="x", messages=[{"role": "user", "content": "uniform_a_mid"}])
        assert len(llm.calls) == 1


@requires_causalchamber
class TestEndToEndMockLlmRun:
    """Full CLI run with --mock-llm against a tiny custom sweep."""

    def test_tiny_sweep_writes_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tiny.parquet")
            rc = main(
                [
                    "--chambers",
                    "lt",
                    "--budgets",
                    "0.10",
                    "--variants",
                    "random",
                    "--seeds",
                    "2",
                    "--out",
                    out,
                    "--quiet",
                ]
            )
            assert rc == 0
            assert os.path.exists(out)
            df = pd.read_parquet(out)
            assert len(df) == 2
            assert set(df["status"]) <= {"ok", "skipped", "error"}

    def test_tiny_llm_sweep_with_mock(self) -> None:
        """LLM variant runs end-to-end through CLI with --mock-llm."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tiny-llm.parquet")
            rc = main(
                [
                    "--chambers",
                    "lt",
                    "--budgets",
                    "0.10",
                    "--variants",
                    "llm_pc",
                    "--seeds",
                    "1",
                    "--mock-llm",
                    "--out",
                    out,
                    "--quiet",
                ]
            )
            assert rc == 0
            df = pd.read_parquet(out)
            assert len(df) == 1
            assert df.iloc[0]["status"] == "ok"
            # LLM-call count populated (FakeLLM exposes .calls).
            assert df.iloc[0]["n_llm_calls"] is not None

    def test_csv_extension_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tiny.csv")
            rc = main(
                [
                    "--chambers",
                    "lt",
                    "--budgets",
                    "0.10",
                    "--variants",
                    "random",
                    "--seeds",
                    "1",
                    "--out",
                    out,
                    "--quiet",
                ]
            )
            assert rc == 0
            assert os.path.exists(out)
            # CSV is human-readable; verify it parses back.
            df = pd.read_csv(out)
            assert len(df) == 1


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """CLI failure-mode behavior."""

    def test_missing_out_without_dry_run_errors(self) -> None:
        with pytest.raises(SystemExit):
            main(["--pilot"])


# ---------------------------------------------------------------------------
# Tests added in M4a.1 (post-review polish)
# ---------------------------------------------------------------------------


class TestCellTimeoutFlag:
    """--cell-timeout-seconds plumbs through to the SweepSpec."""

    def test_timeout_default_is_none(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--pilot", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        assert sweep.cell_timeout_seconds is None

    def test_timeout_propagates_on_pilot(self) -> None:
        """The pilot preset is immutable; we replace via dataclass.replace."""
        parser = build_arg_parser()
        args = parser.parse_args(["--pilot", "--cell-timeout-seconds", "60", "--out", "x.parquet"])
        sweep = _build_sweep_from_args(args)
        # Original PILOT_SPEC is unchanged.
        assert PILOT_SPEC.cell_timeout_seconds is None
        # New sweep has the override.
        assert sweep.cell_timeout_seconds == 60.0

    def test_timeout_propagates_on_custom(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--chambers",
                "lt",
                "--budgets",
                "0.5",
                "--seeds",
                "1",
                "--cell-timeout-seconds",
                "30",
                "--out",
                "x.parquet",
            ]
        )
        sweep = _build_sweep_from_args(args)
        assert sweep.cell_timeout_seconds == 30.0


class TestCliMockLlmPromoted:
    """CliMockLLM is now a top-level class (importable for debugging)."""

    def test_can_be_imported_directly(self) -> None:
        # If this raises an ImportError, the class wasn't properly promoted.
        from evaluation.chamber_pipeline.run_experiment import CliMockLLM as _CliMock

        instance = _CliMock()
        assert hasattr(instance, "calls")
        assert callable(instance)

    def test_records_call_count(self) -> None:
        mock = CliMockLLM()
        mock(model="x", messages=[{"role": "user", "content": "Menu:\nuniform_a_mid"}])
        assert len(mock.calls) == 1


class TestOutputExtensionValidation:
    """--out validation: only .parquet and .csv accepted."""

    def test_unrecognized_extension_errors(self) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "--chambers",
                    "lt",
                    "--budgets",
                    "0.10",
                    "--variants",
                    "random",
                    "--seeds",
                    "1",
                    "--out",
                    "/tmp/m4a1-bad.txt",
                    "--quiet",
                ]
            )


class TestEmptySeedsDryRun:
    """--seeds 0 produces an empty seed range; dry-run shouldn't crash on max()."""

    def test_zero_seeds_dry_run_does_not_crash(self) -> None:
        rc = main(
            [
                "--chambers",
                "lt",
                "--budgets",
                "0.10",
                "--variants",
                "random",
                "--seeds",
                "0",
                "--dry-run",
            ]
        )
        assert rc == 0
