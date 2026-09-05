"""Tests for the M7 Phase 1 mechanism analysis.

The load-bearing piece is `experiment_variable`: every hypothesis in the phase
is stated in terms of DISTINCT VARIABLES, so a parser that mis-splits a name
moves the answer rather than raising. LT variable names contain underscores
(`osr_angle_1`, `diode_ir_3`), so the split has to be on the known strength
suffix and not on a token count.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from evaluation.chamber_pipeline.analyze_mechanism import (
    experiment_strength,
    experiment_variable,
    mde,
    mechanism_table,
    parse_roster,
    summarize,
    verdict,
)


class TestExperimentVariable:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("uniform_blue_strong", "blue"),
            ("uniform_osr_c_mid", "osr_c"),
            ("uniform_diode_ir_1_strong", "diode_ir_1"),
            ("uniform_osr_angle_1_strong", "osr_angle_1"),
            ("uniform_reference", "reference"),
        ],
    )
    def test_real_lt_menu_names(self, name: str, expected: str) -> None:
        assert experiment_variable(name) == expected

    def test_a_multi_token_variable_survives(self) -> None:
        """The defect a token-count split would introduce.

        `rsplit("_", 1)` gives `diode_ir_1` for the name below only by luck of
        it carrying a strength; drop the strength and the same rule eats a
        real token. Distinguishing the two is the whole point of matching the
        suffix against a known set.
        """
        name = "uniform_diode_ir_1"
        naive = name[len("uniform_") :].rsplit("_", 1)[0]
        assert experiment_variable(name) == "diode_ir_1"
        assert naive == "diode_ir", "the naive split eats a real token"

    def test_strength_is_read_from_the_same_suffix(self) -> None:
        assert experiment_strength("uniform_osr_c_mid") == "mid"
        assert experiment_strength("uniform_reference") == "none"

    def test_three_strengths_collapse_to_one_variable(self) -> None:
        names = ["uniform_v_1_weak", "uniform_v_1_mid", "uniform_v_1_strong"]
        assert len({experiment_variable(n) for n in names}) == 1


class TestParseRoster:
    def test_empty_and_null_are_empty(self) -> None:
        assert parse_roster(None) == []
        assert parse_roster("") == []
        assert parse_roster(float("nan")) == []

    def test_order_is_preserved(self) -> None:
        assert parse_roster("b,a,c") == ["b", "a", "c"]


def _row(
    agent: str, seed: int, roster: list[str], f1: float, budget_k: int | None = None
) -> dict[str, object]:
    """One ok-row. `budget_k` defaults to the roster length: every arm spends
    its whole budget, so a fixture that disagrees is describing a shortfall."""
    return {
        "agent_name": agent,
        "seed": seed,
        "budget_k": len(roster) if budget_k is None else budget_k,
        "status": "ok",
        "f1": f1,
        "chosen_experiments": ",".join(roster),
        "n_zero_variance_dropped": 0,
        "n_selection_fallbacks": 0,
    }


class TestMechanismTable:
    def test_counts_variables_not_experiments(self) -> None:
        roster = ["uniform_a_weak", "uniform_a_strong", "uniform_b_mid"]
        table = mechanism_table(pd.DataFrame([_row("llm_pc", 0, roster, 0.4)]))
        assert table.loc[0, "n_experiments"] == 3
        assert table.loc[0, "n_variables"] == 2
        assert table.loc[0, "n_repeat_variables"] == 1
        assert table.loc[0, "max_depth"] == 2

    def test_team_split_is_positional_and_halves_the_roster(self) -> None:
        roster = [
            "uniform_a_weak",
            "uniform_b_weak",
            "uniform_b_strong",
            "uniform_c_mid",
        ]
        table = mechanism_table(pd.DataFrame([_row("team", 0, roster, 0.3)]))
        # ceil(4/2) = 2 -> scout_a bought {a, b}, scout_b bought {b, c}
        assert table.loc[0, "a_variables"] == 2
        assert table.loc[0, "b_variables"] == 2
        assert table.loc[0, "shared_variables"] == 1

    def test_shared_variables_is_the_h3_signal_at_zero_experiment_overlap(self) -> None:
        """Disjoint experiments, one shared variable: exactly H3."""
        roster = ["uniform_v_weak", "uniform_w_mid", "uniform_v_strong", "uniform_x_mid"]
        table = mechanism_table(pd.DataFrame([_row("team", 0, roster, 0.3)]))
        assert len(set(roster)) == len(roster), "no experiment is repeated"
        assert table.loc[0, "shared_variables"] == 1

    def test_non_team_arms_get_no_scout_split(self) -> None:
        table = mechanism_table(pd.DataFrame([_row("llm_pc", 0, ["uniform_a_mid"], 0.4)]))
        assert table.loc[0, "a_variables"] is None

    def test_a_repeated_purchase_raises_rather_than_mis_attributing(self) -> None:
        roster = ["uniform_a_mid", "uniform_a_mid"]
        with pytest.raises(ValueError, match="repeats a name"):
            mechanism_table(pd.DataFrame([_row("team", 0, roster, 0.3)]))

    def test_rows_predating_the_instrument_raise(self) -> None:
        row = _row("llm_pc", 0, [], 0.4)
        row["chosen_experiments"] = None
        with pytest.raises(ValueError, match="predate the instrument"):
            mechanism_table(pd.DataFrame([row]))

    def test_errored_cells_are_excluded(self) -> None:
        bad = _row("llm_pc", 1, [], 0.0)
        bad["status"] = "error"
        bad["chosen_experiments"] = None
        good = _row("llm_pc", 0, ["uniform_a_mid"], 0.4)
        assert len(mechanism_table(pd.DataFrame([good, bad]))) == 1


class TestVerdict:
    @staticmethod
    def _frame(loop_vars: int, team_vars: int, n: int = 10) -> pd.DataFrame:
        rows = []
        for seed in range(n):
            # 30 experiments each; only the variable spread differs.
            loop = [f"uniform_v{i}_mid" for i in range(loop_vars)]
            loop += [f"uniform_v{i}_strong" for i in range(30 - loop_vars)]
            team = [f"uniform_w{i}_mid" for i in range(team_vars)]
            team += [f"uniform_w{i}_strong" for i in range(30 - team_vars)]
            rows.append(_row("llm_pc", seed, loop, 0.42))
            rows.append(_row("team", seed, team, 0.37))
        return mechanism_table(pd.DataFrame(rows))

    def test_a_large_variable_deficit_reads_as_h1_h3(self) -> None:
        cells = self._frame(loop_vars=28, team_vars=18)
        lines = verdict(summarize(cells), cells)
        assert any("H1/H3: team buys measurably fewer" in ln for ln in lines)

    def test_matched_coverage_does_not_read_as_h1_h3(self) -> None:
        cells = self._frame(loop_vars=24, team_vars=24)
        lines = verdict(summarize(cells), cells)
        assert any("H1/H3 not supported" in ln for ln in lines)

    def test_a_missing_arm_refuses_to_rule(self) -> None:
        cells = mechanism_table(pd.DataFrame([_row("llm_pc", 0, ["uniform_a_mid"], 0.4)]))
        assert "required" in verdict(summarize(cells), cells)[0]

    def test_mde_is_nan_with_one_arm(self) -> None:
        cells = mechanism_table(pd.DataFrame([_row("llm_pc", 0, ["uniform_a_mid"], 0.4)]))
        assert math.isnan(mde(cells, "f1"))


class TestScoutSeam:
    """The seam is `ceil(k/2)` from the contract, never `len(roster)/2`."""

    def test_odd_budget_puts_the_extra_pick_on_scout_a(self) -> None:
        # run_cell sets scout_a_budget = k - k//2, so k=5 -> 3 and 2.
        roster = [f"uniform_v{i}_mid" for i in range(5)]
        row = _row("team", 0, roster, 0.3, budget_k=5)
        table = mechanism_table(pd.DataFrame([row]))
        assert table.loc[0, "a_variables"] == 3
        assert table.loc[0, "b_variables"] == 2

    def test_an_underspending_scout_raises(self) -> None:
        roster = [f"uniform_v{i}_mid" for i in range(28)]
        row = _row("team", 0, roster, 0.3, budget_k=30)
        with pytest.raises(ValueError, match="bought 28 of 30"):
            mechanism_table(pd.DataFrame([row]))
