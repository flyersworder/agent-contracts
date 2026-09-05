"""Tests for the two-factor model that predicts `team_varsplit`'s gain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.analyze_headroom import (
    LLM_FREE_ARMS,
    a_priori_headroom,
    coverage_slope,
    distinct_variables,
    parse_roster,
    predict,
    variable_resolver,
    variables_recovered,
)


#: Test resolver: the variable is everything before the final `_`. Lets these
#: tests plant a known number of distinct variables without a real menu.
def _fake_resolver(_chamber: str):
    return lambda name: name.rsplit("_", 1)[0]


def _row(chamber, k, agent, roster, value, status="ok"):
    return {
        "chamber": chamber,
        "budget_k": k,
        "agent_name": agent,
        "chosen_experiments": ",".join(roster),
        "status": status,
        "f1_rescored": value,
    }


def _roster(n_variables: int, n_entries: int) -> list[str]:
    """`n_entries` names touching exactly `n_variables` distinct variables."""
    names = [f"v{i}_a" for i in range(n_variables)]
    names += [f"v{i % n_variables}_b" for i in range(n_entries - n_variables)]
    return names


def test_parse_roster_keeps_order_and_drops_empties():
    assert parse_roster("a,b,c") == ["a", "b", "c"]
    assert parse_roster("a,,b") == ["a", "b"]
    assert parse_roster(None) == []
    assert parse_roster("") == []
    assert parse_roster(3.5) == []


def test_distinct_variables_counts_variables_not_entries():
    resolve = _fake_resolver("lt")
    assert distinct_variables(_roster(3, 3), resolve) == 3
    # six entries, still three variables
    assert distinct_variables(_roster(3, 6), resolve) == 3


def test_variable_resolver_dispatches_per_chamber():
    assert variable_resolver("lt", [])("uniform_osr_c_strong") == "osr_c"
    assert (
        variable_resolver("wt", ["load_out", "pressure_intake"])(
            "validate_load_out_pressure_intake"
        )
        == "load_out"
    )
    with pytest.raises(ValueError, match="unknown chamber"):
        variable_resolver("kitchen", [])


def test_coverage_slope_recovers_a_planted_exchange_rate():
    slope = 0.02
    rows = [
        _row("lt", 30, "random", _roster(n_var, 30), 0.3 + slope * n_var) for n_var in range(10, 30)
    ]
    got = coverage_slope(pd.DataFrame(rows), resolver_for=_fake_resolver)
    assert got.loc[0, "chamber"] == "lt"
    assert got.loc[0, "slope"] == pytest.approx(slope, abs=1e-9)


def test_coverage_slope_removes_the_budget_confound():
    """A between-budget trend of the OPPOSITE sign must not leak into the slope.

    Planted so a naive pooled fit gets the SIGN wrong: the large-budget cells
    buy far more variables yet score lower, which is the shape of a confound
    that centring within budget has to remove. Within each budget the true
    exchange rate is positive.
    """
    slope = 0.01
    rows = []
    for budget_k, offset, base_vars in ((6, 0.5, 4), (45, 0.0, 25)):
        rows += [
            _row("lt", budget_k, "random", _roster(n_var, budget_k), offset + slope * n_var)
            for n_var in range(base_vars, base_vars + 4)
        ]
    resolve = _fake_resolver("lt")
    naive = np.polyfit(
        [distinct_variables(parse_roster(r["chosen_experiments"]), resolve) for r in rows],
        [r["f1_rescored"] for r in rows],
        1,
    )[0]
    assert naive < 0, "the planted confound must flip a naive fit's sign"
    got = coverage_slope(pd.DataFrame(rows), resolver_for=_fake_resolver)
    assert got.loc[0, "slope"] == pytest.approx(slope, abs=1e-9)


def test_coverage_slope_ignores_llm_arms():
    """An LLM arm with an inverted slope must not move the estimate."""
    rows = [
        _row("lt", 30, "random", _roster(n_var, 30), 0.3 + 0.02 * n_var) for n_var in range(10, 30)
    ]
    rows += [
        _row("lt", 30, "llm_pc", _roster(n_var, 30), 0.9 - 0.05 * n_var) for n_var in range(10, 30)
    ]
    got = coverage_slope(pd.DataFrame(rows), resolver_for=_fake_resolver)
    assert got.loc[0, "slope"] == pytest.approx(0.02, abs=1e-9)
    assert got.loc[0, "n_cells"] == 20


def test_coverage_slope_skips_errored_cells():
    rows = [
        _row("lt", 30, "random", _roster(n_var, 30), 0.3 + 0.02 * n_var) for n_var in range(10, 30)
    ]
    rows.append(_row("lt", 30, "random", _roster(29, 30), 99.0, status="error"))
    got = coverage_slope(pd.DataFrame(rows), resolver_for=_fake_resolver)
    assert got.loc[0, "slope"] == pytest.approx(0.02, abs=1e-9)


def test_coverage_slope_refuses_a_frame_with_no_rescored_column():
    rows = [_row("lt", 30, "random", _roster(12, 30), 0.4)]
    frame = pd.DataFrame(rows).drop(columns=["f1_rescored"])
    with pytest.raises(ValueError, match="re-scored"):
        coverage_slope(frame, resolver_for=_fake_resolver)


def test_llm_free_arms_excludes_every_llm_arm():
    for arm in ("llm_pc", "one_shot", "critique", "shared_blackboard", "team", "team_varsplit"):
        assert arm not in LLM_FREE_ARMS


def test_variables_recovered_differences_treatment_minus_baseline():
    rows = [
        _row("wt", 14, "team", _roster(10, 14), 0.2),
        _row("wt", 14, "team", _roster(12, 14), 0.2),
        _row("wt", 14, "team_varsplit", _roster(13, 14), 0.2),
        _row("wt", 14, "team_varsplit", _roster(13, 14), 0.2),
    ]
    got = variables_recovered(pd.DataFrame(rows), resolver_for=_fake_resolver)
    assert got.loc[0, "n_variables_baseline"] == pytest.approx(11.0)
    assert got.loc[0, "n_variables_treatment"] == pytest.approx(13.0)
    assert got.loc[0, "variables_recovered"] == pytest.approx(2.0)


def test_variables_recovered_refuses_a_missing_arm():
    rows = [_row("wt", 14, "team", _roster(10, 14), 0.2)]
    with pytest.raises(ValueError, match="must be present"):
        variables_recovered(pd.DataFrame(rows), resolver_for=_fake_resolver)


def test_predict_multiplies_the_two_factors():
    slopes = pd.DataFrame([{"chamber": "wt", "slope": 0.01}])
    recovered = pd.DataFrame([{"chamber": "wt", "budget_k": 21, "variables_recovered": 1.5}])
    got = predict(slopes, recovered)
    assert got.loc[0, "predicted_gain"] == pytest.approx(0.015)


def test_a_priori_headroom_refuses_a_budget_the_menu_cannot_split(monkeypatch):
    """Two disjoint pools must each hold their scout's whole budget."""
    import evaluation.chamber_pipeline.analyze_headroom as mod

    class _Adapter:
        def available_experiments(self):
            return [f"v{i}_a" for i in range(10)]

        def ground_truth(self):
            return pd.DataFrame(index=[f"v{i}" for i in range(10)])

    monkeypatch.setattr(mod, "_adapter", lambda _chamber: _Adapter())
    monkeypatch.setattr(mod, "variable_resolver", lambda _c, _n: _fake_resolver(_c))
    with pytest.raises(ValueError, match="does not fit"):
        a_priori_headroom("lt", 12, trials=1)


def test_a_priori_headroom_is_higher_for_a_redundant_action_space(monkeypatch):
    """The claim the model rests on: duplication is a property of the menu.

    Two menus of the SAME size and the same budget, differing only in how many
    entries share a variable. The redundant one must give blind scouts more to
    duplicate; if it did not, the moderator would have no content.
    """
    import evaluation.chamber_pipeline.analyze_headroom as mod

    def _menu(n_variables: int):
        class _Adapter:
            def available_experiments(self):
                return [f"v{i % n_variables}_{i // n_variables}" for i in range(24)]

            def ground_truth(self):
                return pd.DataFrame(index=[f"v{i}" for i in range(n_variables)])

        return _Adapter()

    monkeypatch.setattr(mod, "variable_resolver", lambda _c, _n: _fake_resolver(_c))

    monkeypatch.setattr(mod, "_adapter", lambda _c: _menu(24))  # 1 entry per variable
    unique = a_priori_headroom("lt", 10, trials=2000, seed=1)
    monkeypatch.setattr(mod, "_adapter", lambda _c: _menu(8))  # 3 entries per variable
    redundant = a_priori_headroom("lt", 10, trials=2000, seed=1)

    assert unique == pytest.approx(0.0, abs=1e-9), "a 1:1 menu affords no duplication"
    assert redundant > 1.0
