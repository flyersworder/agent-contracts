"""The WT coverage-curve lists must hit their coverage exactly and never repeat an entry."""

import random

import pytest

from evaluation.chamber_pipeline import wt_coverage_curve_probe as probe

# The real WT shape: 21 variables, three with several entries (3, 3, 4), 28 entries.
MULTI = {"hatch": 3, "load_in": 3, "load_out": 4}
MENU = [f"validate_{v}_{i}" for v, n in MULTI.items() for i in range(n)] + [
    f"validate_single{i}_0" for i in range(18)
]


def var_of(entry: str) -> str:
    return entry.removeprefix("validate_").rsplit("_", 1)[0]


def test_menu_fixture_matches_the_wind_tunnel():
    assert len(MENU) == 28
    assert len({var_of(e) for e in MENU}) == 21


@pytest.mark.parametrize("m", list(probe.LEVELS))
def test_exact_coverage_and_distinct_entries(m):
    for s in range(20):
        buy = probe.exact_coverage_list(MENU, var_of, probe.K, m, random.Random(s))
        assert len(buy) == probe.K == len(set(buy))
        assert len({var_of(e) for e in buy}) == m
        assert set(buy) <= set(MENU)


def test_lowest_coverage_forces_every_multi_entry_variable():
    buy = probe.exact_coverage_list(MENU, var_of, probe.K, 14, random.Random(0))
    held = {var_of(e) for e in buy}
    assert set(MULTI) <= held
    assert sum(var_of(e) in MULTI for e in buy) == sum(MULTI.values())


def test_levels_span_every_feasible_coverage():
    assert list(probe.LEVELS) == list(range(21 - (sum(MULTI.values()) - len(MULTI)), 22))


def test_generate_is_seeded_and_sized():
    a = probe.generate(MENU, var_of)
    b = probe.generate(MENU, var_of)
    assert len(a) == len(probe.LEVELS) * probe.LISTS_PER_LEVEL
    assert a["chosen_experiments"].equals(b["chosen_experiments"])
    assert (a["chamber"] == "wt").all() and (a["budget_k"] == probe.K).all()


@pytest.mark.parametrize(
    ("lo", "hi", "letter"),
    [(-0.01, 0.01, "A"), (0.005, 0.03, "B"), (-0.03, -0.001, "C"), (-0.02, 0.01, "D")],
)
def test_verdict_is_keyed_on_the_interval(lo, hi, letter):
    assert probe._verdict(lo, hi).startswith(letter)
