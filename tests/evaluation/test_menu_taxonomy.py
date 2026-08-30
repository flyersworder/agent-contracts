"""Tests for the LT menu taxonomy and the two coverage-manipulation orderings.

These arms exist to answer one question -- does distinct-variable coverage
predict F1 at fixed budget -- so the ONLY property that matters is that the two
orderings actually attain the extremes they claim. A `coverage_min` that
quietly touches 20 variables instead of 11 would answer a weaker question while
reporting the strong one, and nothing downstream would notice.

Every bound is asserted as a tight IDENTITY against a formula, not as a loose
inequality: `<= 30` is satisfied by a broken implementation.
"""

from __future__ import annotations

from collections import Counter

import pytest

from evaluation.chamber_pipeline.menu_taxonomy import (
    coverage_ordered,
    experiment_strength,
    experiment_variable,
    group_by_variable,
)

# The real LT shape: 10 variables with one entry, 11 with two, 9 with three,
# = 59 entries over 30 variables. Rebuilt rather than loaded so the test runs
# without the dataset.
LT_SHAPE = {1: 10, 2: 11, 3: 9}


def _synthetic_menu() -> list[str]:
    menu, n = [], 0
    for width, count in sorted(LT_SHAPE.items()):
        for _ in range(count):
            var = f"v_{n}"
            n += 1
            for strength in ("weak", "mid", "strong")[:width]:
                menu.append(f"uniform_{var}_{strength}")
    return menu


def _n_variables(names: list[str]) -> int:
    return len({experiment_variable(n) for n in names})


class TestTaxonomy:
    def test_the_synthetic_menu_matches_the_real_lt_shape(self) -> None:
        menu = _synthetic_menu()
        assert len(menu) == 59
        assert len(group_by_variable(menu)) == 30
        widths = Counter(len(v) for v in group_by_variable(menu).values())
        assert dict(sorted(widths.items())) == LT_SHAPE

    @pytest.mark.parametrize(
        ("name", "variable", "strength"),
        [
            ("uniform_osr_c_mid", "osr_c", "mid"),
            ("uniform_diode_ir_1_strong", "diode_ir_1", "strong"),
            ("uniform_reference", "reference", "none"),
        ],
    )
    def test_real_names(self, name: str, variable: str, strength: str) -> None:
        assert experiment_variable(name) == variable
        assert experiment_strength(name) == strength


class TestCoverageOrdering:
    def test_max_attains_the_optimum_exactly(self) -> None:
        """`min(budget, n_variables)` is an upper bound no selection can beat."""
        menu = _synthetic_menu()
        for budget in (5, 11, 23, 30, 45, 59):
            picks = coverage_ordered(menu, budget, seed=0, maximize=True)
            assert len(picks) == budget
            assert _n_variables(picks) == min(budget, 30)

    def test_min_attains_the_optimum_exactly(self) -> None:
        """Fewest variables = fewest fresh variables, i.e. fattest-first.

        The bound is computed independently of the implementation: walk the
        widths in descending order and count how many are needed to reach the
        budget.
        """
        menu = _synthetic_menu()
        widths = sorted((len(v) for v in group_by_variable(menu).values()), reverse=True)
        for budget in (5, 11, 23, 30, 45, 59):
            spent = used = 0
            for w in widths:
                if spent >= budget:
                    break
                spent += w
                used += 1
            picks = coverage_ordered(menu, budget, seed=0, maximize=False)
            assert len(picks) == budget
            assert _n_variables(picks) == used

    def test_the_lt_k30_manipulation_spans_11_to_30(self) -> None:
        """The headline range the follow-up sweep depends on."""
        menu = _synthetic_menu()
        lo = coverage_ordered(menu, 30, seed=0, maximize=False)
        hi = coverage_ordered(menu, 30, seed=0, maximize=True)
        assert _n_variables(lo) == 11
        assert _n_variables(hi) == 30

    def test_picks_are_distinct(self) -> None:
        menu = _synthetic_menu()
        for maximize in (True, False):
            picks = coverage_ordered(menu, 30, seed=3, maximize=maximize)
            assert len(set(picks)) == len(picks)

    def test_every_pick_is_on_the_menu(self) -> None:
        menu = _synthetic_menu()
        picks = coverage_ordered(menu, 30, seed=1, maximize=True)
        assert set(picks) <= set(menu)

    def test_a_budget_over_the_menu_returns_the_whole_menu(self) -> None:
        menu = _synthetic_menu()
        for maximize in (True, False):
            picks = coverage_ordered(menu, 999, seed=0, maximize=maximize)
            assert sorted(picks) == sorted(menu)

    def test_seeds_change_which_names_but_never_the_coverage(self) -> None:
        """The seed must perturb membership without weakening the extreme.

        Ties are broken by a seeded shuffle rather than menu order -- the menu
        groups by variable family, so menu-order ties would hand every seed the
        same families and make the blind spot identical in all of them.
        """
        menu = _synthetic_menu()
        for maximize, expected in ((True, 30), (False, 11)):
            seen = set()
            for seed in range(8):
                picks = coverage_ordered(menu, 30, seed=seed, maximize=maximize)
                assert _n_variables(picks) == expected
                seen.add(tuple(sorted(picks)))
            assert len(seen) > 1, "the seed must change which names are bought"

    def test_ordering_is_deterministic_for_one_seed(self) -> None:
        menu = _synthetic_menu()
        a = coverage_ordered(menu, 30, seed=7, maximize=True)
        b = coverage_ordered(menu, 30, seed=7, maximize=True)
        assert a == b

    def test_min_is_strictly_narrower_than_max_at_every_useful_budget(self) -> None:
        menu = _synthetic_menu()
        for budget in (11, 20, 30, 40):
            lo = _n_variables(coverage_ordered(menu, budget, seed=0, maximize=False))
            hi = _n_variables(coverage_ordered(menu, budget, seed=0, maximize=True))
            assert lo < hi, f"budget {budget}: {lo} !< {hi}"
