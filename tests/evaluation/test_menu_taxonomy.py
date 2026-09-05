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
    partition_pools_by_variable,
)

# The real LT shape, verified against the live menu: 59 entries over 30
# variables -- 10 with one entry, 11 with two, 9 with three. Rebuilt rather
# than loaded so the tests run without the dataset.
#
# WHICH strengths each width carries is load-bearing and was wrong in the first
# draft. All 9 `weak` entries in the LT menu belong to the 9 three-entry
# variables; the two-entry variables are (mid, strong) and the one-entry
# variables are (mid,). Modelling width 2 as (weak, mid) made excluding `weak`
# shrink those variables to width 1 and moved every deconfounded bound.
LT_WIDTH_STRENGTHS: dict[int, tuple[str, ...]] = {
    1: ("mid",),
    2: ("mid", "strong"),
    3: ("weak", "mid", "strong"),
}
LT_SHAPE = {1: 10, 2: 11, 3: 9}


def _synthetic_menu() -> list[str]:
    menu, n = [], 0
    for width, count in sorted(LT_SHAPE.items()):
        for _ in range(count):
            var = f"v_{n}"
            n += 1
            for strength in LT_WIDTH_STRENGTHS[width]:
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
        # All 9 weak entries sit on three-entry variables -- the fact the
        # deconfounding filter turns on.
        strengths = Counter(experiment_strength(m) for m in menu)
        assert strengths["weak"] == 9
        assert {
            len(group_by_variable(menu)[experiment_variable(m)])
            for m in menu
            if experiment_strength(m) == "weak"
        } == {3}
        no_weak = [m for m in menu if experiment_strength(m) != "weak"]
        assert len(no_weak) == 50
        assert dict(
            sorted(Counter(len(v) for v in group_by_variable(no_weak).values()).items())
        ) == {1: 10, 2: 20}

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


class TestStrengthExclusion:
    """The deconfounding filter, which is the whole point of the `_ms` arms.

    The unrestricted manipulation moves breadth and intervention strength
    together -- the fattest variables are exactly the ones carrying a `weak`
    level -- so the filter has to actually close the strength channel, not
    merely narrow the span.
    """

    def test_no_excluded_strength_survives(self) -> None:
        menu = _synthetic_menu()
        for maximize in (True, False):
            picks = coverage_ordered(
                menu, 30, seed=0, maximize=maximize, exclude_strengths=("weak",)
            )
            assert not [p for p in picks if experiment_strength(p) == "weak"]

    def test_excluding_weak_narrows_the_span_to_15_30(self) -> None:
        """The real LT shape: 50 entries over 30 variables, 20 of width 2."""
        menu = _synthetic_menu()
        lo = coverage_ordered(menu, 30, seed=0, maximize=False, exclude_strengths=("weak",))
        hi = coverage_ordered(menu, 30, seed=0, maximize=True, exclude_strengths=("weak",))
        assert _n_variables(lo) == 15
        assert _n_variables(hi) == 30

    def test_the_filter_changes_the_lower_end_and_not_the_upper(self) -> None:
        """Max already spends one pick per variable, so dropping a level
        cannot reduce its coverage; min is the end the confound lived at."""
        menu = _synthetic_menu()
        raw_lo = _n_variables(coverage_ordered(menu, 30, seed=0, maximize=False))
        raw_hi = _n_variables(coverage_ordered(menu, 30, seed=0, maximize=True))
        f_lo = _n_variables(
            coverage_ordered(menu, 30, seed=0, maximize=False, exclude_strengths=("weak",))
        )
        f_hi = _n_variables(
            coverage_ordered(menu, 30, seed=0, maximize=True, exclude_strengths=("weak",))
        )
        assert (raw_lo, f_lo) == (11, 15), "the lower end moves"
        assert raw_hi == f_hi == 30, "the upper end does not"

    def test_budget_is_still_spent_in_full_after_filtering(self) -> None:
        menu = _synthetic_menu()
        for maximize in (True, False):
            picks = coverage_ordered(
                menu, 30, seed=2, maximize=maximize, exclude_strengths=("weak",)
            )
            assert len(picks) == 30
            assert len(set(picks)) == 30

    def test_an_empty_exclusion_is_the_unrestricted_ordering(self) -> None:
        menu = _synthetic_menu()
        a = coverage_ordered(menu, 30, seed=5, maximize=False)
        b = coverage_ordered(menu, 30, seed=5, maximize=False, exclude_strengths=())
        assert a == b


class TestVariablePartition:
    """The one property the arm exists for: no variable in both pools.

    Everything else about `team_varsplit` is held identical to `team`, so if
    the partition leaks a variable the contrast measures nothing.
    """

    @staticmethod
    def _pools(claim_a, claim_b, seed=0, budget=15):
        return partition_pools_by_variable(
            _synthetic_menu(), claim_a, claim_b, budget, budget, seed
        )

    def test_no_variable_appears_in_both_pools(self) -> None:
        for seed in range(10):
            a, b = self._pools([], [], seed=seed)
            va = {experiment_variable(n) for n in a}
            vb = {experiment_variable(n) for n in b}
            assert not va & vb, f"seed {seed} leaked {va & vb}"

    def test_every_entry_of_a_variable_travels_together(self) -> None:
        menu = _synthetic_menu()
        a, b = self._pools([], [])
        for variable, names in group_by_variable(menu).items():
            assert set(names) <= a or set(names) <= b, variable

    def test_the_pools_are_disjoint_and_cover_the_menu(self) -> None:
        menu = _synthetic_menu()
        a, b = self._pools([], [])
        assert not a & b
        assert a | b == set(menu)

    def test_a_claim_wins_its_variable(self) -> None:
        a, _ = self._pools(["uniform_v_0_mid"], [])
        assert "uniform_v_0_mid" in a

    def test_a_contested_variable_goes_to_a_matching_the_name_level_rule(self) -> None:
        """`team_agents` resolves name conflicts in A's favour; so does this."""
        a, b = self._pools(["uniform_v_25_weak"], ["uniform_v_25_strong"])
        assert {"uniform_v_25_weak", "uniform_v_25_strong"} <= a
        assert not {"uniform_v_25_weak", "uniform_v_25_strong"} & b

    def test_a_variable_claimed_at_one_strength_carries_the_others(self) -> None:
        """The whole point: claiming `weak` also removes `mid` and `strong`
        from the peer's reach, which the name-level split does not do."""
        menu = _synthetic_menu()
        three_wide = next(v for v, names in group_by_variable(menu).items() if len(names) == 3)
        one_name = group_by_variable(menu)[three_wide][0]
        a, _ = self._pools([one_name], [])
        assert set(group_by_variable(menu)[three_wide]) <= a

    def test_free_variables_are_dealt_to_balance_entry_counts(self) -> None:
        """Variables carry 1-3 entries, so an alternating deal balances the
        wrong quantity. Pools should come out close in entry count."""
        for seed in range(10):
            a, b = self._pools([], [], seed=seed)
            assert abs(len(a) - len(b)) <= 3, f"seed {seed}: {len(a)} vs {len(b)}"

    def test_both_pools_strictly_exceed_their_budget(self) -> None:
        for seed in range(10):
            a, b = self._pools([], [], seed=seed)
            assert len(a) > 15 and len(b) > 15

    def test_an_infeasible_budget_raises_rather_than_going_inert(self) -> None:
        with pytest.raises(ValueError, match="selection loop is inert"):
            partition_pools_by_variable(_synthetic_menu(), [], [], 40, 15, 0)

    def test_the_seed_changes_the_split(self) -> None:
        seen = {tuple(sorted(self._pools([], [], seed=s)[0])) for s in range(8)}
        assert len(seen) > 1
