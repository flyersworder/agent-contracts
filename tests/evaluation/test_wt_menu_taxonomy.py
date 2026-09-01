"""Tests for the wind-tunnel menu parse.

The parse decides what "a distinct variable" means, and the coverage arms are
scored on exactly that count — so a wrong parse does not crash, it silently
changes the number the arm exists to manipulate.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from evaluation.chamber_pipeline.wt_menu_taxonomy import (
    coverage_ordered,
    experiment_variable,
    group_by_variable,
    partition_pools_by_variable,
)

# A faithful miniature of the real menu: single-entry settings, multi-entry
# drivers, and the osr_in / osr_intake prefix collision.
NODES = [
    "hatch",
    "load_in",
    "load_out",
    "osr_in",
    "osr_intake",
    "osr_1",
    "pressure_intake",
    "current_in",
    "mic",
]


def test_exact_node_name_resolves_to_itself() -> None:
    assert experiment_variable("validate_load_in", NODES) == "load_in"


def test_suffix_after_the_variable_is_ignored() -> None:
    """`validate_load_out_pressure_intake` intervenes on load_out."""
    assert experiment_variable("validate_load_out_pressure_intake", NODES) == "load_out"
    assert experiment_variable("validate_hatch_mic", NODES) == "hatch"


def test_longest_match_wins_when_both_candidates_clear_the_boundary() -> None:
    """Discriminating case for longest-match, found by mutation testing.

    On the real WT menu the underscore boundary ALREADY separates `osr_in`
    from `osr_intake`, so longest-match never fires there and a first-match
    rule passes every realistic test. The guards are redundant on today's
    menu. This constructs the case where only longest-match is correct: two
    node names where one plus an underscore prefixes the other, so both clear
    the boundary and only length breaks the tie.
    """
    nodes = ["a_b", "a_b_c"]
    assert experiment_variable("validate_a_b_c_d", nodes) == "a_b_c"


def test_osr_in_and_osr_intake_stay_separate() -> None:
    """The real collision, kept as a regression guard on live names."""
    assert experiment_variable("validate_osr_intake", NODES) == "osr_intake"
    assert experiment_variable("validate_osr_in", NODES) == "osr_in"


def test_bare_prefix_without_an_underscore_is_not_a_match() -> None:
    """Discriminating case for the boundary, also found by mutation testing.

    A plain `startswith` rule matches `load` against `loadX` and returns a
    variable that was never intervened on. The real menu has no bare-prefix
    node, so only a constructed case exposes it.
    """
    with pytest.raises(ValueError, match="no node name prefixes"):
        experiment_variable("validate_loadX", ["load", "load_in"])


def test_partial_token_is_not_a_match() -> None:
    """`load` must not match `load_in` — the boundary is an underscore."""
    with pytest.raises(ValueError, match="no node name prefixes"):
        experiment_variable("validate_loadX", NODES)


def test_unparseable_entry_raises_rather_than_inventing_a_variable() -> None:
    with pytest.raises(ValueError, match="no node name prefixes"):
        experiment_variable("validate_nonsense", NODES)


def test_group_by_variable_partitions_the_menu() -> None:
    menu = [
        "validate_hatch_mic",
        "validate_hatch_rpms",
        "validate_load_in",
        "validate_load_in_current_out",
        "validate_osr_1",
        "validate_osr_intake",
    ]
    groups = group_by_variable(menu, NODES)
    assert sum(len(v) for v in groups.values()) == len(menu)
    assert set(groups) == {"hatch", "load_in", "osr_1", "osr_intake"}
    assert len(groups["hatch"]) == 2


def test_maximize_takes_one_per_variable_before_a_second() -> None:
    menu = [
        "validate_hatch_mic",
        "validate_hatch_rpms",
        "validate_hatch_pressures",
        "validate_osr_1",
        "validate_osr_in",
    ]
    chosen = coverage_ordered(menu, 3, seed=0, node_names=NODES, maximize=True)
    variables = {experiment_variable(c, NODES) for c in chosen}
    assert len(chosen) == 3
    assert len(variables) == 3, "round-robin must reach every variable first"


def test_minimize_exhausts_the_fattest_variable_first() -> None:
    menu = [
        "validate_hatch_mic",
        "validate_hatch_rpms",
        "validate_hatch_pressures",
        "validate_osr_1",
        "validate_osr_in",
    ]
    chosen = coverage_ordered(menu, 3, seed=0, node_names=NODES, maximize=False)
    assert {experiment_variable(c, NODES) for c in chosen} == {"hatch"}


def test_ordering_is_deterministic_in_the_seed() -> None:
    menu = ["validate_hatch_mic", "validate_osr_1", "validate_osr_in", "validate_load_in"]
    a = coverage_ordered(menu, 3, seed=7, node_names=NODES, maximize=True)
    b = coverage_ordered(menu, 3, seed=7, node_names=NODES, maximize=True)
    assert a == b


def test_budget_above_menu_size_returns_the_whole_menu() -> None:
    menu = ["validate_hatch_mic", "validate_osr_1"]
    chosen = coverage_ordered(menu, 10, seed=0, node_names=NODES, maximize=True)
    assert sorted(chosen) == sorted(menu)


class TestWtPartitionPools:
    """The variable partition on the wind-tunnel menu."""

    MENU: ClassVar[list[str]] = [
        "validate_hatch_mic",
        "validate_hatch_rpms",
        "validate_hatch_pressures",
        "validate_load_in",
        "validate_load_in_current_out",
        "validate_osr_1",
        "validate_osr_in",
        "validate_osr_intake",
    ]

    def test_pools_partition_the_menu_exactly(self) -> None:
        a, b = partition_pools_by_variable(self.MENU, NODES, [], [], budget_a=1, budget_b=1, seed=0)
        assert a | b == set(self.MENU)
        assert not (a & b)

    def test_every_entry_of_a_variable_lands_on_one_side(self) -> None:
        """The whole point: no variable may straddle the two pools.

        A straddling variable is exactly the failure `team` has — pools
        disjoint as sets of experiments while both scouts buy the same
        variable — so this is the property the arm exists to guarantee.
        """
        a, b = partition_pools_by_variable(self.MENU, NODES, [], [], budget_a=1, budget_b=1, seed=3)
        for side in (a, b):
            variables = {experiment_variable(n, NODES) for n in side}
            other = b if side is a else a
            assert not (variables & {experiment_variable(n, NODES) for n in other})

    def test_claims_are_honoured_and_ties_go_to_a(self) -> None:
        a, b = partition_pools_by_variable(
            self.MENU,
            NODES,
            claim_a=["validate_hatch_mic"],
            claim_b=["validate_hatch_rpms"],
            budget_a=1,
            budget_b=1,
            seed=0,
        )
        assert "validate_hatch_pressures" in a, "a variable claimed by both goes to A"
        assert not any(experiment_variable(n, NODES) == "hatch" for n in b)

    def test_infeasible_split_raises_rather_than_running_inert(self) -> None:
        """A pool at or below budget makes that scout's selection loop inert."""
        with pytest.raises(ValueError, match="selection loop is inert"):
            partition_pools_by_variable(self.MENU, NODES, [], [], budget_a=99, budget_b=1, seed=0)

    def test_deal_balances_entries_not_variable_count(self) -> None:
        """Variables carry 1-4 entries, so an alternating deal unbalances picks.

        Constructed so the two strategies visibly differ: one fat variable of
        three entries against three singles. Balancing entries gives 3 vs 3;
        alternating variables gives 4 vs 2.
        """
        menu = [
            "validate_hatch_mic",
            "validate_hatch_rpms",
            "validate_hatch_pressures",
            "validate_osr_1",
            "validate_osr_in",
            "validate_osr_intake",
        ]
        a, b = partition_pools_by_variable(menu, NODES, [], [], budget_a=1, budget_b=1, seed=1)
        assert abs(len(a) - len(b)) <= 1, f"entries unbalanced: {len(a)} vs {len(b)}"
