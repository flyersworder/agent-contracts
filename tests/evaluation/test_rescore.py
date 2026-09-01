"""Tests for offline re-scoring of recorded cells.

The load-bearing behaviours are the selection key (which decides what counts
as the same design, and therefore what gets averaged together) and the
deduplication that follows from it. Both are asserted against constructed
inputs whose correct answer is known, not against a golden output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.chamber_pipeline.rescore import (
    LT_CASE_STUDY_NODES,
    SELECTION_KEY_COLUMN,
    attach_rescored,
    parse_selection,
    selection_key,
)


def test_selection_key_ignores_order() -> None:
    """The buy is a SET: same experiments, different order, one key.

    Two cells that bought the same experiments pooled byte-identical data and
    must average together. Keying on the raw string would split them and
    inflate the design count.
    """
    a = selection_key("lt", "standard", ["b", "a", "c"])
    b = selection_key("lt", "standard", ["c", "b", "a"])
    assert a == b


def test_selection_key_separates_chambers_and_configurations() -> None:
    """Identical names in different chambers are different designs."""
    names = ["uniform_blue_mid"]
    assert selection_key("lt", "standard", names) != selection_key("wt", "standard", names)
    assert selection_key("lt", "standard", names) != selection_key("lt", "pressure-control", names)


def test_selection_key_distinguishes_different_buys() -> None:
    assert selection_key("lt", "standard", ["a", "b"]) != selection_key(
        "lt", "standard", ["a", "c"]
    )


def test_selection_key_is_not_a_prefix_collision() -> None:
    """Joining on a separator that can appear in a name would collide.

    ['ab','c'] and ['a','bc'] join to the same string under a bare
    concatenation; the delimiter must keep them apart.
    """
    assert selection_key("lt", "standard", ["ab", "c"]) != selection_key(
        "lt", "standard", ["a", "bc"]
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a,b,c", ["a", "b", "c"]),
        (" a , b ", ["a", "b"]),
        ("a", ["a"]),
        ("", []),
        ("   ", []),
        (None, []),
        (float("nan"), []),
        ("a,,b", ["a", "b"]),
    ],
)
def test_parse_selection(raw: object, expected: list[str]) -> None:
    assert parse_selection(raw) == expected


def test_attach_rescored_averages_over_pc_seeds_not_over_cells() -> None:
    """A design repeated across cells gets ONE mean, joined to every cell.

    This is register §24's guard in code form: `one_shot` re-picks the same
    design, so the averaged value must be a property of the DESIGN. If the
    join instead averaged per cell, repeated cells would each look like an
    independent draw.
    """
    cells = pd.DataFrame(
        {
            "chamber": ["lt"] * 3,
            "configuration": ["standard"] * 3,
            "status": ["ok"] * 3,
            "chosen_experiments": ["a,b", "b,a", "a,c"],
            "f1": [0.10, 0.20, 0.30],
        }
    )
    key_ab = selection_key("lt", "standard", ["a", "b"])
    key_ac = selection_key("lt", "standard", ["a", "c"])
    rescored = pd.DataFrame(
        {
            SELECTION_KEY_COLUMN: [key_ab, key_ab, key_ac, key_ac],
            "pc_seed": [0, 1, 0, 1],
            "f1": [0.10, 0.30, 0.50, 0.70],
            "f1_skeleton": [0.15, 0.35, 0.55, 0.75],
            "f1_core": [0.20, 0.40, 0.60, 0.80],
        }
    )
    out = attach_rescored(cells, rescored)
    # Cells 0 and 1 bought the same set in different order -> same design mean.
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.20)
    assert out.loc[1, "f1_rescored"] == pytest.approx(0.20)
    assert out.loc[2, "f1_rescored"] == pytest.approx(0.60)
    assert list(out["n_pc_seeds"]) == [2, 2, 2]
    # The undirected companion must be averaged the same way, per DESIGN.
    assert out.loc[0, "f1_skeleton_rescored"] == pytest.approx(0.25)
    assert out.loc[2, "f1_skeleton_rescored"] == pytest.approx(0.65)
    assert out.loc[0, "f1_core_rescored"] == pytest.approx(0.30)
    # The original column must survive untouched for comparison.
    assert list(out["f1"]) == [0.10, 0.20, 0.30]


def test_attach_rescored_leaves_unscorable_cells_null() -> None:
    """A cell with no recorded buy cannot be re-scored and must not guess."""
    cells = pd.DataFrame(
        {
            "chamber": ["lt"],
            "configuration": ["standard"],
            "status": ["ok"],
            "chosen_experiments": [""],
            "f1": [0.4],
        }
    )
    out = attach_rescored(
        cells,
        pd.DataFrame({SELECTION_KEY_COLUMN: [], "pc_seed": [], "f1": [], "f1_skeleton": []}),
    )
    assert out[SELECTION_KEY_COLUMN].isna().all()
    assert out["f1_rescored"].isna().all()


def test_lt_case_study_nodes_are_the_published_twenty() -> None:
    """Pinned against the chambers' own notebook, not re-derived.

    The list is a citation, so it must not drift with our node set. A test
    that recomputed it from the ground truth would pass whatever we changed
    it to, which is the failure mode the register keeps recording.
    """
    assert len(LT_CASE_STUDY_NODES) == 20
    assert len(set(LT_CASE_STUDY_NODES)) == 20
    # The three families the case study keeps, and the ones it drops.
    assert "current" in LT_CASE_STUDY_NODES
    assert "angle_1" in LT_CASE_STUDY_NODES
    for dropped in ("t_ir_1", "osr_c", "v_angle_1", "diode_ir_1"):
        assert dropped not in LT_CASE_STUDY_NODES


def test_attach_rescored_tolerates_a_missing_optional_metric() -> None:
    """A frame written before `f1_skeleton` existed must still join.

    `DataFrame.agg` raises `KeyError` on a named column it cannot find, so
    hard-coding every metric would make the joiner refuse older outputs —
    exactly the incompatibility this project keeps hitting when a column is
    added mid-milestone.
    """
    cells = pd.DataFrame(
        {
            "chamber": ["lt"],
            "configuration": ["standard"],
            "status": ["ok"],
            "chosen_experiments": ["a,b"],
            "f1": [0.4],
        }
    )
    key = selection_key("lt", "standard", ["a", "b"])
    legacy = pd.DataFrame({SELECTION_KEY_COLUMN: [key, key], "pc_seed": [0, 1], "f1": [0.2, 0.4]})
    out = attach_rescored(cells, legacy)
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.3)
    assert "f1_skeleton_rescored" not in out.columns
