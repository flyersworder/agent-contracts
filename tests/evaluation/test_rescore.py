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
        }
    )
    out = attach_rescored(cells, rescored)
    # Cells 0 and 1 bought the same set in different order -> same design mean.
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.20)
    assert out.loc[1, "f1_rescored"] == pytest.approx(0.20)
    assert out.loc[2, "f1_rescored"] == pytest.approx(0.60)
    assert list(out["n_pc_seeds"]) == [2, 2, 2]
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
    out = attach_rescored(cells, pd.DataFrame({SELECTION_KEY_COLUMN: [], "pc_seed": [], "f1": []}))
    assert out[SELECTION_KEY_COLUMN].isna().all()
    assert out["f1_rescored"].isna().all()
