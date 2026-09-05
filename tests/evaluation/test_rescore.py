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
    DESIGN_KEY_COLUMN,
    LT_CASE_STUDY_NODES,
    SELECTION_KEY_COLUMN,
    attach_rescored,
    design_key,
    parse_selection,
    rescore_selections,
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

    "Same design" means the same buy in the same ORDER. An earlier version of
    this test asserted that `a,b` and `b,a` share a mean, which encoded the
    ordering defect as intended behaviour — pooling concatenates in sequence,
    so those two cells handed PC different rows and really did score
    differently.
    """
    cells = pd.DataFrame(
        {
            "chamber": ["lt"] * 4,
            "configuration": ["standard"] * 4,
            "status": ["ok"] * 4,
            "chosen_experiments": ["a,b", "a,b", "b,a", "a,c"],
            "f1": [0.10, 0.20, 0.25, 0.30],
        }
    )
    d_ab = design_key("lt", "standard", ["a", "b"])
    d_ba = design_key("lt", "standard", ["b", "a"])
    d_ac = design_key("lt", "standard", ["a", "c"])
    rescored = pd.DataFrame(
        {
            DESIGN_KEY_COLUMN: [d_ab, d_ab, d_ba, d_ba, d_ac, d_ac],
            SELECTION_KEY_COLUMN: [selection_key("lt", "standard", ["a", "b"])] * 4
            + [selection_key("lt", "standard", ["a", "c"])] * 2,
            "pc_seed": [0, 1, 0, 1, 0, 1],
            "f1": [0.10, 0.30, 0.90, 0.94, 0.50, 0.70],
            "f1_skeleton": [0.15, 0.35, 0.95, 0.99, 0.55, 0.75],
            "f1_core": [0.20, 0.40, 0.10, 0.14, 0.60, 0.80],
        }
    )
    out = attach_rescored(cells, rescored)
    # Cells 0 and 1 are the same ordered buy -> one shared design mean.
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.20)
    assert out.loc[1, "f1_rescored"] == pytest.approx(0.20)
    # Cell 2 is the same SET in the other order -> its own mean.
    assert out.loc[2, "f1_rescored"] == pytest.approx(0.92)
    assert out.loc[3, "f1_rescored"] == pytest.approx(0.60)
    assert list(out["n_pc_seeds"]) == [2, 2, 2, 2]
    # ...while still clustering as one buy for analysis.
    assert out.loc[0, SELECTION_KEY_COLUMN] == out.loc[2, SELECTION_KEY_COLUMN]
    # The undirected companion must be averaged the same way, per DESIGN.
    assert out.loc[0, "f1_skeleton_rescored"] == pytest.approx(0.25)
    assert out.loc[3, "f1_skeleton_rescored"] == pytest.approx(0.65)
    assert out.loc[0, "f1_core_rescored"] == pytest.approx(0.30)
    # The original column must survive untouched for comparison.
    assert list(out["f1"]) == [0.10, 0.20, 0.25, 0.30]


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
        pd.DataFrame(
            {
                DESIGN_KEY_COLUMN: [],
                SELECTION_KEY_COLUMN: [],
                "pc_seed": [],
                "f1": [],
                "f1_skeleton": [],
            }
        ),
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
    legacy = pd.DataFrame(
        {
            DESIGN_KEY_COLUMN: [design_key("lt", "standard", ["a", "b"])] * 2,
            SELECTION_KEY_COLUMN: [key, key],
            "pc_seed": [0, 1],
            "f1": [0.2, 0.4],
        }
    )
    out = attach_rescored(cells, legacy)
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.3)
    assert "f1_skeleton_rescored" not in out.columns


def _cells_and_rescored(
    source_backend: str | None, rescore_backend: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ok cell plus its two-seed re-scoring, backends set independently."""
    cells = pd.DataFrame(
        {
            "chamber": ["lt"],
            "configuration": ["standard"],
            "status": ["ok"],
            "chosen_experiments": ["a,b"],
            "f1": [0.4],
            "blas_backend": [source_backend],
        }
    )
    key = selection_key("lt", "standard", ["a", "b"])
    dkey = design_key("lt", "standard", ["a", "b"])
    rescored = pd.DataFrame(
        {
            DESIGN_KEY_COLUMN: [dkey, dkey],
            SELECTION_KEY_COLUMN: [key, key],
            "pc_seed": [0, 1],
            "f1": [0.2, 0.4],
            "blas_backend": [rescore_backend, rescore_backend],
            "platform_tag": ["Darwin-arm64", "Darwin-arm64"],
        }
    )
    return cells, rescored


def test_attach_rescored_refuses_a_cross_backend_join() -> None:
    """A backend mismatch is an error, not a warning.

    `f1_rescored` substitutes for `f1` downstream, and PC forks structurally
    on a ~1e-10 linear-algebra difference, so the joined frame would carry
    Accelerate numbers under an OpenBLAS provenance stamp — the one column
    that exists to prevent exactly this mispooling.
    """
    cells, rescored = _cells_and_rescored("scipy-openblas", "accelerate")
    with pytest.raises(ValueError, match="not comparable"):
        attach_rescored(cells, rescored)


def test_attach_rescored_allows_a_declared_cross_backend_join() -> None:
    """The escape hatch is explicit, and records which backend scored."""
    cells, rescored = _cells_and_rescored("scipy-openblas", "accelerate")
    out = attach_rescored(cells, rescored, allow_backend_mismatch=True)
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.3)
    assert out.loc[0, "blas_backend"] == "scipy-openblas"
    assert out.loc[0, "rescore_blas_backend"] == "accelerate"
    assert out.loc[0, "rescore_platform_tag"] == "Darwin-arm64"


def test_attach_rescored_joins_when_backends_agree() -> None:
    cells, rescored = _cells_and_rescored("scipy-openblas", "scipy-openblas")
    out = attach_rescored(cells, rescored)
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.3)
    assert out.loc[0, "rescore_blas_backend"] == "scipy-openblas"


def test_attach_rescored_joins_when_the_source_declares_no_backend() -> None:
    """Pre-provenance files have no `blas_backend`; they must still join.

    The guard can only fire on evidence of a mismatch. A null column is
    ignorance, not disagreement, and raising on it would lock out every file
    written before the provenance columns existed.
    """
    cells, rescored = _cells_and_rescored(None, "accelerate")
    out = attach_rescored(cells, rescored)
    assert out.loc[0, "f1_rescored"] == pytest.approx(0.3)


def test_rescore_stamps_the_executing_machine_not_the_source() -> None:
    """`rescore_selections` must report the backend it actually ran under."""
    from evaluation.chamber_pipeline.inference import runtime_fingerprint

    expected = runtime_fingerprint()
    cells = pd.DataFrame(
        {
            "chamber": ["lt"],
            "configuration": ["standard"],
            "status": ["ok"],
            "chosen_experiments": ["uniform_reference"],
            "blas_backend": ["a-backend-that-cannot-be-this-machine"],
        }
    )
    out = rescore_selections(cells, n_pc_seeds=1, progress_every=0)
    assert set(out["blas_backend"]) == {expected["blas"]}
    assert set(out["platform_tag"]) == {expected["platform"]}


def test_rescore_output_is_identical_serial_and_parallel() -> None:
    """`--max-workers` must change wall time and nothing else.

    Two ways a worker pool could corrupt this, both guarded in the
    implementation and both checked here: rows reassembled in completion
    order rather than submission order, and a per-worker difference in the
    numbers themselves. The second is why the flag was probed against BLAS
    thread count before it shipped — a reassociated threaded reduction would
    fork PC's accept/reject cascade, which is the documented failure mode
    behind the `blas_backend` column.
    """
    cells = pd.DataFrame(
        {
            "chamber": ["lt", "lt", "lt"],
            "configuration": ["standard"] * 3,
            "status": ["ok"] * 3,
            "chosen_experiments": [
                "uniform_reference,uniform_red_strong",
                "uniform_reference,uniform_green_strong",
                "uniform_blue_strong,uniform_pol_1_strong",
            ],
        }
    )
    serial = rescore_selections(cells, n_pc_seeds=2, progress_every=0, max_workers=1)
    parallel = rescore_selections(cells, n_pc_seeds=2, progress_every=0, max_workers=3)
    pd.testing.assert_frame_equal(serial, parallel)
    # Order is the submission order, so the two frames agreeing is not
    # vacuous: it pins the sequence as well as the values.
    assert list(serial[SELECTION_KEY_COLUMN]) == [
        k
        for k in (
            selection_key("lt", "standard", ["uniform_reference", "uniform_red_strong"]),
            selection_key("lt", "standard", ["uniform_reference", "uniform_green_strong"]),
            selection_key("lt", "standard", ["uniform_blue_strong", "uniform_pol_1_strong"]),
        )
        for _ in range(2)
    ]


def test_the_same_buy_in_two_orders_is_scored_twice() -> None:
    """Order changes the pool, so it must not be collapsed away.

    `pool_experiment_data` concatenates in sequence and PC subsamples 300
    rows under a seed, so `[a, b, c]` and `[c, b, a]` hand PC different rows
    and score differently — measured on real LT data, 0.133 against 0.105 at
    the same seed. Keying the WORK by the set (as this module first did)
    re-scores one ordering and silently hands its value to the other cell.
    Caught by measurement, not review: of 296 cells checkable against their
    production score, the 256 whose design was recorded in one order matched
    256/256, while the 40 in multi-order designs forked at 45%.

    `selection_key` stays order-insensitive on purpose — clustering for
    analysis is about what was bought, not the sequence — so both keys are
    asserted here to keep them from drifting into each other.
    """
    names = ["uniform_reference", "uniform_red_strong", "uniform_green_strong"]
    cells = pd.DataFrame(
        {
            "chamber": ["lt", "lt"],
            "configuration": ["standard", "standard"],
            "status": ["ok", "ok"],
            "chosen_experiments": [",".join(names), ",".join(reversed(names))],
        }
    )
    rescored = rescore_selections(cells, n_pc_seeds=2, progress_every=0)

    assert rescored[DESIGN_KEY_COLUMN].nunique() == 2, "each ordering is its own unit of work"
    assert rescored[SELECTION_KEY_COLUMN].nunique() == 1, "but both are the same buy"

    joined = attach_rescored(cells, rescored)
    forward, backward = joined.loc[0, "f1_rescored"], joined.loc[1, "f1_rescored"]
    # Each cell gets the mean over ITS OWN ordering. Were the join keyed by
    # the set, both would receive the pooled mean of all four scorings and
    # these two numbers would be equal.
    assert forward != backward
    for row, key in ((0, names), (1, list(reversed(names)))):
        own = rescored[rescored[DESIGN_KEY_COLUMN] == design_key("lt", "standard", key)]
        assert joined.loc[row, "f1_rescored"] == pytest.approx(own["f1"].mean())
