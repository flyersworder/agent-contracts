"""Ground-truth scoring functions for the Causal Chamber pillar.

Pure functions that compare a predicted causal graph against a known
ground-truth graph (from `causalchamber.ground_truth.graph(...)`). Used by
the chamber-pipeline orchestrator to score each run; also called by the
M2 smoke test in `tests/integrations/test_causalchamber.py`.

These live in the pipeline (not in `src/agent_contracts/validators/`) per
the M1 Q2 decision in `docs/causal_chamber_M1_decisions.md` §3.

All adjacency-matrix inputs are pandas DataFrames whose row and column
indices are node names. Values are interpreted as **binary edge presence**:
any nonzero entry is treated as an edge, zero as no-edge. Self-loops (the
diagonal) are scored the same as off-diagonal cells — that matches the
ground-truth convention used by the Causal Chambers package.

Convention for graph alignment: predicted and reference must cover the
same node set. Predicted is reindexed to match reference's ordering
before comparison; if any node is missing on either side, we raise.
This catches the classic silent bug where two equal-up-to-permutation
graphs score as completely different.
"""

import pandas as pd


def _align_to_reference(
    predicted: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex `predicted` to match `reference`'s node order; sanity-check shapes.

    Args:
        predicted: Agent's reported adjacency matrix.
        reference: Ground-truth adjacency matrix.

    Returns:
        (predicted_aligned, reference) — both with identical row/column order.

    Raises:
        ValueError: If reference is not square, or if predicted is missing
            any node present in reference (or vice versa), or if any input
            is not a DataFrame.
    """
    if not isinstance(predicted, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
        raise ValueError("Both predicted and reference must be pandas DataFrames")

    if reference.shape[0] != reference.shape[1]:
        raise ValueError(f"Reference adjacency must be square, got shape {reference.shape}")

    ref_nodes = list(reference.index)
    if list(reference.columns) != ref_nodes:
        raise ValueError("Reference adjacency must have identical row/column ordering")

    pred_rows = set(predicted.index)
    pred_cols = set(predicted.columns)
    ref_set = set(ref_nodes)

    missing_in_pred = ref_set - pred_rows
    if missing_in_pred:
        raise ValueError(f"predicted is missing rows for nodes: {sorted(missing_in_pred)}")
    missing_in_pred = ref_set - pred_cols
    if missing_in_pred:
        raise ValueError(f"predicted is missing columns for nodes: {sorted(missing_in_pred)}")
    extra_in_pred = pred_rows - ref_set
    if extra_in_pred:
        raise ValueError(f"predicted has unknown rows not in reference: {sorted(extra_in_pred)}")

    return predicted.loc[ref_nodes, ref_nodes], reference


def shd(predicted: pd.DataFrame, reference: pd.DataFrame) -> int:
    """Structural Hamming Distance between predicted and reference adjacency.

    Cell-wise Hamming distance on binarized adjacency matrices. For each
    cell `(i, j)`, the contribution to SHD is 1 if `(predicted[i,j] != 0)`
    differs from `(reference[i,j] != 0)`, else 0. Under this definition a
    reversed edge contributes 2 (one for the missing forward edge, one for
    the spurious reverse edge), which matches the `causal-learn` library's
    convention.

    Args:
        predicted: Agent's reported adjacency matrix.
        reference: Ground-truth adjacency matrix from
            `causalchamber.ground_truth.graph(chamber, configuration)`.

    Returns:
        SHD as a non-negative integer. Bounded above by `n²` where `n` is
        the number of nodes.

    Raises:
        ValueError: If shapes/indices don't align — see `_align_to_reference`.
    """
    pred, ref = _align_to_reference(predicted, reference)
    pred_bin = pred.values != 0
    ref_bin = ref.values != 0
    return int((pred_bin != ref_bin).sum())


def f1_edges(predicted: pd.DataFrame, reference: pd.DataFrame) -> float:
    """F1 score on edge presence, treating each cell as a binary classification.

    A True Positive is a cell `(i, j)` where both predicted and reference
    have a nonzero entry. F1 = 2·precision·recall / (precision + recall).

    Edge cases:
        - If reference has no edges and predicted also has no edges, returns
          1.0 (perfect agreement on 'no edges').
        - If reference has edges but predicted has none, returns 0.0
          (zero recall).
        - If predicted has edges but reference has none, returns 0.0
          (zero precision).

    Args:
        predicted: Agent's reported adjacency matrix.
        reference: Ground-truth adjacency matrix.

    Returns:
        F1 score in [0, 1].

    Raises:
        ValueError: If shapes/indices don't align — see `_align_to_reference`.
    """
    pred, ref = _align_to_reference(predicted, reference)
    pred_bin = pred.values != 0
    ref_bin = ref.values != 0

    tp = int((pred_bin & ref_bin).sum())
    fp = int((pred_bin & ~ref_bin).sum())
    fn = int((~pred_bin & ref_bin).sum())

    # Perfect agreement on the empty graph.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    # Either side empty but not both — F1 is 0.
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def ci_coverage(
    intervals: dict[tuple[str, str], tuple[float, float]],
    reference: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calibration-coverage scoring on edge-presence confidence intervals.

    For each (i, j) pair the agent reports a (lower, upper) interval on
    edge-presence probability. The interval is said to *cover* the
    reference if the binary indicator `1{reference[i,j] != 0}` falls inside
    `[lower, upper]`. Coverage is the fraction of cells where this holds;
    mean width is the average `upper - lower` across all reported cells.

    Together they form a precision-coverage tradeoff: an agent can game
    coverage by reporting `[0, 1]` everywhere, which pegs coverage at 1.0
    but mean width at 1.0; useful agents push coverage near `1 - alpha`
    while keeping mean width small.

    Args:
        intervals: Dict mapping (row_node, col_node) to (lower, upper) on
            P(edge present). Pairs not in the dict are treated as
            unreported and excluded from both numerator and denominator.
        reference: Ground-truth adjacency matrix.
        alpha: Nominal miscoverage rate. Used only for documentation /
            target setting; the function returns observed coverage.

    Returns:
        (coverage, mean_interval_width) — both in [0, 1] when intervals
        are well-formed.

    Raises:
        ValueError: If reference is not square, any (i, j) refers to a
            node not in reference, or any (lower, upper) is malformed
            (lower > upper, or values outside [0, 1]).
    """
    if reference.shape[0] != reference.shape[1]:
        raise ValueError(f"Reference adjacency must be square, got shape {reference.shape}")

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not intervals:
        return 0.0, 0.0

    nodes = set(reference.index)
    covered = 0
    total_width = 0.0

    for (i, j), (lo, hi) in intervals.items():
        if i not in nodes or j not in nodes:
            raise ValueError(
                f"Interval references unknown node: ({i!r}, {j!r}) — "
                f"reference covers {sorted(nodes)[:5]}..."
            )
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"Malformed interval at ({i!r}, {j!r}): ({lo}, {hi}) — "
                "must satisfy 0 <= lower <= upper <= 1"
            )
        truth = float(reference.loc[i, j] != 0)
        if lo <= truth <= hi:
            covered += 1
        total_width += hi - lo

    n = len(intervals)
    return covered / n, total_width / n
