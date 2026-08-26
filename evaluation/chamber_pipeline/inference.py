"""Causal-discovery inference helpers for the chamber pillar.

Currently wraps the PC algorithm from `causal-learn` so the Random and
LLM+PC variants in `agents.py` (and any future score-based variants) can
share a single conversion step from raw chamber measurements to a
directed adjacency-matrix DataFrame.

Data convention used here: chamber experiments record measurements of
all chamber variables plus an `intervention` meta-column logging which
variable was perturbed. We pool the rows from all spent experiments and
drop the meta-column before passing to PC. This is the simple "PC on
pooled interventional data" baseline. M3+ may swap in interventional-
aware methods (IPM, GIES, etc.) when method-axis breadth becomes the
primary contribution; for the AAMAS submission, PC-on-pool is the
single classical inference step shared across the LLM+PC and Random
variants per plan §5.

CPDAG → directed-adjacency convention:

    PC outputs a CPDAG (Markov equivalence class), not a DAG. For an
    undirected edge i — j, the data is consistent with both i → j and
    j → i. We convert by reporting *both* directions in the binary
    adjacency matrix, surfacing the agent's uncertainty to the SHD
    scorer rather than making an arbitrary orientation choice. Cost:
    exactly +1 false positive per undirected edge under the cell-wise
    SHD convention used by `evaluation.chamber_pipeline.scoring.shd`.

    causal-learn's `pc_graph[i, j]` represents the endpoint of the i—j
    edge **at node i** (NOT j — easy to misread). Values:

        pc_graph[i, j] =  1  →  arrowhead at i
        pc_graph[i, j] = -1  →  tail at i
        pc_graph[i, j] =  0  →  no edge

    So a definite arrow i → j (head at j, tail at i) is encoded as
    `pc_graph[j, i] = 1` (head at j) AND `pc_graph[i, j] = -1` (tail at i).
    An undirected edge i — j is `pc_graph[i, j] = -1` AND
    `pc_graph[j, i] = -1` (tails at both ends).

    Therefore:
        `adj[i, j] = 1` iff
            `pc_graph[j, i] == 1`                                      (arrowhead at j from i)
            OR (`pc_graph[i, j] == -1` AND `pc_graph[j, i] == -1`)     (undirected — both dirs)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The exact error phrase causal-learn's Fisher-Z raises on a singular
# sub-correlation matrix (verified against causallearn==0.1.4 at
# .venv/lib/.../causallearn/utils/cit.py:218 and :482) and the phrase
# numpy raises from `np.linalg.inv` on a singular input matrix.
# Both are matched as tight substring filters so the fallback path
# doesn't accidentally swallow unrelated future errors that happen to
# mention "singular" elsewhere in their message (e.g., a hypothetical
# "singular value decomposition did not converge" from a different code
# path in numpy or causal-learn).
_FISHERZ_SINGULAR_PHRASE = "correlation matrix is singular"
_LINALG_SINGULAR_PHRASE = "singular matrix"

# `causal-learn` is part of the chambers extra; importing it lazily so
# tests of unrelated chamber-pipeline pieces don't fail at collection
# time when the extra is uninstalled.
try:
    from causallearn.search.ConstraintBased.PC import pc as _causallearn_pc

    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    _causallearn_pc = None  # type: ignore[assignment]


def pool_experiment_data(experiment_dfs: list[pd.DataFrame], node_names: list[str]) -> pd.DataFrame:
    """Concatenate per-experiment rows and project to ground-truth nodes.

    Args:
        experiment_dfs: One DataFrame per spent experiment, in spending
            order. Each must include all columns in `node_names` (extra
            columns are dropped).
        node_names: The chamber's ground-truth node names. Used to align
            and order columns so PC's index-by-position output can be
            mapped back to node names.

    Returns:
        Single DataFrame with rows from all experiments, columns ordered
        as `node_names`. No meta-columns.

    Raises:
        ValueError: If any experiment is missing a required node column,
            or if `experiment_dfs` is empty.
    """
    if not experiment_dfs:
        raise ValueError("experiment_dfs is empty — at least one experiment is required")

    missing_per_df: list[set[str]] = []
    for df in experiment_dfs:
        missing = set(node_names) - set(df.columns)
        if missing:
            missing_per_df.append(missing)

    if missing_per_df:
        # Report only the first to keep the message tight.
        raise ValueError(
            f"Experiment DataFrames are missing required node columns; "
            f"first offender missing: {sorted(missing_per_df[0])}"
        )

    # Project each DataFrame to just the node columns (drop intervention
    # and any other meta-columns) and stack.
    projected = [df[node_names].copy() for df in experiment_dfs]
    pooled = pd.concat(projected, axis=0, ignore_index=True)
    return pooled


def cpdag_to_directed_adjacency(pc_graph: np.ndarray, node_names: list[str]) -> pd.DataFrame:
    """Convert a `causal-learn` PC `graph` matrix into a directed adjacency DataFrame.

    See module docstring for the convention. In short: undirected edges
    become bidirectional in the output DataFrame, definite arrows stay
    in their oriented direction.

    Args:
        pc_graph: The 2-D ndarray returned as `result.G.graph` from
            `causallearn.search.ConstraintBased.PC.pc(...)`. Values are
            in `{-1, 0, 1}`.
        node_names: Ordered list of variable names; must match the
            column ordering of the data passed to `pc(...)`.

    Returns:
        Square DataFrame with rows/columns = `node_names`, entries in
        `{0, 1}`, diagonal forced to 0.
    """
    if pc_graph.shape != (len(node_names), len(node_names)):
        raise ValueError(
            f"pc_graph shape {pc_graph.shape} does not match len(node_names)={len(node_names)}"
        )

    # Definite arrowhead at j coming from i: pc_graph[j, i] == 1.
    arrowhead_at_j_from_i = pc_graph.T == 1
    # Undirected i — j: tail at both endpoints. Report both directions
    # so the agent's lack-of-orientation surfaces to the SHD scorer.
    undirected = (pc_graph == -1) & (pc_graph.T == -1)

    adj = (arrowhead_at_j_from_i | undirected).astype(int)
    np.fill_diagonal(adj, 0)
    return pd.DataFrame(adj, index=node_names, columns=node_names)


# Default row cap for run_pc. Fisher-Z is asymptotic in N — past a few
# hundred samples on typical chamber-grade SNR, additional rows slow PC
# without improving inference quality. Each chamber experiment is 1000
# rows of largely-redundant samples (same intervention, slight noise
# variation), so subsampling to a few hundred discards minimal signal.
# At unbounded N, PC at 38 nodes by 2000 rows can take >1h; at N=300
# it's seconds. Set max_rows=None to disable.
DEFAULT_MAX_ROWS = 300

# Pearson |r| above which two columns are treated as the same variable.
#
# Motivated by the WT chamber's four barometers -- pressure_upwind,
# pressure_downwind, pressure_ambient and pressure_intake -- which in the
# `standard` configuration all read essentially ambient atmospheric
# pressure. Measured over the full wt_validate_v1 menu, all six pairs sit
# above r=0.9998 and NONE of the six is a true edge. Four variables spanning
# roughly one dimension leaves the correlation matrix rank-deficient by
# three (cond(R) ~ 1e7), and Fisher-Z's sub-matrix inversion then raises --
# aborting the entire PC run and returning all-zeros for all 32 nodes.
#
# Dropping the redundant columns and padding them back with zeros is the
# same policy this module already applies to zero-variance columns, for the
# same reason: a variable numerically indistinguishable from another
# carries no independent signal, so "no claim" is the honest output. The
# difference it makes is scope -- the loss becomes the edges incident to
# the dropped nodes instead of the whole graph.
#
# 0.999 rather than something closer to 1.0 because float64 Fisher-Z fails
# well before exact collinearity; the WT clique sits at 0.9998-0.99998.
DEFAULT_COLLINEARITY_THRESHOLD = 0.999


def select_noncollinear_columns(
    data: pd.DataFrame, columns: list[str], threshold: float
) -> tuple[list[str], list[str]]:
    """Greedily keep columns that are not near-duplicates of a kept column.

    Walks `columns` in order and keeps each one whose absolute Pearson
    correlation with every already-kept column is below `threshold`. Order
    matters and is therefore taken from the caller's node ordering, never
    from a set or dict iteration, so the choice of which member of a
    duplicate group survives is reproducible across runs and platforms.

    Args:
        data: Frame containing at least `columns`; already subsampled and
            already filtered for zero variance by the caller.
        columns: Candidate column names, in the order that decides which
            member of a near-duplicate group is kept.
        threshold: Absolute-correlation cutoff. See
            `DEFAULT_COLLINEARITY_THRESHOLD`.

    Returns:
        `(kept, dropped)`, together a partition of `columns`.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for name in columns:
        col = data[name].to_numpy(dtype=float)
        redundant = False
        for keeper in kept:
            other = data[keeper].to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.corrcoef(col, other)[0, 1]
            # NaN means one side had no variance, which the caller already
            # excluded; treat it as "not redundant" rather than silently
            # dropping a column for an unrelated reason.
            if np.isfinite(r) and abs(r) >= threshold:
                redundant = True
                break
        (dropped if redundant else kept).append(name)
    return kept, dropped


def pc_call_defaults() -> dict[str, Any]:
    """The values `run_pc` will actually use when called without them.

    Read off the BOUND signature, not the module constants. Python
    evaluates default arguments at def time, so reassigning
    `DEFAULT_MAX_ROWS` after import moves the constant while leaving
    `run_pc` untouched -- provenance stamped from the constant would then
    describe a configuration that never ran. Every agent calls `run_pc`
    without `max_rows` or `collinearity_threshold`, so these defaults are
    what produced the graph.
    """
    params = inspect.signature(run_pc).parameters
    return {name: p.default for name, p in params.items() if p.default is not p.empty}


def run_pc(
    pooled_data: pd.DataFrame,
    node_names: list[str],
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    show_progress: bool = False,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    seed: int = 0,
    collinearity_threshold: float | None = DEFAULT_COLLINEARITY_THRESHOLD,
    **pc_kwargs: Any,
) -> pd.DataFrame:
    """Run PC on pooled chamber data, return directed-adjacency DataFrame.

    Args:
        pooled_data: Output of `pool_experiment_data(...)`. Must have
            columns matching `node_names` in the same order.
        node_names: Ordered list of node names — the chamber's
            ground-truth row/column index.
        alpha: Significance level for the independence test. Default
            0.05 matches `causal-learn`'s default and standard practice.
        indep_test: Independence test name passed to `causal-learn`.
            `"fisherz"` (default) is the linear-Gaussian-appropriate
            choice for chamber data.
        show_progress: If True, `causal-learn`'s tqdm progress bars
            are printed. Default False because they're verbose in
            test output.
        max_rows: If set and `pooled_data` has more rows than this,
            subsample uniformly (with `seed`-controlled RNG) before
            calling PC. See `DEFAULT_MAX_ROWS` rationale at module
            level. Pass None to use all rows (slow at 38 nodes).
        seed: RNG seed for the subsampling. Has no effect when
            `max_rows is None` or `len(pooled_data) <= max_rows`.
        collinearity_threshold: Absolute-correlation cutoff above which a
            column is treated as a duplicate of an earlier one and dropped
            (padded back with zeros). Pass None to disable and restore the
            pre-2026-08-25 behaviour, where a numerically duplicate pair
            aborted the whole run. See `DEFAULT_COLLINEARITY_THRESHOLD`.
        **pc_kwargs: Forwarded to `causallearn.search.ConstraintBased.PC.pc`.

    Returns:
        Directed-adjacency DataFrame (see `cpdag_to_directed_adjacency`).

    Raises:
        ImportError: If `causal-learn` is not installed.
        ValueError: If `pooled_data` columns don't match `node_names`.
    """
    if not CAUSAL_LEARN_AVAILABLE:
        raise ImportError(
            "causal-learn is required for PC inference. "
            "Install with: pip install 'ai-agent-contracts[chambers]'"
        )

    if list(pooled_data.columns) != list(node_names):
        raise ValueError(
            "pooled_data columns must match node_names in order — "
            "use pool_experiment_data() to build the input"
        )

    if max_rows is not None and len(pooled_data) > max_rows:
        pooled_data = pooled_data.sample(n=max_rows, random_state=seed)

    # Drop zero-variance columns. PC's Fisher-Z test produces a singular
    # correlation matrix on constant columns and hangs (or returns
    # garbage). This is structurally inevitable in chamber data: each
    # experiment perturbs one variable, holding others constant, so any
    # variable not downstream of a perturbed variable in the spent set
    # has zero variance in the pooled view. We drop them, run PC on the
    # remaining variables, and pad the result back to full node-set
    # shape with zeros in the dropped rows/cols. Zero = "no signal, no
    # claim" — the honest representation when the agent hasn't
    # perturbed something that drives that variable.
    variances = pooled_data.var()
    valid_cols = [n for n in node_names if variances.get(n, 0.0) > 1e-12]

    if not valid_cols:
        # Nothing has signal — return all-zeros on the full node set.
        return pd.DataFrame(
            np.zeros((len(node_names), len(node_names)), dtype=int),
            index=node_names,
            columns=node_names,
        )

    if collinearity_threshold is not None and len(valid_cols) > 1:
        valid_cols, collinear_dropped = select_noncollinear_columns(
            pooled_data, valid_cols, collinearity_threshold
        )
        if collinear_dropped:
            # Logged, not merely returned, because the orchestrator scrapes
            # this per cell (`_PcCollinearHandler`). A degradation path whose
            # rate can vary with the experiment's independent variable and
            # that leaves no trace is how a harness ends up measuring itself.
            logger.warning(
                "PC dropped %d collinear column(s) at |r|>=%.4g: %s",
                len(collinear_dropped),
                collinearity_threshold,
                ", ".join(collinear_dropped),
            )
        if not valid_cols:
            return pd.DataFrame(
                np.zeros((len(node_names), len(node_names)), dtype=int),
                index=node_names,
                columns=node_names,
            )

    valid_data = pooled_data[valid_cols]
    data_array = valid_data.to_numpy(dtype=float)
    try:
        result = _causallearn_pc(
            data_array,
            alpha=alpha,
            indep_test=indep_test,
            show_progress=show_progress,
            verbose=False,
            **pc_kwargs,
        )
    except (np.linalg.LinAlgError, ValueError) as exc:
        # Fisher-Z's CI step inverts sub-correlation matrices, which can
        # be singular even when no full column is constant. This happens
        # on highly-collinear pooled chamber data (e.g., when selected
        # experiments perturb downstream variables that are deterministic
        # functions of each other in the spent set). The honest fallback
        # is "PC made no claim" -> all-zeros adjacency on the full node
        # set.
        #
        # Both error types are filtered on a specific phrase so unrelated
        # bugs (a non-singular ValueError from causal-learn input
        # validation, or a non-singular LinAlgError from a different
        # numpy code path) still surface as exceptions rather than
        # silent zero-adjacency results. The two phrases cover both
        # legs of the failure: causal-learn raises ValueError with
        # `_FISHERZ_SINGULAR_PHRASE` when its `np.linalg.inv` fails;
        # numpy may also surface the underlying `LinAlgError("Singular
        # matrix")` directly if the inversion happens outside
        # causal-learn's wrapper.
        msg = str(exc).lower()
        is_known_singular_failure = (
            isinstance(exc, ValueError) and _FISHERZ_SINGULAR_PHRASE in msg
        ) or (isinstance(exc, np.linalg.LinAlgError) and _LINALG_SINGULAR_PHRASE in msg)
        if not is_known_singular_failure:
            raise
        # Log for M5 sweep visibility — degenerate-cell counts are
        # publishable as a finding ("X% of cells under variant V at
        # budget k hit PC degeneracy"). Without the warning, the count
        # is invisible at run-time and only inferable from output
        # all-zeros, which is ambiguous (could also be "no signal").
        logger.warning(
            "PC inference fell back to all-zeros adjacency on %d-node input (%s): %s",
            len(node_names),
            type(exc).__name__,
            str(exc).strip(),
        )
        return pd.DataFrame(
            np.zeros((len(node_names), len(node_names)), dtype=int),
            index=node_names,
            columns=node_names,
        )
    valid_adj = cpdag_to_directed_adjacency(result.G.graph, valid_cols)

    if len(valid_cols) == len(node_names):
        return valid_adj

    # Pad back to full node set: zeros for dropped rows/columns.
    full = pd.DataFrame(
        np.zeros((len(node_names), len(node_names)), dtype=int),
        index=node_names,
        columns=node_names,
    )
    full.loc[valid_cols, valid_cols] = valid_adj.values
    return full


__all__ = [
    "CAUSAL_LEARN_AVAILABLE",
    "cpdag_to_directed_adjacency",
    "pool_experiment_data",
    "run_pc",
]
