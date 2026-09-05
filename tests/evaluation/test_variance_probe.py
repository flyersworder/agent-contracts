"""Tests for the selection/measurement variance decomposition.

`decompose` is the only place in the pipeline that makes a claim about
*why* cells differ, so its arithmetic is load-bearing for the interpretive
section it feeds. These tests build frames whose true components are known
by construction and check recovery, rather than asserting that the output
is merely finite or ordered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.chamber_pipeline.variance_probe import decompose


def _frame(values: dict[int, list[float]], budget_k: int = 6) -> pd.DataFrame:
    """Build a probe frame from {selection_seed: [f1 per pc_seed]}."""
    rows = []
    for sel, f1s in values.items():
        for pc_seed, f1 in enumerate(f1s):
            rows.append(
                {
                    "budget_k": budget_k,
                    "selection_seed": sel,
                    "pc_seed": pc_seed,
                    "f1": f1,
                    "n_rows_pooled": 100,
                }
            )
    return pd.DataFrame(rows)


def test_no_measurement_noise_attributes_everything_to_selection() -> None:
    """Identical scores within a selection => within-sd is exactly zero."""
    frame = _frame({0: [0.10, 0.10, 0.10], 1: [0.20, 0.20, 0.20], 2: [0.30] * 3})
    out = decompose(frame).iloc[0]
    assert out.sd_within_pc == pytest.approx(0.0, abs=1e-12)
    # With zero within-variance the correction subtracts nothing, so the
    # selection sd is the plain sd of the three group means.
    assert out.sd_selection == pytest.approx(pd.Series([0.1, 0.2, 0.3]).std(ddof=1))


def test_identical_selection_means_yield_zero_selection_variance() -> None:
    """Every selection scoring the same on average => nothing to attribute.

    This is the k=M invariant in miniature: when the buy cannot differ, the
    decomposition must return 0.0 rather than a small positive artifact of
    the measurement noise leaking into the group means.
    """
    frame = _frame({0: [0.1, 0.3], 1: [0.3, 0.1], 2: [0.2, 0.2]})
    out = decompose(frame).iloc[0]
    assert out.sd_within_pc > 0.05  # noise is real and large here
    assert out.sd_selection == pytest.approx(0.0, abs=1e-12)


def test_bias_correction_is_unbiased_under_pure_noise() -> None:
    """With ZERO true between-variance, the correction must center on zero.

    Each group mean over m draws carries sigma_within^2 / m of measurement
    noise, so the RAW between-estimate reads pure noise as a selection
    effect — the exact error this probe exists to avoid. The correction
    cannot null that out sample-by-sample (the residual is itself random),
    so the claim being tested is unbiasedness ACROSS replications, on the
    unclamped variance. Testing the clamped value here would fail for a
    correct implementation: max(0, x) on a zero-mean quantity is positive
    in expectation.
    """
    rng = np.random.default_rng(20260831)
    residuals = []
    for _ in range(200):
        frame = _frame({sel: list(rng.normal(0.4, 0.05, 12)) for sel in range(8)})
        out = decompose(frame).iloc[0]
        assert out.sd_selection < out.sd_between_raw, "correction must shrink"
        residuals.append(out.sd_between_raw**2 - out.sd_within_pc**2 / out.n_pc_seeds)
    mean_residual = float(np.mean(residuals))
    # Scale reference: the bias the correction removes is sigma^2/m.
    removed = 0.05**2 / 12
    assert abs(mean_residual) < 0.15 * removed, (
        f"corrected variance should center on 0, got {mean_residual:.2e} "
        f"against a removed bias of {removed:.2e}"
    )


def test_recovers_known_components() -> None:
    """Build in a true between-sd and within-sd; check both come back."""
    rng = np.random.default_rng(7)
    true_between, true_within = 0.08, 0.03
    offsets = rng.normal(0.0, true_between, 60)
    frame = _frame(
        {sel: list(0.4 + off + rng.normal(0.0, true_within, 40)) for sel, off in enumerate(offsets)}
    )
    out = decompose(frame).iloc[0]
    assert out.sd_within_pc == pytest.approx(true_within, rel=0.15)
    assert out.sd_selection == pytest.approx(true_between, rel=0.25)


def test_correction_never_returns_negative_or_nan() -> None:
    """Clamping is required: between_raw^2 - within^2/m can go negative."""
    frame = _frame({0: [0.0, 1.0], 1: [0.0, 1.0]})
    out = decompose(frame).iloc[0]
    assert out.sd_selection >= 0.0
    assert not np.isnan(out.sd_selection)


def test_budgets_are_decomposed_independently() -> None:
    """A quiet budget must not borrow variance from a noisy one."""
    quiet = _frame({0: [0.5, 0.5], 1: [0.5, 0.5]}, budget_k=59)
    loud = _frame({0: [0.1, 0.5], 1: [0.9, 0.4]}, budget_k=6)
    out = decompose(pd.concat([quiet, loud])).set_index("budget_k")
    assert out.loc[59, "sd_total"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc[6, "sd_total"] > 0.2


def test_within_variance_uses_the_correct_degrees_of_freedom() -> None:
    """Pooled within-variance divides by (N - groups), not N.

    Asserted deterministically because the two divisors differ by only a
    few percent on large frames — a tolerance loose enough to accommodate
    sampling noise there would wave the bug through. With exactly two
    draws per group the divisors differ by a factor of two in variance
    (sqrt(2) in sd), which no reasonable tolerance can absorb.

    Construction: each group is {x, x + d}, so each contributes d^2 / 2 to
    the residual sum of squares against N - G = G degrees of freedom, and
    the pooled within-sd is exactly d / sqrt(2).
    """
    d = 0.2
    frame = _frame({sel: [0.1 * sel, 0.1 * sel + d] for sel in range(6)})
    out = decompose(frame).iloc[0]
    assert out.sd_within_pc == pytest.approx(d / np.sqrt(2), rel=1e-9)
