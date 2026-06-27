"""Tests for the Track-3 Salmon⊕SPM soft-vote ensemble core (CPU-only)."""

import numpy as np
import pytest

from trap.analysis.boundary_calibration import softmax
from trap.analysis.ensemble import (
    ensemble_soft_labels,
    optimize_ensemble_weight,
    resample_indices_to_priors,
    soft_vote,
)

pytestmark = pytest.mark.unit

CLASSES = ["L1HS", "L1PA", "NEGATIVE"]


def test_soft_vote_convex_and_normalised():
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.2, 0.7]])
    mid = soft_vote(a, b, 0.5)
    assert np.allclose(mid, [[0.4, 0.2, 0.4]])
    assert np.allclose(mid.sum(axis=1), 1.0)


def test_soft_vote_endpoints_are_single_models():
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.2, 0.7]])
    assert np.allclose(soft_vote(a, b, 1.0), a)
    assert np.allclose(soft_vote(a, b, 0.0), b)


def test_ensemble_beats_both_when_errors_decorrelate():
    # Model A nails L1PA, flips NEGATIVE; model B is the mirror. Each alone has a
    # broken class; the 0.5 soft vote should recover both -> higher macro-F1.
    n = 900
    labels = np.repeat([0, 1, 2], n // 3)
    probs_a = np.zeros((n, 3))
    probs_b = np.zeros((n, 3))
    for i, y in enumerate(labels):
        if y == 2:  # NEGATIVE: A wrong (says L1PA), B right
            probs_a[i] = [0.1, 0.6, 0.3]
            probs_b[i] = [0.05, 0.05, 0.9]
        elif y == 1:  # L1PA: A right, B wrong (says NEGATIVE)
            probs_a[i] = [0.05, 0.9, 0.05]
            probs_b[i] = [0.1, 0.3, 0.6]
        else:  # L1HS: both right
            probs_a[i] = [0.9, 0.05, 0.05]
            probs_b[i] = [0.9, 0.05, 0.05]
    from trap.analysis.boundary_calibration import macro_f1

    f1_a = macro_f1(labels, probs_a.argmax(1), 3)
    f1_b = macro_f1(labels, probs_b.argmax(1), 3)
    alpha, f1_ens = optimize_ensemble_weight(probs_a, probs_b, labels, 3)
    assert f1_ens > max(f1_a, f1_b)
    assert 0.0 <= alpha <= 1.0


def test_optimize_ensemble_weight_not_worse_than_good_model():
    # B is pure noise; A near-perfect. The tuned ensemble must not underperform A
    # and must not collapse onto the noise model (α≈0).
    from trap.analysis.boundary_calibration import macro_f1

    rng = np.random.default_rng(1)
    n = 600
    labels = rng.integers(0, 3, n)
    probs_a = np.full((n, 3), 0.02)
    probs_a[np.arange(n), labels] = 0.96
    probs_b = rng.dirichlet([1, 1, 1], size=n)
    alpha, f1_ens = optimize_ensemble_weight(probs_a, probs_b, labels, 3)
    assert f1_ens >= macro_f1(labels, probs_a.argmax(1), 3)
    assert alpha > 0.3


def test_resample_to_priors_matches_target_proportions():
    rng = np.random.default_rng(2)
    # Sample is L1-heavy (per-subfamily cap); priors want 89% NEGATIVE.
    labels = np.concatenate(
        [np.full(5000, 0), np.full(20000, 1), np.full(20000, 2)]
    )  # L1HS, L1PA, NEGATIVE
    priors = {"L1HS": 0.003, "L1PA": 0.104, "NEGATIVE": 0.893}
    idx = resample_indices_to_priors(labels, CLASSES, priors, rng)
    sub = labels[idx]
    frac_neg = (sub == 2).mean()
    frac_pa = (sub == 1).mean()
    assert frac_neg == pytest.approx(0.893, abs=0.02)
    assert frac_pa == pytest.approx(0.104, abs=0.02)


def test_ensemble_soft_labels_matches_manual_combine():
    la = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    lb = np.array([[0.5, 0.5, 0.0], [-1.0, 2.0, 0.0]])
    out = ensemble_soft_labels(la, lb, temperature_a=1.5, temperature_b=0.8, alpha=0.5)
    manual = soft_vote(softmax(la, 1.5), softmax(lb, 0.8), 0.5)
    assert np.allclose(out, manual)
    assert np.allclose(out.sum(axis=1), 1.0)  # convex combo of distributions


def test_ensemble_soft_labels_alpha_one_is_calibrated_model_a():
    la = np.array([[2.0, 0.0, -1.0]])
    lb = np.array([[0.0, 3.0, 0.0]])
    out = ensemble_soft_labels(la, lb, temperature_a=2.0, temperature_b=1.0, alpha=1.0)
    assert np.allclose(out, softmax(la, 2.0))


def test_resample_never_exceeds_available():
    rng = np.random.default_rng(3)
    labels = np.concatenate([np.full(10, 0), np.full(1000, 1), np.full(1000, 2)])
    idx = resample_indices_to_priors(
        labels, CLASSES, {"L1HS": 0.003, "L1PA": 0.104, "NEGATIVE": 0.893}, rng
    )
    # L1HS is the rare limiter at 10 reads; nothing is upsampled past availability.
    assert (labels[idx] == 0).sum() <= 10
