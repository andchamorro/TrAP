"""Tests for the aggregate-quantification scorecard core (CPU-only)."""

import numpy as np
import pytest

from trap.analysis.quantification import (
    bootstrap_compositions,
    collapse_l1_vs_negative,
    concordance_correlation,
    filter_purity_yield,
    hard_counts,
    recovery_metrics,
    soft_counts,
    yield_at_purity,
)

pytestmark = pytest.mark.unit

CLASSES = ["L1HS", "L1PA", "NEGATIVE"]


def _onehot(labels, num_classes):
    p = np.zeros((len(labels), num_classes))
    p[np.arange(len(labels)), labels] = 1.0
    return p


def test_soft_counts_sum_to_n():
    probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    sc = soft_counts(probs)
    assert np.allclose(sc, [0.3, 0.4, 1.3])
    assert sc.sum() == pytest.approx(2.0)


def test_hard_counts_from_argmax():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])
    assert list(hard_counts(probs, 3)) == [1.0, 1.0, 1.0]


def test_concordance_perfect_and_biased():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    assert concordance_correlation(y, y) == pytest.approx(1.0)
    # A constant offset (bias) drops CCC below 1 even with perfect correlation.
    assert concordance_correlation(y, y + 0.2) < 1.0


def test_soft_counts_less_biased_than_hard_for_rare_class():
    # The rare-class (L1HS) case: on true-minority reads the model is unsure and
    # the argmax tips to the majority — hard counts MISS the minority entirely,
    # but the soft expected counts still capture its probability mass.
    rng = np.random.default_rng(0)
    n = 6000
    labels = np.where(rng.random(n) < 0.7, 1, 0)  # 70% majority (class 1)
    probs = np.zeros((n, 2))
    for i, y in enumerate(labels):
        # minority reads: p(minority)=0.4 -> argmax wrongly picks the majority.
        probs[i] = [0.4, 0.6] if y == 0 else [0.2, 0.8]
    true_frac = np.array([(labels == 0).mean(), (labels == 1).mean()])
    soft_err = np.abs(soft_counts(probs) / n - true_frac).sum()
    hard_err = np.abs(hard_counts(probs, 2) / n - true_frac).sum()
    assert soft_err < hard_err
    assert hard_counts(probs, 2)[0] == 0  # hard misses every minority read


def test_bootstrap_perfect_classifier_recovers_composition():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], 2000)
    probs = _onehot(labels, 3)  # perfect predictions
    tf, ef = bootstrap_compositions(
        probs, labels, 3, n_samples=50, rng=rng, soft=True, sample_size=3000
    )
    assert tf.shape == ef.shape == (50, 3)
    metrics = recovery_metrics(tf, ef, CLASSES)
    for c in CLASSES:
        assert metrics[c]["ccc"] > 0.99
        assert abs(metrics[c]["bias"]) < 0.02


def test_collapse_l1_vs_negative():
    probs = np.array([[0.5, 0.3, 0.2], [0.05, 0.05, 0.9]])
    labels = np.array([0, 2])  # L1HS, NEGATIVE
    p2, l2, c2 = collapse_l1_vs_negative(probs, labels, CLASSES)
    assert c2 == ["L1", "NEGATIVE"]
    assert np.allclose(p2[:, 1], [0.2, 0.9])  # P(NEGATIVE) carried through
    assert np.allclose(p2[:, 0], [0.8, 0.1])  # P(L1) = 1 - P(NEGATIVE)
    assert list(l2) == [0, 1]  # L1HS->L1(0), NEGATIVE->1


def test_filter_purity_yield_monotonicity_and_bounds():
    # P(NEG): true-L1 reads low, NEGATIVE reads high -> a mid threshold separates well.
    p_neg = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
    is_l1 = np.array([1, 1, 1, 0, 0, 0])
    rows = {
        r["threshold"]: r for r in filter_purity_yield(p_neg, is_l1, np.array([0.05, 0.5, 0.95]))
    }
    assert rows[0.05]["yield"] == 0.0  # nothing passes
    assert rows[0.5]["yield"] == 1.0 and rows[0.5]["purity"] == 1.0  # clean split
    assert rows[0.95]["kept_frac"] == 1.0  # everything passes
    # Yield is non-decreasing as τ rises (looser filter keeps more L1).
    ys = [filter_purity_yield(p_neg, is_l1, np.array([t]))[0]["yield"] for t in (0.05, 0.5, 0.95)]
    assert ys == sorted(ys)


def test_yield_at_purity_frontier():
    rows = [
        {"threshold": 0.1, "yield": 0.3, "purity": 0.95, "kept_frac": 0.03},
        {"threshold": 0.3, "yield": 0.5, "purity": 0.80, "kept_frac": 0.06},
        {"threshold": 0.5, "yield": 0.7, "purity": 0.60, "kept_frac": 0.10},
    ]
    assert yield_at_purity(rows, 0.90) == 0.3  # only the τ=0.1 row qualifies
    assert yield_at_purity(rows, 0.80) == 0.5  # best yield among purity≥0.80
    assert yield_at_purity(rows, 0.55) == 0.7  # all qualify → max yield
    import math

    assert math.isnan(yield_at_purity(rows, 0.99))  # unreachable purity


def test_recovery_metrics_keys():
    tf = np.random.default_rng(2).random((20, 3))
    ef = tf + 0.01
    m = recovery_metrics(tf, ef, CLASSES)
    assert set(m["L1HS"]) == {"pearson", "ccc", "mae", "bias"}
