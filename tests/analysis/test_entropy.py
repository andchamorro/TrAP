"""Unit tests for the entropy/redundancy estimators.

Each estimator is checked against an analytically known value, plus the
edge cases the reviewers care about (under-sampling, degeneracy, empty input)
and the determinism guarantee for the bootstrap CI.
"""

import math

import pytest

from trap.analysis import entropy as E
from trap.utils.seeding import seeded_rng

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shannon entropy (MLE)
# ---------------------------------------------------------------------------


def test_uniform_entropy_equals_log2_n():
    assert E.shannon_entropy_mle([1] * 8) == pytest.approx(3.0)  # log2(8)


def test_hand_computed_entropy():
    # p = [1/4, 1/4, 1/2] -> H = 1.5 bits
    assert E.shannon_entropy_mle([1, 1, 2]) == pytest.approx(1.5)


def test_degenerate_distribution_is_zero():
    assert E.shannon_entropy_mle([5]) == 0.0


def test_zero_counts_are_ignored():
    assert E.shannon_entropy_mle([0, 3, 0, 1]) == E.shannon_entropy_mle([3, 1])


def test_empty_input_is_zero():
    assert E.shannon_entropy_mle([]) == 0.0


# ---------------------------------------------------------------------------
# Bias-corrected estimators
# ---------------------------------------------------------------------------


def test_miller_madow_adds_expected_correction():
    counts = [1, 1, 1, 1]  # K=4, N=4
    expected_correction = (4 - 1) / (2 * 4 * math.log(2))
    assert E.miller_madow_entropy(counts) == pytest.approx(2.0 + expected_correction)


def test_miller_madow_exceeds_mle_when_undersampled():
    counts = [1, 1, 1, 2, 5]
    assert E.miller_madow_entropy(counts) > E.shannon_entropy_mle(counts)


def test_chao_shen_at_least_mle_when_undersampled():
    counts = [1, 1, 1, 2, 5]
    assert E.chao_shen_entropy(counts) >= E.shannon_entropy_mle(counts) - 1e-9


def test_chao_shen_falls_back_when_all_singletons():
    counts = [1, 1, 1, 1]  # coverage undefined -> plug-in fallback
    assert E.chao_shen_entropy(counts) == pytest.approx(E.shannon_entropy_mle(counts))


# ---------------------------------------------------------------------------
# KL divergence vs uniform
# ---------------------------------------------------------------------------


def test_kl_uniform_zero_for_uniform_observed():
    assert E.kl_divergence_uniform([1] * 8, support=8) == pytest.approx(0.0)


def test_kl_uniform_positive_for_skewed():
    assert E.kl_divergence_uniform([1, 1, 100], support=3) > 0.0


def test_kl_uniform_full_support_uses_4_to_the_k():
    # k=2 full space = 16 uniform k-mers -> divergence 0 against 4**2.
    assert E.kl_divergence_uniform([1] * 16, support=4 ** 2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Redundancy (three operational definitions)
# ---------------------------------------------------------------------------


def test_redundancy_shannon_zero_for_full_uniform_kmer_space():
    # Uniform over all 4**2 di-nucleotides -> H = 2k = 4 bits -> redundancy 0.
    r = E.redundancy([1] * 16, k=2)
    assert r["redundancy_shannon"] == pytest.approx(0.0)
    assert r["redundancy_observed"] == pytest.approx(0.0)


def test_redundancy_degenerate_is_maximal():
    r = E.redundancy([5], k=2)
    assert r["redundancy_observed"] == 1.0
    assert r["redundancy_duplication"] == pytest.approx(0.8)  # 1 - 1/5


def test_redundancy_duplication_counts_repeats():
    # 3 distinct k-mers over 10 occurrences -> 70% are repeats.
    assert E.redundancy([5, 3, 2], k=3)["redundancy_duplication"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_good_turing_coverage():
    # f1 = 2 singletons over N = 7 occurrences.
    assert E.good_turing_coverage([1, 1, 2, 3]) == pytest.approx(1 - 2 / 7)


def test_n_singletons():
    assert E.n_singletons([1, 1, 2, 3, 1]) == 3


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_with_same_seed():
    counts = [10, 20, 30, 40, 5, 5]
    low_a, high_a = E.bootstrap_ci(counts, E.shannon_entropy_mle, 100, seeded_rng(7))
    low_b, high_b = E.bootstrap_ci(counts, E.shannon_entropy_mle, 100, seeded_rng(7))
    assert (low_a, high_a) == (low_b, high_b)
    assert low_a <= E.shannon_entropy_mle(counts) <= high_a


def test_bootstrap_ci_collapses_for_single_type():
    point = E.shannon_entropy_mle([42])
    assert E.bootstrap_ci([42], E.shannon_entropy_mle, 100, seeded_rng(1)) == (point, point)


# ---------------------------------------------------------------------------
# entropy_summary aggregation
# ---------------------------------------------------------------------------


def test_entropy_summary_has_all_metrics():
    summary = E.entropy_summary([1, 2, 3, 4], k=3, bootstrap=50, rng=seeded_rng(0))
    for key in (
        "h_mle",
        "h_miller_madow",
        "h_chao_shen",
        "kl_uniform_observed",
        "kl_uniform_full",
        "redundancy_shannon",
        "redundancy_observed",
        "redundancy_duplication",
        "good_turing_coverage",
        "n_singletons",
        "h_ci_low",
        "h_ci_high",
    ):
        assert key in summary
    assert summary["h_ci_low"] <= summary["h_mle"] <= summary["h_ci_high"]


def test_entropy_summary_requires_rng_when_bootstrapping():
    with pytest.raises(ValueError):
        E.entropy_summary([1, 2, 3], k=2, bootstrap=10, rng=None)


def test_entropy_summary_ci_collapses_without_bootstrap():
    summary = E.entropy_summary([1, 2, 3, 4], k=2, bootstrap=0)
    assert summary["h_ci_low"] == summary["h_mle"] == summary["h_ci_high"]
