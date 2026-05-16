"""Shannon entropy, redundancy, and bias-corrected estimators (in bits).

All estimators operate on a *count vector* ``counts`` — the observed
multiplicities of the distinct k-mers in a corpus (the empirical k-mer
frequency distribution).  Zeros are ignored; only ``counts > 0`` contribute a
"type" (an observed k-mer).

The plug-in (maximum-likelihood) Shannon entropy systematically *under*
estimates the true entropy in the under-sampled regime where the number of
possible k-mers (``4**k``) approaches or exceeds the number of observed
positions ``N``.  This is exactly the regime that produced the manuscript's
apparent entropy "plateau" for large ``k`` and the reviewer concern that it may
reflect sampling saturation rather than a true information plateau.  Two
bias-corrected estimators are provided:

* **Miller-Madow** — a first-order analytic correction
  ``H_MM = H_MLE + (K - 1) / (2 N ln 2)`` (bits).
* **Chao-Shen** — a coverage-adjusted Horvitz-Thompson estimator that uses
  Good-Turing coverage to account for unseen k-mers; more robust when many
  k-mers are singletons.

References
----------
Miller, G. (1955). Note on the bias of information estimates.
Chao, A. & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of
diversity when there are unseen species in sample. *Environ. Ecol. Stat.*
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np
from scipy.special import erfinv as _erfinv

_LN2 = math.log(2.0)

# Above this count of distinct k-mer types, allocating (n_boot, n_types) float64
# is too expensive (>~400 MB per bootstrap batch).  The analytical asymptotic CI
# is used instead — accurate when N >> n_types and conservative otherwise.
_BOOTSTRAP_TYPE_THRESHOLD = 500_000


def _clean_counts(counts) -> np.ndarray:
    """Return a 1-D float array of strictly-positive counts.

    Args:
        counts: Array-like of non-negative k-mer multiplicities.

    Returns:
        ``float64`` array containing only the entries ``> 0``.
    """
    arr = np.asarray(counts, dtype=np.float64).ravel()
    return arr[arr > 0]


def shannon_entropy_mle(counts) -> float:
    """Plug-in (maximum-likelihood) Shannon entropy ``H = -Sum p log2 p``.

    Args:
        counts: Observed k-mer multiplicities.

    Returns:
        Entropy in bits.  Returns ``0.0`` for an empty or single-type input.
    """
    c = _clean_counts(counts)
    total = c.sum()
    if total <= 0 or c.size <= 1:
        return 0.0
    p = c / total
    return float(-np.sum(p * np.log2(p)))


def miller_madow_entropy(counts) -> float:
    """Miller-Madow bias-corrected Shannon entropy (bits).

    Adds the first-order correction ``(K - 1) / (2 N ln 2)`` to the plug-in
    estimate, where ``K`` is the number of observed types and ``N`` the total
    number of k-mer occurrences.

    Args:
        counts: Observed k-mer multiplicities.

    Returns:
        Bias-corrected entropy in bits.
    """
    c = _clean_counts(counts)
    total = c.sum()
    if total <= 0 or c.size <= 1:
        return 0.0
    correction = (c.size - 1) / (2.0 * total * _LN2)
    return shannon_entropy_mle(c) + correction


def chao_shen_entropy(counts) -> float:
    """Chao-Shen coverage-adjusted Shannon entropy (bits).

    Uses Good-Turing sample coverage ``C = 1 - f1 / N`` (``f1`` = number of
    singletons) to inflate observed probabilities for unseen k-mers, then
    applies a Horvitz-Thompson correction for detection probability.

    Args:
        counts: Observed k-mer multiplicities.

    Returns:
        Coverage-adjusted entropy in bits.  Falls back to the plug-in value
        when coverage cannot be estimated (e.g. every k-mer is a singleton).
    """
    c = _clean_counts(counts)
    total = c.sum()
    if total <= 0 or c.size <= 1:
        return 0.0

    singletons = float(np.sum(c == 1))
    coverage = 1.0 - singletons / total
    if coverage <= 0.0:
        # Every k-mer is a singleton: coverage is undefined; the corpus is too
        # under-sampled for the Chao-Shen adjustment, so report the plug-in.
        return shannon_entropy_mle(c)

    p = c / total
    p_adj = p * coverage
    detection = 1.0 - np.power(1.0 - p_adj, total)
    # Guard against detection -> 0 for vanishingly small adjusted probabilities.
    detection = np.where(detection > 0.0, detection, 1.0)
    return float(-np.sum(p_adj * np.log2(p_adj) / detection))


def kl_divergence_uniform(counts, support: Optional[int] = None) -> float:
    """Kullback-Leibler divergence of the empirical distribution from uniform.

    For a uniform background ``U`` over ``support`` k-mers,
    ``D_KL(P || U) = log2(support) - H(P)``.

    Args:
        counts: Observed k-mer multiplicities.
        support: Size of the uniform support.  Defaults to the number of
            *observed* k-mers; pass ``4 ** k`` to compare against the full
            k-mer space.

    Returns:
        Divergence in bits (``>= 0``).
    """
    c = _clean_counts(counts)
    if c.size == 0:
        return 0.0
    k_support = c.size if support is None else int(support)
    if k_support <= 1:
        return 0.0
    divergence = math.log2(k_support) - shannon_entropy_mle(c)
    return float(max(divergence, 0.0))


def redundancy(counts, k: int) -> Dict[str, float]:
    """Three operational redundancy measures for a k-mer distribution.

    Args:
        counts: Observed k-mer multiplicities.
        k: K-mer length (defines the maximum possible alphabet ``4 ** k``).

    Returns:
        Mapping with:

        * ``redundancy_shannon`` — ``1 - H / log2(4**k)`` (Shannon's classical
          redundancy versus the maximum possible entropy ``2k`` bits).
        * ``redundancy_observed`` — ``1 - H / log2(K)`` (versus the realised
          alphabet of ``K`` observed k-mers).
        * ``redundancy_duplication`` — ``1 - K / N`` (fraction of k-mer
          occurrences that are repeats of an already-seen k-mer).
    """
    c = _clean_counts(counts)
    total = c.sum()
    n_types = c.size
    entropy = shannon_entropy_mle(c)

    max_entropy_full = 2.0 * k  # log2(4**k)
    redundancy_shannon = 1.0 - entropy / max_entropy_full if max_entropy_full > 0 else 0.0

    if n_types <= 1:
        redundancy_observed = 1.0
    else:
        redundancy_observed = 1.0 - entropy / math.log2(n_types)

    redundancy_duplication = 1.0 - n_types / total if total > 0 else 0.0

    return {
        "redundancy_shannon": float(redundancy_shannon),
        "redundancy_observed": float(redundancy_observed),
        "redundancy_duplication": float(redundancy_duplication),
    }


def good_turing_coverage(counts) -> float:
    """Good-Turing sample coverage ``C = 1 - f1 / N``.

    The complement (``1 - C``) estimates the total probability mass of k-mers
    not yet observed — a direct diagnostic for whether a corpus is saturated.

    Args:
        counts: Observed k-mer multiplicities.

    Returns:
        Estimated coverage in ``[0, 1]``.
    """
    c = _clean_counts(counts)
    total = c.sum()
    if total <= 0:
        return 0.0
    singletons = float(np.sum(c == 1))
    return float(1.0 - singletons / total)


def n_singletons(counts) -> int:
    """Number of k-mers observed exactly once (``f1``).

    Args:
        counts: Observed k-mer multiplicities.

    Returns:
        Count of singleton k-mers.
    """
    c = _clean_counts(counts)
    return int(np.sum(c == 1))


def bootstrap_ci(
    counts,
    estimator: Callable[[np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-adjusted bootstrap confidence interval for an entropy estimator.

    For corpora with up to ``_BOOTSTRAP_TYPE_THRESHOLD`` distinct k-mers, all
    bootstrap samples are drawn at once via ``rng.multinomial(..., size=n_boot)``
    and entropy is computed in a single vectorised pass — eliminating the Python
    loop and reducing wall time from O(n_boot) sequential allocations to one
    batched call.

    For larger type counts (large k on transcriptome-scale corpora), allocating
    the ``(n_boot, n_types)`` matrix is infeasible.  Instead an asymptotic
    normal CI is returned using the plug-in variance formula
    ``Var(H_MLE) ≈ (E_p[log²_2 p] - H²) / N``, which is accurate when the
    corpus is well-sampled and conservative otherwise.

    In both cases the interval is bias-adjusted (recentered on the point
    estimate) and deterministic given *rng*.

    Args:
        counts: Observed k-mer multiplicities.
        estimator: Function mapping a count vector to a scalar (e.g.
            :func:`shannon_entropy_mle`).  Used only for the vectorised path.
        n_boot: Number of bootstrap replicates.
        rng: Seeded numpy ``Generator`` (build via
            :func:`trap.utils.seeding.seeded_rng`).
        alpha: Two-sided significance level.

    Returns:
        ``(low, high)`` bounds with ``low <= point <= high``.  Returns
        ``(point, point)`` when there is no sampling variability to resample
        (empty or single-type input).
    """
    c = _clean_counts(counts)
    total = int(c.sum())
    point = estimator(c)
    if n_boot <= 0 or c.size <= 1 or total <= 0:
        return point, point

    p = c / c.sum()

    if c.size <= _BOOTSTRAP_TYPE_THRESHOLD:
        # Vectorised path: draw all replicates at once — shape (n_boot, n_types).
        all_counts = rng.multinomial(total, p, size=n_boot).astype(np.float64)
        row_totals = all_counts.sum(axis=1, keepdims=True)
        p_boot = all_counts / row_totals
        mask = all_counts > 0
        boot_h = -np.sum(
            np.where(mask, p_boot * np.log2(np.where(mask, p_boot, 1.0)), 0.0),
            axis=1,
        )
    else:
        # Analytical asymptotic CI (avoids (n_boot × n_types) allocation).
        # Var(H_MLE) ≈ (Σ p_i log²_2(p_i) − H²) / N  (bits²).
        log2_p = np.log2(np.where(p > 0.0, p, 1.0))
        second_moment = float(np.dot(p, log2_p ** 2))
        variance = max(0.0, (second_moment - point ** 2) / total)
        se = math.sqrt(variance)
        # z = Φ⁻¹(1 − α/2) = √2 · erfinv(1 − α)
        z = math.sqrt(2.0) * float(_erfinv(1.0 - alpha))
        return max(0.0, point - z * se), point + z * se

    # Bias-adjusted percentile interval: recenter on the point estimate to
    # correct the systematic downward shift of bootstrap entropy replicates.
    shift = point - float(boot_h.mean())
    low = float(np.percentile(boot_h, 100.0 * (alpha / 2.0))) + shift
    high = float(np.percentile(boot_h, 100.0 * (1.0 - alpha / 2.0))) + shift
    return low, high


def entropy_summary(
    counts,
    k: int,
    *,
    bootstrap: int = 0,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute every estimator + diagnostic for one k-mer distribution.

    This is the single entry point used by :mod:`trap.analysis.kmer_counting`
    to build one tidy row per ``(corpus, k, subsample, replicate)``.

    Args:
        counts: Observed k-mer multiplicities.
        k: K-mer length.
        bootstrap: Number of bootstrap replicates for the entropy CI
            (``0`` disables; bounds collapse to the point estimate).
        rng: Seeded generator required when ``bootstrap > 0``.
        alpha: Two-sided level for the bootstrap interval.

    Returns:
        Flat mapping of metric name to value (all entropies in bits).
    """
    c = _clean_counts(counts)
    total = float(c.sum())
    n_types = int(c.size)

    h_mle = shannon_entropy_mle(c)
    summary: Dict[str, float] = {
        "n_kmers_total": total,
        "n_kmers_distinct": float(n_types),
        "h_mle": h_mle,
        "h_miller_madow": miller_madow_entropy(c),
        "h_chao_shen": chao_shen_entropy(c),
        "kl_uniform_observed": kl_divergence_uniform(c, support=None),
        "kl_uniform_full": kl_divergence_uniform(c, support=4 ** k),
        "good_turing_coverage": good_turing_coverage(c),
        "n_singletons": float(n_singletons(c)),
    }
    summary.update(redundancy(c, k))

    if bootstrap > 0:
        if rng is None:
            raise ValueError("bootstrap > 0 requires a seeded rng")
        low, high = bootstrap_ci(c, shannon_entropy_mle, bootstrap, rng, alpha)
        summary["h_ci_low"] = low
        summary["h_ci_high"] = high
    else:
        summary["h_ci_low"] = h_mle
        summary["h_ci_high"] = h_mle

    return summary
