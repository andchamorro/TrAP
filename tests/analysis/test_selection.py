"""Tests for the quantitative k-selection criterion."""

import pandas as pd
import pytest

from trap.analysis.selection import compute_selection

pytestmark = pytest.mark.unit


def _spectrum(per_k, corpus="c"):
    """Build a spectrum frame from {k: (h_full, h_half)}."""
    rows = []
    for k, (h_full, h_half) in per_k.items():
        rows.append(
            {"corpus": corpus, "k": k, "subsample_frac": 1.0, "replicate": 0, "h_miller_madow": h_full}
        )
        rows.append(
            {"corpus": corpus, "k": k, "subsample_frac": 0.5, "replicate": 0, "h_miller_madow": h_half}
        )
    return pd.DataFrame(rows)


def _ablation(per_k):
    """Build an ablation frame from {k: (f1_mean, f1_std)}."""
    return pd.DataFrame(
        [{"k": k, "macro_f1_mean": m, "macro_f1_std": s} for k, (m, s) in per_k.items()]
    )


def test_real_plateau_within_reliable_regime():
    # Entropy plateaus at k=5 and the corpus is well-sampled everywhere.
    spectrum = _spectrum(
        {2: (2.0, 2.0), 3: (3.0, 3.0), 4: (3.90, 3.90), 5: (3.92, 3.92), 6: (3.93, 3.93)}
    )
    ablation = _ablation({2: (0.6, 0.02), 3: (0.8, 0.02), 4: (0.9, 0.02), 5: (0.92, 0.02), 6: (0.92, 0.02)})
    rep = compute_selection(spectrum, ablation, epsilon=0.05, tau=0.05)["c"]
    assert rep["diminishing_returns_k"] == 5
    assert rep["saturation_safe_k_max"] == 6
    assert rep["diminishing_returns_within_reliable_regime"] is True
    assert rep["recommended_k"] == 5
    assert rep["task_plateau_k"] == 4


def test_saturation_artefact_defers_to_saturation_and_task():
    # Plateau (dH<tau) only appears where the corpus is still under-sampled.
    spectrum = _spectrum(
        {2: (2.0, 2.0), 3: (3.0, 3.0), 4: (4.0, 4.0), 5: (4.04, 3.0), 6: (4.07, 2.9)}
    )
    ablation = _ablation({2: (0.6, 0.02), 3: (0.7, 0.02), 4: (0.8, 0.02), 5: (0.85, 0.02), 6: (0.9, 0.02)})
    rep = compute_selection(spectrum, ablation, epsilon=0.05, tau=0.05)["c"]
    assert rep["diminishing_returns_k"] == 5
    assert rep["saturation_safe_k_max"] == 4
    assert rep["diminishing_returns_within_reliable_regime"] is False
    expected = max(rep["saturation_safe_k_max"], rep["task_plateau_k"])
    assert rep["recommended_k"] == expected


def test_overall_summary_lists_each_corpus():
    spectrum = pd.concat(
        [
            _spectrum({2: (2.0, 2.0), 3: (3.0, 3.0), 4: (3.9, 3.9), 5: (3.92, 3.92)}, corpus="a"),
            _spectrum({2: (2.0, 2.0), 3: (3.0, 3.0), 4: (3.9, 3.9), 5: (3.92, 3.92)}, corpus="b"),
        ]
    )
    report = compute_selection(spectrum, None, epsilon=0.05, tau=0.05)
    assert set(report["overall"]["recommended_k_by_corpus"]) == {"a", "b"}


def test_works_without_ablation():
    spectrum = _spectrum({2: (2.0, 2.0), 3: (3.0, 3.0), 4: (3.9, 3.9), 5: (3.92, 3.92)})
    rep = compute_selection(spectrum, None, epsilon=0.05, tau=0.05)["c"]
    assert rep["task_plateau_k"] is None
    assert rep["recommended_k"] is not None
