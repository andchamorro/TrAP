"""Tests for the stratified statistics of the synthetic insertion benchmark.

The fixture is a 4-stratum, 4-method miniature of the real design: one method
dominates everywhere, one within-quantifier pair is won in every stratum, and
the other loses a single stratum.  That is enough to pin the behaviours the
response depends on — above all that the Wilcoxon p floor is reported, since
p = 0.0039 at nine strata is the attainable minimum rather than an effect size.
"""

import numpy as np
import pandas as pd
import pytest

from trap.analysis.synthetic_strata import (
    bootstrap_stratum_delta,
    load_all_strata,
    load_strata,
    stratum_contrasts,
    stratum_gradients,
    stratum_summary,
    wilcoxon_p_floor,
)

pytestmark = pytest.mark.unit

# Ascending stratum order, as in the stored tables.  AlbertEM loses the first
# stratum to L1EM, mirroring the real 2^5 exception.
_INSERTION = pd.DataFrame(
    {
        "32": [0.99, 0.80, 0.50, 0.52],
        "64": [0.98, 0.75, 0.48, 0.45],
        "128": [0.95, 0.60, 0.40, 0.38],
        "256": [0.90, 0.40, 0.30, 0.25],
    },
    index=["AlbertSalmon", "Salmon", "AlbertEM", "L1EM"],
)

_DELETION = pd.DataFrame(
    {
        "0.000": [0.994, 0.953, 0.868, 0.858],
        "0.050": [0.994, 0.952, 0.868, 0.858],
        "0.100": [0.994, 0.952, 0.860, 0.844],
    },
    index=["AlbertSalmon", "Salmon", "AlbertEM", "L1EM"],
)


@pytest.fixture
def synthetic_dir(tmp_path):
    _INSERTION.to_csv(tmp_path / "by_insertion_rate.csv")
    _DELETION.to_csv(tmp_path / "by_del_probability.csv")
    return tmp_path


@pytest.fixture
def strata(synthetic_dir):
    return load_all_strata(synthetic_dir)


# ── Loading ───────────────────────────────────────────────────────────────────


def test_load_strata_tidies_the_stored_matrix(synthetic_dir):
    tidy = load_strata(synthetic_dir / "by_insertion_rate.csv", "insertion_level")
    assert list(tidy.columns) == ["axis", "stratum", "method", "r2"]
    assert len(tidy) == _INSERTION.size
    assert set(tidy["method"]) == set(_INSERTION.index)
    row = tidy[(tidy["method"] == "AlbertSalmon") & (tidy["stratum"] == "32")]
    assert row["r2"].iloc[0] == pytest.approx(0.99)


def test_load_all_strata_covers_both_axes(strata):
    assert set(strata["axis"]) == {"insertion_level", "deletion_probability"}
    assert len(strata) == _INSERTION.size + _DELETION.size


def test_load_all_strata_raises_when_nothing_is_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_strata(tmp_path)


def test_load_all_strata_skips_a_missing_axis(tmp_path):
    _INSERTION.to_csv(tmp_path / "by_insertion_rate.csv")
    only_one = load_all_strata(tmp_path)
    assert set(only_one["axis"]) == {"insertion_level"}


# ── Summary ───────────────────────────────────────────────────────────────────


def test_stratum_summary_counts_wins_and_dispersion(strata):
    summary = stratum_summary(strata).set_index(["axis", "method"])
    top = summary.loc[("insertion_level", "AlbertSalmon")]
    assert top["n_top"] == len(_INSERTION.columns)
    assert top["mean_r2"] == pytest.approx(_INSERTION.loc["AlbertSalmon"].mean())
    assert top["sd_r2"] == pytest.approx(_INSERTION.loc["AlbertSalmon"].std(ddof=1))
    assert top["min_r2"] == pytest.approx(0.90)
    assert top["max_r2"] == pytest.approx(0.99)
    # Exactly one method is best per stratum, so n_top sums to the stratum count.
    insertion = summary.xs("insertion_level", level="axis")
    assert insertion["n_top"].sum() == len(_INSERTION.columns)


# ── Contrasts ─────────────────────────────────────────────────────────────────


def test_wilcoxon_p_floor_is_the_attainable_minimum():
    assert wilcoxon_p_floor(9) == pytest.approx(0.00390625)
    assert wilcoxon_p_floor(5) == pytest.approx(0.0625)
    assert np.isnan(wilcoxon_p_floor(0))


def test_stratum_contrasts_report_wins_losses_and_the_floor(strata):
    contrasts = stratum_contrasts(strata, n_boot=200)
    insertion = contrasts[contrasts["axis"] == "insertion_level"].set_index(["family", "contrast"])
    clean = insertion.loc[("within_quantifier", "AlbertSalmon-Salmon")]
    assert clean["n_won"] == len(_INSERTION.columns)
    assert clean["strata_lost"] == ""
    # The EM pair loses the first stratum, where L1EM is higher.
    mixed = insertion.loc[("within_quantifier", "AlbertEM-L1EM")]
    assert mixed["n_won"] == len(_INSERTION.columns) - 1
    assert mixed["strata_lost"] == "32"
    assert mixed["min_delta_r2"] < 0
    for _, row in contrasts.iterrows():
        assert row["wilcoxon_p"] >= row["p_floor"] or np.isnan(row["wilcoxon_p"])


def test_stratum_contrasts_cover_both_families(strata):
    contrasts = stratum_contrasts(strata, n_boot=100)
    assert set(contrasts["family"]) == {"reference", "within_quantifier"}
    reference = contrasts[
        (contrasts["family"] == "reference") & (contrasts["axis"] == "insertion_level")
    ]
    # One row per non-reference method.
    assert len(reference) == len(_INSERTION.index) - 1
    assert not reference["contrast"].str.fullmatch("AlbertSalmon-AlbertSalmon").any()


def test_stratum_contrasts_mean_delta_matches_the_tables(strata):
    contrasts = stratum_contrasts(strata, n_boot=100)
    row = contrasts[
        (contrasts["axis"] == "insertion_level")
        & (contrasts["contrast"] == "AlbertSalmon-Salmon")
        & (contrasts["family"] == "within_quantifier")
    ].iloc[0]
    expected = (_INSERTION.loc["AlbertSalmon"] - _INSERTION.loc["Salmon"]).mean()
    assert row["mean_delta_r2"] == pytest.approx(expected)
    assert row["ci_lo"] <= row["mean_delta_r2"] <= row["ci_hi"]


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def test_bootstrap_stratum_delta_is_seed_deterministic():
    deltas = [0.2, 0.15, 0.3, 0.25, 0.1]
    assert bootstrap_stratum_delta(deltas, n_boot=200, seed=3469) == bootstrap_stratum_delta(
        deltas, n_boot=200, seed=3469
    )
    assert bootstrap_stratum_delta(deltas, n_boot=200, seed=3469) != bootstrap_stratum_delta(
        deltas, n_boot=200, seed=7
    )


def test_bootstrap_stratum_delta_declines_too_few_strata():
    lo, hi = bootstrap_stratum_delta([0.2, 0.1], n_boot=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_interval_brackets_the_mean():
    deltas = [0.2, 0.15, 0.3, 0.25, 0.1]
    lo, hi = bootstrap_stratum_delta(deltas, n_boot=2000, seed=3469)
    assert lo <= np.mean(deltas) <= hi


# ── Gradients ─────────────────────────────────────────────────────────────────


def test_stratum_gradients_separate_drop_from_spread(strata):
    gradients = stratum_gradients(strata).set_index(["axis", "method"])
    row = gradients.loc[("insertion_level", "AlbertSalmon")]
    assert row["r2_first"] == pytest.approx(0.99)
    assert row["r2_last"] == pytest.approx(0.90)
    assert row["drop"] == pytest.approx(0.09)
    assert row["spread"] >= row["drop"]
    # The deletion axis has no monotone trend, which is the point of `spread`:
    # every method moves by a negligible amount across it.
    deletion = gradients.xs("deletion_probability", level="axis")
    assert (deletion["spread"] < 0.05).all()
    # Insertion load, by contrast, separates the methods.
    insertion = gradients.xs("insertion_level", level="axis")
    assert insertion.loc["AlbertSalmon", "drop"] < insertion.loc["Salmon", "drop"]
