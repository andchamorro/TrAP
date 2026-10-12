"""Tests for the class-imbalance and deficit-propagation analysis.

The propagation arithmetic is the part worth pinning: the factor that recovers
a class's true abundance from its assigned abundance is ``precision / recall``,
not ``1 / recall``, because the contamination precision measures partly offsets
the reads recall loses.  Getting that backwards would overstate the bias by a
factor of nearly three on these numbers, so it is tested against an
independently constructed confusion matrix rather than against itself.
"""

import numpy as np
import pandas as pd
import pytest

from trap.analysis.imbalance import (
    CLASSES,
    balanced_class_weights,
    imbalance_ratios,
    load_composition,
    load_matched_loci,
    paired_contrasts,
    propagation_factors,
    race5_l1hs_sensitivity,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def composition_path(tmp_path):
    path = tmp_path / "dataset_composition.csv"
    path.write_text(
        "Split,L1HS,L1PA,NEGATIVE,LINE-1 (L1HS+L1PA),Total\n"
        'Training,"60,639","2,439,877","2,161,900","2,500,516","4,662,416"\n'
        'Validation,"19,641","811,233","723,285","830,874","1,554,159"\n'
        'Total,"80,280","3,251,110","2,885,185","3,331,390","6,216,575"\n'
    )
    return path


# ── Composition ───────────────────────────────────────────────────────────────


def test_load_composition_parses_thousands_separators(composition_path):
    frame = load_composition(composition_path)
    assert frame.loc["Training", "L1HS"] == 60639
    assert frame.loc["Training", "Total"] == 4662416
    assert frame["L1HS"].dtype.kind in "iu"


def test_imbalance_ratios_cover_every_split(composition_path):
    frame = load_composition(composition_path)
    ratios = imbalance_ratios(frame).set_index("split")
    assert ratios.loc["Training", "ratio"] == pytest.approx(2439877 / 60639)
    assert ratios.loc["Training", "minority_frequency"] == pytest.approx(60639 / 4662416)
    # The training ratio and the all-splits ratio must both be reported, since
    # the reviewer quoted the all-splits count as a training count.
    assert {"Training", "Total"} <= set(ratios.index)


# ── Class weights ─────────────────────────────────────────────────────────────


def test_balanced_class_weights_match_the_training_scheme():
    counts = {"L1HS": 100, "L1PA": 400, "NEGATIVE": 500}
    weights = balanced_class_weights(counts).set_index("class")
    total = 1000
    for name, count in counts.items():
        assert weights.loc[name, "weight"] == pytest.approx(total / (3 * count))
    # Rarer class, larger weight.
    assert weights.loc["L1HS", "weight"] > weights.loc["NEGATIVE", "weight"]


def test_balanced_class_weights_preserve_the_loss_scale():
    """The frequency-weighted mean weight is 1, so the loss is not rescaled."""
    counts = {"L1HS": 7, "L1PA": 913, "NEGATIVE": 1080}
    weights = balanced_class_weights(counts)
    assert (weights["frequency"] * weights["weight"]).sum() == pytest.approx(1.0)


def test_balanced_class_weights_agree_with_the_trainer_implementation():
    """Cross-check against `trap.modeling.train.compute_class_weights`."""
    from trap.modeling.train import compute_class_weights

    counts = {"L1HS": 120, "L1PA": 2400, "NEGATIVE": 1480}
    labels = np.concatenate([np.full(counts[c], i) for i, c in enumerate(CLASSES)])
    reference = compute_class_weights(labels, num_labels=len(CLASSES))
    ours = balanced_class_weights(counts)["weight"].to_numpy()
    np.testing.assert_allclose(ours, reference, rtol=1e-6)


# ── Propagation arithmetic ────────────────────────────────────────────────────


def test_net_factor_is_precision_over_recall():
    factors = propagation_factors(recall=0.624, precision=0.723).set_index("correction")
    assert factors.loc["net", "factor"] == pytest.approx(0.723 / 0.624)
    assert factors.loc["recall_only", "factor"] == pytest.approx(1 / 0.624)
    # The net bias is far smaller than recall alone implies.
    assert (
        factors.loc["net", "implied_underestimate"]
        < factors.loc["recall_only", "implied_underestimate"]
    )


def test_net_factor_recovers_truth_on_a_constructed_confusion_matrix():
    """Build a confusion matrix, derive recall/precision, recover the truth.

    This is the check that matters: the factor is validated against counts it
    was not derived from.
    """
    true_minority = 1000
    recovered = 624  # true positives
    contaminating = 239  # false positives from other classes
    assigned = recovered + contaminating
    recall = recovered / true_minority
    precision = recovered / assigned
    factors = propagation_factors(recall, precision).set_index("correction")
    assert assigned * factors.loc["net", "factor"] == pytest.approx(true_minority)
    # Recall alone over-corrects, because it ignores the contamination.
    assert assigned * factors.loc["recall_only", "factor"] > true_minority


def test_perfect_classifier_needs_no_correction():
    factors = propagation_factors(recall=1.0, precision=1.0).set_index("correction")
    assert factors.loc["net", "factor"] == pytest.approx(1.0)
    assert factors.loc["net", "implied_underestimate"] == pytest.approx(0.0)


# ── 5'RACE sensitivity ────────────────────────────────────────────────────────


@pytest.fixture
def locus_path(tmp_path):
    """Matched loci where estimates track race5, with L1HS deflated."""
    rng = np.random.default_rng(3469)
    rows = []
    for i in range(40):
        target = "L1HS" if i < 12 else "L1PA2"
        race5 = float(rng.uniform(0.2, 8.0))
        for method in ("AlbertSalmon", "Salmon", "AlbertEM", "L1EM"):
            value = race5 * 100 + rng.normal(0, 8)
            # The classifier-dependent methods under-report L1HS loci.
            if target == "L1HS" and method in ("AlbertSalmon", "AlbertEM"):
                value *= 0.624
            rows.append(
                {
                    "region": f"chr1:{1000 * i}-{1000 * i + 900}",
                    "chrom": "chr1",
                    "chromStart": 1000 * i,
                    "chromEnd": 1000 * i + 900,
                    "target": target,
                    "category": 1,
                    "strand": "+",
                    "race5": race5,
                    "method": method,
                    "value": value,
                }
            )
    path = tmp_path / "l1em_salmon_fpm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_matched_loci_keeps_the_target_column(locus_path):
    matched = load_matched_loci(locus_path)
    assert set(matched["target"]) == {"L1HS", "L1PA2"}
    assert (matched["race5"] > 0).all()
    assert len(matched) == 40


def test_sensitivity_leaves_classifier_independent_methods_untouched(locus_path):
    matched = load_matched_loci(locus_path)
    factors = propagation_factors(0.624, 0.723)
    out = race5_l1hs_sensitivity(matched, factors)
    for method in ("Salmon", "L1EM"):
        rows = out[out["method"] == method]
        assert not rows["correction_applied"].any()
        assert rows["pearson_r2"].nunique() == 1
        assert rows["delta_vs_published"].abs().max() == pytest.approx(0.0)


def test_sensitivity_improves_the_deflated_methods(locus_path):
    """Undoing a planted deflation must raise concordance, not lower it."""
    matched = load_matched_loci(locus_path)
    factors = propagation_factors(0.624, 0.723)
    out = race5_l1hs_sensitivity(matched, factors).set_index(["correction", "method"])
    for method in ("AlbertSalmon", "AlbertEM"):
        published = out.loc[("published", method), "pearson_r2"]
        corrected = out.loc[("recall_only", method), "pearson_r2"]
        assert corrected > published


def test_sensitivity_records_how_many_loci_were_corrected(locus_path):
    matched = load_matched_loci(locus_path)
    factors = propagation_factors(0.624, 0.723)
    out = race5_l1hs_sensitivity(matched, factors)
    applied = out[out["correction_applied"]]
    assert (applied["n_corrected_loci"] == 12).all()
    assert (out[~out["correction_applied"]]["n_corrected_loci"] == 0).all()


def test_paired_contrasts_cover_every_correction(locus_path):
    matched = load_matched_loci(locus_path)
    factors = propagation_factors(0.624, 0.723)
    out = race5_l1hs_sensitivity(matched, factors)
    contrasts = paired_contrasts(out)
    assert set(contrasts["correction"]) == {"published", "recall_only", "net"}
    assert set(contrasts["contrast"]) == {"AlbertSalmon-Salmon", "AlbertEM-L1EM"}
    # Correcting a deficit that only the TrAP configurations carry can only
    # widen the paired margin.
    for contrast in contrasts["contrast"].unique():
        rows = contrasts[contrasts["contrast"] == contrast].set_index("correction")
        assert rows.loc["net", "delta_r2"] > rows.loc["published", "delta_r2"]
