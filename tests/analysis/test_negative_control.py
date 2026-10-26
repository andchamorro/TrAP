"""Tests for the NEGATIVE-class control diagnostics.

Two fixtures with known structure: one where the classifier's NEGATIVE class is
pure noise (so adjusting for a covariate should kill any apparent association)
and one where NEGATIVE is constructed as a fixed fraction of LINE-1 signal plus
noise (so the association must survive adjustment).  Those are the two outcomes
the real analysis has to be able to distinguish, and getting them backwards is
the failure mode that would have let a hand-wave through.
"""

import numpy as np
import pandas as pd
import pytest

from trap.analysis.negative_control import (
    CLASSES,
    MLVS_BWA_FIGURE,
    adjusted_correlations,
    chrom_length_correlations,
    composition_correlations,
    jackknife_correlations,
    leakage_rates,
    load_annotation_density,
    load_by_chromosome,
    load_chrom_lengths,
    negative_control,
    operating_point_leakage,
)

pytestmark = pytest.mark.unit

_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _tidy(classifier: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    """Pack two wide frames into the stored tidy schema."""
    rows = []
    for method, frame in (("Albert", classifier), ("Alignment", alignment)):
        for chrom in frame.index:
            for klass in CLASSES:
                rows.append(
                    {
                        "sample": "TEST",
                        "figure": MLVS_BWA_FIGURE,
                        "chrom": chrom,
                        "method": method,
                        "class": klass,
                        "value": float(frame.loc[chrom, klass]),
                    }
                )
    return pd.DataFrame(rows)


def _frames(leak: float, seed: int = 3469):
    """Alignment counts scaled by chromosome, and a classifier that leaks.

    ``leak`` is the fraction of alignment LINE-1 signal that reappears in the
    classifier's NEGATIVE class.  At ``leak = 0`` NEGATIVE is pure noise.
    """
    rng = np.random.default_rng(seed)
    scale = np.linspace(1.0, 0.2, len(_CHROMS))  # chromosome "size" gradient
    alignment = pd.DataFrame(
        {
            "L1HS": 400 * scale * rng.uniform(0.8, 1.2, len(_CHROMS)),
            "L1PA2": 1500 * scale * rng.uniform(0.8, 1.2, len(_CHROMS)),
            "NEGATIVE": 60 * scale * rng.uniform(0.8, 1.2, len(_CHROMS)),
        },
        index=_CHROMS,
    )
    classifier = pd.DataFrame(
        {
            "L1HS": alignment["L1HS"] * 0.7 + rng.normal(0, 5, len(_CHROMS)),
            "L1PA2": alignment["L1PA2"] * 1.2 + rng.normal(0, 5, len(_CHROMS)),
            "NEGATIVE": leak * (alignment["L1HS"] + alignment["L1PA2"])
            + rng.normal(100, 30, len(_CHROMS)),
        },
        index=_CHROMS,
    )
    return classifier, alignment


@pytest.fixture
def leaky(tmp_path):
    classifier, alignment = _frames(leak=0.3)
    path = tmp_path / "aggregate_by_chr.csv"
    _tidy(classifier, alignment).to_csv(path, index=False)
    return path, classifier, alignment


@pytest.fixture
def clean(tmp_path):
    classifier, alignment = _frames(leak=0.0)
    path = tmp_path / "clean_by_chr.csv"
    _tidy(classifier, alignment).to_csv(path, index=False)
    return path, classifier, alignment


@pytest.fixture
def annotation_bed(tmp_path):
    """Annotation BED in the L1EM name format, with a density gradient."""
    path = tmp_path / "L1EM.400.bed"
    lines = []
    for position, chrom in enumerate(_CHROMS):
        for family, count in (("L1HS", 120 - 4 * position), ("L1PA2", 320 - 10 * position)):
            for i in range(max(count, 1)):
                start = 1000 * (i + 1)
                lines.append(
                    f"{chrom}\t{start}\t{start + 900}\t"
                    f"{family}.{i}.{chrom}:{start + 400}-{start + 500}.+\t0\t+"
                )
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Loading ───────────────────────────────────────────────────────────────────


def test_load_by_chromosome_splits_the_methods(leaky):
    path, classifier, alignment = leaky
    loaded_classifier, loaded_alignment = load_by_chromosome(path)
    assert list(loaded_classifier.columns) == list(CLASSES)
    assert len(loaded_classifier) == len(_CHROMS)
    # Row order must be shared, or every correlation is garbage.
    assert list(loaded_classifier.index) == list(loaded_alignment.index)
    np.testing.assert_allclose(
        loaded_classifier["L1PA2"].to_numpy(),
        classifier.loc[loaded_classifier.index, "L1PA2"].to_numpy(),
    )


def test_load_by_chromosome_rejects_a_missing_figure(leaky):
    path, _, _ = leaky
    with pytest.raises(ValueError, match="not present"):
        load_by_chromosome(path, figure="nope")


def test_load_annotation_density_counts_families(annotation_bed):
    counts = load_annotation_density(annotation_bed)
    assert set(counts.columns) == {"annotated_l1", "L1HS", "L1PA2"}
    assert (counts["annotated_l1"] == counts["L1HS"] + counts["L1PA2"]).all()
    # The planted gradient must be recovered.
    assert counts.loc["chr1", "L1PA2"] > counts.loc["chrY", "L1PA2"]


# ── The discriminating test ───────────────────────────────────────────────────


def test_leakage_survives_adjustment(leaky, annotation_bed):
    """A NEGATIVE class built from LINE-1 signal stays correlated after control."""
    path, _, _ = leaky
    classifier, alignment = load_by_chromosome(path)
    annotation = load_annotation_density(annotation_bed)
    table = adjusted_correlations(classifier, alignment, annotation=annotation).set_index(
        ["classifier_class", "alignment_class"]
    )
    row = table.loc[("NEGATIVE", "L1PA2")]
    assert row["pearson_r"] > 0.5
    assert row["partial_r_given_annotated_l1"] > 0.3
    assert row["partial_p_given_annotated_l1"] < 0.05
    assert not row["composition_explains"]


def test_pure_noise_negative_is_not_significant_after_adjustment(clean, annotation_bed):
    """With no leakage the adjusted association is not significant.

    Judged on the p value rather than the coefficient: at n = 24 the standard
    error of a correlation is ~0.21, so a partial r of 0.3-0.4 is unremarkable
    noise.  That is also why the real analysis reports p values beside every
    adjusted coefficient.
    """
    path, _, _ = clean
    classifier, alignment = load_by_chromosome(path)
    annotation = load_annotation_density(annotation_bed)
    table = adjusted_correlations(classifier, alignment, annotation=annotation).set_index(
        ["classifier_class", "alignment_class"]
    )
    row = table.loc[("NEGATIVE", "L1PA2")]
    assert row["pearson_p"] > 0.05
    assert row["partial_p_given_annotated_l1"] > 0.05


def test_adjusted_correlations_omits_annotation_columns_without_annotation(leaky):
    path, _, _ = leaky
    classifier, alignment = load_by_chromosome(path)
    table = adjusted_correlations(classifier, alignment)
    assert "partial_r_given_l1_total" in table.columns
    assert "partial_r_given_annotated_l1" not in table.columns
    assert len(table) == len(CLASSES) ** 2


# ── Jackknife ─────────────────────────────────────────────────────────────────


def test_jackknife_identifies_a_planted_outlier(leaky):
    path, _, _ = leaky
    classifier, alignment = load_by_chromosome(path)
    # Make chrY an extreme point on one axis only.
    spiked = alignment.copy()
    spiked.loc["chrY", "L1HS"] = spiked["L1HS"].max() * 20
    table = jackknife_correlations(classifier, spiked).set_index(
        ["classifier_class", "alignment_class"]
    )
    row = table.loc[("NEGATIVE", "L1HS")]
    assert row["most_influential"] == "chrY"
    assert row["jackknife_range"] > 0.1
    assert abs(row["r_without_most_influential"] - row["pearson_r"]) > 0.1


def test_jackknife_is_stable_for_a_genuine_association(leaky):
    path, _, _ = leaky
    classifier, alignment = load_by_chromosome(path)
    table = jackknife_correlations(classifier, alignment).set_index(
        ["classifier_class", "alignment_class"]
    )
    row = table.loc[("L1PA2", "L1PA2")]
    assert row["jackknife_range"] < 0.2
    assert not row["swings_sign"]


# ── Composition transform ─────────────────────────────────────────────────────


def test_composition_correlations_are_flagged_as_closure_artifacts(leaky):
    path, _, _ = leaky
    classifier, alignment = load_by_chromosome(path)
    table = composition_correlations(classifier, alignment)
    assert table["closure_artifact"].all()
    assert len(table) == len(CLASSES) ** 2


# ── Leakage mechanism ─────────────────────────────────────────────────────────


def test_leakage_rates_are_read_weighted(tmp_path):
    confusion = pd.DataFrame(
        [
            {"subfamily": "L1HS", "collapsed": "L1HS", "n": 4000, "pct_NEGATIVE": 31.3},
            {"subfamily": "L1PA2", "collapsed": "L1PA", "n": 4000, "pct_NEGATIVE": 31.5},
            {"subfamily": "L1P3B", "collapsed": "L1PA", "n": 1000, "pct_NEGATIVE": 47.8},
            {"subfamily": "NEGATIVE", "collapsed": "NEGATIVE", "n": 4000, "pct_NEGATIVE": 96.2},
        ]
    )
    path = tmp_path / "subfamily_confusion.csv"
    confusion.to_csv(path, index=False)
    table = leakage_rates(path).set_index("group")
    assert table.loc["L1HS", "leak_to_negative_weighted_mean"] == pytest.approx(0.313)
    expected = (4000 * 31.5 + 1000 * 47.8) / 5000 / 100
    assert table.loc["L1PA", "leak_to_negative_weighted_mean"] == pytest.approx(expected)
    assert table.loc["L1PA", "leak_to_negative_max"] == pytest.approx(0.478)
    assert table.loc["NEGATIVE", "leak_to_negative_weighted_mean"] > 0.9


# ── Operating point ───────────────────────────────────────────────────────────


def test_operating_point_leakage_reports_both_decision_rules(tmp_path):
    """argmax and the filter threshold are different rules and both are shown."""
    # Three reads per class with hand-set logits: for the L1PA rows, P(NEGATIVE)
    # is the largest of the three but still below 0.5, so argmax calls NEGATIVE
    # while the 0.5 filter keeps the read.
    logits = np.array(
        [
            [5.0, 0.0, 0.0],  # L1HS, confidently correct
            [0.0, 0.0, 5.0],  # L1HS, confidently leaked
            [0.0, 1.0, 1.2],  # L1PA, argmax NEGATIVE but P < 0.5
            [0.0, 1.0, 1.2],  # L1PA, same
            [0.0, 0.0, 9.0],  # NEGATIVE, correct
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2])
    path = tmp_path / "logits.npz"
    np.savez(path, logits=logits, labels=labels, classes=np.array(["L1HS", "L1PA", "NEGATIVE"]))
    table = operating_point_leakage(path).set_index("true_class")
    assert table.loc["L1HS", "leak_argmax"] == pytest.approx(0.5)
    assert table.loc["L1HS", "leak_threshold"] == pytest.approx(0.5)
    # The rules disagree for L1PA: argmax leaks both reads, the threshold none.
    assert table.loc["L1PA", "leak_argmax"] == pytest.approx(1.0)
    assert table.loc["L1PA", "leak_threshold"] == pytest.approx(0.0)
    assert table.loc["NEGATIVE", "leak_argmax"] == pytest.approx(1.0)
    assert table["prior"].sum() == pytest.approx(1.0)


def test_operating_point_leakage_requires_a_negative_class(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        logits=np.zeros((2, 2), dtype=np.float32),
        labels=np.array([0, 1]),
        classes=np.array(["L1HS", "L1PA"]),
    )
    with pytest.raises(ValueError, match="no NEGATIVE class"):
        operating_point_leakage(path)


# ── Orchestration ─────────────────────────────────────────────────────────────


def test_negative_control_returns_every_table(leaky, annotation_bed, tmp_path):
    path, _, _ = leaky
    confusion = pd.DataFrame(
        [{"subfamily": "L1HS", "collapsed": "L1HS", "n": 10, "pct_NEGATIVE": 30.0}]
    )
    confusion_path = tmp_path / "cm.csv"
    confusion.to_csv(confusion_path, index=False)
    tables = negative_control(path, confusion_path=confusion_path, annotation_path=annotation_bed)
    assert set(tables) == {
        "negative_control_correlations",
        "negative_control_jackknife",
        "negative_control_composition",
        "negative_control_leakage",
    }


def test_negative_control_skips_optional_inputs(leaky):
    path, _, _ = leaky
    tables = negative_control(path)
    assert "negative_control_leakage" not in tables
    assert "negative_control_correlations" in tables


# ── Chromosome length ─────────────────────────────────────────────────────────


@pytest.fixture
def chrom_sizes(tmp_path):
    """A chrom.sizes carrying a comment header and a descending length gradient."""
    path = tmp_path / "test.chrom.sizes"
    lines = ["# a comment, as the real reference file has"]
    for position, chrom in enumerate(_CHROMS):
        lines.append(f"{chrom}\t{250_000_000 - 8_000_000 * position}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _length_split_frames(lengths: pd.Series, seed: int = 3469):
    """One pairing driven only by length, one carrying no length dependence.

    NEGATIVE on both axes is a multiple of chromosome length plus independent
    noise, so their correlation is entirely mediated by length and must collapse
    under adjustment.  L1HS is built flat in length but shared between the axes,
    so its correlation has nothing for the adjustment to take away.  Those are
    the two outcomes the adjustment has to tell apart.
    """
    rng = np.random.default_rng(seed)
    n = len(lengths)
    gb = lengths.to_numpy(dtype=float) / 1e9
    shared = rng.uniform(200, 600, n)
    alignment = pd.DataFrame(
        {
            "L1HS": shared + rng.normal(0, 5, n),
            "L1PA2": rng.uniform(800, 1200, n),
            "NEGATIVE": 500 * gb + rng.normal(0, 3, n),
        },
        index=lengths.index,
    )
    classifier = pd.DataFrame(
        {
            "L1HS": 0.7 * shared + rng.normal(0, 5, n),
            "L1PA2": rng.uniform(800, 1200, n),
            "NEGATIVE": 900 * gb + rng.normal(0, 4, n),
        },
        index=lengths.index,
    )
    return classifier, alignment


def test_load_chrom_lengths_skips_comments_and_extra_columns(chrom_sizes, tmp_path):
    lengths = load_chrom_lengths(chrom_sizes)
    assert len(lengths) == len(_CHROMS)
    assert lengths["chr1"] == pytest.approx(250_000_000)
    # A .fai carries three further columns; only the first two may be read.
    fai = tmp_path / "ref.fa.fai"
    fai.write_text("chr1\t248956422\t112\t60\t61\nchr2\t242193529\t253104910\t60\t61\n")
    from_fai = load_chrom_lengths(fai)
    assert from_fai["chr2"] == pytest.approx(242193529)


def test_chrom_length_correlations_reject_an_unmeasured_chromosome(leaky, chrom_sizes):
    _, classifier, alignment = leaky
    lengths = load_chrom_lengths(chrom_sizes).drop("chrX")
    with pytest.raises(ValueError, match="chrX"):
        chrom_length_correlations(classifier, alignment, lengths)


def test_length_mediated_association_collapses_under_adjustment(chrom_sizes):
    lengths = load_chrom_lengths(chrom_sizes)
    classifier, alignment = _length_split_frames(lengths)
    table = chrom_length_correlations(classifier, alignment, lengths)
    row = table[
        (table["classifier_class"] == "NEGATIVE") & (table["alignment_class"] == "NEGATIVE")
    ].iloc[0]
    assert row["pearson_r"] > 0.9
    assert abs(row["partial_r_given_chrom_length"]) < 0.4
    assert row["delta_r"] < -0.5


def test_length_free_association_survives_adjustment(chrom_sizes):
    lengths = load_chrom_lengths(chrom_sizes)
    classifier, alignment = _length_split_frames(lengths)
    table = chrom_length_correlations(classifier, alignment, lengths)
    row = table[(table["classifier_class"] == "L1HS") & (table["alignment_class"] == "L1HS")].iloc[
        0
    ]
    assert row["pearson_r"] > 0.9
    assert row["partial_r_given_chrom_length"] > 0.9
    assert abs(row["delta_r"]) < 0.05


def test_chrom_length_correlations_report_each_axis_dependence(chrom_sizes):
    lengths = load_chrom_lengths(chrom_sizes)
    classifier, alignment = _length_split_frames(lengths)
    table = chrom_length_correlations(classifier, alignment, lengths)
    assert len(table) == len(CLASSES) ** 2
    assert (table["n_chrom"] == len(_CHROMS)).all()
    # The marginal columns are a property of the axis, so they repeat by class.
    for klass in CLASSES:
        own = table[table["classifier_class"] == klass]["classifier_r_vs_length"]
        assert own.nunique() == 1
        other = table[table["alignment_class"] == klass]["alignment_r_vs_length"]
        assert other.nunique() == 1
    negative = table[table["classifier_class"] == "NEGATIVE"].iloc[0]
    l1hs = table[table["classifier_class"] == "L1HS"].iloc[0]
    assert negative["classifier_r_vs_length"] > 0.9
    assert abs(l1hs["classifier_r_vs_length"]) < 0.4
