"""Is the NEGATIVE class behaving as a clean control? (Figure 13, R2-9)

Reviewer 2's ninth comment observes that the NEGATIVE class correlates with
BWA LINE-1 counts at r = 0.75 for L1PA2, and concludes it "is not behaving as a
clean control".  The manuscript's answer — that this "may reflect background
composition" — is an unsupported hand-wave, and this module replaces it with
tests that can come out either way.

Three things get computed, and the order matters because the first two were
expected to exonerate the NEGATIVE class and do not:

1. :func:`adjusted_correlations` — the composition explanation, tested by
   partial correlation on two covariates.  If the correlation is a composition
   artefact it should collapse.  **It does not, on either.**
2. :func:`jackknife_correlations` — leave-one-chromosome-out ranges for every
   pairing, because n = 24 and a single chromosome can carry a Pearson
   coefficient.  This is what shows the "NEGATIVE tracks L1PA2 but not L1HS"
   argument to be an artefact of one chromosome rather than a real asymmetry.
3. :func:`leakage_rates` — the mechanism, read off the subfamily confusion
   table: roughly a third of genuine L1HS and L1PA reads are assigned to
   NEGATIVE at the operating point.  A class containing that much LINE-1 *must*
   correlate with LINE-1 abundance, so the reviewer's reading is correct and the
   correlation needs no exotic explanation.

A note on what is deliberately **not** used as the headline.
:func:`composition_correlations` renormalises each class by its chromosome
total, which does drive the NEGATIVE/L1PA2 correlation to ~0 — but it drives
*every* off-diagonal strongly negative at the same time, because dividing three
components by their own sum imposes a sum-to-one constraint that induces
negative correlation by construction (the closure problem in compositional
data).  Quoting the ~0 from that transform while ignoring the induced
negatives would be a statistical sleight of hand.  It is computed, reported
beside its own caveat, and labelled ``closure_artifact = True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

#: Classes in the per-chromosome aggregation.
CLASSES: Tuple[str, ...] = ("L1HS", "L1PA2", "NEGATIVE")

#: The figure pairing the classifier against the alignment baseline.
MLVS_BWA_FIGURE = "SRR4099955_agreggate_mlvs_bwa_by_chr"

#: Method labels within that figure.
CLASSIFIER_METHOD = "Albert"
ALIGNMENT_METHOD = "Alignment"


def load_by_chromosome(
    path: Path,
    figure: str = MLVS_BWA_FIGURE,
    classifier: str = CLASSIFIER_METHOD,
    alignment: str = ALIGNMENT_METHOD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-chromosome class aggregation for one figure.

    Args:
        path: ``aggregate_by_chr.csv`` (``sample, figure, chrom, method, class,
            value``).
        figure: Figure key to select.
        classifier: Method label for the classifier.
        alignment: Method label for the alignment baseline.

    Returns:
        ``(classifier_frame, alignment_frame)``, each indexed by chromosome with
        one column per class, row order shared.
    """
    tidy = pd.read_csv(path)
    subset = tidy[tidy["figure"] == figure]
    if subset.empty:
        raise ValueError(f"figure {figure!r} not present in {path}")
    wide = subset.pivot_table(
        index="chrom", columns=["method", "class"], values="value", aggfunc="first"
    )
    missing = {classifier, alignment} - set(wide.columns.get_level_values(0))
    if missing:
        raise ValueError(f"methods absent from {figure}: {sorted(missing)}")
    logger.info(f"loaded {len(wide)} chromosomes from {path.name} ({figure})")
    return wide[classifier], wide[alignment]


def load_annotation_density(
    bed_path: Path, classes: Sequence[str] = ("L1HS", "L1PA2")
) -> pd.DataFrame:
    """Annotated LINE-1 loci per chromosome, from the benchmark annotation.

    This is the covariate the composition hypothesis actually needs: how much
    LINE-1 each chromosome *contains*, independent of how deeply it was
    sequenced.  The per-chromosome class totals in the aggregation cannot serve
    that role — in this sample the alignment baseline's three classes are 96 %
    LINE-1 (NEGATIVE is 692 of 28,435), so their sum is a LINE-1 abundance
    measure, not a library-depth measure.

    Args:
        bed_path: Annotation BED whose name field starts with the family
            (``L1PA6.0.chr1:33528-34041.-``).
        classes: Families to count individually.

    Returns:
        Indexed by chromosome, with one column per family plus ``annotated_l1``
        (all LINE-1 intervals on that chromosome).
    """
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 3],
        names=["chrom", "name"],
        dtype=str,
    )
    bed["family"] = bed["name"].str.split(".").str[0]
    counts = pd.DataFrame({"annotated_l1": bed.groupby("chrom").size()})
    for family in classes:
        counts[family] = bed[bed["family"] == family].groupby("chrom").size()
    counts = counts.fillna(0).astype(int)
    logger.info(
        f"annotation density from {bed_path.name}: {len(counts)} chromosomes, "
        f"{int(counts['annotated_l1'].sum()):,} LINE-1 intervals"
    )
    return counts


def _partial(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> Tuple[float, float]:
    """Pearson correlation of *x* and *y* after regressing both on *control*."""
    design = np.column_stack([np.ones_like(control), control])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    result = pearsonr(residual_x, residual_y)
    return float(result.statistic), float(result.pvalue)


def adjusted_correlations(
    classifier: pd.DataFrame,
    alignment: pd.DataFrame,
    classes: Sequence[str] = CLASSES,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Raw and covariate-adjusted correlations for every class pairing.

    Two controls, named for what they actually are:

    * ``l1_total`` — the alignment baseline's per-chromosome sum over classes.
      Because that sum is almost entirely LINE-1, adjusting for it asks whether
      the association survives *within* a given level of observed LINE-1
      signal.  It is **not** a library-depth control, and is not described as
      one.
    * ``annotated_l1`` — LINE-1 intervals per chromosome from the annotation
      (see :func:`load_annotation_density`).  This is external to the
      sequencing and is the honest test of the composition hypothesis.

    Args:
        classifier: Classifier frame from :func:`load_by_chromosome`.
        alignment: Alignment frame from :func:`load_by_chromosome`.
        classes: Classes to cross.
        annotation: Optional annotation-density frame; the second adjustment is
            omitted when ``None``.

    Returns:
        One row per (classifier class, alignment class).
    """
    l1_total = alignment[list(classes)].sum(axis=1).to_numpy()
    annotated = (
        annotation.reindex(classifier.index)["annotated_l1"].to_numpy(dtype=float)
        if annotation is not None
        else None
    )
    rows = []
    for own in classes:
        x = classifier[own].to_numpy(dtype=float)
        for other in classes:
            y = alignment[other].to_numpy(dtype=float)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            partial_r, partial_p = _partial(x, y, l1_total)
            row = {
                "classifier_class": own,
                "alignment_class": other,
                "n_chrom": len(x),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "partial_r_given_l1_total": partial_r,
                "partial_p_given_l1_total": partial_p,
            }
            if annotated is not None:
                ann_r, ann_p = _partial(x, y, annotated)
                row["partial_r_given_annotated_l1"] = ann_r
                row["partial_p_given_annotated_l1"] = ann_p
                row["composition_explains"] = bool(abs(ann_r) < 0.3 * abs(pearson.statistic))
            rows.append(row)
    return pd.DataFrame(rows)


def jackknife_correlations(
    classifier: pd.DataFrame,
    alignment: pd.DataFrame,
    classes: Sequence[str] = CLASSES,
) -> pd.DataFrame:
    """Leave-one-chromosome-out range of each correlation.

    With 24 chromosomes a single outlier can set a Pearson coefficient, so a
    coefficient is only usable as an argument if it survives dropping any one
    chromosome.  ``most_influential`` names the chromosome whose removal moves
    the correlation furthest, and ``swings_sign`` / ``range`` say whether the
    coefficient is stable enough to carry a claim.

    Args:
        classifier: Classifier frame from :func:`load_by_chromosome`.
        alignment: Alignment frame from :func:`load_by_chromosome`.
        classes: Classes to cross.

    Returns:
        One row per pairing with the full-sample r and the jackknife spread.
    """
    rows = []
    for own in classes:
        for other in classes:
            x = classifier[own].to_numpy(dtype=float)
            y = alignment[other].to_numpy(dtype=float)
            full = float(pearsonr(x, y).statistic)
            dropped = {}
            for position, chrom in enumerate(classifier.index):
                keep = np.arange(len(x)) != position
                dropped[chrom] = float(pearsonr(x[keep], y[keep]).statistic)
            values = pd.Series(dropped)
            influential = (values - full).abs().idxmax()
            rows.append(
                {
                    "classifier_class": own,
                    "alignment_class": other,
                    "pearson_r": full,
                    "jackknife_min": float(values.min()),
                    "jackknife_max": float(values.max()),
                    "jackknife_range": float(values.max() - values.min()),
                    "most_influential": influential,
                    "r_without_most_influential": float(values[influential]),
                    "swings_sign": bool(values.min() * values.max() < 0),
                }
            )
    return pd.DataFrame(rows)


def composition_correlations(
    classifier: pd.DataFrame,
    alignment: pd.DataFrame,
    classes: Sequence[str] = CLASSES,
) -> pd.DataFrame:
    """Correlations after expressing each class as a share of its own total.

    Reported for completeness and flagged, not used as evidence: see the module
    docstring.  Dividing three components by their sum makes them sum to one,
    which induces negative correlation between components regardless of the
    underlying biology, so a collapse to ~0 here cannot be read as "the
    association was a depth artefact".

    Returns:
        One row per pairing, every row carrying ``closure_artifact = True``.
    """
    own_total = classifier[list(classes)].sum(axis=1)
    other_total = alignment[list(classes)].sum(axis=1)
    rows = []
    for own in classes:
        x = (classifier[own] / own_total).to_numpy(dtype=float)
        for other in classes:
            y = (alignment[other] / other_total).to_numpy(dtype=float)
            result = pearsonr(x, y)
            rows.append(
                {
                    "classifier_class": own,
                    "alignment_class": other,
                    "pearson_r_composition": float(result.statistic),
                    "pearson_p_composition": float(result.pvalue),
                    "closure_artifact": True,
                }
            )
    return pd.DataFrame(rows)


def leakage_rates(confusion_path: Path) -> pd.DataFrame:
    """How much genuine LINE-1 the NEGATIVE class absorbs, by subfamily group.

    Read from the subfamily confusion table: ``pct_NEGATIVE`` is the percentage
    of reads from each annotated subfamily that the classifier assigns to
    NEGATIVE.  This is the mechanism behind Figure 13's correlation — it needs
    no appeal to library composition.

    Args:
        confusion_path: ``results/subfamily_confusion.<tokenizer>.csv``.

    Returns:
        One row per collapsed group (L1HS, L1PA, OTHER, NEGATIVE) with the
        read-weighted mean, median and range of the leak rate.
    """
    confusion = pd.read_csv(confusion_path)
    rows = []
    for group, frame in confusion.groupby("collapsed"):
        weights = frame["n"]
        rows.append(
            {
                "group": group,
                "n_subfamilies": len(frame),
                "n_reads": int(weights.sum()),
                "leak_to_negative_weighted_mean": float(
                    np.average(frame["pct_NEGATIVE"], weights=weights) / 100
                ),
                "leak_to_negative_median": float(frame["pct_NEGATIVE"].median() / 100),
                "leak_to_negative_min": float(frame["pct_NEGATIVE"].min() / 100),
                "leak_to_negative_max": float(frame["pct_NEGATIVE"].max() / 100),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def operating_point_leakage(logits_path: Path, negative_threshold: float = 0.5) -> pd.DataFrame:
    """Leak-to-NEGATIVE rates at the deployment prior, by both decision rules.

    :func:`leakage_rates` reads a *balanced* per-subfamily sample, which gives
    the spread across subfamilies but not the rate the pipeline actually runs
    at.  This reads the held-out evaluation logits, whose class mix is the
    deployment prior, and reports the rate under both rules that matter:

    * ``argmax`` — what Figure 13 plots, since the per-chromosome aggregation
      assigns each read to its highest-scoring class;
    * ``threshold`` — what the quantification filter applies, keeping reads with
      ``P(NEGATIVE) < negative_threshold`` (``trap.modeling.postprocessing``).

    The two are reported side by side because they are different operating
    points and should not be conflated, even when they agree numerically.

    Args:
        logits_path: ``.npz`` with ``logits``, ``labels``, ``classes``.
        negative_threshold: Discard threshold used by the filter.

    Returns:
        One row per true class.
    """
    from scipy.special import softmax

    data = np.load(logits_path, allow_pickle=True)
    logits = data["logits"]
    labels = data["labels"]
    classes = [str(c) for c in data["classes"]]
    if "NEGATIVE" not in classes:
        raise ValueError(f"no NEGATIVE class in {logits_path}: {classes}")
    probabilities = softmax(logits, axis=1)
    negative = classes.index("NEGATIVE")
    predicted = probabilities.argmax(axis=1)

    rows = []
    for index, name in enumerate(classes):
        mask = labels == index
        if not mask.any():
            continue
        rows.append(
            {
                "true_class": name,
                "n_reads": int(mask.sum()),
                "prior": float(mask.mean()),
                "leak_argmax": float((predicted[mask] == negative).mean()),
                "leak_threshold": float(
                    (probabilities[mask, negative] >= negative_threshold).mean()
                ),
                "mean_p_negative": float(probabilities[mask, negative].mean()),
                "negative_threshold": negative_threshold,
            }
        )
    return pd.DataFrame(rows)


#: Output stem -> builder.
OUTPUT_TABLES: Tuple[str, ...] = (
    "negative_control_correlations",
    "negative_control_jackknife",
    "negative_control_composition",
    "negative_control_leakage",
    "negative_control_operating_point",
)


def negative_control(
    by_chrom_path: Path,
    confusion_path: Path | None = None,
    annotation_path: Path | None = None,
    logits_path: Path | None = None,
    figure: str = MLVS_BWA_FIGURE,
) -> Dict[str, pd.DataFrame]:
    """Compute every NEGATIVE-class diagnostic table.

    Args:
        by_chrom_path: ``aggregate_by_chr.csv``.
        confusion_path: Subfamily confusion CSV; the leakage table is skipped
            when ``None``.
        annotation_path: LINE-1 annotation BED; the annotation-density
            adjustment is omitted when ``None``.
        logits_path: Held-out evaluation logits; the operating-point table is
            skipped when ``None``.
        figure: Figure key within the aggregation.

    Returns:
        Mapping of output stem (see :data:`OUTPUT_TABLES`) to table.
    """
    classifier, alignment = load_by_chromosome(by_chrom_path, figure=figure)
    annotation = load_annotation_density(annotation_path) if annotation_path is not None else None
    correlations = adjusted_correlations(classifier, alignment, annotation=annotation)
    tables = {
        "negative_control_correlations": correlations,
        "negative_control_jackknife": jackknife_correlations(classifier, alignment),
        "negative_control_composition": composition_correlations(classifier, alignment),
    }
    if confusion_path is not None:
        tables["negative_control_leakage"] = leakage_rates(confusion_path)
    else:
        logger.warning("no confusion table given; skipping the leakage mechanism")
    if logits_path is not None:
        leakage = operating_point_leakage(logits_path)
        tables["negative_control_operating_point"] = leakage
        for _, row in leakage.iterrows():
            logger.info(
                f"  {row['true_class']}: {100 * row['leak_argmax']:.1f}% assigned NEGATIVE "
                f"by argmax, {100 * row['leak_threshold']:.1f}% at the filter threshold"
            )

    focus = correlations[
        (correlations["classifier_class"] == "NEGATIVE")
        & (correlations["alignment_class"] == "L1PA2")
    ].iloc[0]
    logger.success(
        f"NEGATIVE vs alignment L1PA2: r={focus['pearson_r']:.3f}, "
        f"rho={focus['spearman_rho']:.3f}, "
        f"partial r | L1 total = {focus['partial_r_given_l1_total']:.3f} "
        f"(p={focus['partial_p_given_l1_total']:.4f})"
    )
    if "composition_explains" in focus:
        logger.success(
            f"  partial r | annotated L1 per chromosome = "
            f"{focus['partial_r_given_annotated_l1']:.3f} "
            f"(p={focus['partial_p_given_annotated_l1']:.4f})"
        )
        if not focus["composition_explains"]:
            logger.warning(
                "the composition explanation does NOT hold: the association survives "
                "adjustment for annotated LINE-1 content per chromosome"
            )
    return {stem: tables[stem] for stem in OUTPUT_TABLES if stem in tables}
