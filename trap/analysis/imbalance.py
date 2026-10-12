"""Class imbalance: what was done about it, and how the deficit propagates (R2-5).

Reviewer 2's fifth comment asks three things — whether mitigation was explored,
how big the imbalance really is, and how the L1HS recall deficit propagates into
abundance estimates.  The first two are documentation; the third is an analysis
that had been promised in the response letter without being computed, and is the
substance of this module.

**The imbalance.** :func:`imbalance_ratios` reports the training-split counts
alongside the all-splits totals, because the reviewer quoted the all-splits L1HS
count (99,779) as a training count (60,639).  Worth noting in the response that
their *ratio* is right either way — 40.2:1 on the training split, 40.7:1 across
all splits — so this is a labelling correction, not a correction to their point.

**The mitigation.** :func:`balanced_class_weights` reproduces the weight vector
the reported model was actually trained with, from
``trap.modeling.train.compute_class_weights``' ``balanced`` scheme
(``w_c = N / (k * count_c)``).  It was used and never documented.

**The propagation.** This is the part that matters, and the arithmetic is not
the arithmetic of recall alone.  If a class has true abundance ``T`` and the
classifier assigns ``A`` reads to it, then ``precision = T * recall / A``, so
``A = T * recall / precision`` and the factor that recovers the true abundance
is ``precision / recall`` — **not** ``1 / recall``.  For L1HS at recall 0.624
and precision 0.723 that is 1.159 rather than 1.603: the contamination that
precision measures partly offsets the reads that recall loses, so L1HS abundance
is under-estimated by about 14 %, not the 38 % a recall-only reading implies.
:func:`propagation_factors` computes both so the difference is explicit.

:func:`race5_l1hs_sensitivity` then applies those factors to the L1HS loci of
the 5'RACE benchmark and refits.  The factors are applied **only to the
classifier-dependent configurations** — Salmon and L1EM never see the
classifier, so their estimates cannot carry its deficit, and leaving them
unchanged is the point of the comparison rather than an omission.

The result is reported as a sensitivity analysis and not as a corrected
estimate: a single global factor assumes the deficit is uniform across loci,
which cannot be checked from these data.  What it does establish is the
*direction* — correcting the deficit raises TrAP concordance, so the reported
result is achieved despite the deficit rather than because of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd

#: Task classes, in the label order used by the classifier.
CLASSES: Tuple[str, ...] = ("L1HS", "L1PA", "NEGATIVE")

#: Configurations whose estimates depend on the read classifier.
CLASSIFIER_DEPENDENT: Tuple[str, ...] = ("AlbertSalmon", "AlbertEM")

#: Methods compared on the 5'RACE benchmark.
METHODS: Tuple[str, ...] = ("AlbertSalmon", "Salmon", "AlbertEM", "L1EM")

#: Primary matched locus set: common to all methods and reference-detected.
TARGET_PATTERN = r"^(L1HS|L1PA2)"


def _to_int(value) -> int:
    """Parse a thousands-separated count."""
    return int(str(value).replace(",", "").strip())


def load_composition(path: Path) -> pd.DataFrame:
    """Load ``dataset_composition.csv`` with counts as integers.

    Args:
        path: CSV with a ``Split`` column and one column per class.

    Returns:
        Indexed by split, integer counts.
    """
    frame = pd.read_csv(path).set_index("Split")
    for column in frame.columns:
        frame[column] = frame[column].map(_to_int)
    return frame


def balanced_class_weights(
    counts: Dict[str, int] | pd.Series, classes: Sequence[str] = CLASSES
) -> pd.DataFrame:
    """The ``balanced`` weight vector the reported model was trained with.

    Reproduces ``trap.modeling.train.compute_class_weights``:
    ``w_c = N / (k * count_c)``, so the count-weighted mean weight is 1 and the
    loss scale is preserved.

    Args:
        counts: Per-class training-split counts.
        classes: Class order.

    Returns:
        One row per class with count, frequency and weight.
    """
    values = np.array([float(counts[c]) for c in classes])
    total = values.sum()
    weights = total / (len(classes) * values)
    frame = pd.DataFrame(
        {
            "class": list(classes),
            "n_train": values.astype(int),
            "frequency": values / total,
            "weight": weights,
        }
    )
    # The weighted mean of the weights is 1 by construction; assert it so a
    # future change to the scheme cannot silently rescale the loss.
    assert np.isclose((frame["frequency"] * frame["weight"]).sum(), 1.0)
    return frame


def imbalance_ratios(
    composition: pd.DataFrame,
    minority: str = "L1HS",
    majority: str = "L1PA",
) -> pd.DataFrame:
    """Imbalance ratio per split, with the all-splits total alongside.

    Args:
        composition: Table from :func:`load_composition`.
        minority: Rare class.
        majority: Abundant class.

    Returns:
        One row per split with both counts and their ratio.
    """
    rows = []
    for split in composition.index:
        n_min = int(composition.loc[split, minority])
        n_maj = int(composition.loc[split, majority])
        rows.append(
            {
                "split": split,
                f"n_{minority}": n_min,
                f"n_{majority}": n_maj,
                "ratio": n_maj / n_min if n_min else float("nan"),
                "minority_frequency": n_min / int(composition.loc[split, "Total"]),
            }
        )
    return pd.DataFrame(rows)


def propagation_factors(recall: float, precision: float) -> pd.DataFrame:
    """Abundance-correction factors implied by a class's recall and precision.

    Args:
        recall: Class recall.
        precision: Class precision.

    Returns:
        One row per correction with the factor and the implied bias.  The
        ``net`` row is the defensible one; ``recall_only`` is included because
        it is the reading the comment invites and overstates the bias.
    """
    rows = [
        {
            "correction": "recall_only",
            "factor": 1.0 / recall,
            "implied_underestimate": 1.0 - recall,
            "note": "what recall alone implies; ignores the contamination precision measures",
        },
        {
            "correction": "net",
            "factor": precision / recall,
            "implied_underestimate": 1.0 - recall / precision,
            "note": "A = T * recall / precision, so precision/recall recovers T",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.insert(0, "recall", recall)
    frame.insert(1, "precision", precision)
    return frame


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def load_matched_loci(
    path: Path, methods: Sequence[str] = METHODS, target_pattern: str = TARGET_PATTERN
) -> pd.DataFrame:
    """Matched 5'RACE locus set, retaining each locus' target subfamily.

    Args:
        path: Tidy ``l1em_salmon_fpm.csv``.
        methods: Methods to retain.
        target_pattern: Regex selecting benchmark targets.

    Returns:
        Loci with a finite estimate from every method and ``race5 > 0``.
    """
    tidy = pd.read_csv(path)
    benchmark = tidy[tidy["target"].str.match(target_pattern) & (tidy["category"] > 0)]
    wide = benchmark.pivot_table(
        index=["region", "race5", "target"], columns="method", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    matched = wide[wide[list(methods)].notna().all(axis=1) & (wide["race5"] > 0)]
    logger.info(
        f"matched set n={len(matched)} "
        f"({int((matched['target'] == 'L1HS').sum())} L1HS, "
        f"{int((matched['target'] == 'L1PA2').sum())} L1PA2)"
    )
    return matched.reset_index(drop=True)


def race5_l1hs_sensitivity(
    matched: pd.DataFrame,
    factors: pd.DataFrame,
    methods: Sequence[str] = METHODS,
    classifier_dependent: Sequence[str] = CLASSIFIER_DEPENDENT,
    minority_target: str = "L1HS",
) -> pd.DataFrame:
    """Refit the 5'RACE concordance with the L1HS deficit corrected.

    The correction is applied to the estimates at ``minority_target`` loci, for
    the classifier-dependent configurations only.

    Args:
        matched: Table from :func:`load_matched_loci`.
        factors: Table from :func:`propagation_factors`.
        methods: Methods to evaluate.
        classifier_dependent: Methods whose estimates pass through the
            classifier and can therefore carry its deficit.
        minority_target: Target label of the affected loci.

    Returns:
        One row per (correction, method) with the refit R² and the change.
    """
    is_minority = (matched["target"] == minority_target).to_numpy()
    race5 = matched["race5"].to_numpy(dtype=float)
    rows = []
    corrections = [("published", 1.0, "")] + [
        (row["correction"], row["factor"], row["note"]) for _, row in factors.iterrows()
    ]
    for name, factor, note in corrections:
        for method in methods:
            values = matched[method].to_numpy(dtype=float).copy()
            applied = method in classifier_dependent and name != "published"
            if applied:
                values = np.where(is_minority, values * factor, values)
            rows.append(
                {
                    "correction": name,
                    "factor": factor,
                    "method": method,
                    "classifier_dependent": method in classifier_dependent,
                    "correction_applied": applied,
                    "n": len(matched),
                    "n_corrected_loci": int(is_minority.sum()) if applied else 0,
                    "pearson_r2": _pearson_r2(race5, values),
                    "note": note,
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame["correction"] == "published"].set_index("method")["pearson_r2"]
    frame["delta_vs_published"] = frame.apply(
        lambda r: r["pearson_r2"] - baseline[r["method"]], axis=1
    )
    return frame


def paired_contrasts(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Within-quantifier ΔR² under each correction.

    Args:
        sensitivity: Table from :func:`race5_l1hs_sensitivity`.

    Returns:
        One row per (correction, contrast).
    """
    rows = []
    pivot = sensitivity.pivot(index="correction", columns="method", values="pearson_r2")
    for correction in pivot.index:
        for trap_cfg, baseline in (("AlbertSalmon", "Salmon"), ("AlbertEM", "L1EM")):
            if trap_cfg not in pivot.columns or baseline not in pivot.columns:
                continue
            delta = pivot.loc[correction, trap_cfg] - pivot.loc[correction, baseline]
            rows.append(
                {
                    "correction": correction,
                    "contrast": f"{trap_cfg}-{baseline}",
                    "delta_r2": float(delta),
                    "trap_wins": bool(delta > 0),
                }
            )
    return pd.DataFrame(rows)


#: Output stem -> builder.
OUTPUT_TABLES: Tuple[str, ...] = (
    "imbalance_class_weights",
    "imbalance_ratios",
    "imbalance_propagation_factors",
    "imbalance_race5_sensitivity",
    "imbalance_paired_contrasts",
)


def imbalance(
    composition_path: Path,
    metrics_path: Path,
    locus_table_path: Path,
    minority: str = "L1HS",
) -> Dict[str, pd.DataFrame]:
    """Compute every class-imbalance table.

    Args:
        composition_path: ``dataset_composition.csv``.
        metrics_path: ``per_class_metrics_table.csv``.
        locus_table_path: Tidy 5'RACE comparison table.
        minority: Rare class.

    Returns:
        Mapping of output stem (see :data:`OUTPUT_TABLES`) to table.
    """
    composition = load_composition(composition_path)
    metrics = pd.read_csv(metrics_path).set_index("Class")
    recall = float(metrics.loc[minority, "Recall"])
    precision = float(metrics.loc[minority, "Precision"])

    weights = balanced_class_weights(composition.loc["Training"])
    ratios = imbalance_ratios(composition, minority=minority)
    factors = propagation_factors(recall, precision)
    matched = load_matched_loci(locus_table_path)
    sensitivity = race5_l1hs_sensitivity(matched, factors, minority_target=minority)
    contrasts = paired_contrasts(sensitivity)

    logger.success(
        f"{minority} training-split imbalance {ratios.set_index('split').loc['Training', 'ratio']:.1f}:1 "
        f"(all splits {ratios.set_index('split').loc['Total', 'ratio']:.1f}:1); "
        f"class weight {weights.set_index('class').loc[minority, 'weight']:.2f}"
    )
    net = factors[factors["correction"] == "net"].iloc[0]
    logger.success(
        f"net abundance correction = precision/recall = {net['factor']:.3f} "
        f"({100 * net['implied_underestimate']:.1f}% under-estimate), versus "
        f"{100 * (1 - recall):.1f}% implied by recall alone"
    )
    for _, row in contrasts[contrasts["correction"] == "net"].iterrows():
        logger.success(
            f"  {row['contrast']} under the net correction: dR2 = {row['delta_r2']:+.4f}"
        )
    return {
        "imbalance_class_weights": weights,
        "imbalance_ratios": ratios,
        "imbalance_propagation_factors": factors,
        "imbalance_race5_sensitivity": sensitivity,
        "imbalance_paired_contrasts": contrasts,
    }
