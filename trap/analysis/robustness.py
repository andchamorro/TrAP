"""Robustness of the 5'RACE locus-level concordance benchmark (Figure 11).

Peer review raised three distinct objections to the headline 5'RACE result, and
this module answers all three from the same locus table so the numbers cannot
drift apart:

1. **Locus-set dependence** — the published panel applies
   ``race5 > 1 | value < 10``.  The reference-side arm keeps loci carrying more
   than 1 % of the total 5'RACE signal (``race5`` *is* a percentage); the
   estimate-side arm is an **RPM-scale support cut calibrated on L1EM output**
   so that the L1EM re-run reproduces L1EM's published locus-level values.
   Because the tidy table pools TPM (Salmon family) and RPM (EM family) in one
   ``value`` column, a single numeric cut is *not* scale-neutral: it removes
   ~42-44 % of TPM loci and <0.5 % of RPM loci.  :data:`LOCUS_SET_DEFINITIONS`
   therefore evaluates seven definitions (``F0``-``F6``) side by side, including
   a unit-aware variant that applies the cut only to the methods it was
   calibrated on.  The paired TrAP advantage is reported across all of them.
2. **Dependence on one point** — two separate loci get conflated in review.
   :data:`DOMINANT_LOCUS` is the high-leverage point (the largest ``race5``
   value in the benchmark); :data:`MAX_RESIDUAL_LOCUS` is the point the figure
   circles, which is the largest *residual* by construction and not influential
   at all.  :func:`leave_one_out` reports the fit with each, both, and neither,
   which separates leverage from residual explicitly.
3. **No formal test** — :func:`bootstrap_delta_r2` gives paired percentile
   confidence intervals on ΔR² between methods, resampling loci (not residuals)
   so both fits see the same resample.

Two descriptive tables support the orthogonal-annotation question:
:func:`l1base_overlap_counts` (how many comparison loci overlap an L1Base2
full-length intact L1, an annotation source independent of RepeatMasker) and
:func:`dynamic_range_summary` (per-method locus coverage and value range, which
is what makes the pooled-unit caveat concrete).

The primary analysis for the revision is :data:`PRIMARY_LOCUS_SET` — the common
locus set intersected with reference-detected loci (``race5 > 0``) — a matched
set of identical size for every method, which is the comparison the reviewers
actually asked for.  The published filter and its values are retained
unchanged; this is added material, not a replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Methods compared in Figure 11, in panel order.
METHODS: Tuple[str, ...] = ("AlbertSalmon", "Salmon", "AlbertEM", "L1EM")

#: Methods reporting TPM; the rest report RPM.  The published support cut was
#: calibrated on the RPM scale, so this split is what makes ``F6`` meaningful.
TPM_METHODS = frozenset({"AlbertSalmon", "Salmon"})

#: Reference-side arm: keep loci with more than this percentage of total
#: 5'RACE signal.  ``race5`` is already a percentage of the library total.
RACE5_MIN_PCT = 1.0

#: Estimate-side arm of the published filter, on the RPM scale.
VALUE_MAX_RPM = 10.0

#: Highest-leverage locus: the largest ``race5`` value in the benchmark
#: (L1HS, L1Base2 UID-135, race5 = 12.21).  Removing it lowers every method.
DOMINANT_LOCUS = "chr22:28663283-28669315"

#: The locus circled in the published panels.  It is ``argmax |value - pred|``
#: — the largest residual — so removing it *raises* every R².  Reviewers read
#: it as the influential point; it is not.
MAX_RESIDUAL_LOCUS = "chr1:174843226-174849243"

#: Only L1HS and L1PA2 targets with a non-zero category are in the benchmark.
TARGET_PATTERN = r"^(L1HS|L1PA2)"

#: Pre-specified primary locus set for the revision.
PRIMARY_LOCUS_SET = "F5_common_detected"

DEFAULT_SEED = 3469
DEFAULT_N_BOOT = 5000

#: Paired contrasts reported with bootstrap CIs.  The first two are
#: within-quantifier (both members share the quantifier, so the contrast
#: isolates the ALBERT read classifier); the last two are cross-quantifier and
#: are reported for completeness only.
DEFAULT_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("AlbertSalmon", "Salmon"),
    ("AlbertEM", "L1EM"),
    ("AlbertSalmon", "L1EM"),
    ("AlbertEM", "AlbertSalmon"),
)


# ── Locus-set definitions ─────────────────────────────────────────────────────
# Each definition is a mask over the wide table for one method.  They operate on
# the full table (``_pct`` columns and ``in_common_set`` are precomputed by
# :func:`load_locus_table`), after the caller has restricted to loci where that
# method emits a finite estimate.

MaskFn = Callable[[pd.DataFrame, str], np.ndarray]


def _mask_all_finite(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F0`` — every locus with a finite estimate from this method."""
    return np.ones(len(frame), dtype=bool)


def _mask_common(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F1`` — loci where all four methods emit a finite estimate."""
    return frame["in_common_set"].to_numpy()


def _mask_legacy_published(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F2`` — the filter as published in Figure 11."""
    return ((frame["race5"] > RACE5_MIN_PCT) | (frame[method] < VALUE_MAX_RPM)).to_numpy()


def _mask_legacy_scalefree(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F3`` — the same rule with both arms in percent-of-own-total."""
    return ((frame["race5"] > RACE5_MIN_PCT) | (frame[f"{method}_pct"] < RACE5_MIN_PCT)).to_numpy()


def _mask_support_floor(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F4`` — estimate-side support floor at 0.1 % of the method's own total."""
    return (frame[f"{method}_pct"] >= 0.1).to_numpy()


def _mask_common_detected(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F5`` — primary: common locus set restricted to reference-detected loci."""
    return (frame["in_common_set"] & (frame["race5"] > 0)).to_numpy()


def _mask_legacy_unit_aware(frame: pd.DataFrame, method: str) -> np.ndarray:
    """``F6`` — the published cut applied only to the RPM methods it was calibrated on."""
    if method in TPM_METHODS:
        return (frame["race5"] > RACE5_MIN_PCT).to_numpy()
    return _mask_legacy_published(frame, method)


LOCUS_SET_DEFINITIONS: Dict[str, MaskFn] = {
    "F0_all_finite": _mask_all_finite,
    "F1_common": _mask_common,
    "F2_legacy_published": _mask_legacy_published,
    "F3_legacy_scalefree": _mask_legacy_scalefree,
    "F4_support_floor": _mask_support_floor,
    "F5_common_detected": _mask_common_detected,
    "F6_legacy_unit_aware": _mask_legacy_unit_aware,
}

#: Provenance note carried into the output table, so the supplementary table is
#: readable without the manuscript text.
LOCUS_SET_NOTES: Dict[str, str] = {
    "F0_all_finite": "every locus with a finite estimate; not like-for-like",
    "F1_common": "common locus set, finite estimate from all four methods",
    "F2_legacy_published": (
        "AS PUBLISHED in Figure 11; L1EM-calibrated RPM support cut + 1% reference activity arm"
    ),
    "F3_legacy_scalefree": "same rule with both arms in percent-of-own-total",
    "F4_support_floor": ("estimate-side support floor >=0.1% of own total; not like-for-like"),
    "F5_common_detected": "NEW PRIMARY for the revision: common set restricted to race5>0",
    "F6_legacy_unit_aware": (
        "published cut applied only to the RPM methods it was calibrated on; "
        "n=9 on the TPM side, confirmatory only"
    ),
}

#: Definitions that score every method on the same locus set.  ``F0`` and ``F4``
#: do not: they score each method on its own detected subset, so L1EM is
#: evaluated on the ~15 % of loci where it emits an estimate at all while
#: AlbertEM is evaluated on a 281-locus superset.  Always carry that caveat.
LIKE_FOR_LIKE_DEFINITIONS: Tuple[str, ...] = (
    "F1_common",
    "F2_legacy_published",
    "F3_legacy_scalefree",
    "F5_common_detected",
    "F6_legacy_unit_aware",
)


# ── Statistics ────────────────────────────────────────────────────────────────


def pearson_r2(x: Sequence[float], y: Sequence[float]) -> float:
    """Squared Pearson correlation between *x* and *y*.

    Matches the ``r2`` used by the published notebooks: the square of the
    correlation coefficient, which for an OLS fit with intercept equals the
    coefficient of determination.

    Args:
        x: Reference values (5'RACE percentages).
        y: Estimated abundances.

    Returns:
        R² in ``[0, 1]``, or ``nan`` if either input is constant or shorter
        than three points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, reported alongside R² as a rank-robust check."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# ── Input table ───────────────────────────────────────────────────────────────


def load_locus_table(
    path: Path,
    methods: Sequence[str] = METHODS,
    target_pattern: str = TARGET_PATTERN,
) -> pd.DataFrame:
    """Load the tidy 5'RACE comparison table and pivot it to one row per locus.

    The tidy table (``results/srr_experiment/l1em_salmon_fpm.csv``, written by
    ``.trap/scripts/aggregate_srr_l1em_exp.py``) carries one row per locus x
    method with a single ``value`` column that pools TPM and RPM; the wide form
    keeps one column per method so every locus set is applied identically.

    Args:
        path: Tidy CSV with ``region, chrom, chromStart, chromEnd, target,
            category, strand, race5, method, value`` columns.
        methods: Methods to retain as columns.
        target_pattern: Regex matched against ``target`` to select the
            benchmark's L1HS/L1PA2 loci.

    Returns:
        One row per ``(region, race5)``, with a column per method, a
        ``<method>_pct`` column (percent of that method's own total, which makes
        the two unit scales comparable), ``in_common_set``, and the locus
        coordinates.
    """
    tidy = pd.read_csv(path)
    missing = set(methods) - set(tidy["method"].unique())
    if missing:
        raise ValueError(f"methods absent from {path}: {sorted(missing)}")

    benchmark = tidy[tidy["target"].str.match(target_pattern) & (tidy["category"] > 0)]
    wide = benchmark.pivot_table(
        index=["region", "race5"], columns="method", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for method in methods:
        wide[f"{method}_pct"] = 100.0 * wide[method] / wide[method].sum()
    wide["in_common_set"] = wide[list(methods)].notna().all(axis=1)

    coords = benchmark.drop_duplicates("region")[["region", "chrom", "chromStart", "chromEnd"]]
    wide = wide.merge(coords, on="region", how="left")

    logger.info(
        f"loaded {len(wide)} benchmark loci from {path.name}; "
        f"common set n={int(wide['in_common_set'].sum())}, "
        f"reference-detected n={int((wide['race5'] > 0).sum())}"
    )
    return wide


def locus_subset(wide: pd.DataFrame, locus_set: str, method: str) -> pd.DataFrame:
    """Rows of *wide* retained for *method* under locus-set definition *locus_set*.

    Args:
        wide: Table from :func:`load_locus_table`.
        locus_set: Key of :data:`LOCUS_SET_DEFINITIONS`.
        method: Method column to filter on.

    Returns:
        The retained rows, always first restricted to loci where *method* emits
        a finite estimate.
    """
    if locus_set not in LOCUS_SET_DEFINITIONS:
        raise KeyError(
            f"unknown locus set {locus_set!r}; expected one of {list(LOCUS_SET_DEFINITIONS)}"
        )
    finite = wide[wide[method].notna()]
    return finite[LOCUS_SET_DEFINITIONS[locus_set](finite, method)]


# ── Tables ────────────────────────────────────────────────────────────────────


def filter_definition_grid(wide: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Fit every method under every locus-set definition (answers R2-2).

    Args:
        wide: Table from :func:`load_locus_table`.
        methods: Methods to evaluate.

    Returns:
        One row per (definition, method) with ``n``, R², Spearman ρ, how many
        loci the definition removed, and the provenance note.
    """
    rows = []
    for locus_set in LOCUS_SET_DEFINITIONS:
        for method in methods:
            finite_n = int(wide[method].notna().sum())
            kept = locus_subset(wide, locus_set, method)
            rows.append(
                {
                    "filter_def": locus_set,
                    "method": method,
                    "unit": "TPM" if method in TPM_METHODS else "RPM",
                    "n": len(kept),
                    "pearson_r2": pearson_r2(kept["race5"], kept[method]),
                    "spearman_rho": spearman_rho(kept["race5"], kept[method]),
                    "n_removed": finite_n - len(kept),
                    "frac_removed": 1 - len(kept) / finite_n,
                    "note": LOCUS_SET_NOTES[locus_set],
                }
            )
    return pd.DataFrame(rows)


def paired_wins(grid: pd.DataFrame, like_for_like_only: bool = False) -> pd.DataFrame:
    """Count how often each TrAP configuration beats its matched baseline.

    AlbertSalmon/Salmon and AlbertEM/L1EM are within-quantifier pairs, so each
    contrast isolates the contribution of the ALBERT read classifier.

    Args:
        grid: Output of :func:`filter_definition_grid`.
        like_for_like_only: Restrict to :data:`LIKE_FOR_LIKE_DEFINITIONS`.

    Returns:
        One row per contrast with the number of definitions won and the total.
    """
    pivot = grid.pivot(index="filter_def", columns="method", values="pearson_r2")
    if like_for_like_only:
        pivot = pivot.loc[list(LIKE_FOR_LIKE_DEFINITIONS)]
    rows = []
    for trap_cfg, baseline in (("AlbertSalmon", "Salmon"), ("AlbertEM", "L1EM")):
        won = int((pivot[trap_cfg] > pivot[baseline]).sum())
        rows.append(
            {
                "contrast": f"{trap_cfg}-{baseline}",
                "n_definitions": len(pivot),
                "n_won": won,
                "like_for_like_only": like_for_like_only,
                "definitions_lost": ",".join(
                    pivot.index[pivot[trap_cfg] <= pivot[baseline]].tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out(
    wide: pd.DataFrame,
    locus_set: str = PRIMARY_LOCUS_SET,
    methods: Sequence[str] = METHODS,
    variants: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Refit after dropping the leverage locus, the circled locus, and both.

    This is the analysis that separates the two points reviewers conflate.
    Dropping :data:`DOMINANT_LOCUS` lowers R² for *every* method by a similar
    amount — the benchmark's dynamic range depends on it, but no method's
    advantage does.  Dropping :data:`MAX_RESIDUAL_LOCUS` *raises* every R²,
    because it is the largest residual rather than a high-leverage point.

    Args:
        wide: Table from :func:`load_locus_table`.
        locus_set: Locus-set definition to fit within.
        methods: Methods to evaluate.
        variants: Mapping of variant name to regions to drop.  Defaults to
            ``full`` / ``drop_dominant`` / ``drop_circled`` / ``drop_both``.

    Returns:
        One row per (variant, method) with ``n``, R², and Spearman ρ.
    """
    if variants is None:
        variants = {
            "full": (),
            "drop_dominant": (DOMINANT_LOCUS,),
            "drop_circled": (MAX_RESIDUAL_LOCUS,),
            "drop_both": (DOMINANT_LOCUS, MAX_RESIDUAL_LOCUS),
        }
    rows = []
    for variant, dropped in variants.items():
        for method in methods:
            kept = locus_subset(wide, locus_set, method)
            kept = kept[~kept["region"].isin(list(dropped))]
            rows.append(
                {
                    "primary_set": locus_set,
                    "variant": variant,
                    "n": len(kept),
                    "method": method,
                    "pearson_r2": pearson_r2(kept["race5"], kept[method]),
                    "spearman_rho": spearman_rho(kept["race5"], kept[method]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta_r2(
    wide: pd.DataFrame,
    locus_set: str = PRIMARY_LOCUS_SET,
    contrasts: Iterable[Tuple[str, str]] = DEFAULT_CONTRASTS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired percentile bootstrap confidence intervals on ΔR² (answers R2-3).

    Loci are resampled with replacement and **both** methods are refit on the
    same resample, so the interval is on the paired difference rather than on
    two independent fits.  Requires a locus set that is identical across
    methods (the default :data:`PRIMARY_LOCUS_SET` is, by construction).

    Args:
        wide: Table from :func:`load_locus_table`.
        locus_set: Locus-set definition; must be method-invariant.
        contrasts: ``(method_a, method_b)`` pairs; ΔR² is ``a - b``.
        n_boot: Bootstrap resamples.
        seed: RNG seed (the project-wide 3469).

    Returns:
        One row per contrast with the observed ΔR², the 95 % percentile
        interval, and the fraction of resamples at or below zero (a one-sided
        bootstrap p-value for "no improvement").
    """
    matched = locus_subset(wide, locus_set, METHODS[0])
    for method in METHODS[1:]:
        other = locus_subset(wide, locus_set, method)
        if set(other["region"]) != set(matched["region"]):
            raise ValueError(
                f"locus set {locus_set!r} is not method-invariant "
                f"(AlbertSalmon n={len(matched)}, {method} n={len(other)}); "
                "the paired bootstrap needs a matched set"
            )

    n = len(matched)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_boot, n))
    race5 = matched["race5"].to_numpy()

    rows = []
    for method_a, method_b in contrasts:
        y_a = matched[method_a].to_numpy()
        y_b = matched[method_b].to_numpy()
        draws = np.array(
            [pearson_r2(race5[j], y_a[j]) - pearson_r2(race5[j], y_b[j]) for j in index]
        )
        # A resample can be degenerate (every draw the same locus), which makes
        # R² undefined.  Those draws are dropped rather than silently poisoning
        # the percentiles; on the real benchmark (n = 156) there are none.
        finite = draws[np.isfinite(draws)]
        if len(finite) < len(draws):
            logger.warning(
                f"{method_a}-{method_b}: {len(draws) - len(finite)}/{len(draws)} "
                "bootstrap resamples were degenerate and were dropped"
            )
        lo, hi = np.percentile(finite, [2.5, 97.5])
        rows.append(
            {
                "primary_set": locus_set,
                "n": n,
                "contrast": f"{method_a}-{method_b}",
                "delta_r2": pearson_r2(race5, y_a) - pearson_r2(race5, y_b),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p_le_zero": float(np.mean(finite <= 0)),
                "n_boot": n_boot,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def l1base_overlap_counts(
    wide: pd.DataFrame,
    bed_path: Path,
    methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Count benchmark loci overlapping an L1Base2 full-length intact L1.

    L1Base2 is curated independently of RepeatMasker, so this is the orthogonal
    annotation R2-1 asks about.  It is reported **descriptively only**: an
    L1Base-restricted regression was tested and dropped, because the restricted
    set is dominated by a single active locus (see the returned counts) and so
    would be *more* leverage-dependent, not less.

    Args:
        wide: Table from :func:`load_locus_table`.
        bed_path: L1Base2 FLI-L1 BED (``data/ref/l1base/hsflil1_8438.bed``).
        methods: Methods whose coverage of the overlap set is reported.

    Returns:
        A single-row-per-stratum table of locus counts.
    """
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "uid"],
    )
    by_chrom = {chrom: group for chrom, group in bed.groupby("chrom")}

    def _overlaps(row: pd.Series) -> bool:
        group = by_chrom.get(row["chrom"])
        if group is None:
            return False
        return bool(
            ((group["start"] < row["chromEnd"]) & (group["end"] > row["chromStart"])).any()
        )

    overlap = wide.apply(_overlaps, axis=1)
    rows = [
        {"stratum": "all_benchmark_loci", "n": len(wide)},
        {"stratum": "l1base_overlap", "n": int(overlap.sum())},
        {"stratum": "l1base_overlap_race5_gt_0", "n": int((overlap & (wide["race5"] > 0)).sum())},
        {
            "stratum": "l1base_overlap_race5_gt_1pct",
            "n": int((overlap & (wide["race5"] > RACE5_MIN_PCT)).sum()),
        },
        {
            "stratum": "l1base_overlap_in_common_set",
            "n": int((overlap & wide["in_common_set"]).sum()),
        },
    ]
    for method in methods:
        rows.append(
            {
                "stratum": f"l1base_overlap_estimated_by_{method}",
                "n": int((overlap & wide[method].notna()).sum()),
            }
        )
    return pd.DataFrame(rows)


def dynamic_range_summary(wide: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Per-method locus coverage and value range under the published filter.

    This is the table that makes the pooled-unit caveat concrete: the TPM
    methods return an estimate at nearly every locus with values up to ~2e4,
    while the EM methods return estimates at a small minority of loci with
    values below ~25 RPM.  A single numeric cut on the pooled column therefore
    bites only on the TPM side.

    Args:
        wide: Table from :func:`load_locus_table`.
        methods: Methods to summarise.

    Returns:
        One row per method with coverage, published-filter counts, and range.
    """
    rows = []
    for method in methods:
        finite = wide[wide[method].notna()]
        published = locus_subset(wide, "F2_legacy_published", method)
        rows.append(
            {
                "method": method,
                "unit": "TPM" if method in TPM_METHODS else "RPM",
                "n_loci_estimated": len(finite),
                "frac_of_benchmark": len(finite) / len(wide),
                "n_published_filter": len(published),
                "frac_removed_by_filter": 1 - len(published) / len(finite),
                "value_max": float(finite[method].max()),
                "value_median": float(finite[method].median()),
            }
        )
    return pd.DataFrame(rows)


# ── Orchestration ─────────────────────────────────────────────────────────────

#: Output stem -> builder, in the order the revision reports them.
OUTPUT_TABLES: Tuple[str, ...] = (
    "robustness_filter_definitions",
    "robustness_leave_one_out",
    "robustness_bootstrap_dr2",
    "robustness_paired_wins",
    "robustness_l1base_overlap",
    "robustness_dynamic_range",
)


def robustness(
    table_path: Path,
    bed_path: Path | None = None,
    locus_set: str = PRIMARY_LOCUS_SET,
    methods: Sequence[str] = METHODS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, pd.DataFrame]:
    """Compute every robustness table from the tidy 5'RACE comparison table.

    Args:
        table_path: Tidy CSV (see :func:`load_locus_table`).
        bed_path: L1Base2 BED; the overlap table is skipped when ``None``.
        locus_set: Primary locus set for leave-one-out and the bootstrap.
        methods: Methods to evaluate.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Mapping of output stem (see :data:`OUTPUT_TABLES`) to table.
    """
    wide = load_locus_table(table_path, methods=methods)
    grid = filter_definition_grid(wide, methods=methods)
    tables = {
        "robustness_filter_definitions": grid,
        "robustness_leave_one_out": leave_one_out(wide, locus_set=locus_set, methods=methods),
        "robustness_bootstrap_dr2": bootstrap_delta_r2(
            wide, locus_set=locus_set, n_boot=n_boot, seed=seed
        ),
        "robustness_paired_wins": pd.concat(
            [paired_wins(grid, like_for_like_only=flag) for flag in (False, True)],
            ignore_index=True,
        ),
        "robustness_dynamic_range": dynamic_range_summary(wide, methods=methods),
    }
    if bed_path is not None:
        tables["robustness_l1base_overlap"] = l1base_overlap_counts(
            wide, bed_path, methods=methods
        )
    else:
        logger.warning("no L1Base BED given; skipping the orthogonal-annotation counts")

    wins = tables["robustness_paired_wins"]
    for _, row in wins[wins["like_for_like_only"]].iterrows():
        logger.success(
            f"{row['contrast']}: {row['n_won']}/{row['n_definitions']} like-for-like "
            "locus-set definitions"
        )
    return {stem: tables[stem] for stem in OUTPUT_TABLES if stem in tables}
