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

The orthogonal-annotation question (R2-1) is answered with three L1Base2
tracks rather than one: :data:`L1BASE_TRACKS` registers the ORF-intact sets
(FLI, ORF2 — membership decided by sequence-level ORF criteria, so genuinely
independent of the repeat annotation) and the much larger full-length set
(FLnI, >4500 nt over RepeatMasker-derived intervals, so *not* independent but
large enough to fit).  :func:`l1base_overlap_counts` reports coverage per
track and :func:`l1base_restricted_fits` refits the benchmark inside each
restriction, carrying :func:`leverage_share` so a restriction that is really
one locus cannot be mistaken for a result.  Restricting to element bodies
longer than :data:`MIN_ELEMENT_BP` is the biologically matched choice, since a
5'-truncated element cannot initiate its own transcription.
:func:`dynamic_range_summary` reports per-method locus coverage and value
range, which is what makes the pooled-unit caveat concrete.

The primary analysis for the revision is :data:`PRIMARY_LOCUS_SET` — the common
locus set intersected with reference-detected loci (``race5 > 0``) — a matched
set of identical size for every method, which is the comparison the reviewers
actually asked for.  The published filter and its values are retained
unchanged; this is added material, not a replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, NamedTuple, Sequence, Set, Tuple

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


class L1BaseTrack(NamedTuple):
    """One L1Base2 annotation track.

    Attributes:
        name: Short label used in output tables.
        filename: BED file name under the L1Base reference directory.
        flank_bp: **Total** flank (both sides combined) that L1Base adds around
            the element body in the deposited interval.  Derived empirically,
            not assumed: aligning each track against the unpadded L1 bodies in
            the benchmark annotation gives a median flank of 1006/1004 bp per
            side for FLI and ORF2 (so 2000 total), and a median left+right sum
            of 3999 bp for FLnI.  The FLnI value is corroborated by the track's
            own definition — its minimum span of 8500 bp minus 4000 reproduces
            L1Base's published ">4500 nt" cutoff exactly.
        orthogonal: Whether membership is decided by sequence-level criteria
            that are independent of the repeat annotation.  See
            :data:`L1BASE_TRACKS` for why this is not uniform across tracks.
        description: Human-readable definition.
    """

    name: str
    filename: str
    flank_bp: int
    orthogonal: bool
    description: str


#: L1Base2 (Ens84.38) tracks, in decreasing order of annotation independence.
#:
#: Only FLI and ORF2 are orthogonal in the sense R2-1 means: membership turns
#: on ORF intactness — conserved ORF1/ORF2 reading frames, intactness score —
#: which is judged from the sequence and not from a repeat annotation.  FLnI
#: membership turns on *length* (>4500 nt) over RepeatMasker-derived intervals
#: (hence the ``_rm`` file), so it is far larger and gives the statistical power
#: the ORF-defined tracks lack, but it is **not** an independent annotation
#: source.  Keep the distinction when reporting: FLnI answers "does the result
#: survive on full-length elements?", FLI/ORF2 answer "does it survive under an
#: annotation TrAP's labels were not derived from?".
L1BASE_TRACKS: Tuple[L1BaseTrack, ...] = (
    L1BaseTrack(
        "FLI",
        "hsflil1_8438.bed",
        2000,
        True,
        "full-length intact L1 (ORF1 and ORF2 conserved); 146 elements",
    ),
    L1BaseTrack(
        "ORF2",
        "hsorf2l1_8438.bed",
        2000,
        True,
        "ORF2-intact L1; 107 elements",
    ),
    L1BaseTrack(
        "FLnI",
        "hsflnil1_8438_rm.bed",
        4000,
        False,
        "full-length >4500 nt L1 with disrupted ORFs, RepeatMasker-derived; 13,418 elements",
    ),
)

#: Element-length floor, applied to the body length (span minus
#: :attr:`L1BaseTrack.flank_bp`).  A 5'RACE benchmark measures 5'-initiated
#: transcription, which a 5'-truncated element cannot support, so restricting to
#: near-full-length bodies is the biologically matched choice.  6 kb is just
#: below the ~6,030 bp median body of the annotated full-length L1s.
MIN_ELEMENT_BP = 6000

#: Name of the composite over every loaded track.
UNION_COMPOSITE = "union"

#: Name of the composite over the annotation-independent tracks only.
ORTHOGONAL_COMPOSITE = "orf_defined"


def composite_members(elements: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    """Composite restrictions, derived from the tracks actually loaded.

    Membership is computed from the element table rather than from a fixed list
    of track names, so a caller that loads a different (or partial) set of
    tracks still gets correct composites — and, in particular, a correct
    ``orthogonal`` flag.

    Args:
        elements: Table from :func:`load_l1base_elements`.

    Returns:
        Mapping of composite name to its member track names.  The
        orthogonal-only composite is omitted when it would duplicate the union.
    """
    tracks = tuple(sorted(elements["track"].unique()))
    orthogonal = tuple(sorted(elements.loc[elements["orthogonal"], "track"].unique()))
    composites = {UNION_COMPOSITE: tracks}
    if orthogonal:
        composites[ORTHOGONAL_COMPOSITE] = orthogonal
    return composites


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
    restrict_to: Set[str] | None = None,
    restriction_name: str = "none",
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
        restrict_to: Optional further restriction to these regions, e.g. an
            L1Base-defined subset from :func:`l1base_restrictions`.
        restriction_name: Label for that restriction, written to the output.

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

    if restrict_to is not None:
        matched = matched[matched["region"].isin(restrict_to)]
    n = len(matched)
    if n < 3:
        raise ValueError(f"restriction {restriction_name!r} leaves n={n} loci; too few to fit")
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
                "restriction": restriction_name,
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


def load_l1base_elements(
    l1base_dir: Path,
    tracks: Sequence[L1BaseTrack] = L1BASE_TRACKS,
    min_element_bp: int | None = MIN_ELEMENT_BP,
) -> pd.DataFrame:
    """Load L1Base2 BED tracks and filter them to near-full-length elements.

    The deposited intervals include L1Base's flanks, so the body length is
    ``span - flank_bp`` (see :class:`L1BaseTrack` for how each flank was
    established).  ``track``/``browser`` header lines are skipped.

    Args:
        l1base_dir: Directory holding the L1Base BED files.
        tracks: Tracks to load; missing files are skipped with a warning.
        min_element_bp: Keep elements whose body exceeds this length.  ``None``
            keeps every element.

    Returns:
        One row per element with ``track, chrom, start, end, uid, element_bp,
        orthogonal``.
    """
    frames = []
    for track in tracks:
        path = Path(l1base_dir) / track.filename
        if not path.exists():
            logger.warning(f"L1Base track {track.name} missing at {path}; skipping")
            continue
        bed = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["chrom", "start", "end", "uid"],
            dtype={0: str, 1: str, 2: str, 3: str},
            on_bad_lines="skip",
        )
        # BED files from L1Base carry a `track name=...` line, which pandas
        # would otherwise admit as a row with a junk chromosome.
        bed = bed[bed["chrom"].str.startswith("chr", na=False)].copy()
        bed[["start", "end"]] = bed[["start", "end"]].astype(int)
        bed["element_bp"] = bed["end"] - bed["start"] - track.flank_bp
        kept = bed if min_element_bp is None else bed[bed["element_bp"] > min_element_bp]
        logger.info(
            f"L1Base {track.name}: {len(kept)}/{len(bed)} elements with a body "
            f"> {min_element_bp} bp ({track.description})"
        )
        kept = kept.assign(track=track.name, orthogonal=track.orthogonal)
        frames.append(kept)
    if not frames:
        raise FileNotFoundError(f"no L1Base BED tracks found under {l1base_dir}")
    return pd.concat(frames, ignore_index=True)


def _overlap_mask(wide: pd.DataFrame, elements: pd.DataFrame) -> np.ndarray:
    """Boolean mask of *wide* rows overlapping any interval in *elements*.

    Overlap is half-open: intervals that merely abut do not overlap.
    """
    mask = np.zeros(len(wide), dtype=bool)
    locus_chrom = wide["chrom"].to_numpy()
    locus_start = wide["chromStart"].to_numpy()
    locus_end = wide["chromEnd"].to_numpy()
    for chrom, group in elements.groupby("chrom"):
        rows = np.flatnonzero(locus_chrom == chrom)
        if not len(rows):
            continue
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        hit = (starts[None, :] < locus_end[rows][:, None]) & (
            ends[None, :] > locus_start[rows][:, None]
        )
        mask[rows] = hit.any(axis=1)
    return mask


def annotate_l1base_overlap(wide: pd.DataFrame, elements: pd.DataFrame) -> pd.DataFrame:
    """Add one ``l1base_<track>`` flag per track, plus the composite flags.

    Args:
        wide: Table from :func:`load_locus_table`.
        elements: Table from :func:`load_l1base_elements`.

    Returns:
        A copy of *wide* with boolean membership columns.
    """
    annotated = wide.copy()
    for track in sorted(elements["track"].unique()):
        annotated[f"l1base_{track}"] = _overlap_mask(
            annotated, elements[elements["track"] == track]
        )
    for name, members in composite_members(elements).items():
        columns = [f"l1base_{m}" for m in members]
        annotated[f"l1base_{name}"] = annotated[columns].any(axis=1)
    return annotated


def l1base_restrictions(annotated: pd.DataFrame) -> Dict[str, Set[str]]:
    """Region sets for every L1Base membership column on *annotated*."""
    return {
        column[len("l1base_") :]: set(annotated.loc[annotated[column], "region"])
        for column in annotated.columns
        if column.startswith("l1base_")
    }


def l1base_overlap_counts(
    wide: pd.DataFrame,
    elements: pd.DataFrame,
    methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Count benchmark loci overlapping each L1Base2 track (answers R2-1).

    Args:
        wide: Table from :func:`load_locus_table`.
        elements: Table from :func:`load_l1base_elements`.
        methods: Methods whose coverage of each overlap set is reported.

    Returns:
        One row per restriction with locus counts by reference-activity
        stratum, plus how many loci each method estimates.
    """
    annotated = annotate_l1base_overlap(wide, elements)
    detected = annotated["race5"] > 0
    active = annotated["race5"] > RACE5_MIN_PCT
    rows = [
        {
            "restriction": "none",
            "n_elements": len(elements),
            "orthogonal": False,
            "n_loci": len(annotated),
            "n_race5_gt_0": int(detected.sum()),
            "n_race5_gt_1pct": int(active.sum()),
            "n_common_set": int(annotated["in_common_set"].sum()),
            "n_common_and_detected": int((annotated["in_common_set"] & detected).sum()),
            **{f"n_estimated_by_{m}": int(annotated[m].notna().sum()) for m in methods},
        }
    ]
    composites = composite_members(elements)
    for name, regions in l1base_restrictions(annotated).items():
        subset = annotated[annotated["region"].isin(regions)]
        members = composites.get(name, (name,))
        rows.append(
            {
                "restriction": name,
                "n_elements": int(elements["track"].isin(members).sum()),
                "orthogonal": bool(
                    elements.loc[elements["track"].isin(members), "orthogonal"].all()
                ),
                "n_loci": len(subset),
                "n_race5_gt_0": int((subset["race5"] > 0).sum()),
                "n_race5_gt_1pct": int((subset["race5"] > RACE5_MIN_PCT).sum()),
                "n_common_set": int(subset["in_common_set"].sum()),
                "n_common_and_detected": int(
                    (subset["in_common_set"] & (subset["race5"] > 0)).sum()
                ),
                **{f"n_estimated_by_{m}": int(subset[m].notna().sum()) for m in methods},
            }
        )
    return pd.DataFrame(rows)


def leverage_share(race5: Sequence[float]) -> float:
    """Percentage of the reference signal carried by its single largest locus.

    The feasibility statistic for a restricted regression: at 95 % one locus
    *is* the fit, at ~14 % the set has a usable spread.
    """
    race5 = np.asarray(race5, dtype=float)
    total = race5.sum()
    if total <= 0:
        return float("nan")
    return float(100.0 * race5.max() / total)


def l1base_restricted_fits(
    wide: pd.DataFrame,
    elements: pd.DataFrame,
    locus_set: str = PRIMARY_LOCUS_SET,
    methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Refit the benchmark on L1Base-restricted locus sets (answers R2-1).

    Each restriction is intersected with *locus_set*, so the comparison stays
    matched across methods, and is reported both whole and without its own
    highest-leverage locus.  ``leverage_share`` is what makes a restriction
    interpretable: the ORF-defined tracks alone leave a set in which one locus
    carries over 90 % of the reference signal, so their high R² is an artefact
    of a single point; adding the full-length track brings that down to a
    third, and the full-length track on its own to roughly a seventh.

    Args:
        wide: Table from :func:`load_locus_table`.
        elements: Table from :func:`load_l1base_elements`.
        locus_set: Matched locus set to intersect with.
        methods: Methods to evaluate.

    Returns:
        One row per (restriction, variant, method).
    """
    annotated = annotate_l1base_overlap(wide, elements)
    restrictions: Dict[str, Set[str] | None] = {"none": None}
    restrictions.update(l1base_restrictions(annotated))
    composites = composite_members(elements)

    rows = []
    for name, regions in restrictions.items():
        members = composites.get(name, (name,))
        orthogonal = (
            False
            if regions is None
            else bool(elements.loc[elements["track"].isin(members), "orthogonal"].all())
        )
        base = locus_subset(annotated, locus_set, methods[0])
        if regions is not None:
            base = base[base["region"].isin(regions)]
        if base.empty:
            logger.warning(f"L1Base restriction {name!r} leaves no loci; skipping")
            continue
        dominant = base.loc[base["race5"].idxmax(), "region"]
        for variant, dropped in (("full", ()), ("drop_dominant", (dominant,))):
            kept_regions = set(base["region"]) - set(dropped)
            for method in methods:
                kept = locus_subset(annotated, locus_set, method)
                kept = kept[kept["region"].isin(kept_regions)]
                rows.append(
                    {
                        "restriction": name,
                        "orthogonal": orthogonal,
                        "locus_set": locus_set,
                        "variant": variant,
                        "n": len(kept),
                        "leverage_share_pct": leverage_share(kept["race5"]),
                        "dominant_locus": dominant,
                        "method": method,
                        "pearson_r2": pearson_r2(kept["race5"], kept[method]),
                        "spearman_rho": spearman_rho(kept["race5"], kept[method]),
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
    "robustness_l1base_fits",
    "robustness_l1base_bootstrap",
    "robustness_dynamic_range",
)


def robustness(
    table_path: Path,
    l1base_dir: Path | None = None,
    locus_set: str = PRIMARY_LOCUS_SET,
    methods: Sequence[str] = METHODS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    min_element_bp: int | None = MIN_ELEMENT_BP,
) -> Dict[str, pd.DataFrame]:
    """Compute every robustness table from the tidy 5'RACE comparison table.

    Args:
        table_path: Tidy CSV (see :func:`load_locus_table`).
        l1base_dir: Directory of L1Base2 BED tracks; the orthogonal-annotation
            tables are skipped when ``None``.
        locus_set: Primary locus set for leave-one-out and the bootstrap.
        methods: Methods to evaluate.
        n_boot: Bootstrap resamples.
        seed: RNG seed.
        min_element_bp: Element-body length floor for L1Base elements.

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

    if l1base_dir is not None:
        elements = load_l1base_elements(l1base_dir, min_element_bp=min_element_bp)
        tables["robustness_l1base_overlap"] = l1base_overlap_counts(
            wide, elements, methods=methods
        )
        tables["robustness_l1base_fits"] = l1base_restricted_fits(
            wide, elements, locus_set=locus_set, methods=methods
        )
        # Bootstrap only the restrictions large enough to fit, so a two-locus
        # subset cannot produce a confident-looking interval.
        annotated = annotate_l1base_overlap(wide, elements)
        boots = []
        for name, regions in l1base_restrictions(annotated).items():
            try:
                boots.append(
                    bootstrap_delta_r2(
                        wide,
                        locus_set=locus_set,
                        contrasts=DEFAULT_CONTRASTS[:2],
                        n_boot=n_boot,
                        seed=seed,
                        restrict_to=regions,
                        restriction_name=name,
                    )
                )
            except ValueError as exc:
                logger.warning(f"L1Base restriction {name!r} not bootstrapped: {exc}")
        if boots:
            tables["robustness_l1base_bootstrap"] = pd.concat(boots, ignore_index=True)
    else:
        logger.warning("no L1Base directory given; skipping the orthogonal-annotation tables")

    wins = tables["robustness_paired_wins"]
    for _, row in wins[wins["like_for_like_only"]].iterrows():
        logger.success(
            f"{row['contrast']}: {row['n_won']}/{row['n_definitions']} like-for-like "
            "locus-set definitions"
        )
    return {stem: tables[stem] for stem in OUTPUT_TABLES if stem in tables}
