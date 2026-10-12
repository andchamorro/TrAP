"""Stratified statistics for the synthetic insertion benchmark (Figure 10).

Reviewer 2's third comment makes two distinct complaints about Figure 10: the
headline R² = 0.99 is reported without any statement of what it aggregates over,
and the variation across replicates and across the deletion/mutation strata is
not quantified.  This module answers both from the two tables of record,

* ``results/synthetic/chr1/by_insertion_rate.csv`` — R² per method at each of
  the 9 insertion levels (2⁵ … 2¹³), and
* ``results/synthetic/chr1/by_del_probability.csv`` — R² per method at each of
  the 5 deletion probabilities (0 … 0.1),

which are the stored outputs of the generating notebook and are already tracked
in the repository.  Nothing is recomputed from reads: the design is a 9 × 5 × 146
factorial (insertion level × deletion probability × L1Base2 FLI-L1 element),
pooled per element, raw abundances, OLS R².

Three things the statistics establish, in decreasing order of how much they
matter to the response:

1. **Insertion load is the discriminating factor, not mutation rate.**
   :func:`stratum_gradients` reports the R² lost between the lowest and highest
   insertion level — 0.114 for AlbertSalmon against 0.32-0.62 for every
   comparator — and the total spread across deletion probability, which is
   ≤ 0.014 for every method.  The published caption attributed the comparators'
   weakness to "insertions with high mutation rates"; the figure's own strata do
   not support that, and the substitution is more favourable than the original
   claim.
2. **The ranking is consistent, not just favourable on average.**
   :func:`stratum_contrasts` counts strata won and runs a paired Wilcoxon
   signed-rank test per contrast.  At n = 9 strata the two-sided p floor is
   0.0039, so a "significant" p here means *every* stratum agreed and nothing
   more; the test is reported with ``p_floor`` alongside it so the p value
   cannot be read as an effect size.
3. **A paired interval on the mean difference.** :func:`bootstrap_stratum_delta`
   resamples **strata**, not elements — the per-element fits are not in the
   stored tables — so the interval describes the mean ΔR² across the design and
   is wide by construction at n = 9.  It is reported for consistency with the
   5'RACE analysis, not as the primary evidence, which is the win count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

#: The configuration the manuscript selects, used as the reference in contrasts.
REFERENCE_METHOD = "AlbertSalmon"

#: Within-quantifier pairs.  Each isolates the ALBERT read classifier, because
#: the two members share a quantifier.
PAIRED_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("AlbertSalmon", "Salmon"),
    ("AlbertEM", "L1EM"),
)

#: Stratification axes and their table of record.
AXES: Dict[str, str] = {
    "insertion_level": "by_insertion_rate.csv",
    "deletion_probability": "by_del_probability.csv",
}

DEFAULT_SEED = 3469
DEFAULT_N_BOOT = 5000


def load_strata(path: Path, axis: str) -> pd.DataFrame:
    """Load one stratum table into tidy form.

    The stored tables are method × stratum matrices with the method in the
    unnamed index column.

    Args:
        path: CSV path.
        axis: Axis label written into the ``axis`` column.

    Returns:
        One row per (axis, stratum, method) with an ``r2`` column.
    """
    wide = pd.read_csv(path, index_col=0)
    tidy = (
        wide.rename_axis("method")
        .rename_axis("stratum", axis=1)
        .stack()
        .rename("r2")
        .reset_index()
    )
    tidy["axis"] = axis
    tidy["stratum"] = tidy["stratum"].astype(str)
    logger.info(
        f"loaded {axis}: {tidy['method'].nunique()} methods x "
        f"{tidy['stratum'].nunique()} strata from {path.name}"
    )
    return tidy[["axis", "stratum", "method", "r2"]]


def load_all_strata(synthetic_dir: Path, axes: Dict[str, str] = AXES) -> pd.DataFrame:
    """Load every stratum table under *synthetic_dir* and concatenate."""
    frames = []
    for axis, filename in axes.items():
        path = Path(synthetic_dir) / filename
        if not path.exists():
            logger.warning(f"stratum table missing at {path}; skipping axis {axis}")
            continue
        frames.append(load_strata(path, axis))
    if not frames:
        raise FileNotFoundError(f"no stratum tables found under {synthetic_dir}")
    return pd.concat(frames, ignore_index=True)


def _pivot(strata: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Stratum × method matrix for one axis, strata in table order."""
    subset = strata[strata["axis"] == axis]
    order = list(dict.fromkeys(subset["stratum"]))
    return subset.pivot(index="stratum", columns="method", values="r2").loc[order]


def stratum_summary(strata: pd.DataFrame) -> pd.DataFrame:
    """Per-method dispersion across strata, and how often each method is best.

    Args:
        strata: Tidy table from :func:`load_all_strata`.

    Returns:
        One row per (axis, method) with ``n_strata``, mean, SD, min, max, and
        ``n_top`` — the number of strata in which that method has the highest
        R².  ``n_top`` is the statistic the response leads with, because it is
        what "consistent" means operationally.
    """
    rows = []
    for axis in dict.fromkeys(strata["axis"]):
        matrix = _pivot(strata, axis)
        best = matrix.idxmax(axis=1)
        for method in matrix.columns:
            values = matrix[method]
            rows.append(
                {
                    "axis": axis,
                    "method": method,
                    "n_strata": len(values),
                    "mean_r2": float(values.mean()),
                    "sd_r2": float(values.std(ddof=1)),
                    "min_r2": float(values.min()),
                    "max_r2": float(values.max()),
                    "n_top": int((best == method).sum()),
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_p_floor(n: int) -> float:
    """Smallest two-sided p the signed-rank test can return at *n* strata.

    With every difference the same sign, the two-sided p is ``2 / 2**n``.  The
    response quotes this alongside each p value so that p = 0.0039 at n = 9 is
    read as "all nine strata agreed", which is what it means, rather than as
    evidence of a large effect.

    Args:
        n: Number of paired strata.

    Returns:
        The attainable minimum, or ``nan`` when *n* is zero.
    """
    if n <= 0:
        return float("nan")
    return float(2.0 / (2.0**n))


def bootstrap_stratum_delta(
    deltas: Sequence[float], n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
) -> Tuple[float, float]:
    """Percentile CI on the mean paired ΔR², resampling strata.

    Args:
        deltas: Per-stratum differences (reference minus comparator).
        n_boot: Resamples.
        seed: RNG seed.

    Returns:
        ``(ci_lo, ci_hi)`` at 95 %, or ``(nan, nan)`` for fewer than 3 strata.
    """
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[index].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def stratum_contrasts(
    strata: pd.DataFrame,
    reference: str = REFERENCE_METHOD,
    paired: Iterable[Tuple[str, str]] = PAIRED_CONTRASTS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired per-stratum contrasts: win counts, Wilcoxon, and a bootstrap CI.

    Two families of contrast are emitted and labelled by ``family``:
    ``reference`` compares the selected configuration against every other
    method, and ``within_quantifier`` compares each TrAP configuration against
    the baseline sharing its quantifier — the comparison that actually isolates
    the read classifier.

    Args:
        strata: Tidy table from :func:`load_all_strata`.
        reference: Method used for the ``reference`` family.
        paired: Within-quantifier ``(trap_config, baseline)`` pairs.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        One row per (axis, family, contrast).
    """
    rows = []
    for axis in dict.fromkeys(strata["axis"]):
        matrix = _pivot(strata, axis)
        contrasts = [("reference", reference, m) for m in matrix.columns if m != reference]
        contrasts += [
            ("within_quantifier", a, b)
            for a, b in paired
            if a in matrix.columns and b in matrix.columns
        ]
        for family, method_a, method_b in contrasts:
            deltas = (matrix[method_a] - matrix[method_b]).to_numpy()
            n = len(deltas)
            if np.allclose(deltas, 0.0):
                p_value = float("nan")
            else:
                p_value = float(wilcoxon(deltas).pvalue)
            ci_lo, ci_hi = bootstrap_stratum_delta(deltas, n_boot=n_boot, seed=seed)
            losses = matrix.index[deltas <= 0].tolist()
            rows.append(
                {
                    "axis": axis,
                    "family": family,
                    "contrast": f"{method_a}-{method_b}",
                    "n_strata": n,
                    "n_won": int((deltas > 0).sum()),
                    "strata_lost": ",".join(losses),
                    "mean_delta_r2": float(deltas.mean()),
                    "min_delta_r2": float(deltas.min()),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "wilcoxon_p": p_value,
                    "p_floor": wilcoxon_p_floor(n),
                    "n_boot": n_boot,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def stratum_gradients(strata: pd.DataFrame) -> pd.DataFrame:
    """How much R² each method loses across each stratification axis.

    This is the table that replaces the caption's mutation-rate attribution.
    ``first``/``last`` follow the table's own stratum order, which is ascending
    for both axes, so ``drop`` is the loss from the mildest to the harshest
    stratum.  ``spread`` is the max-minus-min over all strata, which is the
    right summary for an axis with no monotone trend.

    Args:
        strata: Tidy table from :func:`load_all_strata`.

    Returns:
        One row per (axis, method).
    """
    rows = []
    for axis in dict.fromkeys(strata["axis"]):
        matrix = _pivot(strata, axis)
        for method in matrix.columns:
            values = matrix[method]
            rows.append(
                {
                    "axis": axis,
                    "method": method,
                    "first_stratum": matrix.index[0],
                    "last_stratum": matrix.index[-1],
                    "r2_first": float(values.iloc[0]),
                    "r2_last": float(values.iloc[-1]),
                    "drop": float(values.iloc[0] - values.iloc[-1]),
                    "spread": float(values.max() - values.min()),
                }
            )
    return pd.DataFrame(rows)


#: Output stem -> builder, in the order the revision reports them.
OUTPUT_TABLES: Tuple[str, ...] = (
    "synthetic_strata_summary",
    "synthetic_strata_contrasts",
    "synthetic_strata_gradients",
)


def synthetic_strata(
    synthetic_dir: Path,
    reference: str = REFERENCE_METHOD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, pd.DataFrame]:
    """Compute every stratified table from the synthetic tables of record.

    Args:
        synthetic_dir: Directory holding the two stratum CSVs.
        reference: Method used for the ``reference`` contrast family.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Mapping of output stem (see :data:`OUTPUT_TABLES`) to table.
    """
    strata = load_all_strata(synthetic_dir)
    tables = {
        "synthetic_strata_summary": stratum_summary(strata),
        "synthetic_strata_contrasts": stratum_contrasts(
            strata, reference=reference, n_boot=n_boot, seed=seed
        ),
        "synthetic_strata_gradients": stratum_gradients(strata),
    }
    summary = tables["synthetic_strata_summary"]
    for _, row in summary[summary["method"] == reference].iterrows():
        logger.success(
            f"{reference} on {row['axis']}: top in {row['n_top']}/{row['n_strata']} strata, "
            f"R2 {row['mean_r2']:.4f} +/- {row['sd_r2']:.4f}"
        )
    within = tables["synthetic_strata_contrasts"]
    for _, row in within[within["family"] == "within_quantifier"].iterrows():
        logger.success(
            f"{row['contrast']} on {row['axis']}: {row['n_won']}/{row['n_strata']} strata, "
            f"mean dR2 {row['mean_delta_r2']:+.4f}, Wilcoxon p {row['wilcoxon_p']:.4f} "
            f"(floor {row['p_floor']:.4f})"
        )
    return {stem: tables[stem] for stem in OUTPUT_TABLES}
