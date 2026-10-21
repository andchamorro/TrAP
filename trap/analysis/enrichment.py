"""Domain enrichment of top-attribution k-mers, including the ORF0 footprint.

Reviewer 2's tenth comment points at an inconsistency: the manuscript calls ORF0
"a primate-specific 5'UTR feature present within the L1HS/L1PA sequences
analyzed here", then reports no enrichment in ORF0.  The inconsistency is
sharper than the reviewer could see from outside — **there was no ORF0 bin.**
``results/interpret/domain_enrichment.csv`` has eight rows (5UTR, ORF1,
interORF, ORF2_EN, ORF2_mid, ORF2_RT, ORF2_C, 3UTR) and none of them is ORF0, so
the published sentence reported a result that had never been computed.

This module computes it, from the attributions that already exist — no model
re-run, no new inference:

* :func:`domain_enrichment` re-derives the published eight-row table, and is
  verified to reproduce it exactly (same top decile, same one-sided Fisher
  test, same counts).  It exists so the ORF0 row is produced by the same
  statistic as the rows beside it, rather than by a second implementation that
  might differ.
* :func:`subregion_enrichment` tests a nested feature — ORF0 — against a
  choice of universe, because two different questions are worth asking: is the
  ORF0 footprint enriched relative to the whole consensus (comparable to the
  eight published bins), and is it enriched relative to *the rest of the
  5'UTR* (the targeted question, which controls for whatever makes the 5'UTR as
  a whole unattributed).  Both are reported with a two-sided p value and a
  conditional odds-ratio confidence interval, so the answer is a bounded effect
  rather than a bare "not significant".

**What the result can and cannot say.** The footprint test is *positional*: a
k-mer is assigned to ORF0 when its L1.3 coordinate falls inside the ORF0
interval.  It cannot be attributed to ORF0's antisense reading frame
specifically, because the tokenizer is canonical — a k-mer and its reverse
complement collapse to one token — so an antisense ORF0 k-mer and the sense
5'UTR k-mer at the same position are *the same token by construction*.  Strand-
resolved ORF0 attribution is therefore impossible under this design, not merely
unmeasured, and that is the honest answer to the mechanistic half of the
comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio

from trap.analysis._l1_align import L1_3_REGIONS, L1_3_SUBREGIONS

#: Primary partition region names, in consensus order.
DOMAIN_ORDER: Tuple[str, ...] = tuple(name for name, _, _ in L1_3_REGIONS)

#: Attribution quantile defining the "top" set.  0.90 is the published choice.
TOP_QUANTILE = 0.90

#: Classes retained: the two LINE-1 task labels.  NEGATIVE reads carry no
#: meaningful L1 coordinate, and the published analysis excluded them.
POSITIVE_LABELS: Tuple[str, ...] = ("L1HS", "L1PA")


def load_attribution(
    path: Path,
    labels: Sequence[str] = POSITIVE_LABELS,
    domains: Sequence[str] = DOMAIN_ORDER,
) -> pd.DataFrame:
    """Load per-k-mer attributions and apply the published inclusion filter.

    Args:
        path: ``per_kmer_attribution.csv`` from :mod:`trap.analysis.interpret`.
        labels: Task labels to retain.
        domains: Regions to retain; drops ``unaligned`` and ``outside``.

    Returns:
        The retained rows, with an ``is_top`` flag marking the top attribution
        decile *computed on this subset* (as published).
    """
    frame = pd.read_csv(path)
    kept = frame[
        frame["label"].isin(list(labels))
        & (frame["domain"] != "unaligned")
        & frame["l1_pos"].notna()
        & frame["domain"].isin(list(domains))
    ].copy()
    threshold = kept["attribution"].quantile(TOP_QUANTILE)
    kept["is_top"] = kept["attribution"] >= threshold
    logger.info(
        f"{len(kept):,} aligned k-mers from {path.name}; "
        f"top-decile threshold {threshold:.6g}, {int(kept['is_top'].sum()):,} in the top set"
    )
    return kept


def _fisher_row(n_sub_top: int, n_sub: int, n_uni_top: int, n_uni: int) -> Dict[str, float]:
    """Enrichment, Fisher p values and an odds-ratio CI for one 2x2 table.

    The table is ``[[in-set top, in-set rest], [out-of-set top, out-of-set
    rest]]`` within the chosen universe.  ``p_enriched`` is the one-sided test
    the published table used; ``p_two_sided`` and the odds-ratio interval are
    added so a null result can be stated as a bound.
    """
    table = [
        [n_sub_top, n_sub - n_sub_top],
        [n_uni_top - n_sub_top, (n_uni - n_sub) - (n_uni_top - n_sub_top)],
    ]
    enrichment = (
        (n_sub_top / n_uni_top) / (n_sub / n_uni)
        if n_uni_top and n_sub and n_uni
        else float("nan")
    )
    result = odds_ratio(table)
    interval = result.confidence_interval(confidence_level=0.95)
    return {
        "n": n_sub,
        "n_top": n_sub_top,
        "bg_frac": n_sub / n_uni if n_uni else float("nan"),
        "top_frac": n_sub_top / n_uni_top if n_uni_top else float("nan"),
        "enrichment": enrichment,
        "p_enriched": float(fisher_exact(table, alternative="greater").pvalue),
        "p_depleted": float(fisher_exact(table, alternative="less").pvalue),
        "p_two_sided": float(fisher_exact(table, alternative="two-sided").pvalue),
        "odds_ratio": float(result.statistic),
        "or_ci_lo": float(interval.low),
        "or_ci_hi": float(interval.high),
        "expected_top": n_sub * n_uni_top / n_uni if n_uni else float("nan"),
    }


def _significance(p_value: float) -> str:
    """Published significance bands."""
    for bound, label in ((1e-4, "***"), (1e-2, "**"), (0.05, "*")):
        if p_value < bound:
            return label
    return "ns"


def domain_enrichment(
    attribution: pd.DataFrame, domains: Sequence[str] = DOMAIN_ORDER
) -> pd.DataFrame:
    """Re-derive the published eight-row domain enrichment table.

    Args:
        attribution: Table from :func:`load_attribution`.
        domains: Regions to report, in order.

    Returns:
        One row per domain, with the published columns plus the two-sided p and
        odds-ratio interval.
    """
    n_uni = len(attribution)
    n_uni_top = int(attribution["is_top"].sum())
    rows = []
    for domain in domains:
        subset = attribution[attribution["domain"] == domain]
        row = {"domain": domain, "universe": "consensus"}
        row.update(_fisher_row(int(subset["is_top"].sum()), len(subset), n_uni_top, n_uni))
        row["sig"] = _significance(row["p_enriched"])
        rows.append(row)
    return pd.DataFrame(rows)


def subregion_enrichment(
    attribution: pd.DataFrame,
    subregions: Sequence[Tuple[str, int, int, str]] = L1_3_SUBREGIONS,
) -> pd.DataFrame:
    """Test each nested feature against the consensus and against its parent.

    Args:
        attribution: Table from :func:`load_attribution`.
        subregions: ``(name, start, end, strand)`` tuples; defaults to
            :data:`trap.analysis._l1_align.L1_3_SUBREGIONS`, i.e. ORF0.

    Returns:
        Two rows per feature — ``universe = "consensus"`` and
        ``universe = "parent_region"`` — plus a ``strand_resolved`` column that
        is ``False`` for every antisense feature, recording that canonical
        k-mer tokenization cannot separate it from the sense strand.
    """
    position = attribution["l1_pos"]
    rows = []
    for name, start, end, strand in subregions:
        inside = (position >= start) & (position <= end)
        subset = attribution[inside]
        if subset.empty:
            logger.warning(f"subregion {name} contains no aligned k-mers; skipping")
            continue
        parent = attribution[attribution["domain"].isin(subset["domain"].unique())]
        universes = {
            "consensus": attribution,
            "parent_region": parent,
        }
        for universe_name, universe in universes.items():
            row = {
                "domain": name,
                "universe": universe_name,
                "start": start,
                "end": end,
                "strand": strand,
                "parent_region": ",".join(sorted(subset["domain"].unique())),
                # Canonical k-mers collapse a sequence with its reverse
                # complement, so an antisense feature cannot be distinguished
                # from the sense strand at the same coordinate.
                "strand_resolved": strand != "antisense",
            }
            row.update(
                _fisher_row(
                    int(subset["is_top"].sum()),
                    len(subset),
                    int(universe["is_top"].sum()),
                    len(universe),
                )
            )
            row["sig"] = _significance(row["p_enriched"])
            rows.append(row)
    return pd.DataFrame(rows)


#: Output stem -> builder.
OUTPUT_TABLES: Tuple[str, ...] = (
    "domain_enrichment.recomputed",
    "domain_enrichment.orf0",
)


def enrichment(attribution_path: Path) -> Dict[str, pd.DataFrame]:
    """Recompute the domain table and the nested-feature table.

    Args:
        attribution_path: ``per_kmer_attribution.csv``.

    Returns:
        Mapping of output stem (see :data:`OUTPUT_TABLES`) to table.
    """
    attribution = load_attribution(attribution_path)
    tables = {
        "domain_enrichment.recomputed": domain_enrichment(attribution),
        "domain_enrichment.orf0": subregion_enrichment(attribution),
    }
    for _, row in tables["domain_enrichment.orf0"].iterrows():
        direction = "depleted" if row["enrichment"] < 1 else "enriched"
        logger.success(
            f"{row['domain']} vs {row['universe']}: n={row['n']}, "
            f"{row['enrichment']:.3f}-fold ({direction}), "
            f"OR {row['odds_ratio']:.3f} [{row['or_ci_lo']:.3f}, {row['or_ci_hi']:.3f}], "
            f"two-sided p={row['p_two_sided']:.3g}"
        )
        if not row["strand_resolved"]:
            logger.warning(
                f"{row['domain']} is antisense: canonical k-mer tokenization collapses it "
                "with the sense strand, so this is a positional test only"
            )
    return tables


def verify_against_published(
    recomputed: pd.DataFrame, published_path: Path, rtol: float = 1e-6
) -> pd.DataFrame:
    """Check the recomputed domain table against the published CSV.

    Guards the ORF0 row: if the recomputation does not reproduce the eight
    published rows, the new row was not produced by the published statistic and
    must not be reported beside them.

    Args:
        recomputed: Output of :func:`domain_enrichment`.
        published_path: Published ``domain_enrichment.csv``.
        rtol: Relative tolerance for the numeric comparison.

    Returns:
        One row per domain with the published and recomputed values and a
        ``matches`` flag.
    """
    published = pd.read_csv(published_path).set_index("domain")
    rows = []
    for _, row in recomputed.iterrows():
        domain = row["domain"]
        if domain not in published.index:
            rows.append({"domain": domain, "matches": False, "note": "absent from published"})
            continue
        reference = published.loc[domain]
        matches = bool(
            int(reference["n"]) == int(row["n"])
            and int(reference["n_top"]) == int(row["n_top"])
            and np.isclose(reference["enrichment"], row["enrichment"], rtol=rtol)
            and np.isclose(reference["p_value"], row["p_enriched"], rtol=rtol, atol=1e-12)
        )
        rows.append(
            {
                "domain": domain,
                "n_published": int(reference["n"]),
                "n_recomputed": int(row["n"]),
                "enrichment_published": float(reference["enrichment"]),
                "enrichment_recomputed": float(row["enrichment"]),
                "matches": matches,
                "note": "" if matches else "MISMATCH",
            }
        )
    report = pd.DataFrame(rows)
    if report["matches"].all():
        logger.success(f"all {len(report)} published domain rows reproduced exactly")
    else:
        logger.error(f"{int((~report['matches']).sum())} domain rows do not reproduce")
    return report


# --------------------------------------------------------------------------- #
# Read-clustered re-analysis (Reviewer 3, Comment 9)                           #
# --------------------------------------------------------------------------- #


def _odds_ratio_and_enrichment(frame: pd.DataFrame, domain: str):
    """2x2 odds ratio and proportion-scale enrichment for *domain* vs the rest."""
    in_domain = frame["domain"].values == domain
    top = frame["is_top"].values
    a = int(np.count_nonzero(in_domain & top))
    b = int(np.count_nonzero(in_domain & ~top))
    c = int(np.count_nonzero(~in_domain & top))
    d = int(np.count_nonzero(~in_domain & ~top))
    if min(a, b, c, d) == 0:
        return float("nan"), float("nan")
    odds_ratio = (a * d) / (b * c)
    enrichment_ = (a / (a + c)) / ((a + b) / len(frame))
    return float(odds_ratio), float(enrichment_)


def clustered_enrichment(
    attribution_path: Path,
    domain: str = "ORF2_C",
    n_boot: int = 2000,
    seed: int = 3469,
) -> Dict[str, pd.DataFrame]:
    """Re-test the attribution enrichment with the *read* as the unit (R3, C9).

    Reviewer 3's objection is correct as stated: the published Fisher test treats
    overlapping stride-1 k-mers as independent, when in fact every k-mer of a read
    is one observation of the same fragment. This recomputes the same estimand
    three ways, none of which assumes k-mer independence:

    * **cluster bootstrap** — resample whole reads with replacement, so the
      confidence interval carries the within-read dependence rather than ignoring
      it. This is the headline correction and gives the honest interval;
    * **read-level test** — collapse each read to its own top-decile fraction and
      compare reads lying wholly inside the domain against reads lying wholly
      outside. The read is then unambiguously the unit of analysis;
    * **within-read permutation** — among reads that straddle the domain boundary,
      shuffle the top flags inside each read, which holds the read's composition
      fixed and asks only where within a read the attribution falls.

    The design effect (the ratio of the clustered to the naive standard error) is
    reported so the inflation of the published p-value is quantified rather than
    merely acknowledged.

    Returns:
        ``{"clustered_enrichment": frame, "clustered_reads": frame}``.
    """
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(seed)
    frame = load_attribution(attribution_path)
    n_reads = int(frame["read_uid"].nunique())
    point_or, point_enr = _odds_ratio_and_enrichment(frame, domain)

    groups = {uid: sub for uid, sub in frame.groupby("read_uid")}
    uids = np.array(list(groups))
    boot_or = np.empty(n_boot)
    boot_enr = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.choice(uids, size=len(uids), replace=True)
        sample = pd.concat([groups[u] for u in draw], ignore_index=True)
        boot_or[i], boot_enr[i] = _odds_ratio_and_enrichment(sample, domain)
    or_lo, or_hi = np.nanpercentile(boot_or, [2.5, 97.5])
    en_lo, en_hi = np.nanpercentile(boot_enr, [2.5, 97.5])

    in_domain = frame["domain"].values == domain
    top = frame["is_top"].values
    cells = [
        np.count_nonzero(in_domain & top),
        np.count_nonzero(in_domain & ~top),
        np.count_nonzero(~in_domain & top),
        np.count_nonzero(~in_domain & ~top),
    ]
    se_naive = float(np.sqrt(sum(1.0 / max(1, int(x)) for x in cells)))
    se_clustered = float(np.nanstd(np.log(boot_or)))
    design_effect = se_clustered / se_naive

    per_read = (
        frame.assign(_in=in_domain)
        .groupby("read_uid")
        .agg(n_kmers=("_in", "size"), n_in=("_in", "sum"), n_top=("is_top", "sum"))
    )
    per_read["top_fraction"] = per_read["n_top"] / per_read["n_kmers"]
    inside = per_read.loc[per_read["n_in"] == per_read["n_kmers"], "top_fraction"].values
    outside = per_read.loc[per_read["n_in"] == 0, "top_fraction"].values
    _, p_read = mannwhitneyu(inside, outside, alternative="greater")

    straddling = per_read.index[(per_read["n_in"] > 0) & (per_read["n_in"] < per_read["n_kmers"])]
    sub = frame[frame["read_uid"].isin(straddling)]
    observed = int(np.count_nonzero((sub["domain"].values == domain) & sub["is_top"].values))
    parts = [
        (gg["domain"].values == domain, int(gg["is_top"].sum()))
        for _, gg in sub.groupby("read_uid")
    ]
    null = np.empty(10000, dtype=int)
    for i in range(null.size):
        total = 0
        for flags, k in parts:
            if k:
                total += int(flags[rng.choice(flags.size, size=k, replace=False)].sum())
        null[i] = total
    p_within = float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))

    summary = pd.DataFrame(
        [
            {
                "domain": domain,
                "n_kmers": int(len(frame)),
                "n_reads": n_reads,
                "kmers_per_read": float(len(frame) / n_reads),
                "odds_ratio": point_or,
                "or_ci_lo_clustered": float(or_lo),
                "or_ci_hi_clustered": float(or_hi),
                "enrichment": point_enr,
                "enr_ci_lo_clustered": float(en_lo),
                "enr_ci_hi_clustered": float(en_hi),
                "se_log_or_naive": se_naive,
                "se_log_or_clustered": se_clustered,
                "design_effect": float(design_effect),
                "effective_n_kmers": float(len(frame) / design_effect**2),
                "n_reads_inside": int(inside.size),
                "n_reads_outside": int(outside.size),
                "mean_top_fraction_inside": float(inside.mean()),
                "mean_top_fraction_outside": float(outside.mean()),
                "read_level_ratio": float(inside.mean() / outside.mean()),
                "p_read_level": float(p_read),
                "n_reads_straddling": int(len(straddling)),
                "within_read_observed": observed,
                "within_read_null_mean": float(null.mean()),
                "p_within_read": p_within,
            }
        ]
    )
    logger.success(
        f"[clustered] {domain}: OR {point_or:.3f} "
        f"(read-clustered 95% CI {or_lo:.3f}-{or_hi:.3f}); "
        f"design effect x{design_effect:.1f}; read-level p={p_read:.4g}"
    )
    return {
        "clustered_enrichment": summary,
        "clustered_reads": per_read.reset_index(),
    }
