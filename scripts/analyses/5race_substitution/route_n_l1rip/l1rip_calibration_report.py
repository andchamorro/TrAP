#!/usr/bin/env python3
"""Route N: does an L1-ORF1p RIP nanopore library measure what L1 5'RACE measures?

WHY THIS ARM CANNOT BE CALIBRATED THE WAY THE ISO-SEQ ARM WAS
The Iso-Seq arm was decisive because PRJNA1176011 contained an HEK-293 library --
the *same cell line* as the published 5'RACE reference (SRR4099955) -- so protocol
was the only variable.  Route N is VM-Cub-1 and no L1 5'RACE library exists for it,
so concordance against SRR4099955 confounds cell line with protocol.  Three
line-independent things carry the verdict instead:

1. TARGETING.  An ORF1p pull-down should put a large share of its mapped reads on
   L1 intervals.  This is the mechanism that failed in the Iso-Seq arm, where L1
   reads were passengers inside abundant host transcripts, and it is the single
   reason this route is worth running at all.
2. PROTOCOL SHAPE.  5'RACE puts 74.0 % of its L1 signal in L1HS/L1PA2 and
   concentrates it on full-length young elements; Iso-Seq put 4.6 % there and its
   top locus was a 184 bp L1PA8 fragment.  Those numbers bracket the question, and
   neither depends on the cell line.
3. THE PUBLISHED RESULT.  Whongsiri et al. report 90 individual L1 elements
   detected with the top 10 carrying more than 60 % of reads.  Reproducing that
   order of magnitude from the raw deposit is a check on OUR pipeline, not on
   theirs -- their locus assignment and thresholds differ, so agreement is expected
   to be approximate.  A large miss means we are reading the data wrongly.

Route N is L1-targeted but NOT 5'-anchored, so ``five_prime`` and ``body_covered``
are reported separately throughout (see l1rip_fulllength.py).  Quantification uses
the identical ``L1EM.400`` SAF and featureCounts flags as the reference; only the
aligner differs (``minimap2 -ax splice --secondary=no``, ONT scoring).

    python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_calibration_report.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.stats import spearmanr

from trap.analysis.robustness import (
    L1BASE_TRACKS,
    MIN_ELEMENT_BP,
    annotate_l1base_overlap,
    leverage_share,
    load_l1base_elements,
)

app = typer.Typer(add_completion=False, help=__doc__)

#: ``family.category.locus.strand`` as emitted in the SAF GeneID column.
_FAMILY = r"([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)"

L1_TARGETS = ("L1HS", "L1PA2")
RACE5_MIN_PCT = 1.0
G3B_MIN_DENOMINATOR = 1_000
G3B_MIN_LOCUS_READS = 10

#: Protocol-shape anchors, measured on the same SAF with the same flags.
#: 5'RACE: SRR4099955.  Iso-Seq: the 4-run HEK-293 arm of PRJNA1176011, measured
#: by ``route_a_isoseq/isoseq_calibration_report.py`` (arm profiles, young share).
YOUNG_SHARE_RACE5_PCT = 74.0
YOUNG_SHARE_ISOSEQ_PCT = 4.6

#: Whongsiri et al., Int J Mol Sci 2020;21:9433, abstract.
PAPER_LOCI_DETECTED = 90
PAPER_TOP10_SHARE_PCT = 60.0

RUNS = ("Library_8", "Library_10")
RUN_OF_ACCESSION = {"SRR12474736": "Library_8", "SRR12474735": "Library_10"}
MEASURES = ("five_prime", "body_covered", "full_length")


def load_counts(path: Path) -> pd.DataFrame:
    """featureCounts SAF output -> one row per interval, one count column per run."""
    frame = pd.read_csv(path, sep="\t", comment="#")
    family = frame["Geneid"].str.extract(_FAMILY)
    family.columns = ["target", "category", "region", "strand"]
    out = pd.concat(
        [
            frame[["Chr", "Start", "End", "Length"]].rename(
                columns={"Chr": "chrom", "Start": "chromStart",
                         "End": "chromEnd", "Length": "length"}
            ),
            family,
        ],
        axis=1,
    )
    out["element_bp"] = out["chromEnd"] - out["chromStart"] - 800
    for column in frame.columns[6:]:
        accession = Path(column).name.split(".")[0]
        label = RUN_OF_ACCESSION.get(accession, accession)
        out[f"{label}_overlap"] = frame[column].astype(float)
        logger.info(f"{label}: {frame[column].sum():,.0f} whole-interval reads")
    return out


def targeting_rate(summary: Path) -> pd.DataFrame:
    """featureCounts .summary -> per-library assignment rate.

    The rate is over reads featureCounts SAW, which is the aligned set, so it
    answers "did the pull-down enrich L1" rather than "what fraction of the
    library mapped".
    """
    if not summary.exists():
        logger.warning(f"no featureCounts summary at {summary}; targeting rate skipped")
        return pd.DataFrame()
    frame = pd.read_csv(summary, sep="\t")
    frame = frame.set_index(frame.columns[0])
    rows = []
    for column in frame.columns:
        accession = Path(column).name.split(".")[0]
        total = float(frame[column].sum())
        assigned = float(frame.loc["Assigned", column])
        rows.append({
            "run": RUN_OF_ACCESSION.get(accession, accession),
            "reads_seen": total,
            "assigned_to_L1EM400": assigned,
            "l1_assignment_rate_pct": 100.0 * assigned / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def shape(count: pd.Series, element_bp: pd.Series, target: pd.Series) -> dict:
    """The line-independent protocol-shape statistics (see the module docstring)."""
    total = float(count.sum())
    if total <= 0:
        return {"assigned_reads": 0.0}
    pct = 100.0 * count / total
    ordered = count.sort_values(ascending=False)
    top10 = ordered.head(10)
    return {
        "assigned_reads": total,
        "young_share_pct": 100.0 * count[target.isin(L1_TARGETS)].sum() / total,
        "full_length_share_pct": 100.0 * count[element_bp >= MIN_ELEMENT_BP].sum() / total,
        "median_element_bp_top10": float(element_bp.loc[top10.index].median()),
        "loci_detected": int((count > 0).sum()),
        "loci_above_1pct": int((pct > RACE5_MIN_PCT).sum()),
        "top1_share_pct": float(100.0 * ordered.iloc[0] / total),
        "top10_share_pct": float(100.0 * top10.sum() / total),
        "leverage_share_pct": leverage_share(pct.to_numpy()),
    }


def g3b(count: pd.Series) -> dict:
    """Is this arm deep enough for a percentage to mean anything?"""
    total = float(count.sum())
    pct = 100.0 * count / total if total else count * 0.0
    qualifying = count[pct > RACE5_MIN_PCT]
    return {
        "assigned_reads": total,
        "denominator_pass": bool(total >= G3B_MIN_DENOMINATOR),
        "loci_above_1pct": int(len(qualifying)),
        "loci_above_1pct_with_10plus_reads": int((qualifying >= G3B_MIN_LOCUS_READS).sum()),
        "min_reads_at_qualifying_locus": float(qualifying.min()) if len(qualifying) else np.nan,
        "g3b_pass": bool(
            total >= G3B_MIN_DENOMINATOR
            and len(qualifying) > 0
            and bool((qualifying >= G3B_MIN_LOCUS_READS).all())
        ),
    }


def agreement(frame: pd.DataFrame, left: str, right: str, name: str) -> dict:
    """Spearman/Pearson/Jaccard between two count columns."""
    a = frame[left].to_numpy(dtype=float)
    b = frame[right].to_numpy(dtype=float)
    both = (a > 0) & (b > 0)
    either = (a > 0) | (b > 0)
    if frame.shape[0] >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
        rho = float(spearmanr(a, b).statistic)
        logs = np.log10(1.0 + np.column_stack([a, b]))
        pearson = float(np.corrcoef(logs[:, 0], logs[:, 1])[0, 1])
    else:
        rho, pearson = np.nan, np.nan
    return {
        "comparison": name,
        "left": left,
        "right": right,
        "n_loci": int(frame.shape[0]),
        "detected_both": int(both.sum()),
        "jaccard_detected": float(both.sum() / either.sum()) if either.sum() else np.nan,
        "spearman_rho": rho,
        "pearson_r2_log10": pearson**2 if np.isfinite(pearson) else np.nan,
    }


@app.command()
def main(
    l1rip_counts: Path = typer.Option(
        Path("results/l1rip_calibration/counts.l1em.txt"),
        help="featureCounts table for the two GridION runs.",
    ),
    fulllength_counts: Path = typer.Option(
        Path("results/l1rip_calibration/l1rip_fulllength_counts.csv"),
        help="Per-run 5'-end and body-coverage counts from l1rip_fulllength.py.",
    ),
    counts_summary: Path = typer.Option(
        Path("results/l1rip_calibration/counts.l1em.txt.summary"),
        help="featureCounts summary, for the L1 targeting rate.",
    ),
    race5_counts: Path = typer.Option(
        Path("results/isoseq_calibration/counts.l1em.SRR4099955.txt"),
        help="Published 5'RACE featureCounts table (HEK293) -- confounded comparator.",
    ),
    l1base_dir: Path = typer.Option(Path("data/ref/l1base")),
    out_dir: Path = typer.Option(Path("results/l1rip_calibration")),
    min_element_bp: int = typer.Option(MIN_ELEMENT_BP),
):
    """Report Route N's targeting, protocol shape, replicate agreement and depth."""
    overlap = load_counts(l1rip_counts)
    detail = pd.read_csv(fulllength_counts)
    detail["region"] = detail["Geneid"].str.extract(_FAMILY)[2]

    detail_columns = [
        c for c in detail.columns if re.search(r"_(five_prime|body_covered|full_length)$", c)
    ]
    joined = overlap.merge(
        detail[["region"] + detail_columns], on="region", how="left", validate="one_to_one"
    ).fillna(0.0)
    elements = load_l1base_elements(l1base_dir, L1BASE_TRACKS, min_element_bp)
    joined = annotate_l1base_overlap(joined, elements)
    logger.info(f"{len(joined):,} L1EM.400 intervals carrying Route N columns")

    targeting = targeting_rate(counts_summary)

    # --- protocol shape, on every measure the route distinguishes -------------
    profiles = pd.DataFrame([
        {"run": run, "measure": measure,
         **shape(joined[column], joined["element_bp"], joined["target"])}
        for run in RUNS
        for measure, column in
        [("whole-interval overlap", f"{run}_overlap")]
        + [(m, f"{run}_{m}") for m in MEASURES]
        if column in joined
    ])

    # --- the published result, as a check on our pipeline --------------------
    # Restricted to full-length elements, which is the population the paper
    # reports; the thresholds still differ, so this is an order-of-magnitude check.
    full_length_only = joined[joined["element_bp"] >= min_element_bp]
    paper_check = pd.DataFrame([
        {
            "run": run,
            "loci_detected_full_length_elements": int((full_length_only[f"{run}_overlap"] > 0).sum()),
            "paper_loci_detected": PAPER_LOCI_DETECTED,
            "top10_share_pct": shape(
                full_length_only[f"{run}_overlap"],
                full_length_only["element_bp"],
                full_length_only["target"],
            ).get("top10_share_pct", np.nan),
            "paper_top10_share_pct": PAPER_TOP10_SHARE_PCT,
        }
        for run in RUNS
        if f"{run}_overlap" in joined
    ])

    # --- replicate agreement, and the confounded cross-line comparison -------
    comparisons = [
        agreement(joined, f"{RUNS[0]}_{measure}", f"{RUNS[1]}_{measure}",
                  f"replicate agreement ({measure})")
        for measure in ("overlap",) + MEASURES
        if f"{RUNS[0]}_{measure}" in joined and f"{RUNS[1]}_{measure}" in joined
    ]
    race = pd.read_csv(race5_counts, sep="\t", comment="#")
    race["region"] = race["Geneid"].str.extract(_FAMILY)[2]
    race["race5_count"] = race[race.columns[6:-1]].sum(axis=1).astype(float)
    with_race = joined.merge(race[["region", "race5_count"]], on="region", how="inner")
    for run in RUNS:
        for measure in ("overlap", "five_prime"):
            column = f"{run}_{measure}"
            if column in with_race:
                comparisons.append(
                    agreement(with_race, "race5_count", column,
                              f"vs HEK-293 5'RACE -- CONFOUNDED ({run}, {measure})")
                )
    concord = pd.DataFrame(comparisons)

    depth = pd.DataFrame([
        {"run": run, "measure": measure, **g3b(joined[f"{run}_{measure}"])}
        for run in RUNS
        for measure in ("overlap",) + MEASURES
        if f"{run}_{measure}" in joined
    ])

    # --- verdict ------------------------------------------------------------
    pooled_five = joined[[f"{r}_five_prime" for r in RUNS]].sum(axis=1)
    pooled_overlap = joined[[f"{r}_overlap" for r in RUNS]].sum(axis=1)
    five_shape = shape(pooled_five, joined["element_bp"], joined["target"])
    rate = float(targeting["l1_assignment_rate_pct"].mean()) if len(targeting) else np.nan
    verdict = {
        "l1_assignment_rate_pct": rate,
        "targeting_worked": bool(np.isfinite(rate) and rate >= 5.0),
        "young_share_pct_pooled_5p": five_shape.get("young_share_pct", np.nan),
        "young_share_anchor_race5_pct": YOUNG_SHARE_RACE5_PCT,
        "young_share_anchor_isoseq_pct": YOUNG_SHARE_ISOSEQ_PCT,
        "closer_to_race5_than_isoseq": bool(
            abs(five_shape.get("young_share_pct", 0.0) - YOUNG_SHARE_RACE5_PCT)
            < abs(five_shape.get("young_share_pct", 0.0) - YOUNG_SHARE_ISOSEQ_PCT)
        ),
        "top_loci_are_full_length": bool(
            five_shape.get("median_element_bp_top10", 0.0) >= min_element_bp
        ),
        "g3b_pass_pooled_5p": bool(g3b(pooled_five)["g3b_pass"]),
        "replicate_rho_overlap": float(
            concord.loc[concord["comparison"] == "replicate agreement (overlap)",
                        "spearman_rho"].iloc[0]
        ) if len(concord) else np.nan,
    }
    verdict["route_n_survives"] = bool(
        verdict["targeting_worked"]
        and verdict["closer_to_race5_than_isoseq"]
        and verdict["top_loci_are_full_length"]
        and verdict["g3b_pass_pooled_5p"]
    )

    top = (
        joined.assign(pooled_five_prime=pooled_five, pooled_overlap=pooled_overlap)
        .loc[:, ["region", "target", "element_bp", "pooled_overlap", "pooled_five_prime"]
             + [f"{r}_full_length" for r in RUNS if f"{r}_full_length" in joined]]
        .sort_values("pooled_overlap", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "l1rip_loci.csv", index=False)
    profiles.to_csv(out_dir / "l1rip_arm_profiles.csv", index=False)
    targeting.to_csv(out_dir / "l1rip_targeting.csv", index=False)
    paper_check.to_csv(out_dir / "l1rip_published_result_check.csv", index=False)
    concord.to_csv(out_dir / "l1rip_agreement.csv", index=False)
    depth.to_csv(out_dir / "l1rip_g3b.csv", index=False)
    top.to_csv(out_dir / "l1rip_top_loci.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n## L1 targeting -- the mechanism that failed in the Iso-Seq arm\n")
    print(targeting.to_string(index=False) if len(targeting) else "  (no summary file)")
    print("\n## Protocol shape (percent of each measure's own total)\n")
    print(f"   anchors: 5'RACE young share = {YOUNG_SHARE_RACE5_PCT} %, "
          f"Iso-Seq = {YOUNG_SHARE_ISOSEQ_PCT} %\n")
    print(profiles.to_string(index=False))
    print("\n## Reproducing the published result (check on OUR pipeline)\n")
    print(paper_check.to_string(index=False))
    print("\n## Agreement: replicates, then the confounded cross-line comparison\n")
    print(concord.to_string(index=False))
    print("\n## G3b depth gate\n")
    print(depth.to_string(index=False))
    print("\n## Top 15 loci by pooled whole-interval signal\n")
    print(top.to_string(index=False))
    print("\n## Verdict\n")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {out_dir}/l1rip_*.csv")


if __name__ == "__main__":
    app()
