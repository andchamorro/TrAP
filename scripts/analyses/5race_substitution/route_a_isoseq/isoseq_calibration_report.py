#!/usr/bin/env python3
"""HEK-293 calibration arm: does PacBio Iso-Seq reproduce the 5'RACE locus profile?

A second 5'RACE library, in a line with more active L1 loci, would strengthen the
locus-level comparison.  None exists: SRR4099955 is the only L1-targeted 5'RACE
deposit in SRA and ENA, so the fallback is to substitute PacBio Iso-Seq as the
reference protocol.
That substitution is only admissible if Iso-Seq measures the same thing 5'RACE
measures.  PRJNA1176011 contains an HEK-293 Iso-Seq arm, i.e. the *same cell line*
as the published 5'RACE reference (SRR4099955), so the substitution can be
measured rather than assumed.  This script performs that measurement.

Both sides are quantified over the identical ``L1EM.400`` SAF with the identical
featureCounts flags (``-F SAF -M --fraction``); only the aligner differs
(``minimap2 -ax splice:hq --secondary=no`` instead of ``bwa mem``), which long
reads require.  ``race5`` is defined exactly as in
``scripts/aggregate_srr_l1em_exp.py``: ``100 * count / total`` over all
L1EM.400 intervals, so it is a share of total L1 signal, not of the library.

    python scripts/analyses/5race_substitution/route_a_isoseq/isoseq_calibration_report.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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

#: G3b (plan sec. 4a): a percentage is depth-invariant but noise-satisfiable on a
#: shallow untargeted library, so the absolute support has to clear a floor too.
G3B_MIN_DENOMINATOR = 1_000
G3B_MIN_LOCUS_READS = 10

#: Reference-side threshold for "a locus carries real 5' signal".
RACE5_MIN_PCT = 1.0

#: The subfamilies Figure 11 quantifies; the published shape statistics in the
#: plan's sec. 13 are renormalised inside this set.
L1_TARGETS = ("L1HS", "L1PA2")

#: Display name -> column prefix, in reporting order.
ARMS = {
    "5'RACE (SRR4099955, bwa mem)": "race5",
    "Iso-Seq HEK-293 (4 Revio runs, minimap2)": "isoseq",
}


def load_counts(path: Path, label: str) -> pd.DataFrame:
    """featureCounts SAF output -> one row per interval with pooled counts.

    Every BAM column is summed, so a multi-run arm is pooled the way a single
    deeper library would be.  Per-run columns are retained for the replicate
    check.

    Args:
        path: ``counts.l1em.txt`` written by featureCounts.
        label: Column prefix for the emitted count/percentage pair.

    Returns:
        ``region, chrom, chromStart, chromEnd, target, category, strand,
        length, <label>_count, <label>_pct`` plus one ``<label>_count__<run>``
        column per input BAM.
    """
    df = pd.read_csv(path, sep="\t", comment="#")
    bam_columns = [c for c in df.columns[6:]]
    fam = df["Geneid"].str.extract(_FAMILY)
    fam.columns = ["target", "category", "region", "strand"]
    out = pd.concat(
        [
            df[["Chr", "Start", "End", "Length"]].rename(
                columns={
                    "Chr": "chrom",
                    "Start": "chromStart",
                    "End": "chromEnd",
                    "Length": "length",
                }
            ),
            fam,
        ],
        axis=1,
    )
    for column in bam_columns:
        run = Path(column).name.split(".")[0]
        out[f"{label}_count__{run}"] = df[column].astype(float)
    out[f"{label}_count"] = out[[c for c in out.columns if "_count__" in c]].sum(axis=1)
    total = out[f"{label}_count"].sum()
    out[f"{label}_pct"] = 100.0 * out[f"{label}_count"] / total if total else 0.0
    logger.info(
        f"{label}: {len(out)} intervals, {len(bam_columns)} BAM column(s), "
        f"{total:,.0f} assigned reads"
    )
    return out


def profile(pct: pd.Series, count: pd.Series) -> dict:
    """The shape statistics the plan asks for on a reference arm (sec. 5)."""
    ordered = pct.sort_values(ascending=False)
    nonzero = pct[pct > 0]
    return {
        "assigned_reads": float(count.sum()),
        "loci_detected": int((pct > 0).sum()),
        "loci_above_1pct": int((pct > RACE5_MIN_PCT).sum()),
        "top1_share_pct": float(ordered.iloc[0]) if len(ordered) else np.nan,
        "top3_share_pct": float(ordered.iloc[:3].sum()) if len(ordered) else np.nan,
        "median_pct_detected": float(nonzero.median()) if len(nonzero) else np.nan,
        "p90_pct_detected": float(nonzero.quantile(0.90)) if len(nonzero) else np.nan,
        "p99_pct_detected": float(nonzero.quantile(0.99)) if len(nonzero) else np.nan,
        "max_pct": float(pct.max()) if len(pct) else np.nan,
        "leverage_share_pct": leverage_share(pct.to_numpy()),
    }


def concordance(frame: pd.DataFrame, name: str) -> dict:
    """Agreement between the two reference protocols on one locus set."""
    race = frame["race5_pct"].to_numpy(dtype=float)
    iso = frame["isoseq_pct"].to_numpy(dtype=float)
    detected_race = race > 0
    detected_iso = iso > 0
    both = detected_race & detected_iso
    either = detected_race | detected_iso
    if frame.shape[0] >= 3 and np.ptp(race) > 0 and np.ptp(iso) > 0:
        rho = float(spearmanr(race, iso).statistic)
        logs = np.log10(1.0 + np.column_stack([race, iso]))
        r = float(np.corrcoef(logs[:, 0], logs[:, 1])[0, 1])
    else:
        rho, r = np.nan, np.nan
    return {
        "locus_set": name,
        "n_loci": int(frame.shape[0]),
        "race5_detected": int(detected_race.sum()),
        "isoseq_detected": int(detected_iso.sum()),
        "detected_both": int(both.sum()),
        "jaccard_detected": float(both.sum() / either.sum()) if either.sum() else np.nan,
        "spearman_rho": rho,
        "pearson_r2_log10": r**2 if np.isfinite(r) else np.nan,
    }


@app.command()
def main(
    isoseq_counts: Path = typer.Option(
        Path("results/isoseq_calibration/counts.l1em.isoseq_hek293.txt"),
        help="featureCounts table for the four HEK-293 Revio runs.",
    ),
    race5_counts: Path = typer.Option(
        Path("results/isoseq_calibration/counts.l1em.SRR4099955.txt"),
        help="Published 5'RACE featureCounts table (HEK293, SRR4099955).",
    ),
    figure_table: Path = typer.Option(
        Path("results/srr_experiment/l1em_salmon_fpm.csv"),
        help="Stage-A join, used only to recover the Figure 11 common locus set.",
    ),
    l1base_dir: Path = typer.Option(Path("data/ref/l1base")),
    out_dir: Path = typer.Option(Path("results/isoseq_calibration")),
    min_element_bp: int = typer.Option(MIN_ELEMENT_BP),
):
    """Join the two reference arms on the locus id and report the calibration."""
    iso = load_counts(isoseq_counts, "isoseq")
    race = load_counts(race5_counts, "race5")

    meta = ["region", "chrom", "chromStart", "chromEnd", "target", "category", "strand", "length"]
    joined = race.merge(
        iso[["region"] + [c for c in iso.columns if c.startswith("isoseq")]],
        on="region",
        how="inner",
        validate="one_to_one",
    )
    logger.info(f"joined {len(joined)} loci on region (race {len(race)}, iso {len(iso)})")

    elements = load_l1base_elements(l1base_dir, L1BASE_TRACKS, min_element_bp)
    joined = annotate_l1base_overlap(joined, elements)

    # --- arm profiles, under both published denominator conventions ---------
    # Plan sec. 13 quotes "9 loci above 1 %" on the global L1EM.400 denominator
    # but "top locus 16.5 %, top three 25.4 %" renormalised inside the
    # L1HS/L1PA2 set.  Reporting one convention would make the Iso-Seq arm look
    # either better or worse than the reference purely by bookkeeping, so both
    # are computed for both arms.
    conventions = [
        ("all L1EM.400 (global denominator)", joined, False),
        ("L1HS+L1PA2 (global denominator)", joined[joined["target"].isin(L1_TARGETS)], False),
        ("L1HS+L1PA2 (renormalised)", joined[joined["target"].isin(L1_TARGETS)], True),
    ]
    rows = []
    for convention, subset, renormalise in conventions:
        for arm, prefix in ARMS.items():
            count = subset[f"{prefix}_count"]
            pct = (
                100.0 * count / count.sum()
                if renormalise and count.sum()
                else subset[f"{prefix}_pct"]
            )
            rows.append({"convention": convention, "arm": arm, "n_loci": len(subset),
                         **profile(pct, count)})
    profiles = pd.DataFrame(rows)

    # --- concordance on nested locus sets -----------------------------------
    common = set(pd.read_csv(figure_table, usecols=["region"])["region"].unique())
    sets = {
        "all_L1EM.400": joined,
        "L1HS_L1PA2": joined[joined["target"].isin(L1_TARGETS)],
        "figure11_common": joined[joined["region"].isin(common)],
        "race5_detected": joined[joined["race5_pct"] > 0],
        "race5_above_1pct": joined[joined["race5_pct"] > RACE5_MIN_PCT],
    }
    for track in sorted(elements["track"].unique()) + ["union", "orf_defined"]:
        column = f"l1base_{track}"
        if column in joined:
            sets[f"l1base_{track}"] = joined[joined[column]]
    concord = pd.DataFrame([concordance(frame, name) for name, frame in sets.items()])

    # --- G3b: is the Iso-Seq arm deep enough for a percentage to mean anything?
    denominator = float(joined["isoseq_count"].sum())
    qualifying = joined[joined["isoseq_pct"] > RACE5_MIN_PCT]
    g3b = {
        "isoseq_assigned_reads": denominator,
        "denominator_floor": G3B_MIN_DENOMINATOR,
        "denominator_pass": bool(denominator >= G3B_MIN_DENOMINATOR),
        "loci_above_1pct": int(len(qualifying)),
        "loci_above_1pct_with_10plus_reads": int(
            (qualifying["isoseq_count"] >= G3B_MIN_LOCUS_READS).sum()
        ),
        "min_reads_at_qualifying_locus": float(qualifying["isoseq_count"].min())
        if len(qualifying)
        else np.nan,
        "locus_read_floor": G3B_MIN_LOCUS_READS,
    }
    g3b["g3b_pass"] = bool(
        g3b["denominator_pass"]
        and len(qualifying) > 0
        and g3b["loci_above_1pct_with_10plus_reads"] == len(qualifying)
    )

    # --- what Iso-Seq says about the loci 5'RACE calls dominant --------------
    dominant_columns = meta[:1] + ["target", "race5_count", "race5_pct", "isoseq_count", "isoseq_pct"]
    dominant = (
        joined.loc[joined["race5_pct"] > RACE5_MIN_PCT, dominant_columns]
        .sort_values("race5_pct", ascending=False)
        .reset_index(drop=True)
    )
    top_iso = (
        joined.loc[:, dominant_columns]
        .sort_values("isoseq_pct", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "isoseq_vs_race5_loci.csv", index=False)
    profiles.to_csv(out_dir / "isoseq_arm_profiles.csv", index=False)
    concord.to_csv(out_dir / "isoseq_concordance.csv", index=False)
    pd.DataFrame([g3b]).to_csv(out_dir / "isoseq_g3b.csv", index=False)
    dominant.to_csv(out_dir / "isoseq_at_race5_dominant_loci.csv", index=False)
    top_iso.to_csv(out_dir / "isoseq_top_loci.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\n## Arm profiles (shared L1EM.400 intervals, percent of total L1 signal)\n")
    print(profiles.to_string(index=False))
    print("\n## Protocol concordance\n")
    print(concord.to_string(index=False))
    print("\n## G3b depth gate (Iso-Seq arm)\n")
    for key, value in g3b.items():
        print(f"  {key}: {value}")
    print("\n## Iso-Seq at the loci 5'RACE calls dominant (> 1 % of 5'RACE signal)\n")
    print(dominant.to_string(index=False))
    print("\n## Iso-Seq's own top 15 loci\n")
    print(top_iso.to_string(index=False))
    print(f"\nwrote {out_dir}/isoseq_*.csv")


if __name__ == "__main__":
    app()
