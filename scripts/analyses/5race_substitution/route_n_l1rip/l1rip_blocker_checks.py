#!/usr/bin/env python3
"""Route N blockers 2 and 3: turn the Grace outputs into verdicts.

BLOCKER 2 -- "not 5'-anchored by design".
``l1rip_fiveprime_profile.py`` measured where along an element reads start, with the
published 5'RACE library through the identical code path as a positive control.  The
test is read off two numbers: the control must show a pile-up at position 0 (if it
does not, the measurement is broken and nothing else here counts), and route N's
enrichment over the flat null says whether its ``five_prime`` column is a real 5'
signal or an artifact of the 909 bp window being 15 % of a 6 kb element.

BLOCKER 3 -- "the published result is not reproduced".
Whongsiri et al. report 90 elements detected with the top 10 carrying >60 % of
reads; our ``-M --fraction`` table gave 2,351 loci and 22.9 %.  The paper required
UNIQUE assignment, so the recount is the test.  The whole matrix of plausible
definitions is printed rather than the one that fits best: if several land near the
published pair the discrepancy was a flag choice, and if none does, the mismatch is
real and the arm stays unvalidated.  ``-M --fraction`` remains the primary table
either way -- it is what the reference used, and changing it would break the join.

    python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_blocker_checks.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

app = typer.Typer(add_completion=False, help=__doc__)

_FAMILY = r"([^.]+)\.([^.]+)\.([^:]+):(\d+)-(\d+)\.([+-])$"
L1_TARGETS = ("L1HS", "L1PA2")
FULL_LENGTH_BP = 6_000
RUNS = ("Library_8", "Library_10")
PAPER_LOCI_DETECTED = 90
PAPER_TOP10_SHARE_PCT = 60.0
#: How close a definition has to land to count as reproducing the paper.
LOCI_TOLERANCE = 3.0        # within 3x on locus count
SHARE_TOLERANCE = 15.0      # within 15 points on the top-10 share


def element_table(counts: Path, labels: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(counts, sep="\t", comment="#")
    bam_cols = [c for c in frame.columns if c.endswith(".bam")]
    if len(bam_cols) != len(labels):
        raise ValueError(f"{counts}: {len(bam_cols)} BAM columns vs {len(labels)} labels")
    meta = frame["Geneid"].str.extract(_FAMILY)
    out = pd.DataFrame({
        "Geneid": frame["Geneid"],
        "target": meta[0],
        "element_bp": meta[4].astype(int) - meta[3].astype(int),
    })
    for label, col in zip(labels, bam_cols):
        out[label] = frame[col].round().astype(np.int64)
    return out


def locus_stats(counts: np.ndarray) -> dict:
    nz = counts[counts > 0]
    order = np.sort(nz)[::-1]
    total = order.sum()
    return {
        "loci_detected": int(nz.size),
        "loci_ge10": int((nz >= 10).sum()),
        "reads": int(total),
        "top10_share_pct": float(100 * order[:10].sum() / total) if total else np.nan,
        "top1_share_pct": float(100 * order[0] / total) if total else np.nan,
    }


@app.command()
def main(
    unique_counts: Path = typer.Option(Path("results/l1rip_calibration/counts.l1em.unique.txt")),
    multi_counts: Path = typer.Option(Path("results/l1rip_calibration/counts.l1em.txt")),
    position_summary: Path = typer.Option(Path("results/l1rip_calibration/l1rip_position_summary.csv")),
    position_hist: Path = typer.Option(Path("results/l1rip_calibration/l1rip_position_hist.csv")),
    out_dir: Path = typer.Option(Path("results/l1rip_calibration")),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- blocker 2
    if position_summary.exists():
        pos = pd.read_csv(position_summary)
        control = pos[(pos["label"] == "reference_5pRACE")
                      & (pos["element_set"] == "full_length_young")]
        control_ok = bool(len(control) and control["enrich_first_15pct"].iloc[0] > 2.0)
        print("\n== BLOCKER 2: where do reads start along an element? ==")
        cols = ["label", "element_set", "reads", "median_norm_pos", "pct_in_first_1pct",
                "pct_in_first_15pct", "enrich_first_15pct", "pct_in_909bp_window"]
        print(pos[cols].to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
        print(f"\npositive control (5'RACE pile-up detected): {control_ok}")
        if not control_ok:
            logger.warning("control FAILED -- the positional measure is not trustworthy; "
                           "do not read route N's row")
        if position_hist.exists():
            hist = pd.read_csv(position_hist)
            lead = hist[hist["element_set"] == "full_length_young"].pivot(
                index="bin_lo", columns="label", values="enrichment_vs_uniform")
            print("\nenrichment vs uniform, by 5 % bin along the element "
                  "(1.00 = flat; full-length young elements):")
            print(lead.head(6).to_string(float_format=lambda v: f"{v:0.2f}"))
        pos.to_csv(out_dir / "l1rip_blocker2_position.csv", index=False)
    else:
        logger.warning(f"{position_summary} missing -- run l1rip_blocker_checks.slurm first")

    # ------------------------------------------------------------- blocker 3
    if not unique_counts.exists():
        logger.warning(f"{unique_counts} missing -- run l1rip_blocker_checks.slurm first")
        raise typer.Exit(0)

    uniq = element_table(unique_counts, RUNS)
    multi = element_table(multi_counts, RUNS)
    young = uniq["target"].isin(L1_TARGETS)
    full = uniq["element_bp"] >= FULL_LENGTH_BP
    subsets = {
        "all elements": np.ones(len(uniq), dtype=bool),
        "full-length (>=6 kb)": full.to_numpy(),
        "full-length young (L1HS/L1PA2)": (full & young).to_numpy(),
    }

    rows = []
    for assignment, table in (("unique (paper's convention)", uniq),
                              ("-M --fraction (ours, primary)", multi)):
        for set_name, mask in subsets.items():
            for arm in RUNS + ("pooled",):
                counts = (table[list(RUNS)].sum(axis=1) if arm == "pooled"
                          else table[arm]).to_numpy(dtype=np.int64)[mask]
                rows.append({"assignment": assignment, "element_set": set_name,
                             "arm": arm, **locus_stats(counts)})
    blocker3 = pd.DataFrame(rows)
    blocker3["loci_fold_vs_paper"] = blocker3["loci_detected"] / PAPER_LOCI_DETECTED
    blocker3["share_gap_vs_paper_pts"] = blocker3["top10_share_pct"] - PAPER_TOP10_SHARE_PCT
    blocker3["reproduces_paper"] = (
        (blocker3["loci_fold_vs_paper"].between(1 / LOCI_TOLERANCE, LOCI_TOLERANCE))
        & (blocker3["share_gap_vs_paper_pts"].abs() <= SHARE_TOLERANCE)
    )
    # the same question asked of the >=10-read set, which is the G3b-qualifying one
    blocker3["loci_ge10_fold_vs_paper"] = blocker3["loci_ge10"] / PAPER_LOCI_DETECTED
    blocker3.to_csv(out_dir / "l1rip_blocker3_published_check.csv", index=False)

    print(f"\n== BLOCKER 3: can the published 90 loci / >60 % top-10 be reproduced? "
          f"(tolerance: {LOCI_TOLERANCE:g}x on loci, {SHARE_TOLERANCE:g} pts on share) ==")
    show = ["assignment", "element_set", "arm", "reads", "loci_detected", "loci_ge10",
            "top10_share_pct", "loci_fold_vs_paper", "reproduces_paper"]
    print(blocker3[show].to_string(index=False, float_format=lambda v: f"{v:0.2f}"))
    hits = blocker3[blocker3["reproduces_paper"]]
    print(f"\ndefinitions landing near the published pair: {len(hits)} of {len(blocker3)}")
    if len(hits):
        print(hits[["assignment", "element_set", "arm", "loci_detected",
                    "top10_share_pct"]].to_string(index=False, float_format=lambda v: f"{v:0.1f}"))
    logger.success(f"wrote blocker CSVs to {out_dir}")


if __name__ == "__main__":
    app()
