#!/usr/bin/env python3
"""Route O: does long-read CAGE measure what L1 5'RACE measures?

WHY THIS ARM CANNOT BE CALIBRATED THE WAY THE ISO-SEQ ARM WAS
The Iso-Seq arm was decisive because PRJNA1176011 contained an HEK-293 library --
the *same cell line* as the published 5'RACE reference (SRR4099955) -- so protocol
was the only variable and a direct locus-by-locus concordance was meaningful.
Route O is NCI-H1299 and no L1 5'RACE library exists for it, so concordance against
SRR4099955 confounds cell line with protocol and cannot on its own accept or reject
the route.  Two things are line-independent and carry the verdict instead:

1. PROTOCOL SHAPE.  What *kind* of L1 signal a protocol produces is a property of
   the protocol, not of the line.  5'RACE puts 74.0 % of its L1 signal in
   L1HS/L1PA2 and concentrates it on full-length young elements; Iso-Seq put 4.6 %
   there and its top locus was a 184 bp L1PA8 fragment.  Those two numbers bracket
   the question.
2. THE DEPOSIT'S OWN CONTROLS.  PRJNA936447 ships LRhex -- random-primed long reads
   from the same cells, platform and read length, with cap selection the ONLY
   difference -- and a DACSB arm in which L1s are driven up by decitabine + SB939.
   LRCAGE-vs-LRhex is an internal calibration of exactly the mechanism that killed
   Iso-Seq (passenger L1 fragments inside abundant host transcripts), and it needs
   no HEK-293 counterpart.

Both arms are quantified over the identical ``L1EM.400`` SAF with the identical
featureCounts flags (``-F SAF -M --fraction``); only the aligner differs
(``minimap2 -ax splice:hq --secondary=no`` instead of ``bwa mem``), as long reads
require.  ``race5`` keeps its published definition: ``100 * count / total`` over all
L1EM.400 intervals, a share of total L1 signal rather than of the library.

    python scripts/analyses/5race_substitution/route_o_lrcage/lrcage_calibration_report.py
"""

from __future__ import annotations

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

#: The subfamilies Figure 11 quantifies.
L1_TARGETS = ("L1HS", "L1PA2")

#: Reference-side threshold for "a locus carries real 5' signal".
RACE5_MIN_PCT = 1.0

#: G3b (plan sec. 4a): a percentage is depth-invariant but noise-satisfiable on a
#: shallow library, so absolute support has to clear a floor too.
G3B_MIN_DENOMINATOR = 1_000
G3B_MIN_LOCUS_READS = 10

#: Protocol-shape anchors, measured on the same SAF with the same flags.
#: 5'RACE: SRR4099955.  Iso-Seq: the 4-run HEK-293 arm of PRJNA1176011, measured
#: by ``route_a_isoseq/isoseq_calibration_report.py`` (arm profiles, young share).
YOUNG_SHARE_RACE5_PCT = 74.0
YOUNG_SHARE_ISOSEQ_PCT = 4.6

#: Run labels, in reporting order, matching lrcage_calibration_5p_ends.slurm.
RUNS = ("LRCAGE_DMSO", "LRCAGE_DACSB", "LRhex_DMSO")

#: Accession -> run label, for the featureCounts BAM columns.
RUN_OF_ACCESSION = {
    "SRR23517790": "LRCAGE_DMSO",
    "SRR23517787": "LRCAGE_DACSB",
    "SRR23517789": "LRhex_DMSO",
}


def load_counts(path: Path) -> pd.DataFrame:
    """featureCounts SAF output -> one row per interval, one count column per run."""
    frame = pd.read_csv(path, sep="\t", comment="#")
    family = frame["Geneid"].str.extract(_FAMILY)
    family.columns = ["target", "category", "region", "strand"]
    out = pd.concat(
        [
            frame[["Chr", "Start", "End", "Length"]].rename(
                columns={
                    "Chr": "chrom",
                    "Start": "chromStart",
                    "End": "chromEnd",
                    "Length": "length",
                }
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


def shape(count: pd.Series, element_bp: pd.Series, target: pd.Series) -> dict:
    """The line-independent protocol-shape statistics (see the module docstring)."""
    total = float(count.sum())
    if total <= 0:
        return {"assigned_reads": 0.0}
    pct = 100.0 * count / total
    young = count[target.isin(L1_TARGETS)].sum()
    ordered = pct.sort_values(ascending=False)
    top10 = count.sort_values(ascending=False).head(10)
    full_length = count[element_bp >= MIN_ELEMENT_BP].sum()
    return {
        "assigned_reads": total,
        "young_share_pct": 100.0 * young / total,
        "full_length_share_pct": 100.0 * full_length / total,
        "median_element_bp_top10": float(element_bp.loc[top10.index].median()),
        "loci_detected": int((count > 0).sum()),
        "loci_above_1pct": int((pct > RACE5_MIN_PCT).sum()),
        "top1_share_pct": float(ordered.iloc[0]),
        "top3_share_pct": float(ordered.iloc[:3].sum()),
        "leverage_share_pct": leverage_share(pct.to_numpy()),
    }


def g3b(count: pd.Series) -> dict:
    """Is this arm deep enough for a percentage to mean anything?"""
    total = float(count.sum())
    pct = 100.0 * count / total if total else count * 0.0
    qualifying = count[pct > RACE5_MIN_PCT]
    passed = (
        total >= G3B_MIN_DENOMINATOR
        and len(qualifying) > 0
        and bool((qualifying >= G3B_MIN_LOCUS_READS).all())
    )
    return {
        "assigned_reads": total,
        "denominator_pass": bool(total >= G3B_MIN_DENOMINATOR),
        "loci_above_1pct": int(len(qualifying)),
        "loci_above_1pct_with_10plus_reads": int((qualifying >= G3B_MIN_LOCUS_READS).sum()),
        "min_reads_at_qualifying_locus": float(qualifying.min()) if len(qualifying) else np.nan,
        "g3b_pass": bool(passed),
    }


def concordance(frame: pd.DataFrame, reference: str, arm: str, name: str) -> dict:
    """Agreement on one locus set.  CONFOUNDED: different cell lines (see docstring)."""
    ref = frame[reference].to_numpy(dtype=float)
    own = frame[arm].to_numpy(dtype=float)
    both = (ref > 0) & (own > 0)
    either = (ref > 0) | (own > 0)
    if frame.shape[0] >= 3 and np.ptp(ref) > 0 and np.ptp(own) > 0:
        rho = float(spearmanr(ref, own).statistic)
        logs = np.log10(1.0 + np.column_stack([ref, own]))
        pearson = float(np.corrcoef(logs[:, 0], logs[:, 1])[0, 1])
    else:
        rho, pearson = np.nan, np.nan
    return {
        "locus_set": name,
        "arm": arm,
        "n_loci": int(frame.shape[0]),
        "detected_both": int(both.sum()),
        "jaccard_detected": float(both.sum() / either.sum()) if either.sum() else np.nan,
        "spearman_rho": rho,
        "pearson_r2_log10": pearson**2 if np.isfinite(pearson) else np.nan,
    }


@app.command()
def main(
    lrcage_counts: Path = typer.Option(
        Path("results/lrcage_calibration/counts.l1em.txt"),
        help="featureCounts table for the three Sequel II runs.",
    ),
    five_prime_counts: Path = typer.Option(
        Path("results/lrcage_calibration/lrcage_5p_end_counts.csv"),
        help="Per-run 5'-end-restricted counts from lrcage_5p_ends.py.",
    ),
    offsets: Path = typer.Option(
        Path("results/lrcage_calibration/lrcage_5p_end_offsets.csv"),
        help="TSS-precision histogram from lrcage_5p_ends.py.",
    ),
    race5_counts: Path = typer.Option(
        Path("results/isoseq_calibration/counts.l1em.SRR4099955.txt"),
        help="Published 5'RACE featureCounts table (HEK293) -- confounded comparator.",
    ),
    l1base_dir: Path = typer.Option(Path("data/ref/l1base")),
    out_dir: Path = typer.Option(Path("results/lrcage_calibration")),
    min_element_bp: int = typer.Option(MIN_ELEMENT_BP),
):
    """Report Route O's protocol shape, its internal controls, and its depth."""
    overlap = load_counts(lrcage_counts)
    five = pd.read_csv(five_prime_counts)
    five["region"] = five["Geneid"].str.extract(_FAMILY)[2]

    joined = overlap.merge(
        five[["region"] + [c for c in five.columns if c.endswith(
            ("_strand_matched", "_position_matched", "_either_end"))]],
        on="region",
        how="left",
        validate="one_to_one",
    ).fillna(0.0)
    elements = load_l1base_elements(l1base_dir, L1BASE_TRACKS, min_element_bp)
    joined = annotate_l1base_overlap(joined, elements)
    logger.info(f"{len(joined):,} L1EM.400 intervals carrying Route O columns")

    # --- protocol shape, whole-interval vs 5'-end-restricted -----------------
    # The pair is the point: whole-interval overlap is expected to look like
    # Iso-Seq (passengers) in BOTH LRCAGE and LRhex.  Cap selection can only show
    # up once the 5'-end filter is applied, and only relative to LRhex.
    rows = []
    for run in RUNS:
        for measure, column in (
            ("whole-interval overlap", f"{run}_overlap"),
            ("5'-end restricted (position-matched)", f"{run}_position_matched"),
        ):
            if column not in joined:
                continue
            rows.append(
                {"run": run, "measure": measure,
                 **shape(joined[column], joined["element_bp"], joined["target"])}
            )
    profiles = pd.DataFrame(rows)

    # --- the two internal contrasts -----------------------------------------
    contrasts = []
    for label, treated, control in (
        ("cap selection (LRCAGE vs LRhex, DMSO)", "LRCAGE_DMSO", "LRhex_DMSO"),
        ("L1 activation (DACSB vs DMSO, LRCAGE)", "LRCAGE_DACSB", "LRCAGE_DMSO"),
    ):
        for measure, suffix in (
            ("whole-interval overlap", "overlap"),
            ("5'-end restricted", "position_matched"),
        ):
            a, b = f"{treated}_{suffix}", f"{control}_{suffix}"
            if a not in joined or b not in joined:
                continue
            shape_a = shape(joined[a], joined["element_bp"], joined["target"])
            shape_b = shape(joined[b], joined["element_bp"], joined["target"])
            contrasts.append({
                "contrast": label,
                "measure": measure,
                "reads_treated": shape_a.get("assigned_reads", 0.0),
                "reads_control": shape_b.get("assigned_reads", 0.0),
                "young_share_treated_pct": shape_a.get("young_share_pct", np.nan),
                "young_share_control_pct": shape_b.get("young_share_pct", np.nan),
                "young_share_ratio": (
                    shape_a.get("young_share_pct", np.nan)
                    / shape_b.get("young_share_pct", np.nan)
                    if shape_b.get("young_share_pct") else np.nan
                ),
                "full_length_share_treated_pct": shape_a.get("full_length_share_pct", np.nan),
                "full_length_share_control_pct": shape_b.get("full_length_share_pct", np.nan),
            })
    contrast_table = pd.DataFrame(contrasts)

    # --- depth, on the measurement that actually carries the verdict ---------
    depth = pd.DataFrame([
        {"run": run, "measure": measure, **g3b(joined[column])}
        for run in RUNS
        for measure, column in (
            ("whole-interval overlap", f"{run}_overlap"),
            ("5'-end restricted", f"{run}_position_matched"),
        )
        if column in joined
    ])

    # --- confounded concordance with the HEK-293 reference -------------------
    race = pd.read_csv(race5_counts, sep="\t", comment="#")
    race["region"] = race["Geneid"].str.extract(_FAMILY)[2]
    race["race5_count"] = race[race.columns[6:-1]].sum(axis=1).astype(float)
    with_race = joined.merge(race[["region", "race5_count"]], on="region", how="inner")
    sets = {
        "all_L1EM.400": with_race,
        "L1HS_L1PA2": with_race[with_race["target"].isin(L1_TARGETS)],
        "race5_above_1pct": with_race[
            with_race["race5_count"] > RACE5_MIN_PCT * with_race["race5_count"].sum() / 100.0
        ],
    }
    concord = pd.DataFrame([
        concordance(frame, "race5_count", f"{run}_position_matched", name)
        for name, frame in sets.items()
        for run in RUNS
        if f"{run}_position_matched" in frame
    ])

    # --- verdict ------------------------------------------------------------
    cap = contrast_table[
        (contrast_table["contrast"].str.startswith("cap selection"))
        & (contrast_table["measure"] == "5'-end restricted")
    ]
    lrcage_5p = joined.get("LRCAGE_DMSO_position_matched")
    lrcage_shape = shape(lrcage_5p, joined["element_bp"], joined["target"]) if lrcage_5p is not None else {}
    verdict = {
        "cap_selection_enriches_young_L1": bool(
            len(cap) and cap.iloc[0]["young_share_ratio"] > 1.5
        ),
        "young_share_pct_LRCAGE_5p": lrcage_shape.get("young_share_pct", np.nan),
        "young_share_anchor_race5_pct": YOUNG_SHARE_RACE5_PCT,
        "young_share_anchor_isoseq_pct": YOUNG_SHARE_ISOSEQ_PCT,
        "closer_to_race5_than_isoseq": bool(
            abs(lrcage_shape.get("young_share_pct", 0.0) - YOUNG_SHARE_RACE5_PCT)
            < abs(lrcage_shape.get("young_share_pct", 0.0) - YOUNG_SHARE_ISOSEQ_PCT)
        ),
        "top_loci_are_full_length": bool(
            lrcage_shape.get("median_element_bp_top10", 0.0) >= min_element_bp
        ),
        "g3b_pass_5p_restricted": bool(
            len(depth) and depth.loc[
                (depth["run"] == "LRCAGE_DMSO") & (depth["measure"] == "5'-end restricted"),
                "g3b_pass",
            ].all()
        ),
    }
    verdict["route_o_survives"] = bool(
        verdict["cap_selection_enriches_young_L1"]
        and verdict["closer_to_race5_than_isoseq"]
        and verdict["top_loci_are_full_length"]
        and verdict["g3b_pass_5p_restricted"]
    )

    top = (
        joined.loc[
            :, ["region", "target", "element_bp", "LRCAGE_DMSO_position_matched",
                "LRCAGE_DACSB_position_matched", "LRhex_DMSO_position_matched"]
        ]
        .sort_values("LRCAGE_DMSO_position_matched", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "lrcage_loci.csv", index=False)
    profiles.to_csv(out_dir / "lrcage_arm_profiles.csv", index=False)
    contrast_table.to_csv(out_dir / "lrcage_internal_contrasts.csv", index=False)
    depth.to_csv(out_dir / "lrcage_g3b.csv", index=False)
    concord.to_csv(out_dir / "lrcage_concordance_confounded.csv", index=False)
    top.to_csv(out_dir / "lrcage_top_loci.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n## Protocol shape (percent of each run's own L1 signal)\n")
    print(f"   anchors: 5'RACE young share = {YOUNG_SHARE_RACE5_PCT} %, "
          f"Iso-Seq = {YOUNG_SHARE_ISOSEQ_PCT} %\n")
    print(profiles.to_string(index=False))
    print("\n## Internal contrasts -- the line-independent calibration\n")
    print(contrast_table.to_string(index=False))
    print("\n## G3b depth gate\n")
    print(depth.to_string(index=False))
    print("\n## Concordance with HEK-293 5'RACE -- CONFOUNDED by cell line, context only\n")
    print(concord.to_string(index=False))
    if offsets.exists():
        print("\n## TSS precision (5' ends by distance from the element's +1)\n")
        print(pd.read_csv(offsets).pivot(
            index="run", columns="offset_bin", values="reads").to_string())
    print("\n## LRCAGE's own top 15 loci, 5'-end restricted\n")
    print(top.to_string(index=False))
    print("\n## Verdict\n")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {out_dir}/lrcage_*.csv")


if __name__ == "__main__":
    app()
