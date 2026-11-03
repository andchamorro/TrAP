"""Emit route N's reference x-axis in featureCounts shape, for the Figure-11 aggregator.

`scripts/aggregate_srr_l1em_exp.py` takes the reference arm as a
featureCounts table (`--race5-counts`) and converts its last column to ``race5``
as ``100 * count / total``.  For HEK293 that file is SRR4099955's own
`counts.l1em.txt`.  Route N has no such file, because its reference x-axis is
**not** the raw whole-interval count: plan §5 fixes it as the **5'-restricted**
column, since the whole-interval column sits at 14.0 % young share against
50.3 % for the 5'-restricted one and is therefore passenger-dominated.

So this writes the 5'-restricted per-locus counts into the same shape, which
keeps the aggregator, the Stage-B notebook and the published axis definition
untouched — the only thing that changes is which column of route N's own
calibration becomes ``race5``.

Three files are written, not one.  §11 established that the two ORF1p-RIP
libraries differ in composition beyond counting noise (young share 42.7 % vs
56.6 %, non-overlapping bootstrap CIs), so the pooled axis is the primary and the
two single-library axes exist to be run as a declared sensitivity analysis rather
than discovered later.

Columns are featureCounts' own order — Geneid, Chr, Start, End, Strand, Length,
<count> — because that is what ``load_race5`` indexes by position and by name.

Run (python -I, paths as arguments):

    python -I scripts/analyses/5race_substitution/route_n_l1rip/l1rip_query_reference.py \
        results/l1rip_calibration/l1rip_loci.csv results/l1rip_calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

#: The measure plan §5 fixes as route N's published x-axis.
MEASURE = "five_prime"

#: Arms written: the pooled primary plus each library alone (declared sensitivity).
ARMS = {
    "pooled": ("Library_8", "Library_10"),
    "Library_8": ("Library_8",),
    "Library_10": ("Library_10",),
}


def emit(loci: pd.DataFrame, columns: tuple[str, ...], label: str, out_path: Path) -> None:
    """Write one featureCounts-shaped table whose count column is *columns* summed."""
    counts = sum(loci[f"{c}_{MEASURE}"].astype(float) for c in columns)
    table = pd.DataFrame(
        {
            "Geneid": (
                loci["target"].astype(str)
                + "."
                + loci["category"].astype(str)
                + "."
                + loci["region"].astype(str)
                + "."
                + loci["strand"].astype(str)
            ),
            "Chr": loci["chrom"],
            "Start": loci["chromStart"],
            "End": loci["chromEnd"],
            "Strand": loci["strand"],
            "Length": loci["length"],
            label: counts.round().astype("int64"),
        }
    )
    total = int(table[label].sum())
    detected = int((table[label] > 0).sum())
    with out_path.open("w") as fh:
        # featureCounts writes a `# Program:...` line that load_race5 skips as a
        # comment; keep the provenance there so the file is self-describing.
        fh.write(
            f"# route N reference arm, {MEASURE} measure, arm={label}; "
            f"derived from l1rip_loci.csv by route_n_l1rip/l1rip_query_reference.py; "
            f"NOT a featureCounts run -- the counts are the 5'-restricted subset of one\n"
        )
        table.to_csv(fh, sep="\t", index=False)
    print(
        f"  {label:<12} -> {out_path.name}  total={total:,} reads, "
        f"{detected:,} loci detected of {len(table):,}"
    )


def main(loci_csv: Path, out_dir: Path) -> None:
    loci = pd.read_csv(loci_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"route N reference x-axis ({MEASURE}) from {loci_csv}:")
    for label, columns in ARMS.items():
        emit(loci, columns, label, out_dir / f"counts.l1em.route-n.{label}.txt")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
