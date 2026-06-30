#!/usr/bin/env python3
"""Build the synthetic-benchmark abundance table for the R/ggplot analysis (5.02).

For each grid cell (insertion level × del_prob) it pairs, per L1 element:
  * Simulated  = number of insertions of that element (count of the insertion BED's
    name column) — the ground-truth abundance (reads ∝ insertions, elements are ~equal
    length), 0 for elements never inserted.
  * AlbertSalmon_seqlabel = salmon NumReads from the seqlabel filter→salmon quant.sf.

The element key is the unique ``<subfamily>.<promoter>::chrom:start-end`` shared by the
insertion BED (col4), the salmon index, and quant.sf, so the join is 1:1. The subfamily
prefix is emitted too, for per-subfamily aggregation in the R step.

Emits a long CSV: ``power, del_prob, l1_id, subfamily, simulated, albertsalmon_seqlabel``.

  python scripts/sh/synthetic_abundance.py            # → results/synthetic_validation/abundance.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import Counter


def load_simulated(bed_path: str) -> Counter:
    """Per-element insertion count from an insertion BED (col4 = element key)."""
    counts: Counter = Counter()
    with open(bed_path) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                counts[fields[3]] += 1
    return counts


def load_salmon(quant_sf: str) -> dict:
    """``{element_key: NumReads}`` from a salmon quant.sf."""
    est = {}
    with open(quant_sf) as fh:
        next(fh, None)  # header: Name Length EffectiveLength TPM NumReads
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 5:
                est[f[0]] = float(f[4])
    return est


def subfamily(key: str) -> str:
    """``L1PA6.1::chr1:100-6100`` → ``L1PA6`` (subfamily, dropping the promoter flag)."""
    return key.split("::", 1)[0].rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refdir", default="data/ref/GRCh38.p14.genome.chr1.withdel")
    ap.add_argument("--chr", default="chr1")
    ap.add_argument("--fcov", default="5")
    ap.add_argument("--out", default="results/synthetic_validation/abundance.csv")
    args = ap.parse_args()

    bed_re = re.compile(r"insert_level_(\d+)_delprob_([0-9.]+)\.bed$")
    rows, cells = [], 0
    beds = sorted(glob.glob(f"{args.refdir}/GRCh38.p14.{args.chr}.insert_level_*_delprob_*.bed"))
    for bed in beds:
        m = bed_re.search(bed)
        if not m:
            continue
        power, dp = m.group(1), m.group(2)
        base = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}.pair.{args.fcov}x"
        quant_sf = f"{args.refdir}/salmon/filtered_seqlabel/{base}/quant.sf"
        if not os.path.isfile(quant_sf):
            print(f"[abundance] WARN: no quant.sf for {base} — skipping")
            continue
        sim = load_simulated(bed)
        est = load_salmon(quant_sf)
        cells += 1
        # Element set = the salmon index targets (quant.sf); Simulated is 0 where the
        # element was never inserted, so the regression includes true negatives.
        for key, num_reads in est.items():
            rows.append((power, dp, key, subfamily(key), sim.get(key, 0), num_reads))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["power", "del_prob", "l1_id", "subfamily", "simulated", "albertsalmon_seqlabel"])
        w.writerows(rows)
    print(f"[abundance] wrote {len(rows):,} rows from {cells} cells -> {args.out}")


if __name__ == "__main__":
    main()
