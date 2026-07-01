#!/usr/bin/env python3
"""Build the synthetic-benchmark abundance table for the R/ggplot analysis (5.02).

For each grid cell (insertion level × del_prob) it pairs, per L1 element:
  * Simulated  = ground-truth abundance — the per-element copy count from a model-2
    transcript-pool counts TSV, or the insertion count from a model-1 BED.
  * AlbertSalmon_seqlabel = salmon NumReads from the seqlabel filter→salmon quant.sf.

The element key (an L1Base ``UID`` or an RM ``<subfamily>.<flag>::chrom:start-end``) is
shared by the ground truth, the salmon index, and quant.sf, so the join is 1:1. A subfamily
token is emitted too, for aggregation in the R step.

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


def load_simulated_counts(tsv_path: str) -> dict:
    """Per-element copy count from a model-2 counts TSV (``l1_id\\tcount``)."""
    counts = {}
    with open(tsv_path) as fh:
        next(fh, None)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                counts[f[0]] = int(f[1])
    return counts


def load_simulated_bed(bed_path: str) -> Counter:
    """Per-element insertion count from a model-1 insertion BED (col4 = element key)."""
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
    """Subfamily token from an element key.

    ``L1PA6.1::chr1:100-6100`` → ``L1PA6`` (RM source). A bare L1Base ``UID107`` has
    no subfamily encoded, so it returns itself (per-element == per-subfamily there).
    """
    return key.split("::", 1)[0].rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    _src = os.environ.get("L1_SOURCE", "l1base")
    ap.add_argument("--refdir",
                    default=os.environ.get("REFDIR", f"data/ref/GRCh38.p14.genome.chr1.withdel.{_src}"))
    ap.add_argument("--chr", default="chr1")
    ap.add_argument("--fcov", default="5")
    ap.add_argument("--out", default="results/synthetic_validation/abundance.csv")
    args = ap.parse_args()

    rows, cells = [], 0
    # Iterate over the validation outputs; pair each with its ground truth — a model-2
    # counts TSV (preferred) or a model-1 insertion BED.
    qdirs = sorted(glob.glob(f"{args.refdir}/salmon/filtered_seqlabel/*/"))
    for qdir in qdirs:
        base = os.path.basename(qdir.rstrip("/"))
        # Anchor del_prob as digits.digits so the trailing '.pair.5x' isn't captured.
        m = re.search(r"insert_level_(\d+)_delprob_([0-9]+\.[0-9]+)", base)
        quant_sf = os.path.join(qdir, "quant.sf")
        if not m or not os.path.isfile(quant_sf):
            continue
        power, dp = m.group(1), m.group(2)
        gt = f"{args.refdir}/GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}"
        if os.path.isfile(f"{gt}.counts.tsv"):
            sim = load_simulated_counts(f"{gt}.counts.tsv")
        elif os.path.isfile(f"{gt}.bed"):
            sim = load_simulated_bed(f"{gt}.bed")
        else:
            print(f"[abundance] WARN: no ground truth for {base} — skipping")
            continue
        est = load_salmon(quant_sf)
        cells += 1
        # Element set = the salmon index targets (quant.sf); Simulated is 0 where the
        # element was never expressed, so the regression includes true negatives.
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
