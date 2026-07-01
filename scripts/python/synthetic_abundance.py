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

  python scripts/python/synthetic_abundance.py        # → results/synthetic_validation/abundance.csv
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


def _norm_uid(key: str) -> str:
    """Normalize an alignment-method id to the ground-truth UID key (``UID107``)."""
    return key.split(":", 1)[0].split(".", 1)[0].replace("UID-", "UID")


def load_l1em(full_counts: str) -> dict:
    """``{uid: abundance}`` from an L1EM / AlbertEM full_counts.txt.

    Abundance = sum of the sense categories (only + 3prunon + passive_sense +
    passive_antisense), grouped by family (first token of the name column), per 4.08.
    """
    est: dict = {}
    with open(full_counts) as fh:
        header = next(fh, "").rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        cats = [c for c in ("only", "3prunon", "passive_sense", "passive_antisense") if c in idx]
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if not f or not cats or len(f) <= max(idx[c] for c in cats):
                continue
            uid = _norm_uid(f[0])
            est[uid] = est.get(uid, 0.0) + sum(float(f[idx[c]]) for c in cats)
    return est


def load_tecount(cnt_table: str) -> dict:
    """``{uid: count}`` from a TEcount .cntTable (LINE-1 UID features only)."""
    est: dict = {}
    with open(cnt_table) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0].startswith("UID-"):
                est[_norm_uid(f[0])] = est.get(_norm_uid(f[0]), 0.0) + float(f[1])
    return est


def load_htseq(counts_csv: str) -> dict:
    """``{uid: count}`` from an htseq-count table (drops the ``__`` summary rows)."""
    est: dict = {}
    with open(counts_csv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and not f[0].startswith("__"):
                est[_norm_uid(f[0])] = float(f[1])
    return est


# method name → (subdir, filename template relative to <refdir>/<subdir>/<base>/, loader).
# {base} = ...pair.<fcov>x, {proj} = ...insert_level_P_delprob_D (no .pair suffix).
METHODS = {
    "AlbertSalmon_seqlabel": ("salmon/filtered_seqlabel", "quant.sf", load_salmon),
    "Salmon": ("salmon/unfiltered", "quant.sf", load_salmon),
    "L1EM": ("L1EM", "full_counts.txt", load_l1em),
    "AlbertEM": ("MLEM", "full_counts.txt", load_l1em),
    "TEtranscripts": ("TEtranscripts", "{proj}.cntTable", load_tecount),
    "HTseq": ("HTseq", "htseq_counts.csv", load_htseq),
}


def subfamily(key: str) -> str:
    """Subfamily token from an element key.

    ``L1PA6.1::chr1:100-6100`` → ``L1PA6`` (RM source). A bare L1Base ``UID107`` has
    no subfamily encoded, so it returns itself (per-element == per-subfamily there).
    """
    return key.split("::", 1)[0].rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    _src = os.environ.get("L1_SOURCE", "l1base")
    _mtag = ".insert" if os.environ.get("SIM_MODEL", "transcript") == "insert" else ""
    ap.add_argument("--refdir",
                    default=os.environ.get("REFDIR", f"data/ref/GRCh38.p14.genome.chr1.withdel.{_src}{_mtag}"))
    ap.add_argument("--chr", default="chr1")
    ap.add_argument("--fcov", default="5")
    ap.add_argument("--out", default="results/synthetic_validation/abundance.csv")
    ap.add_argument("--methods-out", default="results/synthetic_validation/abundance_methods.csv",
                    help="long table of every baseline method that has outputs.")
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

    # --- multi-method long table (for the method comparison) ---------------
    gt_map = {}  # (power, dp) -> ground-truth path; counts.tsv (model 2) wins over bed.
    for pat in ("counts.tsv", "bed"):
        for p in sorted(glob.glob(f"{args.refdir}/GRCh38.p14.{args.chr}.insert_level_*_delprob_*.{pat}")):
            m = re.search(r"insert_level_(\d+)_delprob_([0-9]+\.[0-9]+)", os.path.basename(p))
            if m:
                gt_map.setdefault((m.group(1), m.group(2)), p)

    mrows, mcells = [], 0
    for (power, dp), gt_path in sorted(gt_map.items()):
        sim = load_simulated_counts(gt_path) if gt_path.endswith(".counts.tsv") else load_simulated_bed(gt_path)
        base = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}.pair.{args.fcov}x"
        proj = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}"
        present = False
        for method, (subdir, fname, loader) in METHODS.items():
            path = f"{args.refdir}/{subdir}/{base}/{fname.format(proj=proj)}"
            if not os.path.isfile(path):
                continue
            try:
                vals = loader(path)
            except Exception as exc:  # noqa: BLE001 — a bad file skips one method, not the run
                print(f"[abundance] WARN: {method} parse failed for {base}: {exc}")
                continue
            present = True
            for uid in sim:
                mrows.append((power, dp, method, uid, sim[uid], vals.get(uid, 0)))
        mcells += present

    if mrows:
        with open(args.methods_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["power", "del_prob", "method", "l1_id", "simulated", "abundance"])
            w.writerows(mrows)
        seen = sorted(set(r[2] for r in mrows))
        print(f"[abundance] wrote {len(mrows):,} method rows from {mcells} cells "
              f"(methods={seen}) -> {args.methods_out}")
    else:
        print("[abundance] no baseline-method outputs found (run scripts/sh/run_*.sh) — methods table skipped")


if __name__ == "__main__":
    main()
