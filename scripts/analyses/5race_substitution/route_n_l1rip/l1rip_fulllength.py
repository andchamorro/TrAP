#!/usr/bin/env python3
"""Measure how much of each L1 element Route N's ORF1p-RIP reads actually cover.

Route N is L1-TARGETED but NOT 5'-anchored.  An ORF1p pull-down enriches L1 RNA by
construction -- which is exactly the failure mode that killed the Iso-Seq arm, where
L1 reads were passengers inside abundant host transcripts -- but ORF1p binds its own
transcript anywhere along it, so a read's 5' end is not a TSS.  "Full-length,
5'-complete" therefore has to be measured per read rather than assumed from the
protocol name, and three quantities have to be kept apart:

  five_prime     the read's 5' end lies in the element's own ~909 bp promoter
                 window.  Comparable to the 5'RACE reference and to the Iso-Seq
                 arm's filter, so it is the column that joins across routes.
  body_covered   the read's ALIGNED blocks cover at least --min-body-frac of the
                 element body.  Introns are excluded, so a spliced alignment that
                 merely spans the element does not count.
  full_length    both of the above.  This is Route N's own claim: a transcript that
                 starts at the L1 promoter and runs through the element.

The deposit's ~790 nt mean read length means full_length must be rare on a 6 kb
element no matter how good the pull-down was.  That is the point of measuring it:
the route's honest claim is "which loci are transcribed", not "intact L1 mRNA
recovered", and the gap between body_covered and five_prime is what shows it.

Counts are emitted PER RUN: the two libraries are replicates of the same pull-down,
and their agreement is one of the few internal checks this route has.

    python l1rip_fulllength.py --saf L1EM.400.saf \
        --bam Library_8=a.bam --bam Library_10=b.bam --out counts.csv
"""

from __future__ import annotations

import argparse
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pysam

#: L1HS 5'UTR length; an element's promoter lives in its first ~900 bp.
UTR_BP = 909

#: Default share of the element body an alignment must cover to count as spanning.
MIN_BODY_FRAC = 0.80

#: `family.category.locus.strand` where locus is `chrom:start-end` (element body,
#: before the SAF's +/-400 flank).
GENEID = re.compile(r"^([^.]+)\.([^.]+)\.(chr[^:]+):(\d+)-(\d+)\.([+-])$")

MEASURES = ("five_prime", "body_covered", "full_length")


def parse_bam_arg(value: str) -> tuple[str, Path]:
    """``label=path`` -> (label, path); a bare path falls back to its filename stem."""
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name.split(".")[0], path


def load_elements(saf: Path):
    """SAF -> per-chromosome element bodies, sorted by start, plus the longest body.

    Coordinates come from the GeneID, which carries the ELEMENT body, not the SAF
    interval's +/-400 flank.  The longest body per chromosome bounds how far back a
    positional query has to scan.
    """
    by_chrom: dict[str, list] = {}
    meta = {}
    for line in saf.open():
        if line.startswith("GeneID"):
            continue
        gene = line.split("\t", 1)[0]
        match = GENEID.match(gene)
        if not match:
            continue
        target, category, chrom, start, end, strand = match.groups()
        start, end = int(start), int(end)
        by_chrom.setdefault(chrom, []).append((start, end, gene, strand))
        meta[gene] = (target, category, chrom, start, end, strand, end - start)
    longest = {}
    for chrom, items in by_chrom.items():
        items.sort()
        longest[chrom] = max(end - start for start, end, _g, _s in items)
    return by_chrom, meta, longest


def overlapping(elements, starts, longest: int, left: int, right: int):
    """Indices of elements whose body intersects [left, right)."""
    index = bisect_right(starts, right)
    out = []
    j = index - 1
    while j >= 0 and left - elements[j][0] <= longest:
        if elements[j][0] < right and left < elements[j][1]:
            out.append(j)
        j -= 1
    return out


def covered_bp(blocks, start: int, end: int) -> int:
    """Aligned bases (introns excluded) falling inside [start, end)."""
    total = 0
    for block_start, block_end in blocks:
        overlap = min(block_end, end) - max(block_start, start)
        if overlap > 0:
            total += overlap
    return total


def count_one_bam(label, path, by_chrom, starts, longest, counts, spans, min_frac):
    """Accumulate one run's counts in place; returns (reads on known chroms, forward)."""
    n_reads = n_forward = 0
    with pysam.AlignmentFile(str(path)) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            elements = by_chrom.get(read.reference_name)
            if elements is None:
                continue
            left, right = read.reference_start, read.reference_end
            if right is None:
                continue
            n_reads += 1
            n_forward += not read.is_reverse
            blocks = read.get_blocks()
            for j in overlapping(elements, starts[read.reference_name], longest[read.reference_name], left, right):
                el_start, el_end, gene, strand = elements[j]
                body = el_end - el_start
                window = min(UTR_BP, body)
                if strand == "+":
                    w0, w1 = el_start, el_start + window
                else:
                    w0, w1 = el_end - window, el_end

                # Position-matched 5' end: the end that WOULD be a sense
                # transcript's 5' end for this element's orientation, whichever
                # cDNA strand was sequenced.
                five = (w0 <= left < w1 and strand == "+") or (w0 <= right - 1 < w1 and strand == "-")
                fraction = covered_bp(blocks, el_start, el_end) / body if body else 0.0
                spanning = fraction >= min_frac

                if five:
                    counts[label]["five_prime"][gene] += 1.0
                if spanning:
                    counts[label]["body_covered"][gene] += 1.0
                if five and spanning:
                    counts[label]["full_length"][gene] += 1.0
                    spans[label].append(fraction)
    return n_reads, n_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saf", type=Path, required=True)
    parser.add_argument("--bam", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-body-frac", type=float, default=MIN_BODY_FRAC)
    args = parser.parse_args()

    by_chrom, meta, longest = load_elements(args.saf)
    print(f"elements with a parseable locus id: {len(meta):,}")
    print(f"body-coverage threshold: {args.min_body_frac:.0%} of the element")
    starts = {chrom: [e[0] for e in items] for chrom, items in by_chrom.items()}

    runs = [parse_bam_arg(value) for value in args.bam]
    counts = {label: {m: defaultdict(float) for m in MEASURES} for label, _ in runs}
    spans = {label: [] for label, _ in runs}

    for label, path in runs:
        n_reads, n_forward = count_one_bam(
            label, path, by_chrom, starts, longest, counts, spans, args.min_body_frac
        )
        totals = " ".join(f"{m}={sum(counts[label][m].values()):,.0f}" for m in MEASURES)
        print(
            f"  {label} ({path.name}): reads on known chroms {n_reads:,}, "
            f"forward fraction {n_forward / max(n_reads, 1):.3f} | {totals}"
        )

    columns = [f"{label}_{m}" for label, _ in runs for m in MEASURES]
    with args.out.open("w") as handle:
        handle.write(
            "Geneid,target,category,chrom,el_start,el_end,strand,element_bp,"
            + ",".join(columns)
            + "\n"
        )
        for gene, (target, category, chrom, start, end, strand, body) in meta.items():
            values = [counts[label][m][gene] for label, _ in runs for m in MEASURES]
            if not any(values):
                continue
            handle.write(
                f"{gene},{target},{category},{chrom},{start},{end},{strand},{body},"
                + ",".join(f"{v:.0f}" for v in values)
                + "\n"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
