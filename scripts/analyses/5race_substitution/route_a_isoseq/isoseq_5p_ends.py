#!/usr/bin/env python3
"""Count Iso-Seq reads whose transcript 5' end lands in an L1's own 5' region.

The calibration arm showed Iso-Seq's L1-overlapping reads are dominated by short,
old L1 fragments embedded in host transcripts.  Whole-interval overlap cannot tell
those passengers from L1-initiated transcription, which is the quantity 5'RACE
measures and the quantity TrAP claims to separate.  The plan (sec. 4a) therefore
requires a 5'-end filter on the reference side before a long-read arm is admitted:
a read supports a locus only if its 5' end lies inside that element's 5' region.

Two conventions are reported, because the deposited reads' orientation is not
documented as transcript-normalised:
  strand_matched  read's mapped strand == element strand, 5' end = leftmost base
                  for a forward read and rightmost for a reverse one.  Correct if
                  the reads are in transcript orientation (lima/refine output).
  position_matched
                  the CORRECT convention when read orientation is mixed: a sense
                  transcript's 5' end sits at the element's own 5' coordinate no
                  matter which cDNA strand was sequenced, so use the alignment's
                  leftmost base for a + element and its rightmost for a -, and
                  ignore the read's strand.
  either_end      either end of the alignment falls in the window, strand ignored.
                  A loose upper bound: it also admits reads merely TERMINATING at
                  the element's 5' end (antisense or truncated).

Reporting all three means the verdict does not rest on an assumption about the
deposit: strand_matched undercounts by ~half when reads are not transcript-
oriented, either_end overcounts, and position_matched is the intended quantity.

    python isoseq_5p_ends.py --saf L1EM.400.saf --bam a.bam --bam b.bam --out counts.csv
"""

from __future__ import annotations

import argparse
import re
from bisect import bisect_right
from pathlib import Path

import pysam

#: L1HS 5'UTR length; an element's promoter lives in its first ~900 bp.
UTR_BP = 909

#: `family.category.locus.strand` where locus is `chrom:start-end` (element body,
#: before the SAF's +/-400 flank).
GENEID = re.compile(r"^([^.]+)\.([^.]+)\.(chr[^:]+):(\d+)-(\d+)\.([+-])$")


def load_elements(saf: Path):
    """SAF -> per-chromosome sorted 5'-window lists.

    Windows are built from the ELEMENT coordinates encoded in the GeneID, not the
    padded SAF interval, so the window is the element's own promoter region.
    """
    by_chrom: dict[str, list] = {}
    meta = {}
    for line in saf.open():
        if line.startswith("GeneID"):
            continue
        gene = line.split("\t", 1)[0]
        m = GENEID.match(gene)
        if not m:
            continue
        target, category, chrom, start, end, strand = m.groups()
        start, end = int(start), int(end)
        body = end - start
        window = min(UTR_BP, body)
        if strand == "+":
            w0, w1 = start, start + window
        else:
            w0, w1 = end - window, end
        by_chrom.setdefault(chrom, []).append((w0, w1, gene, strand))
        meta[gene] = (target, category, chrom, start, end, strand, body)
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom, meta


def hits(windows, starts, pos: int):
    """Genes whose 5' window contains *pos*; windows are short so a scan back is cheap."""
    i = bisect_right(starts, pos)
    out = []
    j = i - 1
    while j >= 0 and pos - windows[j][0] <= UTR_BP:
        if windows[j][0] <= pos < windows[j][1]:
            out.append(j)
        j -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saf", type=Path, required=True)
    ap.add_argument("--bam", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_chrom, meta = load_elements(args.saf)
    print(f"elements with a parseable locus id: {len(meta):,}")
    starts = {c: [w[0] for w in ws] for c, ws in by_chrom.items()}

    matched = {g: 0.0 for g in meta}
    posmatch = {g: 0.0 for g in meta}
    either = {g: 0.0 for g in meta}
    n_reads = n_fwd = 0

    for bam_path in args.bam:
        with pysam.AlignmentFile(str(bam_path)) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                chrom = read.reference_name
                ws = by_chrom.get(chrom)
                if ws is None:
                    continue
                n_reads += 1
                left, right = read.reference_start, read.reference_end
                if right is None:
                    continue
                rev = read.is_reverse
                n_fwd += not rev
                five_prime = right - 1 if rev else left
                read_strand = "-" if rev else "+"
                for j in hits(ws, starts[chrom], five_prime):
                    if ws[j][3] == read_strand:
                        matched[ws[j][2]] += 1.0
                # Position-matched: the end that WOULD be a sense transcript's 5'
                # end for this element's orientation, whichever way it was read.
                for end, want in ((left, "+"), (right - 1, "-")):
                    for j in hits(ws, starts[chrom], end):
                        if ws[j][3] == want:
                            posmatch[ws[j][2]] += 1.0
                seen = set()
                for end in (left, right - 1):
                    for j in hits(ws, starts[chrom], end):
                        seen.add(ws[j][2])
                for gene in seen:
                    either[gene] += 1.0
        print(f"  {bam_path.name}: cumulative reads on known chroms {n_reads:,}")

    print(f"forward-strand fraction of mapped reads: {n_fwd / max(n_reads, 1):.3f}")
    with args.out.open("w") as fh:
        fh.write("Geneid,target,category,chrom,el_start,el_end,strand,element_bp,"
                 "five_prime_strand_matched,five_prime_position_matched,"
                 "five_prime_either_end\n")
        for gene, (target, category, chrom, start, end, strand, body) in meta.items():
            m, pm, e = matched[gene], posmatch[gene], either[gene]
            if m or pm or e:
                fh.write(f"{gene},{target},{category},{chrom},{start},{end},{strand},"
                         f"{body},{m:.0f},{pm:.0f},{e:.0f}\n")
    print(f"wrote {args.out}")
    print(f"totals: strand_matched={sum(matched.values()):,.0f} "
          f"position_matched={sum(posmatch.values()):,.0f} "
          f"either_end={sum(either.values()):,.0f}")


if __name__ == "__main__":
    main()
