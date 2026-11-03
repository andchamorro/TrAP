#!/usr/bin/env python3
"""Count long-read CAGE reads whose 5' end lands in an L1's own promoter window.

Route O differs from the Iso-Seq arm in the one way that matters: the library is
cap-trapped, so a read's 5' end is a transcription start site rather than wherever
reverse transcription happened to stop.  The 5'-end filter is therefore not a
sanity check bolted on after the fact -- it IS the measurement.  Whole-interval
overlap is reported by featureCounts and is expected to be dominated by passengers
in both arms; the question is whether cap selection separates them.

WHY PER-RUN COLUMNS
NCI-H1299 has no 5'RACE library, so unlike the HEK-293 Iso-Seq arm this route
cannot be calibrated against a same-cell-line reference.  The decisive contrast
lives inside the deposit instead:

    LRCAGE DMSO   vs   LRhex DMSO     same cells, same platform, same read length,
                                      cap selection the ONLY difference
    LRCAGE DMSO   vs   LRCAGE DACSB   same protocol, L1s driven up by
                                      decitabine + SB939

Pooling the BAMs the way the Iso-Seq script did would destroy both contrasts, so
every count is emitted per run.

Three 5'-end conventions are kept, for the same reason as the Iso-Seq arm -- the
deposited reads' orientation is not documented as transcript-normalised:
  strand_matched    read's mapped strand == element strand; 5' end is the leftmost
                    base of a forward read, the rightmost of a reverse one.
  position_matched  the CORRECT convention under mixed orientation: a sense
                    transcript's 5' end sits at the element's own 5' coordinate
                    whichever cDNA strand was sequenced.  Primary.
  either_end        either alignment end in the window, strand ignored.  A loose
                    upper bound: it also admits reads merely TERMINATING there.

TSS PRECISION
For a cap-anchored library the offset histogram is stronger evidence than any
single count: real L1-promoter transcripts cluster at the element's +1, while
passengers scatter across the window.  --offsets-out writes that distribution.

    python lrcage_5p_ends.py --saf L1EM.400.saf \
        --bam LRCAGE_DMSO=a.bam --bam LRhex_DMSO=b.bam --out counts.csv
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

#: Offset bins (bp from the element's 5' end) for the TSS-precision histogram.
OFFSET_BINS = (0, 10, 25, 50, 100, 200, 400, 600, UTR_BP)

#: `family.category.locus.strand` where locus is `chrom:start-end` (element body,
#: before the SAF's +/-400 flank).
GENEID = re.compile(r"^([^.]+)\.([^.]+)\.(chr[^:]+):(\d+)-(\d+)\.([+-])$")

CONVENTIONS = ("strand_matched", "position_matched", "either_end")


def parse_bam_arg(value: str) -> tuple[str, Path]:
    """``label=path`` -> (label, path); a bare path falls back to its filename stem."""
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name.split(".")[0], path


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
        match = GENEID.match(gene)
        if not match:
            continue
        target, category, chrom, start, end, strand = match.groups()
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
    """Indices of windows containing *pos*; windows are short so a scan back is cheap."""
    index = bisect_right(starts, pos)
    out = []
    j = index - 1
    while j >= 0 and pos - windows[j][0] <= UTR_BP:
        if windows[j][0] <= pos < windows[j][1]:
            out.append(j)
        j -= 1
    return out


def offset_from_element_5p(window, pos: int) -> int:
    """Distance in bp from the element's own 5' base to *pos*."""
    w0, w1, _gene, strand = window
    return pos - w0 if strand == "+" else w1 - 1 - pos


def bin_offset(offset: int) -> str:
    for low, high in zip(OFFSET_BINS, OFFSET_BINS[1:]):
        if low <= offset < high:
            return f"{low}-{high}"
    return f">={OFFSET_BINS[-1]}"


def count_one_bam(label: str, path: Path, by_chrom, starts, counts, offsets) -> tuple[int, int]:
    """Accumulate one run's counts in place; returns (reads on known chroms, forward)."""
    n_reads = n_forward = 0
    with pysam.AlignmentFile(str(path)) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            windows = by_chrom.get(read.reference_name)
            if windows is None:
                continue
            left, right = read.reference_start, read.reference_end
            if right is None:
                continue
            n_reads += 1
            reverse = read.is_reverse
            n_forward += not reverse
            chrom_starts = starts[read.reference_name]

            read_strand = "-" if reverse else "+"
            five_prime = right - 1 if reverse else left
            for j in hits(windows, chrom_starts, five_prime):
                if windows[j][3] == read_strand:
                    counts[label]["strand_matched"][windows[j][2]] += 1.0

            # Position-matched: the end that WOULD be a sense transcript's 5' end
            # for this element's orientation, whichever way the cDNA was read.
            for end, wanted in ((left, "+"), (right - 1, "-")):
                for j in hits(windows, chrom_starts, end):
                    if windows[j][3] == wanted:
                        counts[label]["position_matched"][windows[j][2]] += 1.0
                        offsets[label][bin_offset(offset_from_element_5p(windows[j], end))] += 1

            seen = set()
            for end in (left, right - 1):
                for j in hits(windows, chrom_starts, end):
                    seen.add(windows[j][2])
            for gene in seen:
                counts[label]["either_end"][gene] += 1.0
    return n_reads, n_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saf", type=Path, required=True)
    parser.add_argument(
        "--bam",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="One per run; the label becomes the output column prefix.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets-out", type=Path, default=None)
    args = parser.parse_args()

    by_chrom, meta = load_elements(args.saf)
    print(f"elements with a parseable locus id: {len(meta):,}")
    starts = {chrom: [w[0] for w in ws] for chrom, ws in by_chrom.items()}

    runs = [parse_bam_arg(value) for value in args.bam]
    counts = {
        label: {convention: defaultdict(float) for convention in CONVENTIONS}
        for label, _ in runs
    }
    offsets = {label: defaultdict(int) for label, _ in runs}

    for label, path in runs:
        n_reads, n_forward = count_one_bam(label, path, by_chrom, starts, counts, offsets)
        totals = " ".join(
            f"{c}={sum(counts[label][c].values()):,.0f}" for c in CONVENTIONS
        )
        print(
            f"  {label} ({path.name}): reads on known chroms {n_reads:,}, "
            f"forward fraction {n_forward / max(n_reads, 1):.3f} | {totals}"
        )

    columns = [f"{label}_{c}" for label, _ in runs for c in CONVENTIONS]
    with args.out.open("w") as handle:
        handle.write(
            "Geneid,target,category,chrom,el_start,el_end,strand,element_bp,"
            + ",".join(columns)
            + "\n"
        )
        for gene, (target, category, chrom, start, end, strand, body) in meta.items():
            values = [counts[label][c][gene] for label, _ in runs for c in CONVENTIONS]
            if not any(values):
                continue
            handle.write(
                f"{gene},{target},{category},{chrom},{start},{end},{strand},{body},"
                + ",".join(f"{v:.0f}" for v in values)
                + "\n"
            )
    print(f"wrote {args.out}")

    if args.offsets_out:
        bins = [f"{a}-{b}" for a, b in zip(OFFSET_BINS, OFFSET_BINS[1:])]
        with args.offsets_out.open("w") as handle:
            handle.write("run,offset_bin,reads\n")
            for label, _ in runs:
                for name in bins:
                    handle.write(f"{label},{name},{offsets[label][name]}\n")
        print(f"wrote {args.offsets_out}")
        print("\nTSS precision -- position-matched 5' ends by distance from the element's +1")
        header = "  " + "run".ljust(16) + "".join(name.rjust(11) for name in bins)
        print(header)
        for label, _ in runs:
            total = sum(offsets[label][name] for name in bins) or 1
            row = "".join(
                f"{100 * offsets[label][name] / total:10.1f}%" for name in bins
            )
            print("  " + label.ljust(16) + row)


if __name__ == "__main__":
    main()
