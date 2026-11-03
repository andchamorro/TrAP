#!/usr/bin/env python3
"""Route N blocker 2: ORF1p-RIP is not 5'-anchored by design -- is it in practice?

THE BLOCKER, STATED FAIRLY
ORF1p binds its own transcript anywhere along it, so nothing in the protocol makes
a read's 5' end a transcription start site.  A 5'RACE substitute has to supply
5'-anchored data, which is exactly what is in doubt here.  Route N's ``five_prime``
column (5' end inside the element's own ~909 bp promoter
window) could therefore be nothing more than the share of reads that happen to start
near the element's 5' edge -- 909 / 6,000 = 15 % of a full-length element by length
alone.

WHAT DECIDES IT
Where along an element do reads actually start?  Under positionally-random binding
the oriented start positions are flat, and ~15 % land in the first 15 % by
construction.  Under genuine 5' anchoring they pile up at position 0.  This script
measures that distribution, with two things that make it a test rather than a
picture:

  * A POSITIVE CONTROL.  The published reference (SRR4099955, real L1-targeted
    5'RACE, 34,128 reads) is run through the IDENTICAL code path.  5'RACE must show
    the pile-up.  If it does not, the measurement is broken and route N's result --
    whatever it is -- means nothing.
  * A FLAT NULL.  Enrichment is reported against the uniform expectation for the
    same bin, so "15 % of reads in the first 15 %" reads as 1.0x, not as a result.

The read end used is the POSITION-MATCHED one -- left end for a + element, right end
for a - element -- the same convention as l1rip_fulllength.py, because the deposit is
not orientation-normalised (forward-strand fraction ~0.5).

    python l1rip_fiveprime_profile.py --saf L1EM.400.saf \
        --bam Library_8=a.bam --bam reference_5pRACE=r.bam \
        --hist-out pos_hist.csv --summary-out pos_summary.csv
"""

from __future__ import annotations

import argparse
import re
from bisect import bisect_right
from pathlib import Path

import pysam

# The interval helpers are duplicated from l1rip_fulllength.py rather than imported:
# these scripts run under `python -I`, which drops the script's own directory from
# sys.path, and each route directory has to stay independently promotable.
GENEID = re.compile(r"^([^.]+)\.([^.]+)\.([^:]+):(\d+)-(\d+)\.([+-])$")

UTR_BP = 909
FULL_LENGTH_BP = 6_000
YOUNG = ("L1HS", "L1PA2")
#: 20 bins of 5 % along the element, oriented 5' -> 3'.
N_BINS = 20
#: Fractions of the element whose leading edge is reported as a headline number.
LEAD_FRACTIONS = (0.01, 0.05, 0.10, 0.15)


def parse_bam_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name.split(".")[0], path


def load_elements(saf: Path):
    by_chrom: dict[str, list] = {}
    for line in saf.open():
        if line.startswith("GeneID"):
            continue
        gene = line.split("\t", 1)[0]
        match = GENEID.match(gene)
        if not match:
            continue
        target, _category, chrom, start, end, strand = match.groups()
        start, end = int(start), int(end)
        by_chrom.setdefault(chrom, []).append((start, end, strand, target))
    starts, longest = {}, {}
    for chrom, items in by_chrom.items():
        items.sort()
        starts[chrom] = [i[0] for i in items]
        longest[chrom] = max(end - start for start, end, _s, _t in items)
    return by_chrom, starts, longest


def overlapping(elements, starts, longest: int, left: int, right: int):
    out = []
    j = bisect_right(starts, right) - 1
    while j >= 0 and left - elements[j][0] <= longest:
        if elements[j][0] < right and left < elements[j][1]:
            out.append(j)
        j -= 1
    return out


def profile_one_bam(label, path, by_chrom, starts, longest):
    """Histogram of oriented read-start positions along element bodies.

    Returns {element_set: {"hist": [...], "reads": n, "in_window": n}}.  A read is
    counted once per element it overlaps, which is the same convention the count
    tables use.
    """
    sets = ("all_elements", "full_length_young")
    acc = {s: {"hist": [0] * N_BINS, "reads": 0, "in_window": 0, "pos_sum": 0.0,
               "positions": []} for s in sets}
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
            for j in overlapping(elements, starts[read.reference_name],
                                 longest[read.reference_name], left, right):
                el_start, el_end, strand, target = elements[j]
                body = el_end - el_start
                if body <= 0:
                    continue
                # position-matched 5' end for this element's orientation
                five = left if strand == "+" else right - 1
                pos = (five - el_start) / body if strand == "+" else (el_end - 1 - five) / body
                if not 0.0 <= pos < 1.0:
                    continue          # 5' end outside the body: the flank, not a start
                window = min(UTR_BP, body)
                in_window = pos * body < window
                which = ["all_elements"]
                if body >= FULL_LENGTH_BP and target in YOUNG:
                    which.append("full_length_young")
                for s in which:
                    acc[s]["hist"][min(int(pos * N_BINS), N_BINS - 1)] += 1
                    acc[s]["reads"] += 1
                    acc[s]["in_window"] += int(in_window)
                    acc[s]["pos_sum"] += pos
                    acc[s]["positions"].append(pos)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saf", type=Path, required=True)
    parser.add_argument("--bam", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--hist-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    by_chrom, starts, longest = load_elements(args.saf)
    print(f"[profile] {sum(len(v) for v in by_chrom.values()):,} elements "
          f"on {len(by_chrom):,} chromosomes", flush=True)

    hist_rows, summary_rows = [], []
    for spec in args.bam:
        label, path = parse_bam_arg(spec)
        print(f"[profile] {label}: {path}", flush=True)
        acc = profile_one_bam(label, path, by_chrom, starts, longest)
        for element_set, data in acc.items():
            n = data["reads"]
            for b in range(N_BINS):
                hist_rows.append({
                    "label": label, "element_set": element_set,
                    "bin_lo": round(b / N_BINS, 3), "bin_hi": round((b + 1) / N_BINS, 3),
                    "reads": data["hist"][b],
                    "pct": 100.0 * data["hist"][b] / n if n else 0.0,
                    "enrichment_vs_uniform": (data["hist"][b] / n) * N_BINS if n else 0.0,
                })
            row = {"label": label, "element_set": element_set, "reads": n,
                   "mean_norm_pos": data["pos_sum"] / n if n else 0.0,
                   "pct_in_909bp_window": 100.0 * data["in_window"] / n if n else 0.0}
            pos = sorted(data["positions"])
            row["median_norm_pos"] = pos[len(pos) // 2] if pos else 0.0
            for frac in LEAD_FRACTIONS:
                k = sum(1 for p in pos if p < frac)
                row[f"pct_in_first_{int(frac * 100)}pct"] = 100.0 * k / n if n else 0.0
                row[f"enrich_first_{int(frac * 100)}pct"] = (k / n) / frac if n else 0.0
            summary_rows.append(row)
            print(f"    {element_set}: {n:,} reads, median pos "
                  f"{row['median_norm_pos']:.3f}, first 15 % = "
                  f"{row['pct_in_first_15pct']:.1f} % "
                  f"({row['enrich_first_15pct']:.2f}x uniform)", flush=True)

    def write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0])
        with path.open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")

    write(args.hist_out, hist_rows)
    write(args.summary_out, summary_rows)
    print(f"[profile] wrote {args.hist_out} and {args.summary_out}")


if __name__ == "__main__":
    main()
