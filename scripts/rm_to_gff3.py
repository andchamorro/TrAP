#!/usr/bin/env python3
"""Convert RepeatMasker .out output to GFF3 for the TrAP LINE-1 detection pipeline.

Parses NCBI RepeatMasker native .out / .out.gz format and emits GFF3 with
``Target=<name>`` attributes consumed by ``scripts/divide_gff.py`` (plan §6.2).
Called by ``scripts/data/fetch_references.sh`` to produce REPEATMASKER_GFF.

NCBI RepeatMasker .out chromosome names use RefSeq accessions (e.g.
NC_000001.11).  An NCBI assembly report is used to translate these to
UCSC-style names (chr1, chr2, …) so the GFF3 matches the STAR-aligned BAM,
which is built from the EBI GENCODE genome (UCSC chromosome naming).

Stdlib only — runs under any Python 3.9+ environment including the HPC system
Python available before ``conda activate trap``.

Usage::

    # From NCBI accession — downloads RM output and assembly report automatically
    python scripts/rm_to_gff3.py --accession GCF_000001405.40 \\
        --output data/external/GCF_000001405.40_GRCh38.p14_rm.gff

    # From local .out or .out.gz with a local assembly report
    python scripts/rm_to_gff3.py --input rm.out.gz \\
        --report-file assembly_report.txt --output rm.gff

    # From local file, no chromosome translation
    python scripts/rm_to_gff3.py --input rm.out --output rm.gff
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


_NCBI_API = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/assembly/{accession}/dataset_report"
)


# ---------------------------------------------------------------------------
# NCBI Datasets API helpers
# ---------------------------------------------------------------------------

def _ncbi_assembly_info(accession: str) -> dict:
    """Return the ``assembly_info`` block from the NCBI Datasets API."""
    url = _NCBI_API.format(accession=accession)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise ValueError(f"NCBI API request failed for {accession}: {exc}") from exc
    reports = data.get("reports", [])
    if not reports:
        raise ValueError(f"No assembly report found for accession {accession}")
    return reports[0].get("assembly_info", {})


def get_ncbi_rm_url(accession: str) -> str:
    """Return the NCBI FTP URL for the RepeatMasker .out.gz of *accession*."""
    info = _ncbi_assembly_info(accession)
    base_path = info.get("base_path", "")
    if not base_path:
        raise ValueError(f"No base_path in NCBI assembly info for {accession}")
    return f"{base_path}/{accession}_rm.out.gz"


def get_assembly_report_url(accession: str) -> str:
    """Return the NCBI FTP URL for the assembly report of *accession*."""
    return _ncbi_assembly_info(accession).get("assembly_report_url", "")


# ---------------------------------------------------------------------------
# Download helper (wget with curl fallback)
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str, timeout: int = 600) -> None:
    """Download *url* to *dest* using wget, falling back to curl."""
    print(f"[rm_to_gff3] downloading {url}")
    try:
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", dest, url],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    except FileNotFoundError:
        result = subprocess.run(
            ["curl", "-sSL", "-C", "-", "-o", dest, url],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Assembly report: RefSeq → UCSC chromosome name translation
# ---------------------------------------------------------------------------

def load_assembly_report(path: str) -> dict[str, str]:
    """Return a {RefSeq-accn: UCSC-name} mapping from an NCBI assembly_report.txt.

    The tab-separated report has:
      column 6  RefSeq-Accn  (e.g. NC_000001.11)
      column 9  UCSC-style-name  (e.g. chr1)
    Rows where the UCSC name is "na" are skipped.
    """
    mapping: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            refseq, ucsc = cols[6], cols[9]
            if refseq and ucsc and ucsc != "na":
                mapping[refseq] = ucsc
    return mapping


# ---------------------------------------------------------------------------
# Chromosome filter
# ---------------------------------------------------------------------------

def is_primary_chrom(name: str) -> bool:
    """Return True if *name* is chr1-22, chrX, chrY, or chrM."""
    primary = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
    if name in primary:
        return True
    for skip in ("chrUn_", "_random", "_alt", "_fix", "chrEBV"):
        if skip in name:
            return False
    for prefix in ("GL", "KI", "JH", "KB", "KT", "MU", "NC", "NT", "NW", "NZ"):
        if name.startswith(prefix):
            return False
    return False


# ---------------------------------------------------------------------------
# RepeatMasker .out parser
# ---------------------------------------------------------------------------

def parse_rm_line(line: str) -> Optional[dict]:
    """Parse one line of RepeatMasker .out; return a dict or None.

    RepeatMasker .out is space-separated with fixed column positions:
      0   SW score
      1   perc div
      2   perc del
      3   perc ins
      4   query sequence name (chromosome)
      5   query begin (1-based, inclusive)
      6   query end   (1-based, inclusive)
      7   query left  (parenthesised)
      8   strand  ('+' or 'C' for complement)
      9   repeat name        e.g. L1HS
     10   repeat class/family  e.g. LINE/L1
     11   repeat begin in element
     12   repeat end   in element
     13   repeat left  in element (parenthesised)
     14   repeat ID
    """
    line = line.strip()
    if not line or line.startswith(("SW", "score", "=")):
        return None
    fields = line.split()
    if len(fields) < 15:
        return None
    try:
        return {
            "chrom":     fields[4],
            "start":     int(fields[5]),
            "end":       int(fields[6]),
            "strand":    "-" if fields[8] == "C" else "+",
            "name":      fields[9],   # repeat name, e.g. L1HS
            "family":    fields[10],  # class/family, e.g. LINE/L1
            "score":     fields[0],   # SW score (kept as string for GFF3 column 6)
            "repeat_id": fields[14],
        }
    except (IndexError, ValueError):
        return None


def read_rm_file(path: str):
    """Yield parsed RM .out entries from *path* (.out or .out.gz)."""
    opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with opener as fh:
        for line in fh:
            entry = parse_rm_line(line)
            if entry is not None:
                yield entry


# ---------------------------------------------------------------------------
# GFF3 conversion
# ---------------------------------------------------------------------------

def convert_to_gff3(
    input_path: str,
    output_path: str,
    chrom_translation: Optional[dict[str, str]] = None,
    skip_scaffold: bool = True,
    family_pattern: Optional[str] = None,
    name_pattern: Optional[str] = None,
) -> int:
    """Convert RepeatMasker .out to GFF3; return the number of features written.

    Attributes written per feature::

        ID=rpt_<id>;Name=<subfamily>;family=<class/family>;Target=<subfamily>

    The ``family=`` attribute stores the raw RepeatMasker class/family string
    (e.g. ``LINE/L1``) so downstream tools (``build_l1_corpus.py``) can use it
    without re-reading the original .out file.

    ``Target=<name>`` is kept for backward compatibility with ``divide_gff.py``.
    GFF3 uses 1-based inclusive coordinates, which match RM .out directly.

    Args:
        family_pattern: If set, only entries whose class/family field matches
            are written (e.g. ``'LINE/L1'`` keeps all L1 subfamilies).
        name_pattern: If set, only entries whose repeat name matches are written
            (e.g. ``'^(?:L1HS|L1PA)'`` keeps only L1HS and L1PA subfamilies).
    """
    import re as _re
    family_pat = _re.compile(family_pattern) if family_pattern else None
    name_pat = _re.compile(name_pattern) if name_pattern else None
    count = 0
    with open(output_path, "w") as out:
        out.write("##gff-version 3\n")
        for entry in read_rm_file(input_path):
            if family_pat and not family_pat.search(entry["family"]):
                continue
            if name_pat and not name_pat.search(entry["name"]):
                continue
            chrom = entry["chrom"]
            if chrom_translation:
                chrom = chrom_translation.get(chrom, chrom)
            if skip_scaffold and not is_primary_chrom(chrom):
                continue
            start, end = entry["start"], entry["end"]
            if start > end:
                start, end = end, start
            attrs = (
                f"ID=rpt_{entry['repeat_id']};"
                f"Name={entry['name']};"
                f"family={entry['family']};"
                f"Target={entry['name']}"
            )
            out.write(
                f"{chrom}\tRepeatMasker\trepeat_region\t{start}\t{end}\t"
                f"{entry['score']}\t{entry['strand']}\t.\t{attrs}\n"
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--accession",
        metavar="ACC",
        help=(
            "NCBI assembly accession (e.g. GCF_000001405.40). "
            "Downloads RepeatMasker .out.gz and assembly report automatically."
        ),
    )
    source.add_argument(
        "--input",
        metavar="FILE",
        help="Local RepeatMasker .out or .out.gz file.",
    )
    p.add_argument("--output", required=True, metavar="FILE", help="Output GFF3 path.")
    p.add_argument(
        "--report-file",
        metavar="FILE",
        help=(
            "NCBI assembly_report.txt for RefSeq → UCSC chromosome name translation. "
            "Auto-downloaded when --accession is used."
        ),
    )
    p.add_argument(
        "--family-pattern",
        metavar="REGEX",
        default="LINE/L1",
        help=(
            "Regex applied to the class/family column; only matching entries are written. "
            "Default: 'LINE/L1'. Pass '' to include all families."
        ),
    )
    p.add_argument(
        "--name-pattern",
        metavar="REGEX",
        default=None,
        help=(
            "Regex applied to the repeat name (subfamily); only matching entries are written. "
            "Example: '^(?:L1HS|L1PA)' keeps only L1HS and L1PA subfamilies. "
            "Default: None (all names passing --family-pattern are kept)."
        ),
    )
    p.add_argument(
        "--skip-scaffold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude non-primary chromosomes (scaffolds, alt, unplaced). Default: true.",
    )
    p.add_argument(
        "--keep-rm",
        action="store_true",
        help="Keep the downloaded RepeatMasker .out.gz after conversion (default: delete when --accession is used).",
    )

    args = p.parse_args()
    out_dir = Path(args.output).parent
    temp_rm: Optional[str] = None
    temp_report: Optional[str] = None

    try:
        if args.accession:
            rm_url = get_ncbi_rm_url(args.accession)
            temp_rm = str(out_dir / f"{args.accession}_rm.out.gz")
            download_file(rm_url, temp_rm)
            input_path = temp_rm

            if not args.report_file:
                report_url = get_assembly_report_url(args.accession)
                if report_url:
                    temp_report = str(out_dir / f"{args.accession}_assembly_report.txt")
                    download_file(report_url, temp_report, timeout=300)
                    args.report_file = temp_report
                else:
                    print(
                        "[rm_to_gff3] WARNING: no assembly report URL found; "
                        "chromosome names will not be translated.",
                        file=sys.stderr,
                    )
        else:
            input_path = args.input

        chrom_translation: Optional[dict[str, str]] = None
        if args.report_file and os.path.exists(args.report_file):
            chrom_translation = load_assembly_report(args.report_file)
            print(f"[rm_to_gff3] loaded {len(chrom_translation)} chromosome mappings")

        if not os.path.exists(input_path):
            print(f"[rm_to_gff3] error: input not found: {input_path}", file=sys.stderr)
            return 1

        count = convert_to_gff3(
            input_path, args.output, chrom_translation, args.skip_scaffold,
            family_pattern=args.family_pattern or None,
            name_pattern=args.name_pattern or None,
        )
        print(f"[rm_to_gff3] wrote {count} features -> {args.output}")

    except Exception as exc:
        print(f"[rm_to_gff3] error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_rm and not args.keep_rm and os.path.exists(temp_rm):
            os.remove(temp_rm)
            print(f"[rm_to_gff3] removed temp {temp_rm}")
        if temp_report and os.path.exists(temp_report):
            os.remove(temp_report)
            print(f"[rm_to_gff3] removed temp {temp_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
