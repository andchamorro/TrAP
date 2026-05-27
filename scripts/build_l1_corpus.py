#!/usr/bin/env python3
"""Extract LINE-1 RepeatMasker sequences for tokenizer training (L1_CORPUS).

Reads the GFF3 produced by ``rm_to_gff3.py``, filters to the LINE/L1 family,
extracts genomic sequences via ``bedtools getfasta``, and writes a FASTA with
``>ID|Name|Family`` headers (e.g. ``>3|L1MC5a|LINE``).

This FASTA is consumed by ``scripts/slurm/10_tokenizer.slurm`` via the
``--extra-corpus`` / ``L1_CORPUS`` env var so that LINE-1 k-mers are pinned in
the tokenizer vocabulary regardless of the random GENCODE subsampling.

Requires: ``bedtools`` >=2.27 (available via ``load_bio_modules`` in
``_common.sh``).  Stdlib only — no conda env required — but must run after
the genome FASTA is available (``fetch_references.sh`` step 2).

Usage::

    # All LINE/L1 entries (GFF pre-filtered by rm_to_gff3.py)
    python scripts/build_l1_corpus.py \\
        --gff    data/external/GCF_000001405.40_GRCh38.p14_rm.gff \\
        --genome data/external/GRCh38.p14.genome.fa \\
        --output data/external/GCF_000001405.40_GRCh38.p14_rm.LINE1.gencode.v48.fa

    # Only L1HS and L1PA subfamilies
    python scripts/build_l1_corpus.py ... --name-pattern '^L1HS|^L1PA'
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ID_RE = re.compile(r"(?:^|;)ID=([^;\s]+)")
_NAME_RE = re.compile(r"(?:^|;)Name=([^;\s]+)")
_FAMILY_RE = re.compile(r"(?:^|;)family=([^;\s]+)")
# rm_to_gff3.py prefixes numeric IDs with "rpt_" to satisfy GFF3 (IDs must not
# start with a digit). Strip it so headers match the >N|Name|Family format.
_RPT_PREFIX = re.compile(r"^rpt_")


def _parse_gff_to_bed(
    gff_path: str,
    family_pattern: str | None,
    name_pattern: str | None,
    bed_path: str,
) -> int:
    """Write a name-encoded BED for entries in *gff_path* (optionally filtered).

    The BED name column encodes ``ID|Name|Family`` so ``bedtools getfasta -name``
    produces headers that only need the ``::coords`` suffix stripped.

    ``family=`` filtering only applies when the attribute is present in a GFF
    record; entries without ``family=`` are always included regardless of
    ``family_pattern``.  The family value in the BED name is taken from the GFF
    ``family=`` attribute (top-level only, e.g. ``LINE`` from ``LINE/L1``); when
    absent the field is left empty so the header becomes ``>ID|Name|``.

    Args:
        gff_path: Input RepeatMasker GFF3 path.
        family_pattern: Regex applied to ``family=`` when present; ``None``
            keeps all entries.
        name_pattern: Regex applied to ``Name=``; ``None`` keeps all entries.
        bed_path: Output BED file path.

    Returns:
        Number of features written to the BED.
    """
    family_pat = re.compile(family_pattern) if family_pattern else None
    name_pat = re.compile(name_pattern) if name_pattern else None
    n = 0
    with open(gff_path) as gf, open(bed_path, "w") as bed:
        for line in gf:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = cols[8]
            id_m = _ID_RE.search(attrs)
            name_m = _NAME_RE.search(attrs)
            family_m = _FAMILY_RE.search(attrs)
            name = name_m.group(1) if name_m else ""
            # Apply family filter only when the attribute is present
            if family_m and family_pat and not family_pat.search(family_m.group(1)):
                continue
            if name_pat and not name_pat.search(name):
                continue
            # Strip "rpt_" prefix that rm_to_gff3.py adds for GFF3 compliance
            raw_id = id_m.group(1) if id_m else str(n + 1)
            feat_id = _RPT_PREFIX.sub("", raw_id)
            # Top-level family only (e.g. "LINE" from "LINE/L1"); empty when absent
            family = family_m.group(1).split("/")[0] if family_m else ""
            chrom = cols[0]
            # GFF3 is 1-based inclusive; BED is 0-based half-open
            start = int(cols[3]) - 1
            end = int(cols[4])
            strand = cols[6] if cols[6] in ("+", "-") else "+"
            bed.write(f"{chrom}\t{start}\t{end}\t{feat_id}|{name}|{family}\t0\t{strand}\n")
            n += 1
    return n


def _bedtools_getfasta(genome: str, bed: str, out_fa: str) -> None:
    """Run ``bedtools getfasta -s -name`` to extract strand-aware sequences.

    Args:
        genome: Plain (uncompressed) genome FASTA path.
        bed: BED file with name-encoded fields.
        out_fa: Output raw FASTA path (headers will still contain ::coords suffix).
    """
    cmd = [
        "bedtools", "getfasta",
        "-fi", genome,
        "-bed", bed,
        "-s",       # strand-aware reverse complement
        "-name",    # use BED name column as header
        "-fo", out_fa,
    ]
    print(f"[build_l1_corpus] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)


def _strip_coords_suffix(raw_fa: str, out_fa: str) -> int:
    """Strip the ``::chrom:start-end(strand)`` suffix bedtools ``-name`` appends.

    Args:
        raw_fa: Raw FASTA produced by bedtools (headers contain ``::coords``).
        out_fa: Output path for the cleaned FASTA.

    Returns:
        Number of sequences written.
    """
    n = 0
    with open(raw_fa) as fi, open(out_fa, "w") as fo:
        for line in fi:
            if line.startswith(">"):
                header = line[1:].rstrip("\n")
                fo.write(f">{header.split('::')[0]}\n")
                n += 1
            else:
                fo.write(line)
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gff", required=True, help="RepeatMasker GFF3 (from rm_to_gff3.py)")
    p.add_argument(
        "--genome",
        required=True,
        help="Reference genome FASTA — must be plain (uncompressed); bedtools does not support .gz",
    )
    p.add_argument("--output", required=True, help="Output FASTA path (data/external/)")
    p.add_argument(
        "--family-pattern",
        default=None,
        help=(
            "Regex applied to the GFF family= attribute to filter features "
            "(e.g. 'LINE/L1'). Only active when the attribute is present; entries "
            "without family= are always included. Default: None (no family filter)."
        ),
    )
    p.add_argument(
        "--name-pattern",
        default=None,
        help=(
            "Regex applied to the GFF Name= attribute to filter features "
            "(e.g. '^(?:L1HS|L1PA)' keeps only L1HS and L1PA subfamilies). "
            "Default: None (all entries)."
        ),
    )
    args = p.parse_args(argv)

    for label, path in [("GFF", args.gff), ("genome", args.genome)]:
        if not os.path.isfile(path):
            print(f"[build_l1_corpus] ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        bed_path = os.path.join(tmp, "line1.bed")
        raw_fa = os.path.join(tmp, "line1_raw.fa")

        print(
            f"[build_l1_corpus] parsing GFF -> BED "
            f"(family_pattern={args.family_pattern!r}, name_pattern={args.name_pattern!r})",
            file=sys.stderr,
        )
        n_features = _parse_gff_to_bed(
            args.gff, args.family_pattern, args.name_pattern, bed_path
        )
        print(f"[build_l1_corpus] {n_features:,} features matched", file=sys.stderr)
        if n_features == 0:
            print(
                "[build_l1_corpus] ERROR: no features matched the family pattern",
                file=sys.stderr,
            )
            return 1

        _bedtools_getfasta(args.genome, bed_path, raw_fa)
        n_seqs = _strip_coords_suffix(raw_fa, args.output)
        print(
            f"[build_l1_corpus] wrote {n_seqs:,} sequences -> {args.output}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
