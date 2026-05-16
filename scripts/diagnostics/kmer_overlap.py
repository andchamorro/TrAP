"""Diagnostic: read ↔ L1-reference k-mer overlap (canonical, Salmon-consistent).

Quantifies, per L1 subfamily group vs NEGATIVE, what fraction of each read's
canonical k-mers are present in the L1 reference jellyfish index (built with
``-C``).  Motivated by the finding on the balanced sample that even L1 reads had
only ~19% k-mer *presence* in ``l1.jf`` — this re-checks it at scale on the full
``data/external/*_R*.fq`` reads.

Method: reservoir-sample up to ``--sample-per-group`` reads per group, standardize
them (strip non-ACTG — matching how training tokenizes), write to one FASTA, and
``jellyfish query`` every k-mer against the index.  jellyfish canonicalises the
query exactly like Salmon, so this measures the same canonical identity the
``SalmonKmerTokenizer`` target index would use.

Usage:
    python scripts/diagnostics/kmer_overlap.py \
        --reads data/external/l1hs_l1pa2_negative.5x_R1.fq \
        --jf data/external/l1hs_l1pa2.k17.canonical.jf --k 17 \
        --sample-per-group 5000 --out results/kmer_overlap.tsv
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Dict, List, Optional

from loguru import logger
import numpy as np
import typer

from trap.utils.sequence import standardize

# Ordered so the printed table reads young → old → background.
_GROUP_ORDER = [
    "L1HS (young)",
    "L1PA2-6 (young)",
    "L1PA7+/L1P (older)",
    "L1M*/L1ME* (old)",
    "other L1",
    "NEGATIVE",
    "other",
]


def _extract_label(read_id: str) -> str:
    """Label from a TrAP record id (mirrors preprocessing_sequences._extract_label)."""
    parts = read_id.split("|")
    return parts[-1].split("-")[0] if len(parts) >= 2 else read_id


def _group(label: str) -> str:
    u = label.upper()
    if u == "NEGATIVE":
        return "NEGATIVE"
    if u.startswith("L1HS"):
        return "L1HS (young)"
    if u in {f"L1PA{n}" for n in range(2, 7)}:
        return "L1PA2-6 (young)"
    if u.startswith("L1PA") or u.startswith("L1P"):
        return "L1PA7+/L1P (older)"
    if u.startswith("L1M"):
        return "L1M*/L1ME* (old)"
    if u.startswith("L1"):
        return "other L1"
    return "other"


def _reservoir_sample(reads_path: Path, sample_per_group: int, seed: int) -> Dict[str, List[str]]:
    """Stream a FASTQ and reservoir-sample up to N standardized reads per group."""
    rng = random.Random(seed)
    samples: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, int] = defaultdict(int)
    read_id = None
    with open(reads_path) as fh:
        for i, line in enumerate(fh):
            phase = i % 4
            if phase == 0:
                read_id = line[1:].strip()
            elif phase == 1:
                group = _group(_extract_label(read_id))
                seq = standardize(line.strip())
                seen[group] += 1
                bucket = samples[group]
                if len(bucket) < sample_per_group:
                    bucket.append(seq)
                else:
                    j = rng.randint(0, seen[group] - 1)
                    if j < sample_per_group:
                        bucket[j] = seq
    logger.info("Scanned reads per group: " + ", ".join(f"{g}={n:,}" for g, n in seen.items()))
    return samples


def _query_counts(fasta_path: str, jf: Path) -> np.ndarray:
    """Run ``jellyfish query`` and return the per-k-mer counts in file order."""
    result = subprocess.run(
        ["jellyfish", "query", "-s", fasta_path, str(jf)],
        capture_output=True,
        text=True,
        check=True,
    )
    return np.array(
        [int(line.split()[-1]) for line in result.stdout.splitlines() if line.split()],
        dtype=np.int64,
    )


def main(
    reads: Path = typer.Option(..., help="R1 (or single) FASTQ with TrAP labelled headers"),
    jf: Path = typer.Option(..., help="jellyfish index built with -C (canonical)"),
    k: int = typer.Option(17, help="K-mer length (must match the index)"),
    sample_per_group: int = typer.Option(5000, help="Reads sampled per subfamily group"),
    conserved: str = typer.Option("10,100", help="Comma-separated count thresholds"),
    seed: int = typer.Option(3469, help="Reservoir-sampling seed"),
    out: Optional[Path] = typer.Option(None, help="Write the table as TSV here"),
):
    """Report per-group canonical k-mer overlap of reads against the L1 index."""
    thresholds = [int(t) for t in conserved.split(",")]
    logger.info(f"Sampling up to {sample_per_group:,}/group from {reads}")
    samples = _reservoir_sample(reads, sample_per_group, seed)

    records: List[tuple] = []  # (group, n_kmers)
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fa:
        for group, seqs in samples.items():
            for seq in seqs:
                if len(seq) < k:
                    continue
                fa.write(f">r\n{seq}\n")
                records.append((group, len(seq) - k + 1))
        fasta_path = fa.name

    logger.info(f"Querying {len(records):,} reads ({sum(n for _, n in records):,} k-mers) vs {jf}")
    try:
        counts = _query_counts(fasta_path, jf)
    finally:
        os.unlink(fasta_path)

    expected = sum(n for _, n in records)
    if len(counts) != expected:
        raise RuntimeError(
            f"jellyfish returned {len(counts):,} counts, expected {expected:,}; "
            "k-mer/segmentation mismatch (non-ACTG leakage?)."
        )

    # Per-group accumulators.
    present = defaultdict(list)
    conserved_hits: Dict[int, dict] = {t: defaultdict(list) for t in thresholds}
    offset = 0
    for group, n_kmers in records:
        seg = counts[offset : offset + n_kmers]
        offset += n_kmers
        present[group].append(float(np.mean(seg >= 1)))
        for t in thresholds:
            conserved_hits[t][group].append(float(np.mean(seg >= t)))

    header = ["group", "reads", "present%(c>=1)"] + [f"c>={t}%" for t in thresholds]
    rows = [header]
    logger.success("Read ↔ L1 reference canonical k-mer overlap:")
    line = "  " + "  ".join(f"{h:>16}" for h in header)
    logger.info(line)
    ordered = [g for g in _GROUP_ORDER if g in present] + [
        g for g in present if g not in _GROUP_ORDER
    ]
    for group in ordered:
        n = len(present[group])
        vals = [group, str(n), f"{100 * np.mean(present[group]):.1f}"]
        vals += [f"{100 * np.mean(conserved_hits[t][group]):.1f}" for t in thresholds]
        rows.append(vals)
        logger.info("  " + "  ".join(f"{v:>16}" for v in vals))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join("\t".join(r) for r in rows) + "\n")
        logger.success(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
