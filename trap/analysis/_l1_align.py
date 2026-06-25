"""minimap2 alignment of reads to the L1.3 consensus — the gold-standard referee.

Exact k-mer overlap is a fast screen but divergence-sensitive (an L1PA8 read
diverged ~15% from L1.3 loses most exact k-mers). Local alignment with minimap2
reports **percent identity over the aligned length**, which is divergence-robust,
and the **target coordinate** on L1.3 (GenBank L19088.1), which lets us bin each
read into a functional L1 region (5′UTR / ORF1 / ORF2[EN, RT] / 3′UTR).

Functional-region coordinates are on full-length human L1.3 (L19088.1, ~6.0 kb),
the canonical active human L1 (Solovyov et al. 2025; Brouha et al. 2003). ORF2 is
split into its endonuclease (EN) and reverse-transcriptase (RT) domains
(approximate; ORF2p domain architecture per Baldwin et al. 2024).

On Grace/HPRC: ``module load GCCcore/13.2.0 minimap2/2.29``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

# (name, start, end) 1-based, inclusive, on L1.3 / L19088.1.
L1_3_REGIONS: Tuple[Tuple[str, int, int], ...] = (
    ("5UTR", 1, 909),
    ("ORF1", 910, 1923),
    ("interORF", 1924, 1990),
    ("ORF2_EN", 1991, 2730),
    ("ORF2_mid", 2731, 3399),
    ("ORF2_RT", 3400, 4350),
    ("ORF2_C", 4351, 5814),
    ("3UTR", 5815, 6100),
)


def l1_region(position: float) -> str:
    """Functional L1.3 region containing ``position`` (1-based)."""
    for name, lo, hi in L1_3_REGIONS:
        if lo <= position <= hi:
            return name
    return "outside"


@dataclass
class L1Alignment:
    """A read's primary minimap2 alignment to an L1 reference."""

    identity: float  # residue matches / alignment block length (BLAST-like)
    aligned_frac: float  # (query end - query start) / query length
    target_mid: float  # midpoint of the alignment on the target (1-based)
    region: str  # functional L1.3 region (only meaningful when target IS L1.3)
    strand: str
    target_name: str = ""  # the reference the read aligned to (e.g. the subfamily consensus)


def parse_paf_line(line: str) -> Optional[Tuple[str, L1Alignment]]:
    """Parse one PAF record into ``(query_name, L1Alignment)`` (None if malformed)."""
    f = line.rstrip("\n").split("\t")
    if len(f) < 12:
        return None
    try:
        qname, qlen, qs, qe, strand = f[0], int(f[1]), int(f[2]), int(f[3]), f[4]
        target_name = f[5]
        ts, te, nmatch, alen = int(f[7]), int(f[8]), int(f[9]), int(f[10])
    except (ValueError, IndexError):
        return None
    identity = nmatch / alen if alen else 0.0
    aligned_frac = (qe - qs) / qlen if qlen else 0.0
    mid = (ts + te) / 2 + 1  # 0-based half-open → ~1-based midpoint
    return qname, L1Alignment(identity, aligned_frac, mid, l1_region(mid), strand, target_name)


def parse_paf(text: str) -> Dict[str, L1Alignment]:
    """Parse PAF text → best (highest identity×coverage) alignment per query."""
    best: Dict[str, L1Alignment] = {}
    for line in text.splitlines():
        parsed = parse_paf_line(line)
        if parsed is None:
            continue
        qname, aln = parsed
        cur = best.get(qname)
        if cur is None or aln.identity * aln.aligned_frac > cur.identity * cur.aligned_frac:
            best[qname] = aln
    return best


def minimap2_available(binary: str = "minimap2") -> bool:
    """Whether the ``minimap2`` executable is on PATH."""
    return shutil.which(binary) is not None


def align_sequences(
    sequences: Sequence[str],
    ref_fasta: str,
    binary: str = "minimap2",
    preset: str = "sr",
    seed_k: int = 13,
    seed_w: int = 6,
    threads: int = 8,
    tmpdir: Optional[str] = None,
) -> List[Optional[L1Alignment]]:
    """Align reads to the L1 reference, one alignment (or None) per input read.

    Reads are written to a temp FASTA named by index (so the result maps back
    positionally regardless of read-id characters). minimap2 runs with the
    short-read preset but a **smaller seed k-mer** (``seed_k``, default 13 vs the
    ``sr`` default 21) so divergent L1 reads (old L1PA, 15–30% from L1.3) still
    seed — the ``sr`` k=21 default never seeds them. Reads with no alignment →
    ``None`` (no detectable L1 sequence). ``seed_k=0`` keeps the preset default.
    """
    if not sequences:
        return []
    tmpdir = tmpdir or os.environ.get("TMPDIR") or tempfile.gettempdir()
    fd, reads_fa = tempfile.mkstemp(suffix=".fa", dir=tmpdir)
    cmd = [binary, "-x", preset, "--secondary=no", "-t", str(threads)]
    if seed_k:
        cmd += ["-k", str(seed_k), "-w", str(seed_w)]
    try:
        with os.fdopen(fd, "w") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f">{i}\n{seq}\n")
        proc = subprocess.run(
            cmd + [ref_fasta, reads_fa], capture_output=True, text=True, check=True
        )
    finally:
        os.unlink(reads_fa)
    by_idx = parse_paf(proc.stdout)
    return [by_idx.get(str(i)) for i in range(len(sequences))]
