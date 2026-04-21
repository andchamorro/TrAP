"""Sequence-level utilities: standardization and long-read fragmentation (DR-2).

These functions were previously inlined inside ``predict.py`` and
``GenomeDataset``.  Centralising them here makes them importable from both
the preprocessing and inference paths without circular-import risk.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np


# Strip everything that is not A, C, T, or G (includes N).
# Decision 2026-05-28: N residues are removed so the vocabulary stays pure
# ACTG for the k=17/v48 SentencePiece tokenizer.
_ACTG_RE = re.compile(r"[^ACTG]")

# Base-complement table; non-ACGT characters (e.g. N) pass through unchanged,
# matching Biopython's reverse_complement behaviour.
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCATGCA")


def standardize(seq: str) -> str:
    """Return *seq* uppercased with non-ACTG characters removed.

    Args:
        seq: Raw nucleotide string.

    Returns:
        Cleaned sequence containing only A, C, T, G.
    """
    return _ACTG_RE.sub("", seq.upper())


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of *seq*.

    ACGT (upper/lower) are complemented; any other character (e.g. ``N``) is
    passed through unchanged. For an already-standardized ACTG string this is
    equivalent to ``standardize(Bio.Seq(seq).reverse_complement())`` but needs
    no Biopython object and is computed on demand (no per-record precompute).

    Args:
        seq: Nucleotide string.

    Returns:
        Reverse-complemented string.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def break_long_read(
    long_read: str,
    read_length: int = 150,
    mean_fragment_size: int = 500,
    std_fragment_size: int = 10,
    coverage: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, str]]:
    """Fragment a long PacBio read into simulated paired Illumina pairs.

    Args:
        long_read: Input nucleotide sequence.
        read_length: Output read length in bp.
        mean_fragment_size: Mean insert size (bp).
        std_fragment_size: Std-dev of insert size (bp).
        coverage: Simulated coverage depth; governs number of fragments.
        rng: numpy ``Generator`` for reproducible fragmentation.  A fresh
            non-seeded generator is used when ``None`` is passed.

    Returns:
        List of ``{"forward": str, "reverse": str}`` dicts.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(long_read) < mean_fragment_size + std_fragment_size:
        return [{"forward": long_read[:read_length], "reverse": long_read[read_length::-1]}]

    num_fragments = int(len(long_read) * coverage / mean_fragment_size)
    fragment_sizes = rng.normal(mean_fragment_size, std_fragment_size, num_fragments).astype(int)

    short_reads: List[Dict[str, str]] = []
    for fragment_size in fragment_sizes:
        max_start = len(long_read) - fragment_size
        forward_start = int(rng.integers(0, max(1, max_start + 1)))
        forward_end = forward_start + read_length
        forward = long_read[forward_start:forward_end]

        insert_size = fragment_size - (read_length * 2)
        reverse_start = forward_end + insert_size
        reverse_end = reverse_start + read_length
        if reverse_end > len(long_read):
            reverse_end = int(rng.integers(read_length, len(long_read)))
            reverse_start = reverse_end - read_length
        reverse = long_read[reverse_start:reverse_end]
        short_reads.append({"forward": forward, "reverse": reverse})

    return short_reads
