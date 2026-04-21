"""Canonical k-mer encoding consistent with Salmon / jellyfish (``mer_dna``).

Base codes are ``A=0, C=1, G=2, T=3`` and the complement of code ``x`` is
``3 - x`` (matching ``salmon``'s ``CanonicalKmer.hpp`` and jellyfish
``mer_dna``).  A k-mer's *canonical* code is ``min(forward, reverse_complement)``
so a k-mer and its reverse complement collapse to a single identity — the same
notion Salmon uses when building its index, and the property that lets a read
and its reverse complement share tokens.

The integer encoding is big-endian (first base most significant):
``code = sum_i base(kmer[i]) * 4**(k-1-i)``.  The absolute integer need not match
Salmon's internal packing; only the equivalence ``code(kmer) == code(revcomp)``
under canonicalisation matters, and that is preserved.

For ``k <= 31`` a code fits in a signed 64-bit integer (``4**31 < 2**63``), so
all array operations use ``numpy.int64``.

Inputs are assumed to be clean upper/lower-case ``ACGT`` (TrAP standardises
sequences to upper-case ``ACTG`` with ``N`` stripped before tokenisation).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Byte → 2-bit code lookup (A=0, C=1, G=2, T=3; upper and lower case).
_BYTE_TO_CODE = np.full(256, -1, dtype=np.int64)
for _base, _code in zip(b"ACGT", (0, 1, 2, 3)):
    _BYTE_TO_CODE[_base] = _code
    _BYTE_TO_CODE[_base + 32] = _code  # lower-case (ASCII +32)

# Scalar lookup for single-k-mer encoding.
_CODE = {ch: i for i, ch in enumerate("ACGT")}
_CODE.update({ch.lower(): i for ch, i in list(_CODE.items())})

# splitmix64 mixing constants (used to spread decoy k-mers across hash buckets).
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def seq_to_base_codes(seq: str) -> np.ndarray:
    """Return the per-base 2-bit codes of *seq* as an ``int64`` array."""
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _BYTE_TO_CODE[raw]


def canonical_codes(seq: str, k: int) -> np.ndarray:
    """Canonical codes of every overlapping k-mer in *seq*.

    Args:
        seq: DNA sequence (clean ``ACGT``).
        k: K-mer length.

    Returns:
        ``int64`` array of length ``len(seq) - k + 1`` (empty if ``len(seq) < k``)
        holding ``min(forward, revcomp)`` for each window.
    """
    base = seq_to_base_codes(seq)
    if base.shape[0] < k:
        return np.empty(0, dtype=np.int64)

    windows = sliding_window_view(base, k)  # (n, k) int64
    pow4_be = np.int64(4) ** np.arange(k - 1, -1, -1, dtype=np.int64)  # 4**(k-1)..4**0
    pow4_le = pow4_be[::-1]  # 4**0..4**(k-1)

    forward = windows @ pow4_be
    revcomp = (np.int64(3) - windows) @ pow4_le
    return np.minimum(forward, revcomp)


def canonical_code(kmer: str) -> int:
    """Canonical code of a single k-mer string."""
    forward = 0
    for ch in kmer:
        forward = (forward << 2) | _CODE[ch]
    revcomp = 0
    for ch in reversed(kmer):
        revcomp = (revcomp << 2) | (3 - _CODE[ch])
    return forward if forward < revcomp else revcomp


def splitmix64(codes: np.ndarray) -> np.ndarray:
    """Deterministic splitmix64 hash of canonical codes (``int64`` → ``uint64``)."""
    with np.errstate(over="ignore"):
        x = codes.astype(np.uint64)
        x = (x ^ (x >> _S30)) * _MIX1
        x = (x ^ (x >> _S27)) * _MIX2
        x = x ^ (x >> _S31)
    return x
