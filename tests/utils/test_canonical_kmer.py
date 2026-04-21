"""Unit tests for canonical k-mer encoding (Salmon/jellyfish-consistent)."""

import numpy as np
import pytest

from trap.utils.canonical_kmer import (
    canonical_code,
    canonical_codes,
    seq_to_base_codes,
    splitmix64,
)


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@pytest.mark.unit
class TestBaseCodes:
    def test_acgt_mapping(self):
        assert seq_to_base_codes("ACGT").tolist() == [0, 1, 2, 3]

    def test_lowercase_matches_uppercase(self):
        assert seq_to_base_codes("acgt").tolist() == [0, 1, 2, 3]


@pytest.mark.unit
class TestCanonicalCode:
    def test_kmer_and_revcomp_share_code(self):
        kmer = "ACGTACGTACGTACGTA"  # k=17
        assert canonical_code(kmer) == canonical_code(_revcomp(kmer))

    def test_palindrome_is_its_own_canonical(self):
        kmer = "ACGT"  # revcomp of ACGT is ACGT
        assert canonical_code(kmer) == canonical_code(_revcomp(kmer))

    def test_distinct_kmers_distinct_codes(self):
        assert canonical_code("AAAAA") != canonical_code("AAAAC")

    def test_scalar_matches_vectorized(self):
        seq = "ACGTTGCAACGTTGCAAC"
        k = 17
        vec = canonical_codes(seq, k)
        manual = [canonical_code(seq[j : j + k]) for j in range(len(seq) - k + 1)]
        assert vec.tolist() == manual


@pytest.mark.unit
class TestCanonicalCodes:
    def test_length_is_num_windows(self):
        seq = "A" * 30
        assert canonical_codes(seq, 17).shape[0] == 30 - 17 + 1

    def test_too_short_returns_empty(self):
        assert canonical_codes("ACGT", 17).shape[0] == 0

    def test_read_and_revcomp_same_canonical_multiset(self):
        seq = "ACGTTGCAACGTACGTTGCAAC"  # 22 bp
        fwd = sorted(canonical_codes(seq, 17).tolist())
        rev = sorted(canonical_codes(_revcomp(seq), 17).tolist())
        assert fwd == rev


@pytest.mark.unit
class TestSplitmix64:
    def test_deterministic(self):
        codes = np.array([0, 1, 2, 12345, 9**9], dtype=np.int64)
        assert splitmix64(codes).tolist() == splitmix64(codes).tolist()

    def test_spreads_small_inputs(self):
        codes = np.arange(1000, dtype=np.int64)
        buckets = splitmix64(codes) % np.uint64(64)
        # A good mix should populate most buckets for 1000 distinct inputs.
        assert len(np.unique(buckets)) >= 60
