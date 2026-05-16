"""Tests for deterministic k-mer counting and rarefaction."""

from collections import Counter

import pytest

from trap.analysis import kmer_counting as KC

pytestmark = pytest.mark.unit


def _brute_force_counts(sequences, k):
    """Reference forward k-mer multiplicities via a plain Counter."""
    counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            counter[seq[i : i + k]] += 1
    return sorted(counter.values())


def test_count_kmers_matches_brute_force():
    sequences = ["ACGTACGT", "TTTTAC", "ACGTA"]
    counts = KC.count_kmers(sequences, k=2)
    assert sorted(int(c) for c in counts) == _brute_force_counts(sequences, 2)


def test_count_kmers_total_equals_window_count():
    sequences = ["ACGTACGTAC", "GGGGCCCC"]
    counts = KC.count_kmers(sequences, k=3)
    expected_total = sum(len(s) - 3 + 1 for s in sequences)
    assert int(counts.sum()) == expected_total


def test_count_kmers_respects_chunk_size():
    sequences = ["ACGTACGT"] * 10
    big = KC.count_kmers(sequences, k=2, chunk_size=100)
    small = KC.count_kmers(sequences, k=2, chunk_size=1)
    assert sorted(big.tolist()) == sorted(small.tolist())


def test_count_kmers_skips_sequences_shorter_than_k():
    assert KC.count_kmers(["AC", "A", ""], k=5).size == 0


def test_canonical_merges_reverse_complements():
    # A sequence and its reverse complement together: canonical collapses pairs,
    # so the distinct count is no larger than the forward count.
    sequences = ["ACGTACGT", "ACGTACGT"[::-1]]
    forward = KC.count_kmers(sequences, k=3, canonical=False)
    canonical = KC.count_kmers(sequences, k=3, canonical=True)
    assert canonical.size <= forward.size
    assert int(canonical.sum()) == int(forward.sum())  # occurrences conserved


def test_iter_forward_sequences_strips_n(fasta_path):
    seqs = list(KC.iter_forward_sequences(fasta_path, "fasta"))
    assert all(set(s) <= set("ACTG") for s in seqs)


def test_kmer_spectrum_row_count_and_keys():
    sequences = ["ACGTACGTACGT", "TTGGCCAATTGG", "ACGTTTGGCCAA", "GGGGCCCCAAAA"]
    rows = KC.kmer_spectrum(
        sequences, k_values=[2, 3], subsample_fracs=[0.5, 1.0], replicates=2, seed=11
    )
    assert len(rows) == 2 * 2 * 2  # k * fracs * replicates
    row = rows[0]
    for key in ("corpus", "k", "subsample_frac", "replicate", "n_sequences", "h_mle"):
        assert key in row


def test_kmer_spectrum_is_deterministic():
    sequences = ["ACGTACGTACGT", "TTGGCCAATTGG", "ACGTTTGGCCAA", "GGGGCCCCAAAA"]
    kwargs = dict(k_values=[2, 3], subsample_fracs=[0.5, 1.0], replicates=2, seed=11, bootstrap=20)
    assert KC.kmer_spectrum(sequences, **kwargs) == KC.kmer_spectrum(sequences, **kwargs)


def test_kmer_spectrum_empty_corpus_raises():
    with pytest.raises(ValueError):
        KC.kmer_spectrum([], k_values=[2])
