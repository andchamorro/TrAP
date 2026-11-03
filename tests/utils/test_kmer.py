"""Tests for trap.utils.kmer — kmer_split, seq_to_encoded, encoded_to_seq, kmer_split_batch."""

import pytest

from trap.utils.kmer import encoded_to_seq, kmer_split, kmer_split_batch, seq_to_encoded


# -----------------------------------------------------------------------
# seq_to_encoded
# -----------------------------------------------------------------------

class TestSeqToEncoded:
    """Tests for n-mer Unicode compression."""

    @pytest.mark.unit
    def test_no_encoding_returns_identity(self):
        assert seq_to_encoded("ACTG") == "ACTG"

    @pytest.mark.unit
    def test_no_encoding_none_explicit(self):
        assert seq_to_encoded("ACTG", encoding=None) == "ACTG"

    @pytest.mark.unit
    @pytest.mark.parametrize("encoding", ["pairs", "2-mers", 2])
    def test_2mer_encoding_aliases(self, encoding):
        """All 2-mer aliases should produce the same result."""
        result = seq_to_encoded("ACTG", encoding=encoding)
        # 2-mers of ACTG: AC, TG -> two Unicode chars
        assert len(result) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("encoding", ["codons", "3-mers", 3])
    def test_3mer_encoding_aliases(self, encoding):
        """All 3-mer aliases should produce the same result."""
        result = seq_to_encoded("ACTGAC", encoding=encoding)
        # 3-mers of ACTGAC: ACT, GAC -> two Unicode chars
        assert len(result) == 2

    @pytest.mark.unit
    def test_2mer_encoding_output_is_uppercase_letters(self):
        result = seq_to_encoded("ACTG", encoding=2)
        for ch in result:
            assert ch.isupper() and ch.isalpha()

    @pytest.mark.unit
    def test_3mer_encoding_output_is_alpha(self):
        """3-mer encoding maps 64 codons to chr(65)..chr(128), some are lowercase."""
        result = seq_to_encoded("ACTGAC", encoding=3)
        for ch in result:
            assert ch.isalpha()

    @pytest.mark.unit
    def test_2mer_deterministic(self):
        """Same input must always produce same output."""
        a = seq_to_encoded("ACTGACTG", encoding=2)
        b = seq_to_encoded("ACTGACTG", encoding=2)
        assert a == b

    @pytest.mark.unit
    def test_empty_string_no_encoding(self):
        assert seq_to_encoded("") == ""


# -----------------------------------------------------------------------
# encoded_to_seq  (round-trip)
# -----------------------------------------------------------------------

class TestEncodedToSeq:
    """Tests for Unicode-to-DNA decoding."""

    @pytest.mark.unit
    def test_no_encoding_identity(self):
        assert encoded_to_seq("ACTG") == "ACTG"

    @pytest.mark.unit
    def test_2mer_roundtrip(self):
        original = "ACTGACTG"
        encoded = seq_to_encoded(original, encoding="pairs")
        decoded = encoded_to_seq(encoded, encoding="pairs")
        assert decoded == original

    @pytest.mark.unit
    def test_3mer_roundtrip(self):
        original = "ACTGAC"  # length divisible by 3
        encoded = seq_to_encoded(original, encoding="codons")
        decoded = encoded_to_seq(encoded, encoding="codons")
        assert decoded == original

    @pytest.mark.unit
    @pytest.mark.parametrize("seq", [
        "AAAA",
        "TTTT",
        "CCCC",
        "GGGG",
        "ACGT",
        "TGCA",
    ])
    def test_2mer_roundtrip_various(self, seq):
        encoded = seq_to_encoded(seq, encoding=2)
        decoded = encoded_to_seq(encoded, encoding="2-mers")
        assert decoded == seq

    @pytest.mark.unit
    @pytest.mark.parametrize("seq", [
        "ACTACT",
        "GGGCCC",
        "AAATTT",
    ])
    def test_3mer_roundtrip_various(self, seq):
        encoded = seq_to_encoded(seq, encoding=3)
        decoded = encoded_to_seq(encoded, encoding="3-mers")
        assert decoded == seq


# -----------------------------------------------------------------------
# kmer_split
# -----------------------------------------------------------------------

class TestKmerSplit:
    """Tests for k-mer splitting."""

    @pytest.mark.unit
    def test_basic_kmer_split(self):
        result = kmer_split(3, "ACTGA")
        tokens = result.split()
        # len("ACTGA") - 3 + 1 = 3 tokens
        assert len(tokens) == 3
        assert tokens == ["ACT", "CTG", "TGA"]

    @pytest.mark.unit
    def test_kmer_count_formula(self, short_dna_seq):
        """Number of k-mers should be len(seq) - k + 1."""
        k = 5
        tokens = kmer_split(k, short_dna_seq).split()
        assert len(tokens) == len(short_dna_seq) - k + 1

    @pytest.mark.unit
    @pytest.mark.parametrize("k", [1, 3, 5, 10, 18])
    def test_kmer_count_various_k(self, k):
        seq = "A" * 50
        tokens = kmer_split(k, seq).split()
        assert len(tokens) == 50 - k + 1

    @pytest.mark.unit
    def test_kmer_equal_to_sequence_length(self):
        seq = "ACTG"
        result = kmer_split(4, seq)
        assert result == "ACTG"

    @pytest.mark.unit
    def test_kmer_larger_than_sequence(self):
        seq = "ACT"
        result = kmer_split(5, seq)
        # len(seq) - k + 1 = -1, so empty
        assert result == ""

    @pytest.mark.unit
    def test_kmer_split_returns_string(self, short_dna_seq):
        result = kmer_split(3, short_dna_seq)
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_kmer_split_space_separated(self, short_dna_seq):
        result = kmer_split(3, short_dna_seq)
        assert " " in result

    @pytest.mark.unit
    def test_kmer_split_with_encoding(self):
        """When encoding is passed, each token should be compressed."""
        result_plain = kmer_split(4, "ACTGACTG", encoding=None)
        result_enc = kmer_split(4, "ACTGACTG", encoding=2)
        # Encoded tokens should be shorter than plain 4-char tokens
        plain_tokens = result_plain.split()
        enc_tokens = result_enc.split()
        assert len(plain_tokens) == len(enc_tokens)
        assert all(len(t) == 4 for t in plain_tokens)
        assert all(len(t) == 2 for t in enc_tokens)


# -----------------------------------------------------------------------
# kmer_split_batch
# -----------------------------------------------------------------------

class TestKmerSplitBatch:
    """Tests for batched k-mer splitting."""

    @pytest.mark.unit
    def test_single_batch(self):
        seqs = ["ACTGAC", "TTTGGG"]
        batches = list(kmer_split_batch(seqs, batch_size=10, k=3))
        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.unit
    def test_multiple_batches(self):
        seqs = ["ACTGAC"] * 5
        batches = list(kmer_split_batch(seqs, batch_size=2, k=3))
        # 5 seqs / batch_size 2 => 3 batches (2, 2, 1)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    @pytest.mark.unit
    def test_batch_contents_are_kmer_strings(self):
        seqs = ["ACTGAC"]
        batches = list(kmer_split_batch(seqs, batch_size=10, k=3))
        token_str = batches[0][0]
        tokens = token_str.split()
        assert tokens == ["ACT", "CTG", "TGA", "GAC"]

    @pytest.mark.unit
    def test_empty_input(self):
        batches = list(kmer_split_batch([], batch_size=10, k=3))
        assert batches == []

    @pytest.mark.unit
    def test_batch_with_encoding(self):
        seqs = ["ACTGACTG"]
        batches = list(kmer_split_batch(seqs, batch_size=10, k=4, encoding=2))
        enc_tokens = batches[0][0].split()
        # Each 4-bp k-mer encoded as 2-mers => 2 chars per token
        assert all(len(t) == 2 for t in enc_tokens)

    @pytest.mark.unit
    def test_generator_input_raises_type_error(self):
        """kmer_split_batch requires a Sequence (len + subscript); a bare generator must fail."""
        gen = (s for s in ["ACTGAC", "TTTTTT"])
        with pytest.raises(TypeError):
            list(kmer_split_batch(gen, batch_size=10, k=3))


# -----------------------------------------------------------------------
# Integer-alias roundtrip (cross-encoding)
# -----------------------------------------------------------------------


class TestIntegerAliasRoundtrip:
    """Integer aliases (2, 3) must be accepted on both encode and decode sides."""

    @pytest.mark.unit
    @pytest.mark.parametrize("enc_alias,dec_alias", [
        (2, 2),
        (2, "pairs"),
        ("pairs", 2),
        (2, "2-mers"),
        ("2-mers", 2),
    ])
    def test_2mer_roundtrip_cross_alias(self, enc_alias, dec_alias):
        original = "ACTGACTG"
        encoded = seq_to_encoded(original, encoding=enc_alias)
        decoded = encoded_to_seq(encoded, encoding=dec_alias)
        assert decoded == original

    @pytest.mark.unit
    @pytest.mark.parametrize("enc_alias,dec_alias", [
        (3, 3),
        (3, "codons"),
        ("codons", 3),
        (3, "3-mers"),
        ("3-mers", 3),
    ])
    def test_3mer_roundtrip_cross_alias(self, enc_alias, dec_alias):
        original = "ACTGAC"
        encoded = seq_to_encoded(original, encoding=enc_alias)
        decoded = encoded_to_seq(encoded, encoding=dec_alias)
        assert decoded == original


# -----------------------------------------------------------------------
# Sequence lengths that are not a multiple of the encoding size
# -----------------------------------------------------------------------


class TestNonDivisibleLengths:
    """The trailing ``len % n`` bases of an n-mer-encoded sequence are dropped.

    No padding is added, so decoding returns the truncated sequence rather
    than the original.  The encoding is lossy for such inputs by design.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("seq,expected_chars", [
        ("", 0),
        ("A", 0),
        ("AC", 1),
        ("ACT", 1),
        ("ACTG", 2),
        ("ACTGA", 2),
    ])
    def test_2mer_length_is_floor_division(self, seq, expected_chars):
        assert len(seq_to_encoded(seq, encoding=2)) == expected_chars

    @pytest.mark.unit
    @pytest.mark.parametrize("seq,expected_chars", [
        ("", 0),
        ("AC", 0),
        ("ACT", 1),
        ("ACTG", 1),
        ("ACTGA", 1),
        ("ACTGAC", 2),
    ])
    def test_3mer_length_is_floor_division(self, seq, expected_chars):
        assert len(seq_to_encoded(seq, encoding=3)) == expected_chars

    @pytest.mark.unit
    def test_2mer_trailing_base_is_ignored(self):
        assert seq_to_encoded("ACTGA", encoding=2) == seq_to_encoded("ACTG", encoding=2)

    @pytest.mark.unit
    def test_3mer_trailing_bases_are_ignored(self):
        assert seq_to_encoded("ACTGA", encoding=3) == seq_to_encoded("ACT", encoding=3)

    @pytest.mark.unit
    @pytest.mark.parametrize("seq", ["ACT", "ACTGA", "A"])
    def test_2mer_roundtrip_returns_truncated_sequence(self, seq):
        decoded = encoded_to_seq(seq_to_encoded(seq, encoding=2), encoding=2)
        assert decoded == seq[: len(seq) // 2 * 2]

    @pytest.mark.unit
    @pytest.mark.parametrize("seq", ["AC", "ACTG", "ACTGA"])
    def test_3mer_roundtrip_returns_truncated_sequence(self, seq):
        decoded = encoded_to_seq(seq_to_encoded(seq, encoding=3), encoding=3)
        assert decoded == seq[: len(seq) // 3 * 3]

    @pytest.mark.unit
    def test_non_acgt_character_raises(self):
        with pytest.raises(KeyError):
            seq_to_encoded("ACNT", encoding=2)

    @pytest.mark.unit
    @pytest.mark.parametrize("k,encoding,seq", [
        (3, 2, "ACTGAC"),
        (5, 2, "ACTGACTG"),
        (4, 3, "ACTGACTG"),
        (7, 3, "ACTGACTGAC"),
    ])
    def test_kmer_split_keeps_one_token_per_window(self, k, encoding, seq):
        """A k that the encoding does not divide still yields len - k + 1 tokens."""
        assert len(kmer_split(k, seq, encoding=encoding).split(" ")) == len(seq) - k + 1

    @pytest.mark.unit
    def test_kmer_split_token_drops_window_remainder(self):
        """k=3 under 2-mer encoding: each 3-mer window keeps only its first pair."""
        tokens = kmer_split(3, "ACTG", encoding=2).split(" ")
        assert tokens == [seq_to_encoded("AC", 2), seq_to_encoded("CT", 2)]
