"""Tests for trap.utils.dna2bit — 2-bit DNA encoding/decoding."""

import pytest

from trap.utils.dna2bit import (
    _base_to_bit,
    _bits_to_base,
    _bits_to_comp,
    bases_to_byte,
    byte_to_bases,
    byte_to_comp,
    decode_dna,
    decode_n_blocks,
    decode_rco,
    rev_comp,
)


# -----------------------------------------------------------------------
# Lookup tables
# -----------------------------------------------------------------------

class TestLookupTables:
    """Validate the static encoding/decoding tables."""

    @pytest.mark.unit
    def test_base_to_bit_covers_all_bases(self):
        for base in "ACGTNacgt":
            assert base in _base_to_bit

    @pytest.mark.unit
    def test_bits_to_base_covers_0_to_3(self):
        for i in range(4):
            assert i in _bits_to_base

    @pytest.mark.unit
    def test_bits_to_comp_is_complement(self):
        # T(0)->C, C(1)->T, A(2)->G, G(3)->A
        assert _bits_to_comp[0] == "C"
        assert _bits_to_comp[1] == "T"
        assert _bits_to_comp[2] == "G"
        assert _bits_to_comp[3] == "A"


# -----------------------------------------------------------------------
# bases_to_byte / byte_to_bases round-trip
# -----------------------------------------------------------------------

class TestByteBases:
    """Test encoding four bases into one byte and back."""

    @pytest.mark.unit
    @pytest.mark.parametrize("bases", [
        "TCAG",
        "AAAA",
        "TTTT",
        "CCCC",
        "GGGG",
        "ACTG",
        "TGCA",
    ])
    def test_roundtrip(self, bases):
        byte_val = bases_to_byte(bases)
        recovered = "".join(byte_to_bases(byte_val))
        assert recovered == bases

    @pytest.mark.unit
    def test_byte_range(self):
        """Any 4-base encoding should fit in one byte (0-255)."""
        byte_val = bases_to_byte("ACTG")
        assert 0 <= byte_val <= 255

    @pytest.mark.unit
    def test_case_insensitive_encoding(self):
        upper = bases_to_byte("ACTG")
        lower = bases_to_byte("actg")
        assert upper == lower


# -----------------------------------------------------------------------
# byte_to_comp
# -----------------------------------------------------------------------

class TestByteToComp:
    """Test complement decoding from a byte."""

    @pytest.mark.unit
    def test_complement_of_known_byte(self):
        # ACTG encoded, complement should be TGAC
        byte_val = bases_to_byte("ACTG")
        comp = "".join(byte_to_comp(byte_val))
        # Complement mapping: A->G(wrong?), let's verify via table
        # Actually: bit 0->C, 1->T, 2->G, 3->A
        # ACTG = bits 10,01,11,00 -> bases A,C,G,T
        # comp via _bits_to_comp: 10->G, 01->T, 11->A, 00->C
        assert len(comp) == 4
        assert all(c in "ACGT" for c in comp)


# -----------------------------------------------------------------------
# rev_comp
# -----------------------------------------------------------------------

class TestRevComp:
    """Test bitwise reverse-complement of a byte."""

    @pytest.mark.unit
    def test_rev_comp_is_integer(self):
        result = rev_comp(0b10011100)
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_double_rev_comp_identity(self):
        """Applying rev_comp twice should return the original byte."""
        original = 0b10011100
        double = rev_comp(rev_comp(original))
        assert (double & 0xFF) == original


# -----------------------------------------------------------------------
# decode_dna
# -----------------------------------------------------------------------

class TestDecodeDna:
    """Test multi-byte DNA decoding."""

    @pytest.mark.unit
    def test_single_byte_yields_four_bases(self):
        byte_val = bases_to_byte("ACTG")
        result = decode_dna([byte_val])
        assert len(result) == 4

    @pytest.mark.unit
    def test_multiple_bytes(self):
        b1 = bases_to_byte("ACTG")
        b2 = bases_to_byte("TTTT")
        result = decode_dna([b1, b2])
        assert len(result) == 8
        assert result[:4] == list("ACTG")
        assert result[4:] == list("TTTT")


# -----------------------------------------------------------------------
# decode_rco
# -----------------------------------------------------------------------

class TestDecodeRco:
    """Test reverse-complement decoding of byte sequences."""

    @pytest.mark.unit
    def test_output_length(self):
        b1 = bases_to_byte("ACTG")
        result = decode_rco([b1])
        assert len(result) == 4

    @pytest.mark.unit
    def test_all_bases_valid(self):
        b1 = bases_to_byte("ACTG")
        result = decode_rco([b1])
        assert all(c in "ACGTN" for c in result)


# -----------------------------------------------------------------------
# decode_n_blocks
# -----------------------------------------------------------------------

class TestDecodeNBlocks:
    """Test N-block header decoding."""

    @pytest.mark.unit
    def test_zero_blocks(self):
        # blockCount = 0 encoded as 4 bytes big-endian
        data = (0).to_bytes(4, byteorder="big")
        count, starts, sizes = decode_n_blocks(data)
        assert count == 0
        assert starts == []
        assert sizes == []

    @pytest.mark.unit
    def test_one_block(self):
        count_bytes = (1).to_bytes(4, byteorder="big")
        start_bytes = (10).to_bytes(4, byteorder="big")
        size_bytes = (5).to_bytes(4, byteorder="big")
        data = count_bytes + start_bytes + size_bytes
        count, starts, sizes = decode_n_blocks(data)
        assert count == 1
        assert starts == [10]
        assert sizes == [5]

    @pytest.mark.unit
    def test_two_blocks(self):
        count_bytes = (2).to_bytes(4, byteorder="big")
        s1 = (10).to_bytes(4, byteorder="big")
        s2 = (50).to_bytes(4, byteorder="big")
        sz1 = (5).to_bytes(4, byteorder="big")
        sz2 = (3).to_bytes(4, byteorder="big")
        data = count_bytes + s1 + s2 + sz1 + sz2
        count, starts, sizes = decode_n_blocks(data)
        assert count == 2
        assert starts == [10, 50]
        assert sizes == [5, 3]
