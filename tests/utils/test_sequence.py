"""Tests for trap.utils.sequence (DR-2)."""
import pytest
import numpy as np

from trap.utils.sequence import break_long_read, standardize


@pytest.mark.unit
class TestStandardize:
    def test_uppercase(self):
        assert standardize("actg") == "ACTG"

    def test_strips_n(self):
        assert standardize("ACTGnNn") == "ACTG"

    def test_strips_other_chars(self):
        assert standardize("ACXTYGRN") == "ACTG"

    def test_empty(self):
        assert standardize("") == ""

    def test_pure_actg_unchanged(self):
        assert standardize("AATTCCGG") == "AATTCCGG"


@pytest.mark.unit
class TestBreakLongRead:
    def test_short_read_returns_one_pair(self):
        seq = "A" * 100
        result = break_long_read(seq, mean_fragment_size=500, std_fragment_size=10)
        assert len(result) == 1
        assert "forward" in result[0]
        assert "reverse" in result[0]

    def test_long_read_produces_multiple_pairs(self):
        rng = np.random.default_rng(42)
        seq = "ATCG" * 500  # 2000 bp
        result = break_long_read(seq, coverage=5, rng=rng)
        assert len(result) > 1

    def test_seeded_is_reproducible(self):
        seq = "ATCG" * 500
        r1 = break_long_read(seq, rng=np.random.default_rng(42))
        r2 = break_long_read(seq, rng=np.random.default_rng(42))
        assert r1 == r2

    def test_different_seeds_differ(self):
        seq = "ATCG" * 500
        r1 = break_long_read(seq, coverage=10, rng=np.random.default_rng(1))
        r2 = break_long_read(seq, coverage=10, rng=np.random.default_rng(2))
        assert r1 != r2

    def test_output_structure(self):
        seq = "ATCG" * 500
        pairs = break_long_read(seq, rng=np.random.default_rng(0))
        for pair in pairs:
            assert set(pair.keys()) == {"forward", "reverse"}
            assert isinstance(pair["forward"], str)
            assert isinstance(pair["reverse"], str)
