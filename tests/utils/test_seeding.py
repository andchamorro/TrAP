"""Tests for trap.utils.seeding — seeded_rng API and Generator isolation.

Key invariant documented in seeding.py:
    ``np.random.seed`` (called by ``set_global_seed``) does NOT affect
    ``np.random.default_rng()`` generators — those must be seeded explicitly
    via ``seeded_rng(seed)`` and threaded through call sites.
"""

import numpy as np
import pytest

from trap.utils.seeding import seeded_rng, set_global_seed
from trap.utils.sequence import break_long_read

_LONG_SEQ = "ACGT" * 200  # 800 bp — long enough to produce multiple fragments


@pytest.mark.unit
class TestSeededRng:
    def test_returns_numpy_generator(self):
        rng = seeded_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_same_seed_same_draw(self):
        assert seeded_rng(1).integers(0, 1000) == seeded_rng(1).integers(0, 1000)

    def test_different_seeds_differ(self):
        assert seeded_rng(1).integers(0, 10_000) != seeded_rng(2).integers(0, 10_000)

    def test_independent_of_set_global_seed(self):
        """set_global_seed must not affect a separately seeded Generator."""
        rng_before = seeded_rng(7)
        draw_before = rng_before.integers(0, 10_000)

        set_global_seed(99999)  # mutate the legacy RNG state

        rng_after = seeded_rng(7)
        draw_after = rng_after.integers(0, 10_000)

        assert draw_before == draw_after, (
            "seeded_rng(7) produced different results before and after "
            "set_global_seed(99999) — Generator is not isolated"
        )


@pytest.mark.unit
class TestBreakLongReadDeterminism:
    def test_same_seed_same_output(self):
        out1 = break_long_read(_LONG_SEQ, rng=seeded_rng(42))
        out2 = break_long_read(_LONG_SEQ, rng=seeded_rng(42))
        assert out1 == out2

    def test_different_seeds_differ(self):
        out1 = break_long_read(_LONG_SEQ, rng=seeded_rng(1))
        out2 = break_long_read(_LONG_SEQ, rng=seeded_rng(2))
        assert out1 != out2

    def test_set_global_seed_does_not_affect_rng(self):
        """set_global_seed between two calls with seeded_rng must not change output."""
        out1 = break_long_read(_LONG_SEQ, rng=seeded_rng(42))
        set_global_seed(12345)
        out2 = break_long_read(_LONG_SEQ, rng=seeded_rng(42))
        assert out1 == out2
