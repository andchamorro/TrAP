"""Cross-stage standardization consistency (plan Step 1.7).

GenomeDataset._standardization and trap.utils.sequence.standardize must
agree on every input so that tokenizer-training sequences and
preprocessing/inference sequences pass through the same cleaning logic.
"""

import pytest

from trap.loaders.dataset import GenomeDataset
from trap.utils.sequence import standardize

_INPUTS = [
    "ACNGNTTGNA",
    "actgnnn",
    "NNNN",
    "",
    "ACGT",
    "ACTGnN",
    "acxtygrn",
    "AATTCCGGNN",
]


@pytest.mark.unit
class TestStandardizeConsistency:
    @pytest.mark.parametrize("seq", _INPUTS)
    def test_both_paths_agree(self, seq):
        assert GenomeDataset._standardization(seq) == standardize(seq), (
            f"Mismatch on {seq!r}: "
            f"dataset={GenomeDataset._standardization(seq)!r} "
            f"sequence={standardize(seq)!r}"
        )

    @pytest.mark.parametrize("seq", _INPUTS)
    def test_n_stripped_not_preserved(self, seq):
        assert "N" not in standardize(seq)
        assert "N" not in GenomeDataset._standardization(seq)

    @pytest.mark.parametrize("seq", _INPUTS)
    def test_vocab_chars_are_actg_only(self, seq):
        cleaned = standardize(seq)
        assert all(c in "ACTG" for c in cleaned), (
            f"unexpected char in cleaned {seq!r} → {cleaned!r}"
        )

    def test_lowercase_actg_uppercased(self):
        assert standardize("actg") == "ACTG"
        assert GenomeDataset._standardization("actg") == "ACTG"

    def test_empty_string_safe(self):
        assert standardize("") == ""
        assert GenomeDataset._standardization("") == ""
