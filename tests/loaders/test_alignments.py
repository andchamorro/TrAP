"""Tests for trap.loaders.alignments — SAMDataset."""

from pathlib import Path

import pytest

pysam = pytest.importorskip("pysam", reason="pysam not installed")

from trap.loaders.alignments import SAMDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAM_FIXTURE = Path(__file__).parent.parent / "fixtures" / "minimal.sam"
_EXPECTED_IDS = ["read1", "read2", "read3", "read4", "read5"]
_EXPECTED_SEQS = [
    "ACTGACTGACTG",
    "CCCCGGGGTTTT",
    "AAAATTTTCCCC",
    "ACTGACTGTTTT",
    "TTTTTTTTTTTT",
]


@pytest.fixture
def sam_ds():
    return SAMDataset(SAM_FIXTURE)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSAMDatasetInit:
    @pytest.mark.integration
    def test_len(self, sam_ds):
        assert len(sam_ds) == 5

    @pytest.mark.integration
    def test_ids_ordered(self, sam_ds):
        ids = [rid for rid, _ in sam_ds]
        assert ids == _EXPECTED_IDS

    @pytest.mark.integration
    def test_sequences_correct(self, sam_ds):
        seqs = [seq for _, seq in sam_ds]
        assert seqs == _EXPECTED_SEQS


# ---------------------------------------------------------------------------
# __getitem__ — index isolation after partial iteration
# ---------------------------------------------------------------------------


class TestSAMDatasetGetItem:
    @pytest.mark.integration
    def test_getitem_positive_index(self, sam_ds):
        rid, seq = sam_ds[0]
        assert rid == "read1"
        assert seq == "ACTGACTGACTG"

    @pytest.mark.integration
    def test_getitem_after_partial_iteration(self, sam_ds):
        """Direct __getitem__ must not be affected by a previous partial iteration."""
        it = iter(sam_ds)
        next(it)
        next(it)
        # index 0 must still return the first read
        rid, seq = sam_ds[0]
        assert rid == "read1"

    @pytest.mark.integration
    def test_getitem_negative_index(self, sam_ds):
        rid, seq = sam_ds[-1]
        assert rid == "read5"
        assert seq == "TTTTTTTTTTTT"

    @pytest.mark.integration
    def test_getitem_last_via_negative(self, sam_ds):
        assert sam_ds[-1] == sam_ds[len(sam_ds) - 1]

    @pytest.mark.integration
    def test_getitem_oob_raises(self, sam_ds):
        with pytest.raises(IndexError):
            _ = sam_ds[1000]

    @pytest.mark.integration
    def test_getitem_negative_oob_raises(self, sam_ds):
        with pytest.raises(IndexError):
            _ = sam_ds[-1000]

    @pytest.mark.integration
    def test_getitem_invalid_type_raises(self, sam_ds):
        with pytest.raises(TypeError):
            _ = sam_ds["read1"]

    @pytest.mark.integration
    def test_getitem_slice(self, sam_ds):
        keys, seqs = sam_ds[1:3]
        assert keys == ["read2", "read3"]
        assert seqs == ["CCCCGGGGTTTT", "AAAATTTTCCCC"]
