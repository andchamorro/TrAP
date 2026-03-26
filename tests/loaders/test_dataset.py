"""Tests for trap.loaders.dataset — GenomeDataset."""

import gzip

import pytest

from trap.loaders.dataset import GenomeDataset


# -----------------------------------------------------------------------
# Construction and loading
# -----------------------------------------------------------------------

class TestGenomeDatasetInit:
    """Test GenomeDataset construction from FASTA files."""

    @pytest.mark.integration
    def test_loads_fasta(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        assert len(ds) == 3

    @pytest.mark.integration
    def test_loads_gzip_fasta(self, fasta_gz_path):
        ds = GenomeDataset(fasta_gz_path, "fasta")
        assert len(ds) == 3

    @pytest.mark.integration
    def test_loads_fastq(self, fastq_path):
        ds = GenomeDataset(fastq_path, "fastq")
        assert len(ds) == 2

    @pytest.mark.integration
    def test_empty_file(self, empty_fasta_path):
        ds = GenomeDataset(empty_fasta_path, "fasta")
        assert len(ds) == 0

    @pytest.mark.integration
    def test_accepts_string_path(self, fasta_path):
        ds = GenomeDataset(str(fasta_path), "fasta")
        assert len(ds) == 3

    @pytest.mark.integration
    def test_stores_ids(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        assert ds.ids[0] == "seq1"
        assert ds.ids[1] == "seq2"
        assert ds.ids[2] == "seq3"


# -----------------------------------------------------------------------
# Standardization
# -----------------------------------------------------------------------

class TestStandardization:
    """Test DNA sequence cleaning."""

    @pytest.mark.unit
    def test_default_standardization_uppercases(self):
        result = GenomeDataset._standardization("actg")
        assert result == "ACTG"

    @pytest.mark.unit
    def test_default_standardization_keeps_n(self):
        result = GenomeDataset._standardization("ACTGnNn")
        # Lowercase 'n' becomes uppercase 'N' (3 total N chars)
        assert result == "ACTGNNN"

    @pytest.mark.unit
    def test_default_standardization_strips_non_actgn(self):
        result = GenomeDataset._standardization("ACXTYGRN")
        assert result == "ACTGN"

    @pytest.mark.integration
    def test_loaded_sequences_are_clean(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        for seq in ds.sequences:
            assert all(c in "ACTGN" for c in seq)

    @pytest.mark.integration
    def test_custom_standardization(self, fasta_path):
        ds = GenomeDataset(
            fasta_path,
            "fasta",
            standardization=lambda s: s.upper().replace("N", ""),
        )
        for seq in ds.sequences:
            assert "N" not in seq

    @pytest.mark.integration
    def test_complement_sequences_are_loaded(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        assert len(ds.complement) == len(ds.sequences)
        for comp in ds.complement:
            assert all(c in "ACTGN" for c in comp)


# -----------------------------------------------------------------------
# __len__, __iter__, __next__
# -----------------------------------------------------------------------

class TestGenomeDatasetIteration:
    """Test iteration protocol."""

    @pytest.mark.integration
    def test_len(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        assert len(ds) == 3

    @pytest.mark.integration
    def test_iter_yields_tuples(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        items = list(ds)
        assert len(items) == 3
        for seq, rev, id_ in items:
            assert isinstance(seq, str)
            assert isinstance(rev, str)
            assert isinstance(id_, str)

    @pytest.mark.integration
    def test_iter_resets(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        first_pass = list(ds)
        second_pass = list(ds)
        assert first_pass == second_pass

    @pytest.mark.integration
    def test_stopiteration(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        it = iter(ds)
        for _ in range(len(ds)):
            next(it)
        with pytest.raises(StopIteration):
            next(it)


# -----------------------------------------------------------------------
# __getitem__
# -----------------------------------------------------------------------

class TestGenomeDatasetGetItem:
    """Test index-based access."""

    @pytest.mark.integration
    def test_getitem_positive_index(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        seq, rev, id_, idx = ds[0]
        assert id_ == "seq1"
        assert idx == 0

    @pytest.mark.integration
    def test_getitem_negative_index(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        seq, rev, id_, idx = ds[-1]
        assert id_ == "seq3"

    @pytest.mark.integration
    def test_getitem_out_of_range(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        with pytest.raises(IndexError, match="out of range"):
            ds[100]

    @pytest.mark.integration
    def test_getitem_negative_out_of_range(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        with pytest.raises(IndexError, match="out of range"):
            ds[-100]

    @pytest.mark.integration
    def test_getitem_invalid_type(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        with pytest.raises(TypeError, match="Invalid argument type"):
            ds["invalid"]

    @pytest.mark.integration
    def test_getitem_slice(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        seqs, comps = ds[0:2]
        assert len(seqs) == 2
        assert len(comps) == 2

    @pytest.mark.integration
    def test_transform_applied(self, fasta_path):
        def upper_transform(seq, rev, id_):
            return seq.lower(), rev.lower(), id_.upper()

        ds = GenomeDataset(fasta_path, "fasta", transform=upper_transform)
        seq, rev, id_, idx = ds[0]
        assert seq == seq.lower()
        assert id_ == id_.upper()

    @pytest.mark.integration
    def test_target_transform_applied(self, fasta_path):
        ds = GenomeDataset(
            fasta_path, "fasta", target_transform=lambda i: i * 10
        )
        seq, rev, id_, idx = ds[1]
        assert idx == 10
