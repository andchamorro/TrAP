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
    def test_default_standardization_strips_n(self):
        # N residues are stripped per the k=17/v48 vocabulary decision
        # (2026-05-28): pure ACTG vocabulary, no N token.
        result = GenomeDataset._standardization("ACTGnNn")
        assert result == "ACTG"

    @pytest.mark.unit
    def test_default_standardization_strips_non_actg(self):
        result = GenomeDataset._standardization("ACXTYGRN")
        assert result == "ACTG"

    @pytest.mark.integration
    def test_loaded_sequences_are_clean(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        for seq in ds.sequences:
            assert all(c in "ACTG" for c in seq)

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
    def test_reverse_complement_not_precomputed(self, fasta_path):
        # The reverse complement is no longer eagerly stored (computed on demand
        # by trap.utils.sequence.reverse_complement where needed).
        ds = GenomeDataset(fasta_path, "fasta")
        assert not hasattr(ds, "complement")

    @pytest.mark.integration
    def test_n_density_tracked(self, tmp_path):
        # Eager mode: stats available immediately after construction.
        clean = tmp_path / "clean.fa"
        clean.write_text(">s1\nACTGACTG\n")
        ds = GenomeDataset(clean, "fasta")
        assert ds.stripped_bases == 0
        assert ds.total_bases == 8

        heavy = tmp_path / "heavy.fa"
        heavy.write_text(">s1\nACTGNNNN\n")
        ds2 = GenomeDataset(heavy, "fasta")
        assert ds2.stripped_bases == 4
        assert ds2.total_bases == 8

    @pytest.mark.integration
    def test_n_density_tracked_lazy(self, tmp_path):
        # Lazy mode: stats are accumulated during iteration, not at construction.
        heavy = tmp_path / "heavy.fa"
        heavy.write_text(">s1\nACTGNNNN\n")
        ds = GenomeDataset(heavy, "fasta", lazy=True)
        assert ds.stripped_bases == 0  # not yet loaded
        list(ds)  # consume the iterator
        assert ds.stripped_bases == 4
        assert ds.total_bases == 8


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
        for seq, id_ in items:
            assert isinstance(seq, str)
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
        seq, id_, idx = ds[0]
        assert id_ == "seq1"
        assert idx == 0

    @pytest.mark.integration
    def test_getitem_negative_index(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta")
        seq, id_, idx = ds[-1]
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
        seqs = ds[0:2]
        assert len(seqs) == 2

    @pytest.mark.integration
    def test_transform_applied(self, fasta_path):
        def upper_transform(seq, id_):
            return seq.lower(), id_.upper()

        ds = GenomeDataset(fasta_path, "fasta", transform=upper_transform)
        seq, id_, idx = ds[0]
        assert seq == seq.lower()
        assert id_ == id_.upper()

    @pytest.mark.integration
    def test_target_transform_applied(self, fasta_path):
        ds = GenomeDataset(
            fasta_path, "fasta", target_transform=lambda i: i * 10
        )
        seq, id_, idx = ds[1]
        assert idx == 10

    @pytest.mark.integration
    def test_unary_transform_raises_type_error(self, fasta_path):
        """A unary lambda (old API) must raise TypeError on first __getitem__."""
        ds = GenomeDataset(fasta_path, "fasta", transform=lambda s: s)
        with pytest.raises(TypeError):
            _ = ds[0]


# -----------------------------------------------------------------------
# Lazy streaming mode
# -----------------------------------------------------------------------

class TestLazyMode:
    """GenomeDataset(lazy=True) streams from disk; no RAM pre-load."""

    @pytest.mark.unit
    def test_lazy_len_raises(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta", lazy=True)
        with pytest.raises(TypeError, match="lazy"):
            len(ds)

    @pytest.mark.unit
    def test_lazy_getitem_raises(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta", lazy=True)
        with pytest.raises(TypeError, match="lazy"):
            _ = ds[0]

    @pytest.mark.integration
    def test_lazy_iter_yields_same_sequences(self, fasta_path):
        eager = GenomeDataset(fasta_path, "fasta", lazy=False)
        lazy  = GenomeDataset(fasta_path, "fasta", lazy=True)
        assert list(eager) == list(lazy)

    @pytest.mark.integration
    def test_lazy_iter_is_reentrant(self, fasta_path):
        """Each iter() call opens a fresh file handle."""
        ds = GenomeDataset(fasta_path, "fasta", lazy=True)
        first  = list(ds)
        second = list(ds)
        assert first == second

    @pytest.mark.integration
    def test_lazy_sequences_not_loaded(self, fasta_path):
        ds = GenomeDataset(fasta_path, "fasta", lazy=True)
        assert ds.sequences is None
