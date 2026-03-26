"""Integration tests verifying cross-module imports and refactoring integrity.

These tests prove that the consolidation refactoring (removing duplicated code
from predict.py, train.py, postprocessing.py, preprocessing_sequences.py,
spm2hugging.py, tokenizer.py and centralizing into utils/io.py, utils/kmer.py,
loaders/dataset.py) left all modules importable and their public APIs intact.
"""

import pytest


# -----------------------------------------------------------------------
# Package-level imports
# -----------------------------------------------------------------------

class TestPackageImports:
    """Verify all packages and subpackages are importable."""

    @pytest.mark.unit
    def test_import_trap(self):
        import trap  # noqa: F401

    @pytest.mark.unit
    def test_import_trap_config(self):
        import trap.config  # noqa: F401
        import trap.config.config  # noqa: F401

    @pytest.mark.unit
    def test_import_trap_loaders(self):
        import trap.loaders  # noqa: F401

    @pytest.mark.unit
    def test_import_trap_modeling(self):
        import trap.modeling  # noqa: F401

    @pytest.mark.unit
    def test_import_trap_utils(self):
        import trap.utils  # noqa: F401


# -----------------------------------------------------------------------
# Consolidated utility modules
# -----------------------------------------------------------------------

class TestConsolidatedImports:
    """Verify the canonical locations export the right symbols."""

    @pytest.mark.unit
    def test_io_exports_try_mkdir(self):
        from trap.utils.io import try_mkdir
        assert callable(try_mkdir)

    @pytest.mark.unit
    def test_io_exports_genome_file_handle(self):
        from trap.utils.io import genome_file_handle
        assert callable(genome_file_handle)

    @pytest.mark.unit
    def test_kmer_exports_kmer_split(self):
        from trap.utils.kmer import kmer_split
        assert callable(kmer_split)

    @pytest.mark.unit
    def test_kmer_exports_kmer_split_batch(self):
        from trap.utils.kmer import kmer_split_batch
        assert callable(kmer_split_batch)

    @pytest.mark.unit
    def test_kmer_exports_seq_to_encoded(self):
        from trap.utils.kmer import seq_to_encoded
        assert callable(seq_to_encoded)

    @pytest.mark.unit
    def test_kmer_exports_encoded_to_seq(self):
        from trap.utils.kmer import encoded_to_seq
        assert callable(encoded_to_seq)

    @pytest.mark.unit
    def test_dataset_exports_GenomeDataset(self):
        from trap.loaders.dataset import GenomeDataset
        assert isinstance(GenomeDataset, type)


# -----------------------------------------------------------------------
# Consumer modules import from consolidated locations
# -----------------------------------------------------------------------

class TestConsumerModuleImports:
    """Verify modules that were refactored still import successfully.

    Each of these modules previously had duplicated code that was removed
    in favour of imports from the canonical utility modules.
    """

    @pytest.mark.integration
    def test_import_modeling_train(self):
        from trap.modeling import train  # noqa: F401

    @pytest.mark.integration
    def test_import_modeling_predict(self):
        from trap.modeling import predict  # noqa: F401

    @pytest.mark.integration
    def test_import_modeling_postprocessing(self):
        from trap.modeling import postprocessing  # noqa: F401

    @pytest.mark.integration
    def test_import_loaders_tokenizer(self):
        from trap.loaders import tokenizer  # noqa: F401

    @pytest.mark.integration
    def test_import_utils_preprocessing(self):
        from trap.utils import preprocessing_sequences  # noqa: F401

    @pytest.mark.integration
    def test_import_utils_spm2hugging(self):
        from trap.utils import spm2hugging  # noqa: F401

    @pytest.mark.integration
    def test_import_utils_dna2bit(self):
        from trap.utils import dna2bit  # noqa: F401


# -----------------------------------------------------------------------
# No duplicate definitions remain
# -----------------------------------------------------------------------

class TestNoDuplicateDefinitions:
    """Verify duplicated code was actually removed from consumer modules."""

    @pytest.mark.unit
    def test_train_has_no_local_try_mkdir(self):
        import inspect
        from trap.modeling import train
        source = inspect.getsource(train)
        # The function definition should NOT exist in train.py
        assert "def try_mkdir" not in source

    @pytest.mark.unit
    def test_predict_has_no_local_genome_file_handle(self):
        import inspect
        from trap.modeling import predict
        source = inspect.getsource(predict)
        assert "def genome_file_handle" not in source

    @pytest.mark.unit
    def test_predict_has_no_local_GenomeDataset(self):
        import inspect
        from trap.modeling import predict
        source = inspect.getsource(predict)
        assert "class GenomeDataset" not in source

    @pytest.mark.unit
    def test_predict_has_no_local_kmer_split_def(self):
        import inspect
        from trap.modeling import predict
        source = inspect.getsource(predict)
        assert "def kmer_split" not in source

    @pytest.mark.unit
    def test_preprocessing_has_no_local_try_mkdir(self):
        import inspect
        from trap.utils import preprocessing_sequences
        source = inspect.getsource(preprocessing_sequences)
        assert "def try_mkdir" not in source

    @pytest.mark.unit
    def test_preprocessing_has_no_local_GenomeDataset(self):
        import inspect
        from trap.utils import preprocessing_sequences
        source = inspect.getsource(preprocessing_sequences)
        assert "class GenomeDataset" not in source

    @pytest.mark.unit
    def test_preprocessing_has_no_local_seq_to_encoded(self):
        import inspect
        from trap.utils import preprocessing_sequences
        source = inspect.getsource(preprocessing_sequences)
        assert "def seq_to_encoded" not in source

    @pytest.mark.unit
    def test_spm2hugging_has_no_local_try_mkdir(self):
        import inspect
        from trap.utils import spm2hugging
        source = inspect.getsource(spm2hugging)
        assert "def try_mkdir" not in source

    @pytest.mark.unit
    def test_postprocessing_has_no_local_genome_file_handle(self):
        import inspect
        from trap.modeling import postprocessing
        source = inspect.getsource(postprocessing)
        assert "def genome_file_handle" not in source

    @pytest.mark.unit
    def test_postprocessing_has_no_local_try_mkdir(self):
        import inspect
        from trap.modeling import postprocessing
        source = inspect.getsource(postprocessing)
        assert "def try_mkdir" not in source

    @pytest.mark.unit
    def test_tokenizer_has_no_local_seq_to_encoded(self):
        import inspect
        from trap.loaders import tokenizer
        source = inspect.getsource(tokenizer)
        assert "def seq_to_encoded" not in source

    @pytest.mark.unit
    def test_tokenizer_has_no_local_kmer_split_def(self):
        import inspect
        from trap.loaders import tokenizer
        source = inspect.getsource(tokenizer)
        assert "def _kmer_split" not in source


# -----------------------------------------------------------------------
# Cross-module functional smoke tests
# -----------------------------------------------------------------------

class TestCrossModuleSmoke:
    """Quick functional tests exercising the refactored import paths."""

    @pytest.mark.integration
    def test_kmer_split_used_from_canonical_location(self):
        from trap.utils.kmer import kmer_split
        result = kmer_split(3, "ACTGACTG")
        assert len(result.split()) == 6

    @pytest.mark.integration
    def test_try_mkdir_from_canonical_location(self, tmp_path):
        from trap.utils.io import try_mkdir
        target = tmp_path / "smoke_test"
        try_mkdir(target)
        assert target.is_dir()

    @pytest.mark.integration
    def test_genome_dataset_uses_consolidated_file_handle(self, fasta_path):
        from trap.loaders.dataset import GenomeDataset
        ds = GenomeDataset(fasta_path, "fasta")
        assert len(ds) > 0

    @pytest.mark.integration
    def test_postprocessing_sanitize_paths(self):
        from trap.modeling.postprocessing import sanitize_paths
        paths = sanitize_paths("/tmp/test", None)
        from pathlib import Path
        assert paths[0] == Path("/tmp/test")
        assert paths[1] is None
