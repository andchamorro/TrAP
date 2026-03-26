"""Tests for trap.utils.io — try_mkdir and genome_file_handle."""

import gzip
from pathlib import Path

import pytest

from trap.utils.io import genome_file_handle, try_mkdir


# -----------------------------------------------------------------------
# try_mkdir
# -----------------------------------------------------------------------

class TestTryMkdir:
    """Tests for the try_mkdir utility."""

    @pytest.mark.unit
    def test_creates_single_directory(self, tmp_path):
        target = tmp_path / "newdir"
        assert not target.exists()
        try_mkdir(target)
        assert target.is_dir()

    @pytest.mark.unit
    def test_creates_nested_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        try_mkdir(target)
        assert target.is_dir()

    @pytest.mark.unit
    def test_existing_directory_no_error(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        # Should not raise
        try_mkdir(target)
        assert target.is_dir()

    @pytest.mark.unit
    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "strdir")
        try_mkdir(target)
        assert Path(target).is_dir()

    @pytest.mark.unit
    def test_accepts_pathlib_path(self, tmp_path):
        target = tmp_path / "pldir"
        try_mkdir(target)
        assert target.is_dir()


# -----------------------------------------------------------------------
# genome_file_handle
# -----------------------------------------------------------------------

class TestGenomeFileHandle:
    """Tests for the genome_file_handle opener."""

    @pytest.mark.unit
    def test_opens_plain_text_file(self, tmp_path):
        p = tmp_path / "test.fa"
        p.write_text("ACTG\n")
        with genome_file_handle(p) as fh:
            content = fh.read()
        assert content == "ACTG\n"

    @pytest.mark.unit
    def test_opens_gzip_file(self, tmp_path):
        p = tmp_path / "test.fa.gz"
        with gzip.open(p, "wt") as fh:
            fh.write("ACTG\n")
        with genome_file_handle(p) as fh:
            content = fh.read()
        assert content == "ACTG\n"

    @pytest.mark.unit
    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        with genome_file_handle(str(p)) as fh:
            content = fh.read()
        assert content == "hello"

    @pytest.mark.unit
    def test_accepts_pathlib_path(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("world")
        with genome_file_handle(p) as fh:
            content = fh.read()
        assert content == "world"

    @pytest.mark.unit
    def test_returns_text_mode(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("ACTG")
        with genome_file_handle(p) as fh:
            assert isinstance(fh.read(), str)

    @pytest.mark.unit
    def test_plain_fasta_extension(self, tmp_path):
        p = tmp_path / "genome.fasta"
        p.write_text(">seq1\nACTG\n")
        with genome_file_handle(p) as fh:
            assert ">seq1" in fh.read()

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        p = tmp_path / "nonexistent.fa"
        with pytest.raises(FileNotFoundError):
            genome_file_handle(p)
