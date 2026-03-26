"""Shared fixtures for the TrAP test suite."""

import gzip
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# DNA sequence fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_dna_seq():
    """A short, clean DNA sequence (20 bp)."""
    return "ACTGACTGACTGACTGACTG"


@pytest.fixture
def dna_seq_with_n():
    """A DNA sequence containing ambiguous N bases."""
    return "ACTGNNNACTG"


@pytest.fixture
def long_dna_seq():
    """A longer DNA sequence (200 bp) for batch/performance tests."""
    unit = "ACTGACTGAC"
    return unit * 20


# ---------------------------------------------------------------------------
# FASTA / FASTQ file fixtures
# ---------------------------------------------------------------------------

FASTA_CONTENT = textwrap.dedent("""\
    >seq1 first sequence
    ACTGACTGACTG
    >seq2 second sequence
    GGGGCCCCTTTTAAAA
    >seq3 with ambiguity
    ACTGNNNACTG
""")

FASTQ_CONTENT = textwrap.dedent("""\
    @read1
    ACTGACTGACTG
    +
    IIIIIIIIIIII
    @read2
    GGGGCCCCTTTTAAAA
    +
    IIIIIIIIIIIIIIII
""")


@pytest.fixture
def fasta_path(tmp_path):
    """Write a small FASTA file and return its path."""
    p = tmp_path / "test.fasta"
    p.write_text(FASTA_CONTENT)
    return p


@pytest.fixture
def fasta_gz_path(tmp_path):
    """Write a gzip-compressed FASTA file and return its path."""
    p = tmp_path / "test.fasta.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(FASTA_CONTENT)
    return p


@pytest.fixture
def fastq_path(tmp_path):
    """Write a small FASTQ file and return its path."""
    p = tmp_path / "test.fastq"
    p.write_text(FASTQ_CONTENT)
    return p


@pytest.fixture
def empty_fasta_path(tmp_path):
    """Write an empty FASTA file and return its path."""
    p = tmp_path / "empty.fasta"
    p.write_text("")
    return p
