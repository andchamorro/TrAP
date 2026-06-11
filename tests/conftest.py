"""Shared fixtures for the TrAP test suite."""

import gzip
from pathlib import Path
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Platform guard: skip tests that need the transformers modeling stack
# ---------------------------------------------------------------------------
# transformers 5.x requires torch>=2.5 (e.g. torch.distributed.tensor, and on
# 5.10+ torch.float8_e8m0fnu which needs torch>=2.7). The newest torch
# installable on Intel macOS is 2.4 (PyTorch dropped x86-64 macOS builds and
# conda-forge caps there), so importing the modeling stack raises at runtime on
# the dev machine. Those tests run on Grace, where torch>=2.7 is present.
#
# This hook converts a failure/error into a skip ONLY when its traceback carries
# one of the modeling-import signatures below — so genuine failures stay failed,
# newly-added modeling tests are covered automatically, and on Grace (where the
# import succeeds) the hook never fires.
_MODELING_IMPORT_SIGNATURES = (
    "float8_e8m0fnu",
    "torch.distributed.tensor",
    "Could not import module 'AlbertForMaskedLM'",
    "Could not import module 'AlbertForSequenceClassification'",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when in ("setup", "call") and report.failed and call.excinfo is not None:
        text = str(call.excinfo.getrepr(style="short"))
        if any(sig in text for sig in _MODELING_IMPORT_SIGNATURES):
            report.outcome = "skipped"
            report.longrepr = (
                str(item.fspath),
                item.location[1] or 0,
                "Skipped: transformers modeling stack requires torch>=2.5 "
                "(unavailable on this platform; runs on Grace).",
            )


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

FASTA_CONTENT = textwrap.dedent(
    """\
    >seq1 first sequence
    ACTGACTGACTG
    >seq2 second sequence
    GGGGCCCCTTTTAAAA
    >seq3 with ambiguity
    ACTGNNNACTG
"""
)

FASTQ_CONTENT = textwrap.dedent(
    """\
    @read1
    ACTGACTGACTG
    +
    IIIIIIIIIIII
    @read2
    GGGGCCCCTTTTAAAA
    +
    IIIIIIIIIIIIIIII
"""
)


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
