"""Paired FASTQ reader with dnaio acceleration.

Tries ``dnaio`` first (10–20× faster than BioPython); falls back to
``Bio.SeqIO`` when dnaio is not installed.

Typical usage
-------------
from trap.utils.fastq import paired_fastq_iter

for read_id, r1_seq, r2_seq in paired_fastq_iter(r1_path, r2_path):
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple


def paired_fastq_iter(
    r1_path: Path,
    r2_path: Path,
) -> Generator[Tuple[str, str, str], None, None]:
    """Iterate over paired FASTQ files, yielding (read_id, r1_seq, r2_seq).

    The read ID is the name of the R1 record with any trailing ``/1`` or
    ``/2`` suffix stripped so paired records share a common ID.

    Args:
        r1_path: Path to the R1 (forward) FASTQ file, optionally gzip/bgz
            compressed.
        r2_path: Path to the R2 (reverse) FASTQ file.

    Yields:
        Tuples of ``(read_id, r1_sequence, r2_sequence)`` where sequences
        are upper-case ASCII strings.
    """
    import re

    _strip_suffix = re.compile(r"/[12]$")

    try:
        import dnaio

        with dnaio.open(str(r1_path)) as f1, dnaio.open(str(r2_path)) as f2:
            for r1, r2 in zip(f1, f2):
                read_id = _strip_suffix.sub("", r1.name.split()[0])
                yield read_id, r1.sequence.upper(), r2.sequence.upper()

    except ImportError:
        from Bio import SeqIO

        from trap.utils.io import genome_file_handle

        with genome_file_handle(r1_path) as h1, genome_file_handle(r2_path) as h2:
            for r1, r2 in zip(SeqIO.parse(h1, "fastq"), SeqIO.parse(h2, "fastq")):
                # BioPython's .id is already the first whitespace token; strip the
                # /1,/2 mate suffix so IDs match the dnaio branch exactly.
                read_id = _strip_suffix.sub("", str(r1.id))
                yield read_id, str(r1.seq).upper(), str(r2.seq).upper()
