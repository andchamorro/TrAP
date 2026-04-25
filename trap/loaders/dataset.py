import re
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union

from Bio import SeqIO
from loguru import logger
from torch.utils.data import Dataset

from trap.utils.io import genome_file_handle

# Warn when more than this fraction of bases are non-ACGT (stripped). GENCODE
# transcripts and ART reads are ~0% N; a high fraction means k-mers are bridging
# deleted N runs (chimeric k-mers) and the input should be checked.
_N_DENSITY_WARN = 0.001


class GenomeDataset(Dataset):
    """A PyTorch Dataset for loading genome sequences from FASTA/FASTQ files.

    Two modes controlled by ``lazy``:

    * ``lazy=False`` (default): loads all sequences into RAM on construction.
      Fast random access via ``__getitem__``. Suitable for small-to-medium
      files (e.g. GENCODE transcripts, ~250K sequences).

    * ``lazy=True``: stores only the file path; sequences are streamed from
      disk on each iteration. No RAM overhead for the sequence strings, so
      safe for large classification FASTQs (14M+ reads). ``__getitem__`` and
      ``__len__`` are not available in this mode; only iteration is supported.

    The reverse complement is **not** precomputed in either mode. Consumers
    that need it (single-FASTQ inference) call
    :func:`trap.utils.sequence.reverse_complement` on demand.

    Args:
        file_path: Path to the genome file (supports .gz, .bgz, plain text).
        file_format: BioPython SeqIO format string (``"fasta"`` or ``"fastq"``).
        transform: Optional callable applied to ``(seq, id)`` at access time.
        target_transform: Optional callable applied to the index.
        standardization: Sequence cleaning function. Defaults to stripping
            non-ACTG characters and uppercasing.
        lazy: If ``True``, stream from disk on each iteration instead of
            pre-loading all sequences into RAM.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        file_format: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        standardization: Optional[Callable] = None,
        lazy: bool = False,
    ):
        self.file_path = Path(file_path)
        self.file_format = file_format
        self.transform = transform
        self.target_transform = target_transform
        self.standardization = (
            standardization if standardization is not None else self._standardization
        )
        self.lazy = lazy
        self.total_bases = 0
        self.stripped_bases = 0

        if lazy:
            self.sequences = None
            self.ids = None
        else:
            self.sequences, self.ids = self._load_sequences()
        self._index = 0

    # ------------------------------------------------------------------
    # Eager mode helpers
    # ------------------------------------------------------------------

    def _load_sequences(self):
        sequences = []
        ids = []
        total_bases = 0
        stripped_bases = 0
        with genome_file_handle(self.file_path) as handle:
            for record in SeqIO.parse(handle, self.file_format):
                raw = str(record.seq)
                clean = self.standardization(raw)
                sequences.append(clean)
                ids.append(str(record.id))
                total_bases += len(raw)
                stripped_bases += len(raw) - len(clean)
        self.total_bases = total_bases
        self.stripped_bases = stripped_bases
        self._log_n_density(stripped_bases, total_bases)
        return sequences, ids

    def _log_n_density(self, stripped: int, total: int) -> None:
        if total and stripped:
            frac = stripped / total
            msg = (
                f"GenomeDataset {self.file_path.name}: stripped {stripped:,}/"
                f"{total:,} non-ACGT bases ({100 * frac:.3f}%)"
            )
            if frac > _N_DENSITY_WARN:
                logger.warning(
                    f"{msg} — k-mers spanning these gaps are chimeric; verify input N density."
                )
            else:
                logger.info(msg)

    # ------------------------------------------------------------------
    # Lazy streaming helper
    # ------------------------------------------------------------------

    def _stream(self) -> Iterator[Tuple[str, str]]:
        """Open the file and yield ``(clean_seq, read_id)`` one record at a time."""
        total = stripped = 0
        with genome_file_handle(self.file_path) as handle:
            for record in SeqIO.parse(handle, self.file_format):
                raw = str(record.seq)
                clean = self.standardization(raw)
                total += len(raw)
                stripped += len(raw) - len(clean)
                yield clean, str(record.id)
        # Accumulate across multiple iterations (e.g., once for dedup, once for generator)
        self.total_bases += total
        self.stripped_bases += stripped
        self._log_n_density(stripped, total)

    # ------------------------------------------------------------------
    # Shared public API
    # ------------------------------------------------------------------

    @staticmethod
    def _standardization(sequence: str) -> str:
        """Remove non-ACTG characters (including N) and uppercase the sequence."""
        return re.sub(r'[^ACTG]', '', sequence.upper())

    def __len__(self):
        if self.lazy:
            raise TypeError(
                "GenomeDataset(lazy=True) does not support len(). "
                "Use iteration only, or construct with lazy=False."
            )
        return len(self.sequences)

    def __iter__(self):
        if self.lazy:
            return self._stream()
        self._index = 0
        return self

    def __next__(self):
        # Only reached in eager mode (lazy mode returns a generator from __iter__)
        if self._index < len(self.sequences):
            seq = self.sequences[self._index]
            id_ = self.ids[self._index]
            self._index += 1
            return seq, id_
        raise StopIteration

    def __getitem__(self, index):
        if self.lazy:
            raise TypeError(
                "GenomeDataset(lazy=True) does not support indexing. "
                "Use iteration only, or construct with lazy=False."
            )
        if isinstance(index, slice):
            return self.sequences[index]
        elif isinstance(index, int):
            if index < 0:
                index += len(self.sequences)
            if index >= len(self.sequences) or index < 0:
                raise IndexError("The index is out of range.")
            seq = self.sequences[index]
            id_ = self.ids[index]
            if self.transform:
                seq, id_ = self.transform(seq, id_)
            if self.target_transform:
                index = self.target_transform(index)
            return seq, id_, index
        else:
            raise TypeError("Invalid argument type.")
