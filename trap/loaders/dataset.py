import re
from pathlib import Path
from typing import Callable, Optional, Union

from Bio import SeqIO
from torch.utils.data import Dataset

from trap.utils.io import genome_file_handle


class GenomeDataset(Dataset):
    """A PyTorch Dataset for loading genome sequences from FASTA/FASTQ files.

    Loads forward sequences and their reverse complements, with optional
    transforms applied at access time.

    Args:
        file_path: Path to the genome file (supports .gz, .bgz, and plain text).
        file_format: BioPython SeqIO format string (e.g. "fasta", "fastq").
        transform: Optional callable applied to (seq, complement, id) tuples.
        target_transform: Optional callable applied to the index.
        standardization: Optional callable to clean sequences. Defaults to
            stripping non-ACTGN characters and uppercasing.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        file_format: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        standardization: Optional[Callable] = None,
    ):
        self.file_path = Path(file_path)
        self.file_format = file_format
        self.transform = transform
        self.target_transform = target_transform
        self.standardization = (
            standardization if standardization is not None else self._standardization
        )
        self.sequences, self.complement, self.ids = self._load_sequences()
        self._index = 0

    def _load_sequences(self):
        sequences = []
        complement = []
        ids = []
        with genome_file_handle(self.file_path) as handle:
            for record in SeqIO.parse(handle, self.file_format):
                sequences.append(self.standardization(str(record.seq)))
                complement.append(
                    self.standardization(str(record.seq.reverse_complement()))
                )
                ids.append(str(record.id))
        return sequences, complement, ids

    @staticmethod
    def _standardization(sequence: str) -> str:
        """Remove non-ACTGN characters and uppercase the sequence."""
        return re.sub(r'[^ACTGN]', '', sequence.upper())

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.sequences):
            seq = self.sequences[self._index]
            rev = self.complement[self._index]
            id_ = self.ids[self._index]
            self._index += 1
            return seq, rev, id_
        else:
            raise StopIteration

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.sequences[index], self.complement[index]
        elif isinstance(index, int):
            if index < 0:
                index += len(self.sequences)
            if index >= len(self.sequences) or index < 0:
                raise IndexError("The index is out of range.")
            seq = self.sequences[index]
            rev = self.complement[index]
            id_ = self.ids[index]
            if self.transform:
                seq, rev, id_ = self.transform(seq, rev, id_)
            if self.target_transform:
                index = self.target_transform(index)
            return seq, rev, id_, index
        else:
            raise TypeError("Invalid argument type.")
