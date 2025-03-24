import re
import gzip
from pathlib import Path
from loguru import logger
from Bio import bgzf, SeqIO

from torch.utils.data import Dataset

from translast.config.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
class GenomeDataset(Dataset):

    def __init__(self, file_path, file_format, transform=None, target_transform=None):
        self.file_path = file_path
        self.file_format = file_format
        self.transform = transform
        self.target_transform = target_transform
        self.sequences, self.complement, self.ids = self._load_sequences()
        self._index = 0  # Initialize the index for iteration

    def _load_sequences(self):
        sequences = []
        complement = []
        ids = []
        with self._file_handle() as handle:
            for record in SeqIO.parse(handle, self.file_format):
                sequences.append(self._standardization(str(record.seq)))
                complement.append(self._standardization(str(record.seq.reverse_complement())))
                ids.append(str(record.id))
                pass
        return sequences, complement, ids
    
    def _standardization(self, sequence):
        return re.sub(r'[^ACTG]', '', sequence.upper())
    
    def _file_handle(self):
        if self.file_path.endswith('.gz'):
            return gzip.open(self.file_path, 'rt')
        elif self.file_path.endswith('.bgz'):
            return bgzf.open(self.file_path, 'rt')
        else :
            return open(self.file_path, 'rt')
    
    def __len__(self):
        return len(self.sequences)
    
    def __iter__(self):
        self._index = 0  # Reset the index for a new iteration
        return self
    
    def __next__(self):
        if self._index < len(self.sequences):
            seq = self.sequences[self._index]
            rev = self.complement[self._index]
            self._index += 1
            return seq, rev
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
            if self.transform:
                seq = self.transform(seq)
                rev = self.transform(rev)
            if self.target_transform:
                index = self.target_transform(index)
            return seq, rev, index
        else:
            raise TypeError("Invalid argument type.")
