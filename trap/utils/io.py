import os
import gzip
from pathlib import Path
from typing import Union

from Bio import bgzf


def try_mkdir(dir_name: Union[str, Path]) -> None:
    """Create directory and parents if they don't exist.

    Args:
        dir_name: Path to the directory to create.
    """
    os.makedirs(dir_name, exist_ok=True)


def genome_file_handle(file_path: Union[str, Path]):
    """Open a genome file with appropriate decompression based on extension.

    Supports plain text, gzip (.gz), and bgzf (.bgz) compressed files.

    Args:
        file_path: Path to the genome file.

    Returns:
        A file handle opened in text read mode.
    """
    file_path = Path(file_path)
    if file_path.suffix == '.gz':
        return gzip.open(file_path, 'rt')
    elif file_path.suffix == '.bgz':
        return bgzf.open(file_path, 'rt')
    else:
        return open(file_path, 'rt')
