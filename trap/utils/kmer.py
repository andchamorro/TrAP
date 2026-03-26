from itertools import product
from typing import Iterator, List, Optional, Union


def seq_to_encoded(seq: str, encoding: Optional[Union[str, int]] = None) -> str:
    """Encode a DNA sequence using n-mer compression to Unicode characters.

    Args:
        seq: Input DNA sequence string.
        encoding: Encoding scheme. Accepts "pairs"/"2-mers"/2 for 2-mer encoding,
            or "codons"/"3-mers"/3 for 3-mer encoding. None returns the
            sequence unchanged.

    Returns:
        Encoded string, or the original sequence if encoding is None.
    """
    nucleotides = ['A', 'C', 'G', 'T']

    # TODO: Handle the padding if the seq is not div by 2
    if encoding in ("pairs", "2-mers", 2):
        pairs = [''.join(p) for p in product(nucleotides, repeat=2)]
        pairs_to_unicode = {pair: chr(65 + i) for i, pair in enumerate(pairs)}
        return ''.join(
            pairs_to_unicode[seq[i:i + 2]] for i in range(0, len(seq) - 1, 2)
        )

    # TODO: Handle the padding if the seq is not div by 3
    if encoding in ("codons", "3-mers", 3):
        codons = [''.join(p) for p in product(nucleotides, repeat=3)]
        codons_to_unicode = {codon: chr(65 + i) for i, codon in enumerate(codons)}
        return ''.join(
            codons_to_unicode[seq[i:i + 3]] for i in range(0, len(seq) - 2, 3)
        )

    return seq


def encoded_to_seq(
    encoded_string: str, encoding: Optional[str] = None
) -> str:
    """Decode a Unicode-encoded string back to a DNA sequence.

    Args:
        encoded_string: The encoded string to decode.
        encoding: Encoding scheme used. Accepts "pairs"/"2-mers" for 2-mer
            decoding, or "codons"/"3-mers" for 3-mer decoding.

    Returns:
        Decoded DNA sequence string.
    """
    nucleotides = ['A', 'C', 'G', 'T']

    if encoding in ("pairs", "2-mers"):
        pairs = [''.join(p) for p in product(nucleotides, repeat=2)]
        unicode_to_pairs = {chr(65 + i): pair for i, pair in enumerate(pairs)}
        return ''.join(unicode_to_pairs[e] for e in encoded_string)

    if encoding in ("codons", "3-mers"):
        codons = [''.join(p) for p in product(nucleotides, repeat=3)]
        unicode_to_codons = {chr(65 + i): codon for i, codon in enumerate(codons)}
        return ''.join(unicode_to_codons[e] for e in encoded_string)

    return encoded_string


def kmer_split(
    k: int, sequence: str, encoding: Optional[Union[str, int]] = None
) -> str:
    """Split a DNA sequence into space-separated k-mers, optionally encoded.

    Args:
        k: K-mer length.
        sequence: Input DNA sequence.
        encoding: Optional encoding scheme passed to ``seq_to_encoded``.

    Returns:
        Space-separated string of k-mer tokens.
    """
    return " ".join([
        seq_to_encoded(sequence[j: j + k], encoding=encoding)
        for j in range(len(sequence) - k + 1)
    ])


def kmer_split_batch(
    raw_datasets: Iterator[str],
    batch_size: int,
    k: int,
    encoding: Optional[Union[str, int]] = None,
) -> Iterator[List[str]]:
    """Yield batches of k-mer-split sequences from a dataset.

    Args:
        raw_datasets: Indexable sequence of DNA strings.
        batch_size: Number of sequences per batch.
        k: K-mer length.
        encoding: Optional encoding scheme.

    Yields:
        Lists of space-separated k-mer strings.
    """
    for i in range(0, len(raw_datasets), batch_size):
        yield [
            kmer_split(k, seq, encoding=encoding)
            for seq in raw_datasets[i:i + batch_size]
        ]
