"""Deterministic k-mer counting and rarefaction for the entropy/redundancy analysis.

Builds the empirical k-mer frequency distribution for a corpus across a range of
k, optionally at several subsampling fractions, and turns each distribution into
one tidy row via :func:`trap.analysis.entropy.entropy_summary`.

Two design choices address the peer review directly:

* **Exact counts, not hashed approximations** — k-mers are counted either via
  Jellyfish (default when available: multi-threaded C++ hash counter, orders of
  magnitude faster for transcriptome-scale corpora) or via the pure-numpy
  fallback (base-4 integer codes + ``numpy.unique``).  Both paths yield
  identical count vectors and therefore identical entropy estimates.
* **Rarefaction by sequence count** — for each subsample fraction we count
  k-mers over a deterministic, *nested* subset of sequences (a fixed seeded
  permutation, truncated).  Comparing entropy across fractions reveals whether a
  high-``k`` plateau is real or an artefact of finite sampling (Reviewer 1).

Sequences are streamed forward-strand only (we do **not** reuse
:class:`trap.loaders.dataset.GenomeDataset`, which also materialises every
reverse complement) and standardised with
:func:`trap.utils.sequence.standardize` so the counted alphabet matches what the
tokenizer sees.  Pass ``canonical=True`` to collapse each k-mer with its reverse
complement (the Salmon/jellyfish convention) for strand-agnostic corpora.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

from Bio import SeqIO
from loguru import logger
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from trap.analysis.entropy import entropy_summary
from trap.utils.canonical_kmer import canonical_codes, seq_to_base_codes
from trap.utils.io import genome_file_handle
from trap.utils.seeding import seeded_rng
from trap.utils.sequence import standardize

# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------


def iter_forward_sequences(path: Union[str, Path], file_format: str) -> Iterator[str]:
    """Yield standardised forward-strand sequences from a FASTA/FASTQ file.

    Args:
        path: Corpus path (``.gz``/``.bgz`` handled by ``genome_file_handle``).
        file_format: BioPython SeqIO format (``"fasta"`` or ``"fastq"``).

    Yields:
        Upper-case ACTG sequences with non-ACTG characters removed; empty
        sequences are skipped.
    """
    with genome_file_handle(Path(path)) as handle:
        for record in SeqIO.parse(handle, file_format):
            seq = standardize(str(record.seq))
            if seq:
                yield seq


def load_corpus(path: Union[str, Path], file_format: str = "fasta") -> List[str]:
    """Load a corpus into a list of standardised forward sequences.

    Args:
        path: Corpus path.
        file_format: BioPython SeqIO format.

    Returns:
        List of ACTG sequences.
    """
    return list(iter_forward_sequences(path, file_format))


# ---------------------------------------------------------------------------
# Jellyfish backend (preferred for large corpora)
# ---------------------------------------------------------------------------


def _jellyfish_available() -> bool:
    """Return True if the ``jellyfish`` binary is on PATH."""
    return shutil.which("jellyfish") is not None


def _write_fasta(sequences: Sequence[str], path: Path) -> None:
    """Write a plain FASTA file (no line-wrapping) for a sequence list."""
    with open(path, "w") as fh:
        for i, seq in enumerate(sequences):
            fh.write(f">s{i}\n{seq}\n")


def _run_jellyfish_count(
    fasta: Path, jf_db: Path, k: int, canonical: bool, threads: int, hash_size: str
) -> None:
    """Run ``jellyfish count`` on an existing FASTA file; raises with stderr on failure."""
    count_cmd = [
        "jellyfish",
        "count",
        "-m",
        str(k),
        "-s",
        hash_size,
        "-t",
        str(max(1, threads)),
        "-o",
        str(jf_db),
    ]
    if canonical:
        count_cmd.append("-C")
    count_cmd.append(str(fasta))

    result = subprocess.run(count_cmd, capture_output=True)
    if result.returncode != 0:
        stderr_txt = result.stderr.decode(errors="replace").strip()
        logger.warning(f"jellyfish count failed (k={k}): {stderr_txt or '(no stderr)'}")
        raise subprocess.CalledProcessError(
            result.returncode, count_cmd, result.stdout, result.stderr
        )


def _run_jellyfish_dump(jf_db: Path, tmp_dir: Optional[Path] = None) -> np.ndarray:
    """Run ``jellyfish dump -c`` and return the counts array.

    Pipes through ``awk`` to extract only the count column before Python sees
    any bytes.  For k≥15 on transcriptome-scale corpora, the full dump can be
    1–4 GB of text; extracting the second column in-pipe reduces Python-side
    allocation to one ``np.frombuffer`` call (~8 bytes × n_distinct_kmers)
    instead of a Python list of ints (28 bytes × n_distinct_kmers) plus the
    raw string and its splitlines list — a 10–15× memory reduction per dump
    that prevents OOM when 16 dumps run concurrently in the thread pool.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        counts_txt = Path(td) / "counts.txt"
        with open(counts_txt, "wb") as fout:
            dump_proc = subprocess.Popen(
                ["jellyfish", "dump", "-c", str(jf_db)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            awk_proc = subprocess.Popen(
                ["awk", "{print $2}"],
                stdin=dump_proc.stdout,
                stdout=fout,
                stderr=subprocess.DEVNULL,
            )
            dump_proc.stdout.close()  # type: ignore[union-attr]
            awk_proc.wait()
            dump_proc.wait()
            if dump_proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    dump_proc.returncode, ["jellyfish", "dump"]
                )
        data = np.fromfile(counts_txt, sep="\n", dtype=np.int64)
    return data if data.size else np.empty(0, dtype=np.int64)


def count_kmers_jellyfish_from_file(
    fasta: Path,
    k: int,
    canonical: bool = False,
    threads: int = 1,
    hash_size: str = "1G",
    tmp_dir: Optional[Path] = None,
) -> np.ndarray:
    """Count k-mers using Jellyfish from a pre-written FASTA file.

    Prefer over :func:`count_kmers_jellyfish` when the same sequences are
    counted at multiple ``k`` values — the caller writes the FASTA once and
    passes the path here, avoiding repeated disk writes.

    Args:
        fasta: Existing plain FASTA file (no gzip).
        k: K-mer length.
        canonical: Collapse each k-mer with its reverse complement (``-C``).
        threads: CPU threads for Jellyfish (``-t``).
        hash_size: Initial hash table size for Jellyfish (``-s``).
        tmp_dir: Directory for the ``.jf`` database; defaults to system temp.

    Returns:
        ``int64`` array of k-mer counts.  Empty when the FASTA has no sequences
        at least ``k`` long.

    Raises:
        subprocess.CalledProcessError: If Jellyfish exits non-zero (stderr logged).
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        td_path = Path(td)
        jf_db = td_path / "counts.jf"
        _run_jellyfish_count(fasta, jf_db, k, canonical, threads, hash_size)
        return _run_jellyfish_dump(jf_db, tmp_dir=td_path)


def count_kmers_jellyfish(
    sequences: Sequence[str],
    k: int,
    canonical: bool = False,
    threads: int = 1,
    hash_size: str = "1G",
    tmp_dir: Optional[Path] = None,
) -> np.ndarray:
    """Count k-mers using Jellyfish (requires ``jellyfish`` on PATH).

    Writes *sequences* to a temporary FASTA, runs ``jellyfish count`` then
    ``jellyfish dump``, and returns the multiplicity vector.  Jellyfish uses
    *threads* CPU threads internally, making this orders of magnitude faster
    than the numpy path for transcriptome-scale corpora.

    When counting the same sequences at multiple ``k`` values, prefer writing
    the FASTA once externally and calling :func:`count_kmers_jellyfish_from_file`
    to avoid redundant disk I/O.

    Args:
        sequences: Standardised ACTG sequences.
        k: K-mer length.
        canonical: Collapse each k-mer with its reverse complement (``-C``).
        threads: CPU threads for Jellyfish (``-t``).
        hash_size: Initial hash table size passed to Jellyfish (``-s``).
        tmp_dir: Directory for temporary files; defaults to the system temp dir.

    Returns:
        ``int64`` array of k-mer counts (length = number of distinct k-mers).
        Empty when no sequence is at least ``k`` long.

    Raises:
        FileNotFoundError: If ``jellyfish`` is not on PATH.
        subprocess.CalledProcessError: If jellyfish exits non-zero (stderr logged).
    """
    if not sequences:
        return np.empty(0, dtype=np.int64)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        td_path = Path(td)
        fasta = td_path / "seqs.fa"
        _write_fasta(sequences, fasta)
        return count_kmers_jellyfish_from_file(
            fasta, k, canonical=canonical, threads=threads, hash_size=hash_size, tmp_dir=td_path
        )


# ---------------------------------------------------------------------------
# SeqKit backend (streaming sub-sampling, pipes into Jellyfish)
# ---------------------------------------------------------------------------


def _seqkit_available() -> bool:
    """Return True if the ``seqkit`` binary is on PATH."""
    return shutil.which("seqkit") is not None


def _seqkit_num_seqs(path: Path) -> int:
    """Return the sequence count for a corpus via ``seqkit stats``.

    Args:
        path: FASTA or gzipped FASTA corpus (seqkit auto-detects format).

    Returns:
        Total number of sequences.

    Raises:
        RuntimeError: If seqkit returns no parseable data.
        subprocess.CalledProcessError: If seqkit exits non-zero.
    """
    r = subprocess.run(
        ["seqkit", "stats", "--tabular", str(path)],
        capture_output=True, text=True, check=True,
    )
    for line in r.stdout.splitlines():
        if not line.startswith("file"):
            cols = line.split("\t")
            if len(cols) > 3:
                return int(cols[3])
    raise RuntimeError(f"seqkit stats returned no parseable data for {path}")


def _seqkit_sample(
    input_path: Path,
    output_path: Path,
    proportion: float,
    seed: int,
) -> None:
    """Write a Bernoulli-sampled FASTA subset via ``seqkit sample``.

    Handles gzipped and plain inputs transparently.  *output_path* should use
    a plain ``.fa`` extension so seqkit writes uncompressed FASTA.

    Args:
        input_path: Source FASTA (plain or ``.gz``; seqkit auto-detects).
        output_path: Destination plain FASTA (caller places in ``$TMPDIR``).
        proportion: Fraction to retain (``0 < proportion ≤ 1``).
        seed: Reproducibility seed; clamped to int31 range for seqkit.

    Raises:
        subprocess.CalledProcessError: If seqkit exits non-zero.
    """
    cmd = [
        "seqkit", "sample",
        "--proportion", f"{proportion:.10f}",
        "--rand-seed", str(seed % (2 ** 31)),
        "--out-file", str(output_path),
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def _count_jellyfish_seqkit_pipe(
    corpus_path: Path,
    k: int,
    canonical: bool,
    threads: int,
    hash_size: str,
    tmp_dir: Optional[Path] = None,
) -> np.ndarray:
    """Count k-mers by piping ``seqkit seq`` directly into ``jellyfish count``.

    No intermediate FASTA file is written; the corpus is streamed from disk.
    Used for frac=1.0 when both backends are active.

    Args:
        corpus_path: Source FASTA (plain or gzipped; seqkit auto-detects).
        k: K-mer length.
        canonical: Collapse reverse complements (``-C``).
        threads: Jellyfish thread count.
        hash_size: Jellyfish initial hash-table size (``-s``).
        tmp_dir: Directory for the ``.jf`` database; defaults to system temp.

    Returns:
        ``int64`` array of k-mer counts.

    Raises:
        subprocess.CalledProcessError: If jellyfish exits non-zero.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        jf_db = Path(td) / "counts.jf"
        count_cmd = [
            "jellyfish", "count",
            "-m", str(k),
            "-s", hash_size,
            "-t", str(max(1, threads)),
            "-o", str(jf_db),
        ]
        if canonical:
            count_cmd.append("-C")
        count_cmd.append("/dev/stdin")

        seqkit_proc = subprocess.Popen(
            ["seqkit", "seq", str(corpus_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        jf_proc = subprocess.Popen(
            count_cmd,
            stdin=seqkit_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        seqkit_proc.stdout.close()  # allow jellyfish to receive SIGPIPE if it exits
        _, jf_err = jf_proc.communicate()
        seqkit_proc.wait()

        if jf_proc.returncode != 0:
            stderr_txt = jf_err.decode(errors="replace").strip()
            logger.warning(
                f"jellyfish count (seqkit pipe, k={k}): {stderr_txt or '(no stderr)'}"
            )
            raise subprocess.CalledProcessError(
                jf_proc.returncode, count_cmd, stderr=jf_err
            )

        return _run_jellyfish_dump(jf_db, tmp_dir=Path(td))


# ---------------------------------------------------------------------------
# NumPy fallback backend
# ---------------------------------------------------------------------------


def _forward_codes(seq: str, k: int) -> np.ndarray:
    """Integer codes of every forward k-mer in *seq* (base-4, big-endian)."""
    base = seq_to_base_codes(seq)
    if base.shape[0] < k:
        return np.empty(0, dtype=np.int64)
    windows = sliding_window_view(base, k)
    pow4_be = np.int64(4) ** np.arange(k - 1, -1, -1, dtype=np.int64)
    return windows @ pow4_be


def _merge_counts(
    acc_codes: np.ndarray,
    acc_counts: np.ndarray,
    new_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a chunk of k-mer codes into the running (codes, counts) tally."""
    if new_codes.size == 0:
        return acc_codes, acc_counts
    uniq, counts = np.unique(new_codes, return_counts=True)
    if acc_codes.size == 0:
        return uniq, counts.astype(np.int64)
    all_codes = np.concatenate([acc_codes, uniq])
    all_counts = np.concatenate([acc_counts, counts])
    merged_codes, inverse = np.unique(all_codes, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    merged_counts = np.bincount(inverse, weights=all_counts).astype(np.int64)
    return merged_codes, merged_counts


def count_kmers(
    sequences: Sequence[str],
    k: int,
    canonical: bool = False,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Count k-mers using the pure-numpy backend (no external dependencies).

    Accumulated in chunks so transient memory stays bounded by the number of
    distinct k-mers plus one chunk.  Prefer :func:`count_kmers_jellyfish` for
    large corpora; this function is the fallback and is used in unit tests.

    Args:
        sequences: Standardised ACTG sequences.
        k: K-mer length (``1 <= k <= 31``).
        canonical: Collapse each k-mer with its reverse complement.
        chunk_size: Number of sequences encoded before each merge.

    Returns:
        ``int64`` array of k-mer counts.
        Empty when no sequence is at least ``k`` long.
    """
    acc_codes = np.empty(0, dtype=np.int64)
    acc_counts = np.empty(0, dtype=np.int64)
    buffer: List[np.ndarray] = []
    encode = canonical_codes if canonical else _forward_codes

    def flush() -> None:
        nonlocal acc_codes, acc_counts, buffer
        if not buffer:
            return
        chunk = np.concatenate(buffer)
        acc_codes, acc_counts = _merge_counts(acc_codes, acc_counts, chunk)
        buffer = []

    for i, seq in enumerate(sequences, start=1):
        codes = encode(seq, k)
        if codes.size:
            buffer.append(codes)
        if i % chunk_size == 0:
            flush()
    flush()
    return acc_counts


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------


def _count_all_k_jellyfish(
    fasta: Path,
    k_values: Sequence[int],
    canonical: bool,
    jf_threads: int,
    jf_hash_size: str,
    n_jobs: int,
    tmp_dir: Optional[Path] = None,
) -> Dict[int, np.ndarray]:
    """Count k-mers for every value in *k_values* using jellyfish from a file.

    When *n_jobs* > 1 a thread pool dispatches one jellyfish process per k
    (each single-threaded); otherwise all *jf_threads* are given to each
    sequential jellyfish call.

    Args:
        fasta: Existing plain FASTA file.
        k_values: K-mer lengths to count.
        canonical, jf_threads, jf_hash_size, tmp_dir: Forwarded to jellyfish.
        n_jobs: Worker threads for the pool.

    Returns:
        Mapping ``k -> int64 count array``.
    """
    worker_threads = 1 if n_jobs > 1 else jf_threads
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            fut_map = {
                k: pool.submit(
                    count_kmers_jellyfish_from_file,
                    fasta, k, canonical, worker_threads, jf_hash_size, tmp_dir,
                )
                for k in k_values
            }
            return {k: fut.result() for k, fut in fut_map.items()}
    return {
        k: count_kmers_jellyfish_from_file(
            fasta, k, canonical, jf_threads, jf_hash_size, tmp_dir,
        )
        for k in k_values
    }


def _count_all_k_seqkit_pipe(
    corpus_path: Path,
    k_values: Sequence[int],
    canonical: bool,
    jf_threads: int,
    jf_hash_size: str,
    n_jobs: int,
    tmp_dir: Optional[Path] = None,
) -> Dict[int, np.ndarray]:
    """Stream the corpus through seqkit into jellyfish for every k — no temp file.

    Args:
        corpus_path: Source FASTA (gzipped or plain; seqkit auto-detects).
        k_values: K-mer lengths to count.
        canonical, jf_threads, jf_hash_size, tmp_dir: Forwarded to jellyfish.
        n_jobs: Parallel seqkit→jellyfish pipelines (one per k).

    Returns:
        Mapping ``k -> int64 count array``.
    """
    worker_threads = 1 if n_jobs > 1 else jf_threads
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            fut_map = {
                k: pool.submit(
                    _count_jellyfish_seqkit_pipe,
                    corpus_path, k, canonical, worker_threads, jf_hash_size, tmp_dir,
                )
                for k in k_values
            }
            return {k: fut.result() for k, fut in fut_map.items()}
    return {
        k: _count_jellyfish_seqkit_pipe(
            corpus_path, k, canonical, jf_threads, jf_hash_size, tmp_dir,
        )
        for k in k_values
    }


def _build_entropy_rows(
    counts_by_k: Dict[int, np.ndarray],
    k_values: Sequence[int],
    corpus_name: str,
    frac: float,
    replicate: int,
    n_sub: int,
    canonical: bool,
    seed: int,
    bootstrap: int,
    alpha: float,
) -> List[Dict[str, object]]:
    """Build tidy entropy rows from pre-counted vectors for one (replicate, frac).

    The per-cell bootstrap seed is derived the same way as in :func:`_spectrum_cell`
    so rows from the cached and the streamed paths are numerically consistent.

    Args:
        counts_by_k: Mapping ``k -> count array``.
        k_values: K-mer lengths (must be keys in *counts_by_k*).
        corpus_name, frac, replicate, n_sub, canonical: Row metadata.
        seed, bootstrap, alpha: Entropy/bootstrap parameters.

    Returns:
        One tidy dict row per k value.
    """
    rows: List[Dict[str, object]] = []
    for k in k_values:
        cell_seed = seed + replicate * 1_000_003 + int(frac * 1000) * 1009 + k
        boot_rng = seeded_rng(cell_seed)
        summary = entropy_summary(
            counts_by_k[k], k, bootstrap=bootstrap, rng=boot_rng, alpha=alpha,
        )
        row: Dict[str, object] = {
            "corpus": corpus_name,
            "k": int(k),
            "subsample_frac": float(frac),
            "replicate": int(replicate),
            "n_sequences": int(n_sub),
            "canonical": bool(canonical),
        }
        row.update(summary)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------


def _spectrum_cell(
    subset: List[str],
    k: int,
    canonical: bool,
    chunk_size: int,
    seed: int,
    replicate: int,
    frac: float,
    bootstrap: int,
    alpha: float,
    corpus_name: str,
    use_jellyfish: bool,
    jf_threads: int,
    jf_fasta: Optional[Path] = None,
    jf_hash_size: str = "1G",
) -> Dict[str, object]:
    """Compute one (corpus, k, frac, replicate) row — used by the thread pool.

    When *jf_fasta* is provided (recommended), Jellyfish reads from that
    pre-written file instead of re-writing the same sequences to disk per k,
    avoiding concurrent I/O pressure in the thread pool.  Falls back to the
    numpy backend if Jellyfish exits non-zero.
    """
    if use_jellyfish:
        try:
            if jf_fasta is not None:
                counts = count_kmers_jellyfish_from_file(
                    jf_fasta, k, canonical=canonical, threads=jf_threads,
                    hash_size=jf_hash_size,
                )
            else:
                counts = count_kmers_jellyfish(
                    subset, k, canonical=canonical, threads=jf_threads,
                    hash_size=jf_hash_size,
                )
        except subprocess.CalledProcessError:
            logger.warning(f"[{corpus_name}] jellyfish failed for k={k} — falling back to numpy")
            counts = count_kmers(subset, k, canonical=canonical, chunk_size=chunk_size)
    else:
        counts = count_kmers(subset, k, canonical=canonical, chunk_size=chunk_size)

    cell_seed = seed + replicate * 1_000_003 + int(frac * 1000) * 1009 + k
    boot_rng = seeded_rng(cell_seed)
    summary = entropy_summary(counts, k, bootstrap=bootstrap, rng=boot_rng, alpha=alpha)
    row: Dict[str, object] = {
        "corpus": corpus_name,
        "k": int(k),
        "subsample_frac": float(frac),
        "replicate": int(replicate),
        "n_sequences": int(len(subset)),
        "canonical": bool(canonical),
    }
    row.update(summary)
    return row


def kmer_spectrum(
    sequences: Sequence[str],
    k_values: Sequence[int],
    subsample_fracs: Sequence[float] = (1.0,),
    replicates: int = 1,
    seed: int = 3469,
    canonical: bool = False,
    bootstrap: int = 0,
    alpha: float = 0.05,
    corpus_name: str = "corpus",
    chunk_size: int = 2000,
    n_jobs: int = 1,
    jf_hash_size: str = "1G",
    corpus_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Compute the entropy/redundancy spectrum with rarefaction.

    For each ``(replicate, subsample fraction, k)`` triple, count k-mers over a
    deterministic nested subset of sequences and summarise the distribution.

    **SeqKit + Jellyfish path** (activated when *corpus_path* is supplied):
    Both ``seqkit`` and ``jellyfish`` must be on PATH; a :exc:`RuntimeError` is
    raised if either is absent.  Sub-1.0 fractions are produced by chaining
    ``seqkit sample`` calls in descending order so each level is a strict nested
    subset of the level above — no sequences are loaded into Python memory and
    no permutation array is built.  Each fraction's FASTA is written once to
    ``$TMPDIR`` and read by all k-value jellyfish workers in parallel, then
    deleted before the next level's file is created.  frac=1.0 is counted via
    ``seqkit seq | jellyfish count /dev/stdin`` (no temp file at all).

    **Legacy path** (when *corpus_path* is ``None``): uses the pre-loaded
    *sequences* list with the jellyfish-or-numpy backends as before.

    **frac=1.0 caching** (both paths): counts at frac=1.0 are order-independent
    and therefore identical across replicates.  They are computed once and
    cached; subsequent replicates re-run only the bootstrap step with their
    per-replicate seed, eliminating ``(replicates - 1)`` redundant passes.

    Args:
        sequences: Pre-loaded ACTG sequences.  Ignored when *corpus_path* is
            set and the seqkit+jellyfish backend is active.
        k_values: K-mer lengths to evaluate.
        subsample_fracs: Fractions of sequences to retain.
        replicates: Independent seeded subsamples per fraction.
        seed: Master seed; per-cell RNGs are derived deterministically.
        canonical: Collapse reverse complements when counting.
        bootstrap: Bootstrap replicates for the entropy CI (``0`` disables).
        alpha: Two-sided level for the bootstrap interval.
        corpus_name: Label written into every row.
        chunk_size: Sequences per merge (numpy backend only).
        n_jobs: Worker threads for k-level parallelism.
        jf_hash_size: Initial hash-table size for ``jellyfish count -s``.
        corpus_path: When provided, enables the seqkit streaming path.
            Requires both ``seqkit`` and ``jellyfish`` on PATH.

    Returns:
        List of tidy dict rows, one per ``(corpus, k, subsample_frac,
        replicate)``, carrying every estimator and diagnostic.

    Raises:
        RuntimeError: When *corpus_path* is given but seqkit or jellyfish is
            not installed.
        ValueError: When *sequences* is empty and *corpus_path* is not given.
    """
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    fracs = sorted({float(f) for f in subsample_fracs})
    rows: List[Dict[str, object]] = []
    jf_threads_serial = n_jobs  # for serial jellyfish (all CPUs per call)

    # ------------------------------------------------------------------
    # SeqKit + Jellyfish path
    # ------------------------------------------------------------------
    if corpus_path is not None:
        if not _seqkit_available():
            raise RuntimeError(
                "seqkit is required for the corpus_path streaming path but was not found on PATH. "
                "Install seqkit (https://bioinf.shenwei.me/seqkit/) and ensure it is on PATH."
            )
        if not _jellyfish_available():
            raise RuntimeError(
                "jellyfish is required for the corpus_path streaming path but was not found on PATH. "
                "Install jellyfish and ensure it is on PATH."
            )

        n_total = _seqkit_num_seqs(corpus_path)
        if n_total == 0:
            raise ValueError(f"corpus '{corpus_name}' has no sequences at {corpus_path}")

        tmpdir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        pid = os.getpid()

        logger.log(
            "STAGE",
            f"[{corpus_name}] seqkit+jellyfish backend — "
            f"n_total={n_total:,} n_jobs={n_jobs} hash={jf_hash_size}",
        )

        # frac=1.0: pipe seqkit seq → jellyfish; no temp FASTA file.
        _full_counts: Dict[int, np.ndarray] = {}
        if 1.0 in fracs:
            _full_counts = _count_all_k_seqkit_pipe(
                corpus_path, k_values, canonical,
                jf_threads_serial, jf_hash_size, n_jobs, tmpdir,
            )
            logger.log(
                "STAGE",
                f"[{corpus_name}] frac=1.0 counts cached via seqkit pipe "
                f"({len(_full_counts)}/{len(k_values)} k values) — "
                f"{replicates - 1} redundant pass(es) eliminated",
            )

        _use_cache = frozenset(_full_counts) >= frozenset(k_values)

        for replicate in range(replicates):
            # frac=1.0 rows: reuse cached counts, re-seed bootstrap per replicate.
            if 1.0 in fracs and _use_cache:
                rows.extend(
                    _build_entropy_rows(
                        _full_counts, k_values, corpus_name,
                        1.0, replicate, n_total, canonical,
                        seed, bootstrap, alpha,
                    )
                )
                logger.log(
                    "STAGE",
                    f"[{corpus_name}] replicate={replicate} frac=1.00 "
                    f"({n_total:,}/{n_total:,} seqs) done [cached counts]",
                )

            # Sub-1.0 fracs: seqkit chain (descending) → one temp FASTA per level.
            sub_fracs = sorted([f for f in fracs if f < 1.0], reverse=True)
            prev_path = corpus_path
            prev_frac = 1.0
            chain_tmps: List[Path] = []

            try:
                for level_idx, frac in enumerate(sub_fracs):
                    proportion = frac / prev_frac
                    # Seed: replicate and level are independently varied so that
                    # different replicates produce different Bernoulli draws at
                    # each fraction level.
                    sample_seed = (seed + replicate * 1_000_003 + level_idx * 997) % (2 ** 31)
                    n_sub = max(1, round(frac * n_total))

                    frac_fasta = tmpdir / f"trap_{corpus_name}_r{replicate}_f{int(frac * 1000):04d}_{pid}.fa"
                    _seqkit_sample(prev_path, frac_fasta, proportion, sample_seed)
                    chain_tmps.append(frac_fasta)

                    frac_counts = _count_all_k_jellyfish(
                        frac_fasta, k_values, canonical,
                        jf_threads_serial, jf_hash_size, n_jobs, tmpdir,
                    )
                    rows.extend(
                        _build_entropy_rows(
                            frac_counts, k_values, corpus_name,
                            frac, replicate, n_sub, canonical,
                            seed, bootstrap, alpha,
                        )
                    )
                    logger.log(
                        "STAGE",
                        f"[{corpus_name}] replicate={replicate} frac={frac:.2f} "
                        f"({n_sub:,}/{n_total:,} seqs) done [seqkit chain]",
                    )

                    # Delete the previous level's temp file immediately — it is
                    # no longer needed and may be large (frac=0.75 → ~300 MB).
                    if prev_path != corpus_path:
                        prev_path.unlink(missing_ok=True)
                        chain_tmps = [f for f in chain_tmps if f != prev_path]

                    prev_path = frac_fasta
                    prev_frac = frac

            finally:
                # Ensure all created chain files are removed on success or error.
                for f in chain_tmps:
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass

        return rows

    # ------------------------------------------------------------------
    # Legacy path (pre-loaded sequences, jellyfish-or-numpy)
    # ------------------------------------------------------------------
    use_jellyfish = _jellyfish_available()
    n_total = len(sequences)
    if n_total == 0:
        raise ValueError(f"corpus '{corpus_name}' has no usable sequences")

    if use_jellyfish:
        logger.debug(
            f"[{corpus_name}] jellyfish backend (n_jobs={n_jobs}, hash={jf_hash_size})"
        )
    else:
        logger.debug(f"[{corpus_name}] jellyfish not found — using numpy backend")

    # frac=1.0 count-vector cache (order-independent; identical across replicates).
    _full_counts_leg: Dict[int, np.ndarray] = {}
    _full_fasta_td: Optional[tempfile.TemporaryDirectory] = None

    if 1.0 in fracs:
        if use_jellyfish:
            _full_fasta_td = tempfile.TemporaryDirectory()
            _full_jf_fasta = Path(_full_fasta_td.name) / "seqs.fa"
            _write_fasta(sequences, _full_jf_fasta)
            # Try jellyfish per k; fall back to numpy individually on failure.
            jf_pool_threads = 1 if n_jobs > 1 else jf_threads_serial
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                    fut_map = {
                        k: pool.submit(
                            count_kmers_jellyfish_from_file,
                            _full_jf_fasta, k, canonical,
                            jf_pool_threads, jf_hash_size,
                        )
                        for k in k_values
                    }
                    for k, fut in fut_map.items():
                        try:
                            _full_counts_leg[k] = fut.result()
                        except subprocess.CalledProcessError:
                            logger.warning(
                                f"[{corpus_name}] jellyfish frac=1.0 k={k} failed; numpy fallback"
                            )
                            _full_counts_leg[k] = count_kmers(
                                sequences, k, canonical=canonical, chunk_size=chunk_size
                            )
            else:
                for k in k_values:
                    try:
                        _full_counts_leg[k] = count_kmers_jellyfish_from_file(
                            _full_jf_fasta, k, canonical,
                            jf_threads_serial, jf_hash_size,
                        )
                    except subprocess.CalledProcessError:
                        logger.warning(
                            f"[{corpus_name}] jellyfish frac=1.0 k={k} failed; numpy fallback"
                        )
                        _full_counts_leg[k] = count_kmers(
                            sequences, k, canonical=canonical, chunk_size=chunk_size
                        )
        else:
            for k in k_values:
                _full_counts_leg[k] = count_kmers(
                    sequences, k, canonical=canonical, chunk_size=chunk_size
                )

        if _full_counts_leg:
            logger.log(
                "STAGE",
                f"[{corpus_name}] frac=1.0 counts cached "
                f"({len(_full_counts_leg)}/{len(k_values)} k values) — "
                f"{replicates - 1} redundant pass(es) eliminated",
            )

    _use_cache_leg = frozenset(_full_counts_leg) >= frozenset(k_values)

    try:
        for replicate in range(replicates):
            permutation = seeded_rng(seed + replicate).permutation(n_total)
            for frac in fracs:
                n_sub = max(1, int(round(frac * n_total)))
                subset = [sequences[i] for i in permutation[:n_sub]]

                if frac == 1.0 and _use_cache_leg:
                    rows.extend(
                        _build_entropy_rows(
                            _full_counts_leg, k_values, corpus_name,
                            1.0, replicate, n_sub, canonical,
                            seed, bootstrap, alpha,
                        )
                    )
                    logger.log(
                        "STAGE",
                        f"[{corpus_name}] replicate={replicate} frac=1.00 "
                        f"({n_sub:,}/{n_total:,} seqs) done [cached counts]",
                    )
                    continue

                # Write FASTA once per (replicate, frac); all k workers share it.
                if use_jellyfish:
                    fasta_td: Optional[tempfile.TemporaryDirectory] = (
                        tempfile.TemporaryDirectory()
                    )
                    jf_fasta: Optional[Path] = Path(fasta_td.name) / "seqs.fa"
                    _write_fasta(subset, jf_fasta)
                else:
                    fasta_td = None
                    jf_fasta = None

                try:
                    if n_jobs > 1:
                        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                            futures = [
                                pool.submit(
                                    _spectrum_cell,
                                    subset, k, canonical, chunk_size,
                                    seed, replicate, frac, bootstrap, alpha,
                                    corpus_name, use_jellyfish,
                                    1, jf_fasta, jf_hash_size,
                                )
                                for k in k_values
                            ]
                            for future in futures:
                                rows.append(future.result())
                    else:
                        for k in k_values:
                            rows.append(
                                _spectrum_cell(
                                    subset, k, canonical, chunk_size,
                                    seed, replicate, frac, bootstrap, alpha,
                                    corpus_name, use_jellyfish,
                                    jf_threads_serial, jf_fasta, jf_hash_size,
                                )
                            )
                finally:
                    if fasta_td is not None:
                        fasta_td.cleanup()

                logger.log(
                    "STAGE",
                    f"[{corpus_name}] replicate={replicate} frac={frac:.2f} "
                    f"({n_sub:,}/{n_total:,} seqs) done",
                )
    finally:
        if _full_fasta_td is not None:
            _full_fasta_td.cleanup()

    return rows
