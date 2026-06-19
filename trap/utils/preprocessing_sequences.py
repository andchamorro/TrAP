"""Dataset preprocessing for MLM and classification tasks (DR-3, DR-6).

Changes from the argparse version
----------------------------------
DR-3  Converted to Typer: ``python -m trap.utils.preprocessing_sequences
      {masking,classification} --config …``.
DR-6  Added ``--split-strategy transcript-level`` (default): groups reads by
      transcript ID *before* splitting, so no transcript contributes reads to
      both training and eval/test splits (reviewer §12/#15 leakage fix).
      Pass ``--split-strategy read-level`` to restore the previous behaviour.

Label extraction
----------------
Labels come from the FASTA/FASTQ header via
``id.split('|')[-1].split('-')[0]``.  This is moved to ``trap/utils/labels.py``
as a stable function so R notebooks can call it directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence

from datasets import Dataset, DatasetDict
from loguru import logger
import numpy as np
from transformers import PreTrainedTokenizerFast
import typer

from trap.config import manifest as manifest_mod
from trap.config.verbosity import set_verbosity
from trap.loaders.dataset import GenomeDataset
from trap.loaders.salmon_tokenizer import SalmonKmerTokenizer
from trap.loaders.tokenizer import load_kmer_tokenizer
from trap.utils.io import try_mkdir
from trap.utils.kmer import kmer_split
from trap.utils.labels import TASK_PRIORITY, analyze_labels, task_label
from trap.utils.seeding import set_global_seed

app = typer.Typer(help="Pre-process genomic sequences for MLM or classification training.")


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def _extract_label(read_id: str) -> str:
    """Extract the class label from a TrAP FASTA/FASTQ record ID.

    Expected header format: ``TRANSCRIPT_ID|LABEL-...``

    Args:
        read_id: Record identifier string.

    Returns:
        Label string (e.g. ``"L1HS"``, ``"L1PA"``, ``"NEGATIVE"``).
    """
    parts = read_id.split("|")
    if len(parts) < 2:
        return read_id
    return parts[-1].split("-")[0]


def _extract_transcript_id(read_id: str) -> str:
    """Return the transcript ID portion of a record identifier.

    The transcript ID is the part before the first ``|`` character.

    Args:
        read_id: Record identifier string.

    Returns:
        Transcript ID string.
    """
    return read_id.split("|")[0]


# Strip a `/1` or `/2` mate marker (followed by `|` or end-of-id) so that the
# two mates of a fragment compare equal. The current build leaves both mates'
# ids identical, but normalising keeps the desync guard correct if a future
# build numbers mates.
_MATE_SUFFIX_RE = re.compile(r"/[12](?=\||$)")


def _mate_key(read_id: str) -> str:
    """Return *read_id* with its ``/1`` / ``/2`` mate marker removed."""
    return _MATE_SUFFIX_RE.sub("", read_id)


def _fragment_key(read_id: str) -> str:
    """Identity of the source fragment, shared by both mates and by every
    per-subfamily labelled copy of the read.

    A read overlapping multiple subfamilies is emitted once per subfamily with a
    different ``|LABEL`` appended; stripping that trailing label (and the
    ``/1``/``/2`` mate marker) yields a key that collapses all those copies.
    """
    base = read_id.rsplit("|", 1)[0]
    return _mate_key(base)


def _resolve_fragment_labels(
    reads: "GenomeDataset", label_fn: "Callable[[str], str]", priority: "Sequence[str]"
) -> Dict[str, str]:
    """Map each fragment key to its single highest-priority label.

    A read overlapping several subfamilies appears under several labels; this
    keeps the one ranked first in *priority* (labels absent from *priority* rank
    lowest), so e.g. an L1HS+L1PA read resolves to L1HS.
    """
    rank = {lbl: i for i, lbl in enumerate(priority)}
    lowest = len(priority)
    best: Dict[str, tuple] = {}
    for _seq, read_id in reads:
        label = label_fn(read_id)
        key = _fragment_key(read_id)
        r = rank.get(label, lowest)
        current = best.get(key)
        if current is None or r < current[0]:
            best[key] = (r, label)
    return {key: label for key, (_r, label) in best.items()}


# ---------------------------------------------------------------------------
# Transcript-level GroupShuffleSplit (DR-6)
# ---------------------------------------------------------------------------


def _dominant_label_per_transcript(transcript_ids: List[str], labels: List[int]) -> Dict[str, int]:
    """Map each transcript to the integer label most common among its reads."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for tid, lbl in zip(transcript_ids, labels):
        counts[tid][lbl] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in counts.items()}


def _transcript_level_split(
    dataset: Dataset,
    test_split: float,
    val_split: Optional[float],
    seed: int = 3469,
) -> DatasetDict:
    """Stratified transcript-level split that prevents read-level leakage.

    All reads from the same source transcript stay in one split (no read-level
    leakage, DR-6). Transcripts are grouped by their *dominant* label (most
    frequent class among that transcript's reads) and each group is allocated
    proportionally, so rare classes like L1HS are represented in every split
    rather than landing by chance in a single one.

    When a class has too few transcripts to populate all splits the function
    falls back gracefully: train gets priority, followed by test, then eval;
    a warning is logged so the caller can decide whether to increase data.

    Args:
        dataset: HuggingFace ``Dataset`` with ``"transcript_id"`` and
            ``"label"`` (``ClassLabel``) columns.
        test_split: Fraction of transcripts reserved for test.
        val_split: Fraction of transcripts reserved for validation
            (``None`` → no eval split).
        seed: RNG seed for reproducible shuffling.

    Returns:
        ``DatasetDict`` with ``"train"`` / ``"test"`` / (optionally) ``"eval"``
        splits where no transcript ID appears in two splits.
    """
    requested = test_split + (val_split or 0.0)
    if requested >= 1.0:
        raise ValueError(
            f"test_split + val_split must be < 1.0 (got {requested}); "
            "no transcripts would remain for training."
        )

    rng = np.random.default_rng(seed)

    transcript_ids: List[str] = dataset["transcript_id"]
    labels: List[int] = dataset["label"]
    label_names: List[str] = dataset.features["label"].names

    dominant = _dominant_label_per_transcript(transcript_ids, labels)

    # Shuffle unique transcripts once so within-class order is random and
    # reproducible, then bucket by dominant label.
    unique = list(dict.fromkeys(transcript_ids))
    rng.shuffle(unique)  # type: ignore[arg-type]

    by_label: Dict[int, List[str]] = defaultdict(list)
    for tid in unique:
        by_label[dominant[tid]].append(tid)

    train_set: set = set()
    eval_set: set = set()
    test_set: set = set()
    log_lines: List[str] = []

    for lbl_int, tids in sorted(by_label.items()):
        n = len(tids)
        name = label_names[lbl_int]
        n_test = max(1, round(n * test_split))
        n_val = max(1, round(n * val_split)) if val_split else 0
        n_train = n - n_test - n_val

        if n_train < 1:
            # Not enough transcripts for this class: shrink splits preserving train.
            if val_split and n >= 3:
                n_test, n_val, n_train = 1, 1, n - 2
            elif n >= 2:
                n_test, n_val, n_train = 1, 0, n - 1
            else:
                n_test, n_val, n_train = 0, 0, 1
            logger.warning(
                f"Class {name!r} has only {n} transcript(s) — adjusted to "
                f"train={n_train}/test={n_test}/eval={n_val}. "
                "Increase dataset size for proper stratification."
            )

        test_set.update(tids[:n_test])
        eval_set.update(tids[n_test : n_test + n_val])
        train_set.update(tids[n_test + n_val :])
        log_lines.append(
            f"  {name}: {len(tids)} transcripts → "
            f"{n_train} train / {n_test} test" + (f" / {n_val} eval" if val_split else "")
        )

    # Guard: both train and test must be populated after per-class allocation.
    if not train_set:
        raise ValueError(
            "Stratified transcript-level split yields an empty training set. "
            f"(total transcripts={len(unique)}, test_split={test_split}, "
            f"val_split={val_split}). Provide more transcripts."
        )
    if not test_set:
        raise ValueError(
            "Stratified transcript-level split yields an empty test set — every class "
            f"had too few transcripts to allocate any to test. "
            f"(total transcripts={len(unique)}, test_split={test_split}). "
            "Provide more transcripts or reduce test_split."
        )

    logger.info(
        "Stratified transcript-level split (dominant label per transcript):\n"
        + "\n".join(log_lines)
        + f"\n  total: {len(train_set)} train / {len(test_set)} test"
        + (f" / {len(eval_set)} eval" if eval_set else "")
        + " transcripts"
    )

    # num_proc=1: set-lookup filters are fast (< 1 min on 5M rows) and forking
    # workers at this stage would multiply parent RSS × num_proc in the cgroup.
    splits: Dict[str, Dataset] = {}
    splits["train"] = dataset.filter(
        lambda ex: ex["transcript_id"] in train_set, num_proc=1, desc="train split"
    )
    splits["test"] = dataset.filter(
        lambda ex: ex["transcript_id"] in test_set, num_proc=1, desc="test split"
    )
    if eval_set:
        splits["eval"] = dataset.filter(
            lambda ex: ex["transcript_id"] in eval_set, num_proc=1, desc="eval split"
        )
    return DatasetDict(splits)


# ---------------------------------------------------------------------------
# Tokenizer statistics
# ---------------------------------------------------------------------------


def _log_tokenizer_stats(
    tokenized_datasets: "DatasetDict",
    tokenizer: "PreTrainedTokenizerFast",
    task: str,
) -> None:
    """Log vocabulary coverage and sequence-length stats over all splits.

    Coverage = fraction of non-``<unk>`` tokens among all real (non-padding)
    tokens. 100 % means the tokenizer can represent every k-mer in the corpus
    without falling back to the unknown token.

    Args:
        tokenized_datasets: DatasetDict with ``input_ids`` and (optionally)
            ``attention_mask`` columns produced by the tokenizer.
        tokenizer: Tokenizer that produced ``tokenized_datasets``.
        task: Label used in the log line (e.g. ``"classification"``).
    """
    unk_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    all_lengths: list = []
    total_real = 0
    total_unk = 0

    for split_ds in tokenized_datasets.values():
        has_attn = "attention_mask" in split_ds.column_names
        for batch in split_ds.iter(batch_size=8192):
            for i, ids in enumerate(batch["input_ids"]):
                arr = np.asarray(ids, dtype=np.int32)
                real_mask = (
                    np.asarray(batch["attention_mask"][i], dtype=bool)
                    if has_attn
                    else arr != pad_id
                )
                real_len = int(real_mask.sum())
                all_lengths.append(real_len)
                total_real += real_len
                if unk_id is not None:
                    total_unk += int(((arr == unk_id) & real_mask).sum())

    lens = np.array(all_lengths, dtype=np.float64)
    coverage = 100.0 * (1.0 - total_unk / total_real) if total_real else 0.0
    logger.log(
        "STAGE",
        f"[preprocessing:{task}] tokenizer stats — "
        f"coverage={coverage:.2f}%  "
        f"(unk={total_unk:,} / {total_real:,} non-pad tokens)  |  "
        f"seq len: mean={lens.mean():.1f} ± {lens.std():.1f}  "
        f"[min={int(lens.min())}  max={int(lens.max())}]",
    )


# ---------------------------------------------------------------------------
# Fast dataset construction (replaces Dataset.from_generator for large inputs)
# ---------------------------------------------------------------------------


def _dataset_from_generator_fast(gen_fn: "Callable[[], Iterator[Dict]]") -> Dataset:
    """Build a HuggingFace Dataset from a generator ~10–100× faster than
    ``Dataset.from_generator`` for large inputs.

    ``Dataset.from_generator`` serialises rows one-at-a-time via the slow
    ``ArrowWriter`` path (≈ 1,000 rows/sec on paired FASTQ).  Collecting into
    Python lists first then calling ``Dataset.from_dict`` converts the entire
    payload to Arrow in one vectorised batch (≈ 200,000 rows/sec), which
    eliminates ≈ 2.5–3 h of wall-time for the 10 M-pair classification dataset.

    The trade-off is holding all rows in Python lists before the Arrow
    conversion.  Memory peaks at roughly ``n_rows × (avg_seq_len + label + id)``
    Python bytes, which is smaller than ``Dataset.from_generator``'s
    intermediate working set because HF's ArrowWriter also buffers in-flight.
    """
    columns: Dict[str, List] = {}
    for i, row in enumerate(gen_fn()):
        for k, v in row.items():
            if k not in columns:
                columns[k] = []
            columns[k].append(v)
        if i % 1_000_000 == 0 and i > 0:
            logger.info(f"  collected {i:,} rows …")
    logger.info(f"  converting {len(next(iter(columns.values()))):,} rows to Arrow …")
    return Dataset.from_dict(columns)


# ---------------------------------------------------------------------------
# Core dataset loader
# ---------------------------------------------------------------------------


def dataset_loader(
    builder: str = "gencode.v48.transcripts.fa.gz",
    pair: Optional[str] = None,
    file_format: str = "fasta",
    k: int = 17,
    batch_size: int = 1000,
    test_split: float = 0.1,
    val_split: Optional[float] = None,
    class_threshold: Optional[int] = None,
    num_proc: int = 4,
    split_strategy: str = "transcript-level",
    label_fn: Callable[[str], str] = _extract_label,
    drop_labels: Optional[frozenset] = None,
    dedup_priority: Optional[Sequence[str]] = None,
) -> DatasetDict:
    """Load genome sequences, build a labelled dataset, and apply a train/test split.

    Args:
        builder: Path to the FASTA/FASTQ input file.
        pair: Path to paired-end R2 file (optional).
        file_format: BioPython format string (``"fasta"`` or ``"fastq"``).
        k: K-mer size for ``kmer_split``.
        batch_size: (unused; kept for API compatibility).
        test_split: Fraction for the test set.
        val_split: Fraction for the validation set; ``None`` → skip.
        class_threshold: Minimum reads per class; classes below are dropped.
        num_proc: Worker count for ``Dataset.filter``.
        split_strategy: ``"transcript-level"`` (default, DR-6) or
            ``"read-level"`` (legacy; retains read-level leakage).
        label_fn: Maps a record id to its label. Defaults to the raw subfamily
            label; classification passes ``task_label`` for the 3-class collapse.
        drop_labels: Labels to remove before encoding (e.g. ``{"OTHER"}`` to drop
            non-target LINE-1 subfamilies from the 3-class task).
        dedup_priority: When set, a read overlapping several subfamilies (so it
            appears under several labels) is emitted once, keeping the label
            ranked first here (e.g. ``("L1HS", "L1PA", "NEGATIVE")``).

    Returns:
        ``DatasetDict`` with ``"train"``, ``"test"``, and optionally ``"eval"``
        splits.
    """
    logger.info(f"Loading sequences from {builder!r}")
    # Use lazy streaming for paired FASTQ (classification): the R1/R2 files can
    # contain 14M+ reads each. Eager loading would pre-allocate ~10 GB of Python
    # strings that then sit in memory when filter(num_proc=48) forks 48 workers,
    # causing OOM. Lazy mode streams from disk on each iteration — two file reads
    # total (once for _resolve_fragment_labels, once for the generator), which
    # is fast on Lustre scratch and keeps the parent RSS well under 80 GB.
    # FASTA inputs (masking, small GENCODE corpus) stay eager for __len__ support.
    _lazy = pair is not None and file_format == "fastq"
    raw_datasets = GenomeDataset(builder, file_format, lazy=_lazy)
    if pair is not None:
        pair_datasets = GenomeDataset(pair, file_format, lazy=_lazy)
        # Skip the count check for lazy mode: __len__ would require an extra full
        # file scan. The per-pair ID desync guard below catches mismatches anyway.
        if not _lazy and len(raw_datasets) != len(pair_datasets):
            raise ValueError(
                f"Paired FASTQs have mismatched read counts: R1={len(raw_datasets):,} "
                f"R2={len(pair_datasets):,}. The mates must align 1:1 — check the "
                "build_dataset.sh primary/secondary-alignment concatenation."
            )

        resolved = (
            _resolve_fragment_labels(raw_datasets, label_fn, dedup_priority)
            if dedup_priority is not None
            else None
        )

        def generator_from_iterator():
            emitted: set = set()
            for (r1, i1), (r2, i2) in zip(raw_datasets, pair_datasets):
                if i1 != i2 and _mate_key(i1) != _mate_key(i2):
                    raise ValueError(
                        f"R1/R2 mate desync: read_1 id {i1!r} does not pair with "
                        f"read_2 id {i2!r}. The paired FASTQs are out of order; "
                        "positional zip would mislabel every subsequent read "
                        "(check build_dataset.sh secondary-alignment handling)."
                    )
                if resolved is not None:
                    key = _fragment_key(i1)
                    if key in emitted:
                        continue
                    emitted.add(key)
                    label = resolved[key]
                else:
                    label = label_fn(i1)
                yield {
                    "read_1": r1,
                    "read_2": r2,
                    "label": label,
                    "transcript_id": _extract_transcript_id(i1),
                }

    else:
        resolved = (
            _resolve_fragment_labels(raw_datasets, label_fn, dedup_priority)
            if dedup_priority is not None
            else None
        )

        def generator_from_iterator():
            emitted: set = set()
            for seq, read_id in raw_datasets:
                if resolved is not None:
                    key = _fragment_key(read_id)
                    if key in emitted:
                        continue
                    emitted.add(key)
                    label = resolved[key]
                else:
                    label = label_fn(read_id)
                yield {
                    "sequence": seq,
                    "label": label,
                    "transcript_id": _extract_transcript_id(read_id),
                }

    logger.info("Building HuggingFace dataset from generator")
    dataset = _dataset_from_generator_fast(generator_from_iterator)

    # Use single-threaded filters here: string-in-set and int-comparison are
    # trivially fast (< 2 min on 10M rows), and forking num_proc workers at
    # this point inherits the parent's full RSS (resolved dict + Arrow table),
    # which SLURM counts per-worker → OOM at num_proc=48.  The expensive
    # kmer_split + tokenization step later gets the full num_proc budget.
    if drop_labels:
        before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["label"] not in drop_labels, num_proc=1)
        logger.info(
            f"Dropped {before - len(dataset):,} reads with labels in {set(drop_labels)} "
            f"(non-target LINE-1 subfamilies); {len(dataset):,} reads remain"
        )
    dataset = dataset.class_encode_column("label")

    if class_threshold is not None:
        label_counts = Counter(dataset["label"])
        dataset = dataset.filter(
            lambda ex: label_counts[ex["label"]] >= class_threshold,
            num_proc=1,
        )

    logger.info(f"Split strategy: {split_strategy!r}")
    if split_strategy == "transcript-level":
        transcripts_dataset = _transcript_level_split(
            dataset, test_split=test_split, val_split=val_split
        )
    else:
        # Legacy read-level split
        if val_split is not None:
            dataset = dataset.train_test_split(
                test_size=(test_split + val_split), stratify_by_column="label"
            )
            test_eval_split = dataset["test"].train_test_split(
                test_size=test_split / (test_split + val_split),
                stratify_by_column="label",
            )
            dataset["test"] = test_eval_split["train"]
            dataset["eval"] = test_eval_split["test"]
            transcripts_dataset = dataset
        else:
            transcripts_dataset = dataset.train_test_split(
                test_size=test_split, stratify_by_column="label"
            )

    return transcripts_dataset


# ---------------------------------------------------------------------------
# Typer commands (DR-3)
# ---------------------------------------------------------------------------


@app.command()
def masking(
    pretrained_model_path: Path = typer.Option(
        ..., help="Path to the pretrained tokenizer directory"
    ),
    builder: str = typer.Option("gencode.v48.transcripts.fa.gz", help="Input FASTA/FASTQ file"),
    file_format: str = typer.Option("fasta", help="BioPython format string"),
    k: int = typer.Option(17, help="K-mer size"),
    max_position_embeddings: int = typer.Option(1280),
    test_split: float = typer.Option(0.1),
    val_split: Optional[float] = typer.Option(None),
    chunk_size: Optional[int] = typer.Option(
        None,
        help="MLM chunk size (tokens); defaults to max_position_embeddings. "
        "Must be <= max_position_embeddings.",
    ),
    preprocessing_dataset: Path = typer.Option(
        ..., help="Output directory for the processed dataset"
    ),
    num_proc: int = typer.Option(32, help="Parallel workers"),
    split_strategy: str = typer.Option(
        "transcript-level",
        help="'transcript-level' (default, DR-6) or 'read-level' (legacy)",
    ),
    seed: int = typer.Option(3469, help="Global RNG seed"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """Build a masked-language-modeling dataset (tokenise + chunk)."""
    import time as _time

    if chunk_size is None:
        chunk_size = max_position_embeddings
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_size > max_position_embeddings:
        raise ValueError(
            f"chunk_size={chunk_size} exceeds max_position_embeddings={max_position_embeddings}; "
            "the model cannot process sequences longer than its position embedding table."
        )

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[preprocessing:masking] builder={builder!r} k={k} chunk_size={chunk_size} "
        f"split_strategy={split_strategy!r} out={preprocessing_dataset}",
    )
    set_global_seed(seed)
    logger.info("Loading pretrained tokenizer")
    tokenizer = load_kmer_tokenizer(pretrained_model_path, max_position_embeddings)

    transcripts_dataset = dataset_loader(
        builder=builder,
        file_format=file_format,
        k=k,
        test_split=test_split,
        val_split=val_split,
        num_proc=num_proc,
        split_strategy=split_strategy,
    )
    _split_sizes = {s: len(transcripts_dataset[s]) for s in transcripts_dataset}
    logger.log(
        "STAGE",
        f"[preprocessing:masking] split sizes — "
        + ", ".join(f"{s}={n:,}" for s, n in _split_sizes.items()),
    )

    if isinstance(tokenizer, SalmonKmerTokenizer):
        # Each canonical k-mer is one token, so whole-k-mer masking == per-token
        # masking and word_ids is just the running k-mer index.
        def tokenize_function(examples):
            out = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "word_ids": []}
            for seq in examples["sequence"]:
                ids = tokenizer.kmer_ids(seq)
                full = tokenizer.build_inputs_with_special_tokens(ids)
                out["input_ids"].append(full)
                out["attention_mask"].append([1] * len(full))
                out["token_type_ids"].append([0] * len(full))
                out["word_ids"].append([None] + list(range(len(ids))) + [None])
            return out

    else:
        # raw_read SPM tokenizers consume the raw read; legacy/k-mer tokenizers
        # consume space-joined overlapping k-mers.
        is_raw = getattr(tokenizer, "trap_raw_read", False)

        def tokenize_function(examples):
            texts = (
                examples["sequence"]
                if is_raw
                else [kmer_split(k, seq) for seq in examples["sequence"]]
            )
            result = tokenizer(
                text=texts,
                return_special_tokens_mask=False,
                truncation=False,
                verbose=False,
            )
            if tokenizer.is_fast:
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
            return result

    remove_cols = [
        c
        for c in transcripts_dataset["train"].column_names
        if c in ("sequence", "transcript_id", "label")
    ]
    _tok_proc = num_proc if isinstance(tokenizer, SalmonKmerTokenizer) else min(num_proc, 4)
    tokenized_datasets = transcripts_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
        num_proc=_tok_proc,
    )
    _log_tokenizer_stats(tokenized_datasets, tokenizer, "masking")
    out_tok = os.path.join(preprocessing_dataset, "masking", "tokenized")
    try_mkdir(out_tok)
    tokenized_datasets.save_to_disk(out_tok)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated[list(examples.keys())[0]]) // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total, chunk_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # load_from_cache_file=False: group_texts captures chunk_size as a closure variable,
    # which is invisible to the HuggingFace fingerprint hash. Without this flag, changing
    # chunk_size between runs silently returns the old cached result with the wrong block size.
    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=False
    )
    out_grouped = os.path.join(preprocessing_dataset, "masking", "grouped")
    try_mkdir(out_grouped)
    lm_datasets.save_to_disk(out_grouped)
    manifest_mod.write(
        os.path.join(preprocessing_dataset, "masking"),
        seed=seed,
        k=k,
        max_position_embeddings=max_position_embeddings,
        dataset={
            "builder": str(builder),
            "sha256_builder": manifest_mod.sha256_file(Path(builder)),
            "split_strategy": split_strategy,
            "task": "masking",
        },
    )
    _elapsed = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[preprocessing:masking] done — elapsed={_elapsed:.1f} s → {preprocessing_dataset}/masking/",
    )
    logger.success(f"MLM dataset saved to {preprocessing_dataset}/masking/")


@app.command()
def classification(
    pretrained_model_path: Path = typer.Option(
        ..., help="Path to the pretrained tokenizer directory"
    ),
    builder: str = typer.Option(..., help="Input FASTA/FASTQ file (R1 or single)"),
    pair: Optional[str] = typer.Option(None, help="Paired-end R2 file"),
    file_format: str = typer.Option("fastq", help="BioPython format string"),
    k: int = typer.Option(17, help="K-mer size"),
    max_position_embeddings: int = typer.Option(1280),
    test_split: float = typer.Option(0.2),
    val_split: Optional[float] = typer.Option(None),
    class_threshold: Optional[int] = typer.Option(10),
    padding: str = typer.Option("max_length"),
    preprocessing_dataset: Path = typer.Option(
        ..., help="Output directory for the processed dataset"
    ),
    num_proc: int = typer.Option(32, help="Parallel workers"),
    split_strategy: str = typer.Option(
        "transcript-level",
        help="'transcript-level' (default, DR-6) or 'read-level' (legacy)",
    ),
    collapse_task_labels: bool = typer.Option(
        True,
        "--collapse-task-labels/--raw-subfamily-labels",
        help="Collapse the ~130 RepeatMasker subfamilies into the 3-class task "
        "(L1HS/L1PA/NEGATIVE), dropping non-target L1s. Disable to keep raw "
        "subfamily labels.",
    ),
    seed: int = typer.Option(3469, help="Global RNG seed"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """Build a classification dataset (tokenise, stratify, save to disk)."""
    import time as _time

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[preprocessing:classification] builder={builder!r} k={k} "
        f"split_strategy={split_strategy!r} out={preprocessing_dataset}",
    )
    set_global_seed(seed)
    logger.info("Loading pretrained tokenizer")
    tokenizer = load_kmer_tokenizer(pretrained_model_path, max_position_embeddings)

    label_fn = task_label if collapse_task_labels else _extract_label
    drop_labels = frozenset({"OTHER"}) if collapse_task_labels else None
    dedup_priority = TASK_PRIORITY if collapse_task_labels else None
    if collapse_task_labels:
        logger.info(
            "Collapsing subfamilies → 3-class task (L1HS/L1PA/NEGATIVE); dropping OTHER; "
            f"multi-subfamily reads resolved by priority {TASK_PRIORITY}"
        )
    transcripts_dataset = dataset_loader(
        builder=builder,
        pair=pair,
        file_format=file_format,
        k=k,
        test_split=test_split,
        val_split=val_split,
        class_threshold=class_threshold,
        num_proc=num_proc,
        split_strategy=split_strategy,
        label_fn=label_fn,
        drop_labels=drop_labels,
        dedup_priority=dedup_priority,
    )

    _split_sizes = {s: len(transcripts_dataset[s]) for s in transcripts_dataset}
    logger.log(
        "STAGE",
        f"[preprocessing:classification] split sizes — "
        + ", ".join(f"{s}={n:,}" for s, n in _split_sizes.items()),
    )

    label_list = transcripts_dataset["train"].features["label"].names
    for split_name in ("eval", "test"):
        if split_name not in transcripts_dataset:
            continue
        diff = set(transcripts_dataset[split_name].features["label"].names) - set(label_list)
        if diff:
            logger.warning(f"Labels {diff!r} in {split_name!r} not in training set")
            label_list += list(diff)
    label_list = sorted(lbl for lbl in label_list if lbl != "-1")
    if len(label_list) <= 1:
        raise ValueError("Classification requires more than one label.")

    analyze_labels(transcripts_dataset)

    is_salmon = isinstance(tokenizer, SalmonKmerTokenizer)
    if pair is not None:
        # Keep transcript_id in the tokenised dataset so stage 21 can verify
        # train/test transcript disjointness (DR-6 leakage check); training drops
        # it in train.load_classification_data so the collator never sees it.
        remove_cols = ["read_1", "read_2"]

        if is_salmon:

            def tokenize_function(examples):
                return tokenizer.batch_encode_sequences(
                    examples["read_1"],
                    examples["read_2"],
                    max_length=max_position_embeddings,
                    padding=padding,
                    pad_to_multiple_of=8,
                    truncation=True,
                )

        else:
            is_raw = getattr(tokenizer, "trap_raw_read", False)

            def tokenize_function(examples):
                t1 = (
                    examples["read_1"]
                    if is_raw
                    else [kmer_split(k, r) for r in examples["read_1"]]
                )
                t2 = (
                    examples["read_2"]
                    if is_raw
                    else [kmer_split(k, r) for r in examples["read_2"]]
                )
                return tokenizer(
                    t1,
                    t2,
                    padding=padding,
                    pad_to_multiple_of=8,
                    truncation=True,
                    # Emit segment ids (read_1=0, read_2=1) so the classifier sees
                    # the R1/R2 boundary, matching the Salmon path; fast tokenizers
                    # otherwise omit token_type_ids and the model defaults to zeros.
                    return_token_type_ids=True,
                    verbose=False,
                )

    else:
        # transcript_id retained for the stage-21 leakage check (see paired branch).
        remove_cols = ["sequence"]

        if is_salmon:

            def tokenize_function(examples):
                return tokenizer.batch_encode_sequences(
                    examples["sequence"],
                    max_length=max_position_embeddings,
                    padding=padding,
                    pad_to_multiple_of=8,
                    truncation=True,
                )

        else:
            is_raw = getattr(tokenizer, "trap_raw_read", False)

            def tokenize_function(examples):
                texts = (
                    examples["sequence"]
                    if is_raw
                    else [kmer_split(k, seq) for seq in examples["sequence"]]
                )
                return tokenizer(
                    texts,
                    padding=padding,
                    pad_to_multiple_of=8,
                    truncation=True,
                    return_token_type_ids=True,
                    verbose=False,
                )

    remove_cols = [c for c in remove_cols if c in transcripts_dataset["train"].column_names]
    # The Salmon canonical tokenizer holds no large COW state, so it can use all
    # workers; the legacy fast tokenizer is capped because each forked worker
    # inherits the parent RSS (~30 GB for two loaded GenomeDatasets) via COW.
    _tok_proc = num_proc if is_salmon else min(num_proc, 4)
    tokenized_datasets = transcripts_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False,
        num_proc=_tok_proc,
        desc="Tokenising",
    )
    _log_tokenizer_stats(tokenized_datasets, tokenizer, "classification")
    out = os.path.join(preprocessing_dataset, "classification", "tokenized")
    try_mkdir(out)
    tokenized_datasets.save_to_disk(out)
    manifest_mod.write(
        os.path.join(preprocessing_dataset, "classification"),
        seed=seed,
        k=k,
        max_position_embeddings=max_position_embeddings,
        dataset={
            "builder": str(builder),
            "sha256_builder": manifest_mod.sha256_file(Path(builder)),
            "pair": str(pair) if pair else None,
            "sha256_pair": manifest_mod.sha256_file(Path(pair) if pair else None),
            "split_strategy": split_strategy,
            "task": "classification",
            "labels": label_list,
        },
    )
    _elapsed = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[preprocessing:classification] done — elapsed={_elapsed:.1f} s "
        f"labels={label_list} → {preprocessing_dataset}/classification/",
    )
    logger.success(f"Classification dataset saved to {preprocessing_dataset}/classification/")


if __name__ == "__main__":
    app()
