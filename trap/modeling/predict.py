"""Inference / quantification entry points for TrAP.

Phase-1 quick wins applied
--------------------------
QW-1   processing_dataset: flat generator {r1_seq, r2_seq, id} + batched=True
        tokenise map (kmer_split runs in parallel via num_proc).
QW-4   save_processing defaults to False; no intermediate Arrow unless asked.
QW-5   quantify: DataLoader + torch.inference_mode + torch.autocast.
QW-6   Dynamic padding (padding=False in tokenise + DataCollatorWithPadding
        with pad_to_multiple_of=64 at inference).
QW-8   quantify: per-rank Parquet writer (id, L1HS, L1PA, NEGATIVE); no pickle
        accumulation in host RAM.
QW-9   torch.compile only on CUDA (mode=reduce-overhead); skipped on MPS/CPU.
QW-10  TemplateProcessing no longer re-applied at runtime in quantify/*
        (it is baked into the tokenizer at save time by trap/loaders/tokenizer.py).
        processing_dataset still applies it as a backward-compat safety net.
QW-11  break_long_read accepts an optional numpy Generator for seeded runs.
QW-12  SAMDataset streams from pysam instead of loading the whole file into RAM.

Phase-2 changes (DR-2)
-----------------------
SAMDataset    → imported from trap.loaders.alignments (canonical module).
break_long_read → imported from trap.utils.sequence (canonical module).
standardize   → imported from trap.utils.sequence (strips N; k=17/v48 decision).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from Bio import SeqIO
from accelerate import Accelerator, PartialState
from datasets import Dataset, load_from_disk
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
import typer

from trap.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from trap.config.verbosity import set_verbosity
from trap.loaders.alignments import SAMDataset
from trap.loaders.dataset import GenomeDataset
from trap.loaders.salmon_tokenizer import SalmonKmerTokenizer
from trap.loaders.tokenizer import load_kmer_tokenizer
from trap.modeling._inference import (
    autocast_ctx,
    build_score_schema,
    resolve_device,
    scores_table,
)
from trap.modeling.train import _resolve_precision
from trap.utils.io import genome_file_handle, try_mkdir
from trap.utils.kmer import kmer_split
from trap.utils.seeding import set_global_seed
from trap.utils.sequence import break_long_read, reverse_complement
from trap.utils.sequence import standardize as _standardize_sequence

# Back-compat aliases — these helpers moved to trap.modeling._inference.
_autocast_ctx = autocast_ctx
_resolve_device = resolve_device

START_TIME = time.strftime("%Y%m%d_%H%M%S")

app = typer.Typer()

debug_mode = False


def debug_callback(debug: bool = typer.Option(False, "--debug", "-d")):
    global debug_mode
    if debug:
        typer.echo("Debug mode enabled")
        debug_mode = True


# Device/dtype helpers (resolve_device, autocast_ctx) and the Parquet score
# helpers (build_score_schema, scores_table) now live in trap.modeling._inference;
# _autocast_ctx / _resolve_device aliases above keep existing imports working.


# NOTE: SAMDataset and break_long_read are now canonical in
# trap.loaders.alignments and trap.utils.sequence respectively.
# They are re-exported here for backward compatibility with existing callers.


# ---------------------------------------------------------------------------
# Stub train commands (implemented in train.py; kept here for CLI discoverability)
# ---------------------------------------------------------------------------


@app.command()
def masking(
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(
        default=..., help="Path to the pretrained tokenizer"
    ),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_base_uncased.json"),
    albert_config_path: Path = typer.Option(CONFIG_DIR / "albert_config_base_uncased.json"),
    builder: str = typer.Option(None),
    k: int = typer.Option(17, help="K-mer size"),
    test_split: float = typer.Option(0.1),
    chunk_size: int = typer.Option(128),
    preprocessing_name: str = typer.Option("gencode.v48.transcripts.k17.32k"),
    num_workers: int = typer.Option(16),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    pass


@app.command()
def classification(
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(default=None),
    pretrained_model_path: Path = typer.Option(default=None),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_base_repeatmasker.json"),
    albert_config_path: Path = typer.Option(CONFIG_DIR / "albert_config_base_uncased.json"),
    builder: str = typer.Option(None),
    k: int = typer.Option(17, help="K-mer size"),
    test_split: float = typer.Option(0.1),
    chunk_size: int = typer.Option(128),
    preprocessing_name: str = typer.Option("gencode.v48.transcripts.k17.32k"),
    num_workers: int = typer.Option(16),
    do_eval: bool = typer.Option(False),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    import json

    import evaluate
    from transformers import AlbertForSequenceClassification, Trainer, TrainingArguments

    debug_callback(debug)
    if not trainer_config_path.exists():
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Trainer config not found.")

    logger.info("Loading pretokenized dataset")
    lm_datasets = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "classification")
    )
    if debug_mode:
        train_size = 1_000
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=int(0.1 * train_size), seed=42
        )
        lm_datasets["eval"] = lm_datasets["train"].train_test_split(
            test_size=int(0.1 * train_size), seed=42
        )["test"]

    model = AlbertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_path, "final")
    )
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

    tokenizer = load_kmer_tokenizer(
        os.path.join(MODELS_DIR, pretrained_model_path, "final"),
        model.config.max_position_embeddings,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64)
    trainer_args = TrainingArguments(**_resolve_precision(json.load(open(trainer_config_path))))
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="micro")["f1"],
            "precision": precision.compute(
                predictions=predictions, references=labels, average="micro"
            )["precision"],
            "recall": recall.compute(predictions=predictions, references=labels, average="micro")[
                "recall"
            ],
        }

    try_mkdir(trainer_args.output_dir)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if do_eval:
        metrics = trainer.evaluate(eval_dataset=lm_datasets["eval"])
        metrics["eval_samples"] = len(lm_datasets["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


# ---------------------------------------------------------------------------
# processing_dataset  (QW-1 / QW-4 / QW-6 / QW-10)
# ---------------------------------------------------------------------------


@app.command()
def processing_dataset(
    input_file: Path = typer.Option(None, help="Path to R1 / single FASTQ or BAM/SAM"),
    pair_file: Path = typer.Option(None, help="Path to R2 FASTQ (paired-end)"),
    output_path: Path = typer.Option(None, help="Base output directory"),
    k: int = typer.Option(17, help="K-mer size"),
    save_processing: bool = typer.Option(False, help="Save intermediate (pre-tokenised) dataset"),
    pretrained_tokenizer_name: Path = typer.Option(
        None, help="Model directory containing tokenizer"
    ),
    padding: bool = typer.Option(
        False, help="Pad sequences to max_length (default: dynamic padding)"
    ),
    is_long: bool = typer.Option(False, "--is-long", "-l", help="Input is long-read sequencing"),
    num_workers: int = typer.Option(16, help="Parallel workers for Dataset.map"),
    seed: int = typer.Option(3469, help="Random seed for long-read fragmentation"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """Tokenise a FASTQ/BAM into a HuggingFace Dataset saved to disk.

    Phase-1 changes
    ---------------
    * Generator now yields raw sequences ``{r1_seq, r2_seq, id}`` instead of
      pre-split k-mer strings.  kmer_split runs *inside* the tokenise map so
      it benefits from ``num_proc`` parallelism (QW-1).
    * ``id`` column is kept in the tokenised dataset (was removed before,
      forcing postprocessing to re-join by position).
    * ``batched=True`` in Dataset.map — activates the HF Rust tokeniser's
      batch path (QW-1).
    * ``padding=False`` by default; DataCollatorWithPadding handles padding
      at inference time (QW-6).
    """
    import time as _time

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[predict:processing_dataset] input={input_file} k={k} out={output_path}",
    )

    # Use canonical standardize (strips N; k=17/v48 decision 2026-05-28).
    standardization = _standardize_sequence
    fragmentation_rng = np.random.default_rng(seed) if is_long else None
    break_fn = (lambda seq: break_long_read(seq, rng=fragmentation_rng)) if is_long else None

    # --- build generator ---
    # Detect format by the filename ending, not Path.suffixes: bioinformatics names
    # carry dots in the stem (e.g. ...delprob_0.025.pair.5x1.fq.gz), which would make
    # "".join(suffixes) collect every segment and never match.
    name = input_file.name.lower()
    if name.endswith((".fq.gz", ".fastq.gz", ".fq.bgz", ".fastq.bgz", ".fq", ".fastq")):
        if pair_file is not None:
            # Paired-end FASTQ (hot path) — raw sequences, no k-mer split yet
            def generator_from_iterator():
                with genome_file_handle(input_file) as h1, genome_file_handle(pair_file) as h2:
                    for r1, r2 in zip(SeqIO.parse(h1, "fastq"), SeqIO.parse(h2, "fastq")):
                        yield {
                            "r1_seq": standardization(str(r1.seq)),
                            "r2_seq": standardization(str(r2.seq)),
                            "id": str(r1.id),
                        }

        else:
            # Single FASTQ — synthesize the paired-end input as
            # (read, reverse_complement(read)) to match the classifier's training
            # format. The revcomp is computed per read on demand (GenomeDataset no
            # longer precomputes a second strand for every record).
            # lazy=True streams reads from disk instead of pre-loading every
            # sequence string into RAM. A release-size single-end FASTQ has 14M+
            # reads; eager loading would hold ~10 GB of Python strings in the
            # parent RSS, which Dataset.from_generator's fork then multiplies in
            # the SLURM cgroup (same CoW-OOM resolved for stage-20 preprocessing).
            raw_reads = GenomeDataset(input_file, "fastq", lazy=True)

            def generator_from_iterator():
                for seq, read_id in raw_reads:
                    if break_fn is not None:
                        for pair in break_fn(seq):
                            yield {
                                "r1_seq": standardization(pair["forward"]),
                                "r2_seq": standardization(pair["reverse"]),
                                "id": read_id,
                            }
                    else:
                        std_seq = standardization(seq)
                        yield {
                            "r1_seq": std_seq,
                            "r2_seq": reverse_complement(std_seq),
                            "id": read_id,
                        }

    elif name.endswith((".sam", ".bam", ".cram")):
        raw_alignments = SAMDataset(input_file)

        def generator_from_iterator():
            for read_id, seq in raw_alignments:
                if break_fn is not None:
                    for pair in break_fn(seq):
                        yield {
                            "r1_seq": standardization(pair["forward"]),
                            "r2_seq": standardization(pair["reverse"]),
                            "id": read_id,
                        }
                else:
                    yield {"r1_seq": standardization(seq), "r2_seq": "", "id": read_id}

    else:
        raise ValueError(f"Unsupported file format: {input_file.name}")

    logger.info("Building dataset from generator")
    raw_dataset = Dataset.from_generator(generator_from_iterator)
    logger.log("STAGE", f"[predict:processing_dataset] loaded {len(raw_dataset):,} records")
    logger.info(f"Generator produced {len(raw_dataset)} records")
    try_mkdir(os.path.join(output_path, "short_reads"))

    if save_processing or pretrained_tokenizer_name is None:
        raw_dataset.save_to_disk(os.path.join(output_path, "short_reads"), num_proc=num_workers)
        logger.info(f"Saved raw dataset to {output_path}/short_reads")

    if pretrained_tokenizer_name is not None:
        logger.info("Loading tokenizer")
        # Resolve the tokenizer dir flexibly: an explicit path, a standalone tokenizer
        # saved at models/<name>/, or a model dir bundling it at models/<name>/final/.
        tok_arg = str(pretrained_tokenizer_name)
        candidates = [
            tok_arg,                                       # explicit path
            os.path.join(MODELS_DIR, tok_arg, "final"),    # model dir bundling a tokenizer
            os.path.join(MODELS_DIR, tok_arg),             # standalone tokenizer models/<name>/
        ]
        tok_path = next((c for c in candidates if os.path.isdir(c)), candidates[1])
        tokenizer = load_kmer_tokenizer(tok_path)

        _k = k  # capture for closure

        _is_salmon = isinstance(tokenizer, SalmonKmerTokenizer)

        def tokenize(examples):
            """kmer_split + tokenise in one batched map step (QW-1, QW-3)."""
            pad = "max_length" if padding else False
            if _is_salmon:
                r2 = examples["r2_seq"] if any(examples["r2_seq"]) else None
                return tokenizer.batch_encode_sequences(
                    examples["r1_seq"],
                    r2,
                    max_length=tokenizer.model_max_length,
                    padding=pad,
                    truncation=True,
                )
            r1_kmers = [kmer_split(_k, s) for s in examples["r1_seq"]]
            r2_kmers = [kmer_split(_k, s) for s in examples["r2_seq"]]
            return tokenizer(
                r1_kmers,
                r2_kmers if any(r2_kmers) else None,
                padding=pad,
                truncation=True,
                verbose=False,
            )

        logger.info("Tokenising dataset (batched=True)")
        tokenized = raw_dataset.map(
            tokenize,
            batched=True,
            batch_size=256,
            remove_columns=["r1_seq", "r2_seq"],  # keep 'id'
            num_proc=num_workers,
            load_from_cache_file=False,
            desc="Tokenising",
        )
        logger.info(f"Tokenised {len(tokenized)} records")
        try_mkdir(os.path.join(output_path, "short_reads", "tokenized"))
        tokenized.save_to_disk(os.path.join(output_path, "short_reads", "tokenized"))
        _elapsed = _time.perf_counter() - _t0
        logger.log(
            "STAGE",
            f"[predict:processing_dataset] done — {len(tokenized):,} records "
            f"elapsed={_elapsed:.1f} s → {output_path}/short_reads/tokenized/",
        )
        logger.info(f"Saved tokenised dataset to {output_path}/short_reads/tokenized")


# ---------------------------------------------------------------------------
# quantify  (QW-5 / QW-6 / QW-8 / QW-9 / QW-10)
# ---------------------------------------------------------------------------


@app.command()
def quantify(
    pretrained_model_name: Path = typer.Option(
        default=None, help="Model directory under MODELS_DIR"
    ),
    output_path: Path = typer.Option(None, help="Directory containing the tokenised dataset"),
    batch_size: int = typer.Option(64, help="Per-device inference batch size"),
    num_shards: int = typer.Option(None),
    shards_index: int = typer.Option(0),
    is_tokenized: bool = typer.Option(False),
    num_workers: int = typer.Option(4, help="DataLoader worker processes"),
    seed: int = typer.Option(3469),
    debug: bool = typer.Option(False, "--debug", "-d"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """Classify tokenised reads and write per-rank Parquet score files.

    Phase-1 changes
    ---------------
    * DataLoader with ``pin_memory`` and ``prefetch_factor`` replaces
      ``pipeline(iter(batch_size=1))`` (QW-5).
    * ``torch.inference_mode`` + ``torch.autocast`` (QW-5).
    * ``torch.compile(mode="reduce-overhead")`` on CUDA only (QW-9).
    * Parquet output per rank; no in-process accumulation (QW-8).
    * TemplateProcessing not re-applied (baked into tokenizer; QW-10).
    """
    import time as _time

    import pyarrow.parquet as pq
    from transformers import AlbertForSequenceClassification

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[predict:quantify] model={pretrained_model_name} out={output_path}",
    )
    debug_callback(debug)
    set_global_seed(seed)

    logger.info("Loading model")
    model = AlbertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"),
        attn_implementation="sdpa",
    )
    logger.log("STAGE", f"[predict:quantify] model parameters: {model.num_parameters() / 1e6:.0f}M")
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

    logger.info("Loading tokenizer")
    tokenizer = load_kmer_tokenizer(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"),
        model.config.max_position_embeddings,
    )

    logger.info("Loading dataset")
    if is_tokenized:
        dataset = load_from_disk(os.path.join(output_path, "short_reads", "tokenized"))
        if num_shards is not None:
            dataset = dataset.shard(num_shards=num_shards, index=shards_index)
    else:
        dataset = load_from_disk(os.path.join(output_path, "short_reads"))

    if debug_mode:
        dataset = dataset.select(range(min(1_000, len(dataset))))

    distributed_state = PartialState()
    device = distributed_state.device.type  # 'cuda', 'mps', or 'cpu'

    model = model.to(distributed_state.device)
    model.eval()

    # Compile on CUDA only; inductor is not available on MPS/CPU (QW-9).
    if device == "cuda":
        logger.info("Compiling model (reduce-overhead)")
        model = torch.compile(model, mode="reduce-overhead")

    if distributed_state.is_main_process:
        try_mkdir(output_path)

    distributed_state.wait_for_everyone()

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    schema = build_score_schema(labels)

    collator = DataCollatorWithPadding(
        tokenizer, padding="longest", pad_to_multiple_of=64, return_tensors="pt"
    )

    with distributed_state.split_between_processes(dataset) as shard:
        # Snapshot IDs before setting torch format (string columns drop out).
        shard_ids = (
            list(shard["id"])
            if "id" in shard.column_names
            else [str(i) for i in range(len(shard))]
        )
        tensor_cols = [c for c in shard.column_names if c != "id"]
        shard.set_format("torch", columns=tensor_cols)

        use_pin = device == "cuda"
        use_workers = num_workers if device != "mps" else 0  # fork safety on Mac
        loader = DataLoader(
            shard,
            batch_size=batch_size,
            collate_fn=collator,
            pin_memory=use_pin,
            num_workers=use_workers,
            persistent_workers=(use_workers > 0),
            prefetch_factor=4 if use_workers > 0 else None,
        )

        parquet_path = (
            Path(output_path) / f"scores_{distributed_state.process_index}_{shards_index}.parquet"
        )

        with pq.ParquetWriter(str(parquet_path), schema) as writer:
            id_cursor = 0
            with torch.inference_mode(), _autocast_ctx(device):
                for batch in tqdm(
                    loader, desc=f"[rank {distributed_state.process_index}] classifying"
                ):
                    batch = {k: v.to(distributed_state.device) for k, v in batch.items()}
                    logits = model(**batch).logits
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                    n = len(probs)
                    batch_ids = shard_ids[id_cursor : id_cursor + n]
                    id_cursor += n
                    writer.write_table(scores_table(batch_ids, probs, labels))

    _elapsed = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[predict:quantify] done — rank={distributed_state.process_index} "
        f"elapsed={_elapsed:.0f} s → {parquet_path}",
    )
    logger.info(f"Scores written to {parquet_path}")


# ---------------------------------------------------------------------------
# cpu_quantify — same as quantify but forces CPU
# ---------------------------------------------------------------------------


@app.command()
def cpu_quantify(
    pretrained_model_name: Path = typer.Option(default=None),
    output_path: Path = typer.Option(None),
    batch_size: int = typer.Option(64),
    num_shards: int = typer.Option(None),
    shards_index: int = typer.Option(0),
    is_tokenized: bool = typer.Option(False),
    num_workers: int = typer.Option(0),
    seed: int = typer.Option(3469),
    debug: bool = typer.Option(False, "--debug", "-d"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """CPU-only quantification (no Accelerate distributed state)."""
    import time as _time

    import pyarrow.parquet as pq
    from transformers import AlbertForSequenceClassification

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log("STAGE", f"[predict:cpu_quantify] model={pretrained_model_name} out={output_path}")
    debug_callback(debug)
    set_global_seed(seed)

    model = AlbertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"), attn_implementation="sdpa"
    )
    model.eval()
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

    tokenizer = load_kmer_tokenizer(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"),
        model.config.max_position_embeddings,
    )

    if is_tokenized:
        dataset = load_from_disk(os.path.join(output_path, "short_reads", "tokenized"))
        if num_shards is not None:
            dataset = dataset.shard(num_shards=num_shards, index=shards_index)
    else:
        dataset = load_from_disk(os.path.join(output_path, "short_reads"))

    if debug:
        dataset = dataset.select(range(min(1_000, len(dataset))))

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    schema = build_score_schema(labels)
    collator = DataCollatorWithPadding(
        tokenizer, padding="longest", pad_to_multiple_of=64, return_tensors="pt"
    )

    shard_ids = (
        list(dataset["id"])
        if "id" in dataset.column_names
        else [str(i) for i in range(len(dataset))]
    )
    tensor_cols = [c for c in dataset.column_names if c != "id"]
    dataset.set_format("torch", columns=tensor_cols)

    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers
    )
    parquet_path = Path(output_path) / f"scores_cpu_{shards_index}.parquet"
    try_mkdir(output_path)

    with pq.ParquetWriter(str(parquet_path), schema) as writer:
        id_cursor = 0
        with torch.inference_mode():
            for batch in tqdm(loader, desc="classifying (CPU)"):
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                n = len(probs)
                writer.write_table(
                    scores_table(shard_ids[id_cursor : id_cursor + n], probs, labels)
                )
                id_cursor += n

    _elapsed_cpu = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[predict:cpu_quantify] done — elapsed={_elapsed_cpu:.0f} s → {parquet_path}",
    )
    logger.info(f"Scores written to {parquet_path}")


# ---------------------------------------------------------------------------
# accelerate_quantify  (QW-9 / QW-10)
# ---------------------------------------------------------------------------


@app.command()
def accelerate_quantify(
    pretrained_model_name: Path = typer.Option(default=None),
    output_path: Path = typer.Option(None),
    batch_size: int = typer.Option(64),
    num_workers: int = typer.Option(4),
    seed: int = typer.Option(3469),
    debug: bool = typer.Option(False, "--debug", "-d"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
):
    """Multi-GPU inference via Accelerate prepare + DataLoader."""
    import time as _time

    import pyarrow.parquet as pq
    from transformers import AlbertForSequenceClassification

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    accelerator = Accelerator()
    logger.log(
        "STAGE",
        f"[predict:accelerate_quantify] model={pretrained_model_name} "
        f"num_processes={accelerator.num_processes}",
    )
    set_global_seed(seed)

    with accelerator.main_process_first():
        debug_callback(debug)
        model = AlbertForSequenceClassification.from_pretrained(
            os.path.join(MODELS_DIR, pretrained_model_name, "final"), attn_implementation="sdpa"
        )
        logger.info(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

        # Compile on CUDA only (QW-9)
        if accelerator.device.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead")

        tokenizer = load_kmer_tokenizer(
            os.path.join(MODELS_DIR, pretrained_model_name, "final"),
            model.config.max_position_embeddings,
        )

        dataset = load_from_disk(os.path.join(output_path, "short_reads", "tokenized"))
        if debug_mode:
            dataset = dataset.select(range(min(1_000, len(dataset))))

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    collator = DataCollatorWithPadding(
        tokenizer, padding="longest", pad_to_multiple_of=64, return_tensors="pt"
    )

    # Shard the dataset across processes ourselves, then snapshot IDs from the
    # shard. Do NOT pass the loader through accelerator.prepare(): that applies a
    # second distributed sampler, which would desync the per-rank batches from
    # the full ordered `shard_ids` list and assign scores to the wrong reads.
    dataset = dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
    shard_ids = (
        list(dataset["id"])
        if "id" in dataset.column_names
        else [str(i) for i in range(len(dataset))]
    )
    tensor_cols = [c for c in dataset.column_names if c != "id"]
    dataset.set_format("torch", columns=tensor_cols)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )
    model = accelerator.prepare(model)
    model.eval()

    schema = build_score_schema(labels)
    parquet_path = Path(output_path) / f"scores_accel_{accelerator.process_index}.parquet"
    try_mkdir(output_path)

    with pq.ParquetWriter(str(parquet_path), schema) as writer:
        id_cursor = 0
        with torch.inference_mode(), _autocast_ctx(accelerator.device.type):
            for batch in tqdm(loader, desc=f"[rank {accelerator.process_index}] classifying"):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                n = len(probs)
                writer.write_table(
                    scores_table(shard_ids[id_cursor : id_cursor + n], probs, labels)
                )
                id_cursor += n

    accelerator.wait_for_everyone()
    _elapsed_accel = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[predict:accelerate_quantify] done — rank={accelerator.process_index} "
        f"elapsed={_elapsed_accel:.0f} s → {parquet_path}",
    )
    logger.info(f"Scores written to {parquet_path}")


# ---------------------------------------------------------------------------
# ONNX export (unchanged)
# ---------------------------------------------------------------------------


@app.command()
def convert_model_to_onnx(
    pretrained_model_name: Path = typer.Option(default=None),
    opset: int = 14,
    use_auth_token: bool = False,
):
    """Export model to ONNX via Optimum."""
    from optimum.exporters.onnx import main_export

    output_dir = os.path.join(MODELS_DIR, pretrained_model_name, "onnx")
    try_mkdir(output_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"), local_files_only=True
    )
    main_export(
        model_name_or_path=os.path.join(MODELS_DIR, pretrained_model_name, "final"),
        output=output_dir,
        task="text-classification",
        opset=opset,
        tokenizer=tokenizer,
        trust_remote_code=True,
        use_auth_token=use_auth_token,
    )
    logger.info(f"ONNX model exported to {output_dir}")


if __name__ == "__main__":
    app()
