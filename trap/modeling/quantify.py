"""Streaming single-pass quantification pipeline (Phase 2, DR-1).

Architecture
------------
::

    _PairedFastqIterableDataset  (worker pool)
    ├── dnaio FASTQ reader  (worker-sharded)
    ├── kmer_split(k)
    └── fast tokenizer (encode_batch)
           │ {id, input_ids, attention_mask, token_type_ids}
           ▼
    DataLoader (pin_memory, persistent_workers, prefetch_factor=4)
           │ batches
           ▼
    PartialState.split_between_processes  OR  Accelerator.prepare
           │ bf16 autocast + torch.inference_mode
           ▼
    pyarrow.parquet.ParquetWriter  (one durably-closed segment per checkpoint)
    ├── scores_<rank>_<segment>.parquet
    ├── checkpoint_<rank>.json  (absolute resume index + next segment)
    └── manifest.json

Compared to the two-step ``processing_dataset → quantify`` path in
``predict.py``, this module:
- Requires no intermediate Arrow save/load (bounded I/O).
- Keeps host RAM flat regardless of FASTQ size.
- Supports checkpoint/restart: on ``--resume`` the IterableDataset skips
  already-written read pairs.
- Emits a ``manifest.json`` with git commit, seed, throughput metrics.

Old commands (``predict.py``) remain available under their original names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from loguru import logger
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorWithPadding
import typer

from trap.config import manifest as manifest_mod
from trap.modeling._inference import (
    build_score_schema,
    scores_table,
)
from trap.modeling._inference import autocast_ctx as _autocast_ctx
from trap.modeling._setup import load_model_and_tokenizer
from trap.utils.fastq import paired_fastq_iter
from trap.utils.kmer import kmer_split
from trap.utils.seeding import set_global_seed
from trap.utils.sequence import standardize

app = typer.Typer(help="Streaming FASTQ → classification scores (Phase 2).")


# ---------------------------------------------------------------------------
# Worker-aware IterableDataset
# ---------------------------------------------------------------------------


class _PairedFastqIterableDataset(IterableDataset):
    """Shard a paired FASTQ across Accelerate ranks **and** DataLoader workers.

    Every read pair is owned by exactly one ``(rank, worker)`` shard, selected
    by the read's **absolute** index so the assignment is stable and consistent
    across processes::

        total_shards = num_replicas * num_workers
        shard_index  = rank * num_workers + worker_id
        owned        <=>  global_idx % total_shards == shard_index

    ``num_replicas``/``rank`` come from the Accelerate ``PartialState`` (set in
    ``run``); ``num_workers``/``worker_id`` come from
    ``torch.utils.data.get_worker_info()``.  Without rank sharding every process
    would re-read the whole FASTQ and the output would be duplicated
    ``num_replicas`` times.

    Args:
        r1_path: R1 FASTQ path (gzip/bgz accepted).
        r2_path: R2 FASTQ path.
        k: K-mer size passed to ``kmer_split``.
        tokenizer: HuggingFace fast tokenizer.
        max_length: Truncation length (``model.config.max_position_embeddings``).
        num_replicas: Number of distributed processes (``PartialState.num_processes``).
        rank: This process's index (``PartialState.process_index``).
        start: Global read-pair index to resume from (checkpoint restart).
    """

    def __init__(
        self,
        r1_path: Path,
        r2_path: Path,
        k: int,
        tokenizer,
        max_length: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        start: int = 0,
    ) -> None:
        self.r1_path = Path(r1_path)
        self.r2_path = Path(r2_path)
        self.k = k
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_replicas = num_replicas
        self.rank = rank
        self.start = start

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        k = self.k
        tokenizer = self.tokenizer
        max_length = self.max_length
        start = self.start

        # Each read pair is owned by exactly one (rank, worker) shard, keyed by
        # absolute index so ranks never overlap and never duplicate.
        total_shards = self.num_replicas * num_workers
        shard_index = self.rank * num_workers + worker_id

        for global_idx, (read_id, r1_seq, r2_seq) in enumerate(
            paired_fastq_iter(self.r1_path, self.r2_path)
        ):
            if global_idx < start:
                continue
            if global_idx % total_shards != shard_index:
                continue

            # Strip non-ACTG + uppercase so k-mers match the training-time
            # vocabulary (mirrors predict.py's processing_dataset generator).
            r1_kmer = kmer_split(k, standardize(r1_seq))
            r2_kmer = kmer_split(k, standardize(r2_seq))

            enc = tokenizer(
                r1_kmer,
                r2_kmer if r2_kmer else None,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_token_type_ids=True,
            )
            # ``_abs_idx`` is the read's absolute position in the FASTQ. The
            # collator pops it back out; ``run`` uses it to checkpoint a true
            # resume offset (not a per-rank count, which would be wrong once
            # records are sharded across ranks).
            yield {"id": read_id, "_abs_idx": global_idx, **enc}


# ---------------------------------------------------------------------------
# Collator that keeps the string 'id' column outside the tensor batch
# ---------------------------------------------------------------------------


class _DataCollatorWithIds:
    """Wrapper that preserves the ``'id'`` and ``'_abs_idx'`` columns.

    Both are non-tensor metadata stripped before padding and re-attached to the
    batch as plain Python lists (``id`` → ``list[str]``, ``abs_idx`` →
    ``list[int]``).
    """

    def __init__(self, base_collator: DataCollatorWithPadding) -> None:
        self._base = base_collator

    def __call__(self, features: list) -> dict:
        ids = [f.pop("id") for f in features]
        abs_idxs = [f.pop("_abs_idx") for f in features]
        batch = self._base(features)
        batch["id"] = ids
        batch["abs_idx"] = abs_idxs
        return batch


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(output_path: Path, rank: int) -> dict:
    """Return ``{last_read_idx, n_written, next_segment}`` for *rank*.

    Defaults to a fresh-start dict when no checkpoint exists. ``last_read_idx``
    is the absolute read index to resume *at*; ``next_segment`` is the index of
    the next Parquet segment to write (earlier segments are durably closed).
    """
    ckpt = output_path / f"checkpoint_{rank}.json"
    if not ckpt.exists():
        return {"last_read_idx": 0, "n_written": 0, "next_segment": 0}
    data = json.loads(ckpt.read_text())
    return {
        "last_read_idx": data.get("last_read_idx", 0),
        "n_written": data.get("n_written", 0),
        "next_segment": data.get("next_segment", 0),
    }


def _save_checkpoint(
    output_path: Path, rank: int, last_read_idx: int, n_written: int, next_segment: int
) -> None:
    ckpt = output_path / f"checkpoint_{rank}.json"
    ckpt.write_text(
        json.dumps(
            {
                "last_read_idx": last_read_idx,
                "n_written": n_written,
                "next_segment": next_segment,
            }
        )
    )


def _remove_incomplete_segments(output_path: Path, rank: int, from_segment: int) -> None:
    """Delete this rank's Parquet segments with index >= *from_segment*.

    A crashed run leaves the in-progress segment without a Parquet footer
    (unreadable). On resume those files must be removed so postprocessing's
    ``scores_*.parquet`` glob never picks up a corrupt or duplicate shard.
    """
    seg = from_segment
    while True:
        stale = output_path / f"scores_{rank}_{seg}.parquet"
        if not stale.exists():
            break
        stale.unlink()
        logger.info(f"[rank {rank}] Removed incomplete segment {stale.name}")
        seg += 1


# ---------------------------------------------------------------------------
# ``run`` — main streaming command
# ---------------------------------------------------------------------------


@app.command()
def run(
    pretrained_model_name: Path = typer.Option(
        default=...,
        help="Model directory under MODELS_DIR (or absolute path)",
    ),
    r1: Path = typer.Option(default=..., help="R1 FASTQ (gzip/bgz accepted)"),
    r2: Path = typer.Option(default=..., help="R2 FASTQ (gzip/bgz accepted)"),
    output_path: Path = typer.Option(
        default=...,
        help="Directory for Parquet score files and manifest",
    ),
    k: int = typer.Option(17, help="K-mer size"),
    batch_size: int = typer.Option(64, help="Per-device inference batch size"),
    num_workers: int = typer.Option(4, help="DataLoader worker count"),
    checkpoint_every: int = typer.Option(
        1000, "--checkpoint-every", help="Flush Parquet + checkpoint every N batches"
    ),
    seed: int = typer.Option(3469),
    resume: bool = typer.Option(False, "--resume", help="Resume from last checkpoint"),
):
    """Single-pass streaming FASTQ → classification scores → Parquet + manifest.

    No intermediate Arrow files; host RAM stays flat; checkpoint/restart
    via ``--resume``.
    """
    from accelerate import PartialState

    set_global_seed(seed)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(pretrained_model_name)

    distributed_state = PartialState()
    device = distributed_state.device.type
    rank = distributed_state.process_index
    num_replicas = distributed_state.num_processes

    model = model.to(distributed_state.device)
    model.eval()

    if device == "cuda":
        logger.info("Compiling model (reduce-overhead)")
        model = torch.compile(model, mode="reduce-overhead")

    distributed_state.wait_for_everyone()

    # Checkpoint restart. Output is written as durably-closed segments
    # (scores_<rank>_<segment>.parquet); the checkpoint records the absolute
    # resume index and the next segment to write.
    if resume:
        ckpt = _load_checkpoint(output_path, rank)
        start_idx = ckpt["last_read_idx"]
        segment_idx = ckpt["next_segment"]
        prior_written = ckpt["n_written"]
        # The crashed run's in-progress segment has no Parquet footer; drop it
        # (and any later strays) so the glob only sees durable shards.
        _remove_incomplete_segments(output_path, rank, segment_idx)
        if start_idx:
            logger.info(
                f"[rank {rank}] Resuming from read-pair index {start_idx}, "
                f"segment {segment_idx}"
            )
    else:
        # Fresh run: clear this rank's prior shards/checkpoint so a re-run in an
        # existing output dir doesn't leave stale segments for postprocessing.
        _remove_incomplete_segments(output_path, rank, 0)
        (output_path / f"checkpoint_{rank}.json").unlink(missing_ok=True)
        start_idx, segment_idx, prior_written = 0, 0, 0

    dataset = _PairedFastqIterableDataset(
        r1,
        r2,
        k,
        tokenizer,
        model.config.max_position_embeddings,
        num_replicas=num_replicas,
        rank=rank,
        start=start_idx,
    )

    base_collator = DataCollatorWithPadding(
        tokenizer, padding="longest", pad_to_multiple_of=64, return_tensors="pt"
    )
    collator = _DataCollatorWithIds(base_collator)

    use_pin = device == "cuda"
    effective_workers = num_workers if device != "mps" else 0

    # Resume is only safe single-worker: with >1 DataLoader worker, batches from
    # different workers interleave, so the per-rank checkpoint high-water mark can
    # overshoot a slower worker and silently drop its un-flushed reads on restart.
    if resume and effective_workers > 1:
        raise typer.BadParameter(
            "--resume requires --num-workers <= 1 (multi-worker resume can drop "
            "reads because worker batches interleave). Re-run the resume with "
            "--num-workers 1, or restart the job from scratch."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=use_pin,
        num_workers=effective_workers,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=4 if effective_workers > 0 else None,
    )

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    schema = build_score_schema(labels)

    def _segment_path(seg: int) -> Path:
        return output_path / f"scores_{rank}_{seg}.parquet"

    t0 = time.perf_counter()
    session_reads = 0
    read_cursor = start_idx

    writer = pq.ParquetWriter(str(_segment_path(segment_idx)), schema)
    seg_rows = 0
    try:
        with torch.inference_mode(), _autocast_ctx(device):
            for batch_idx, batch in enumerate(loader):
                ids = batch.pop("id")
                abs_idxs = batch.pop("abs_idx")
                batch = {k_: v.to(distributed_state.device) for k_, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                n = len(probs)

                writer.write_table(scores_table(ids, probs, labels))
                seg_rows += n
                session_reads += n
                # Checkpoint the absolute resume offset: the next read index after
                # the furthest one flushed by this rank (single-worker => safe).
                read_cursor = max(read_cursor, max(abs_idxs) + 1)

                if (batch_idx + 1) % checkpoint_every == 0:
                    # Close the current segment so it has a valid Parquet footer
                    # (durable), record progress, then rotate to a new segment.
                    writer.close()
                    segment_idx += 1
                    _save_checkpoint(
                        output_path,
                        rank,
                        read_cursor,
                        prior_written + session_reads,
                        segment_idx,
                    )
                    writer = pq.ParquetWriter(str(_segment_path(segment_idx)), schema)
                    seg_rows = 0
                    elapsed = time.perf_counter() - t0
                    rps = session_reads / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"[rank {rank}] batch {batch_idx + 1}: "
                        f"{session_reads:,} reads this session, {rps:,.0f} reads/s"
                    )
    finally:
        writer.close()

    # Drop a trailing empty segment (e.g. resume with nothing new); otherwise
    # the just-closed segment is durable and the next free index advances.
    if seg_rows == 0:
        _segment_path(segment_idx).unlink(missing_ok=True)
        next_segment = segment_idx
    else:
        next_segment = segment_idx + 1
    total_reads = prior_written + session_reads
    # Final checkpoint marks every written segment durable so a later --resume
    # does not treat the last segment as incomplete.
    _save_checkpoint(output_path, rank, read_cursor, total_reads, next_segment)

    elapsed = time.perf_counter() - t0
    rps = session_reads / elapsed if elapsed > 0 else 0.0
    logger.success(
        f"[rank {rank}] Done: {session_reads:,} reads this session "
        f"({total_reads:,} total) in {elapsed:.1f}s ({rps:,.0f} reads/s) "
        f"→ {next_segment} segment(s) under {output_path}"
    )

    # Write manifest on main process only (other ranks wait first)
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        manifest_path = manifest_mod.write(
            output_path,
            seed=seed,
            k=k,
            model={"path": str(pretrained_model_name)},
            dataset={"r1": str(r1), "r2": str(r2)},
            throughput={
                "reads_per_s": round(rps, 1),
                "total_reads": total_reads,
                "elapsed_s": round(elapsed, 2),
            },
        )
        logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    app()
