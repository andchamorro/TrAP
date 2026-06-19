"""Diagnose whether the tokenisation carries learnable MLM signal (Gate 2).

Runs two decisive checks on the REAL tokenizer + data + model:

  1. Input-dependence: forward two different real sequences and measure how much
     the logits change. If the output barely depends on the input, the encoder
     signal is being washed out (a wiring bug) and the model can only ever
     predict the marginal token distribution -> loss pinned at ln(vocab).

  2. Overfit capacity: try to memorise a handful of real sequences with a high
     LR for a few hundred steps. If the loss cannot drop well below the unigram
     floor even on 16 examples, the tokenisation carries no context-conditional
     signal (the Salmon-hash failure mode). If it CAN, MLM is learnable.

This is the SPM-vs-Salmon test: the Salmon tokenizer feature-hashes k-mers
through a non-invertible avalanche, so a masked token is unpredictable from its
neighbours and the loss collapses to H(unigram). Content-preserving SPM pieces
keep neighbour sequence, so masked-piece prediction should beat that floor.

Two data sources:
  * ``--corpus`` (FASTA/FASTQ): tokenise N sequences in memory — no stage-20
    masking dataset needed. Raw-read SPM tokenizers are auto-detected (fed the
    raw read, standard subword MLM masking); k-mer tokenizers get whole-k-mer
    masking. This is the path for the SPM exploration.
  * ``--preprocessing-name``: load a pre-built ``masking/grouped`` dataset
    (the original Salmon path).

Usage (single GPU; CPU works for a tiny --n-seqs):
    srun --gres=gpu:a100:1 --mem=64G --time=00:20:00 --pty bash
    conda activate trap
    TRAP_SPM_EXPERIMENTAL=1 python scripts/diagnose_mlm_learnability.py \
        --tokenizer-path models/tokenizer.gencode.v48.k17.spm \
        --corpus data/external/gencode.v48.transcripts.fa.gz --file-format fasta \
        --albert-config-path config/albert_config_k17_v48.json
"""

import math
import os
from pathlib import Path
from typing import Optional

from loguru import logger
import torch
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    DataCollatorForLanguageModeling,
)
import typer

from trap.config.config import PROCESSED_DATA_DIR
from trap.loaders.dataset import GenomeDataset
from trap.loaders.tokenizer import (
    WholeKmerMaskingDataCollator,
    load_kmer_tokenizer,
    spm_input_mode,
)
from trap.utils.kmer import kmer_split

app = typer.Typer(add_completion=False)


def _make_features(pool, idxs):
    """Copy selected rows from a Dataset or a list of dicts into plain dicts."""
    keep = ("input_ids", "attention_mask", "token_type_ids", "word_ids", "labels")
    feats = []
    for i in idxs:
        row = pool[int(i)]
        feats.append({k: list(row[k]) for k in keep if k in row})
    return feats


def _build_corpus_pool(corpus, file_format, tokenizer, raw_read, k, max_len, n_data):
    """Tokenise up to ``n_data`` sequences in memory (no stage-20 dataset)."""
    seqs = GenomeDataset(corpus, file_format).sequences
    seqs = seqs[: min(n_data, len(seqs))]
    pool = []
    for s in seqs:
        if len(s) < k:
            continue
        text = s if raw_read else kmer_split(k, s)
        enc = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=True)
        pool.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return pool


@app.command()
def run(
    tokenizer_path: Path = typer.Option(..., help="Tokenizer dir (SPM, Salmon, …)"),
    albert_config_path: Path = typer.Option("config/albert_config_k17_v48.json"),
    corpus: Optional[Path] = typer.Option(
        None, help="FASTA/FASTQ to tokenise in memory (preferred for SPM)."
    ),
    file_format: str = typer.Option("fasta", help="BioPython format for --corpus."),
    preprocessing_name: Optional[str] = typer.Option(
        None, help="Pre-built masking dataset (alternative to --corpus, Salmon path)."
    ),
    k: int = typer.Option(17, help="K-mer size (k-mer tokenizers only)."),
    n_data: int = typer.Option(2000, help="Sequences to load into the pool (--corpus)."),
    n_seqs: int = typer.Option(16, help="Sequences to attempt to overfit"),
    steps: int = typer.Option(400),
    lr: float = typer.Option(1e-3),
    mlm_probability: float = typer.Option(0.15, help="Subword MLM mask probability."),
    fixed_mask: bool = typer.Option(
        False,
        help="Mask the SAME positions every step (collate once, reuse). Fixed-mask "
        "memorisation isolates optimisation/wiring from the random-masking draw: if "
        "even this stalls at the unigram floor the bottleneck is structural, not the "
        "per-step masking.",
    ),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AlbertConfig.from_json_file(str(albert_config_path))
    tokenizer = load_kmer_tokenizer(str(tokenizer_path), cfg.max_position_embeddings)
    cfg.vocab_size = len(tokenizer)
    uniform = math.log(cfg.vocab_size)
    raw_read, tok_k = spm_input_mode(tokenizer)
    if getattr(tokenizer, "trap_k", None) is not None:
        k = tok_k
    logger.info(
        f"device={device} vocab={cfg.vocab_size} ln(vocab)={uniform:.3f} "
        f"mode={'raw-read' if raw_read else 'kmer'} k={k}"
    )

    # A subword (raw-read) tokenizer needs standard per-token MLM masking; the
    # whole-k-mer collator masks whole 'words', and a raw read is one word.
    use_standard_mlm = raw_read or corpus is not None
    if use_standard_mlm:
        if corpus is None:
            raise typer.BadParameter("Raw-read/SPM mode requires --corpus.")
        pool = _build_corpus_pool(
            str(corpus),
            file_format,
            tokenizer,
            raw_read,
            k,
            cfg.max_position_embeddings,
            n_data,
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
        )
        logger.info(f"[data] tokenised {len(pool):,} seqs in memory; standard MLM masking")
    else:
        from datasets import load_from_disk

        if preprocessing_name is None:
            raise typer.BadParameter("Provide --corpus or --preprocessing-name.")
        pool = load_from_disk(
            os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "masking", "grouped")
        )["train"]
        collator = WholeKmerMaskingDataCollator(tokenizer)

    model = AlbertForMaskedLM(cfg).to(device)

    # --- Check 1: does the output depend on the input? -----------------------
    model.eval()
    a = collator(_make_features(pool, [0]))
    b = collator(_make_features(pool, [1]))
    with torch.no_grad():
        la = model(**{k_: v.to(device) for k_, v in a.items() if k_ != "labels"}).logits
        lb = model(**{k_: v.to(device) for k_, v in b.items() if k_ != "labels"}).logits
    n = min(la.shape[1], lb.shape[1])
    delta = (la[0, :n] - lb[0, :n]).abs().mean().item()
    logger.info(
        f"[input-dependence] mean|logits(A)-logits(B)|={delta:.4e}  "
        f"logits_std={la.std().item():.4e}  (≈0 => output ignores the input)"
    )

    # --- data sanity: how diverse are the masked labels? ---------------------
    masked = collator(_make_features(pool, list(range(min(n_seqs, len(pool))))))
    lbl = masked["labels"]
    n_masked = int((lbl != -100).sum())
    n_unique = int(torch.unique(lbl[lbl != -100]).numel()) if n_masked else 0
    logger.info(
        f"[data] over {min(n_seqs, len(pool))} seqs: masked_positions={n_masked} "
        f"unique_masked_tokens={n_unique} seq_len={lbl.shape[1]}"
    )

    # --- Check 2: can it overfit n_seqs real sequences? ----------------------
    # The bar is NOT ln(vocab): a model that learns only the marginal token
    # frequency (the unigram prior) already drops to ln(#unique masked tokens)
    # without using any context. Real memorisation must beat that floor by a
    # clear margin, otherwise the model has merely collapsed to the unigram.
    unigram_floor = math.log(max(n_unique, 2))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idxs = list(range(min(n_seqs, len(pool))))
    fixed_batch = None
    if fixed_mask:
        fixed_batch = {k_: v.to(device) for k_, v in collator(_make_features(pool, idxs)).items()}
    logger.info(
        f"[overfit] memorising {len(idxs)} real seqs, lr={lr}, steps={steps}, "
        f"fixed_mask={fixed_mask}; unigram_floor=ln({n_unique})={unigram_floor:.3f}"
    )
    out = None
    for step in range(steps):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            batch = {k_: v.to(device) for k_, v in collator(_make_features(pool, idxs)).items()}
        out = model(**batch)
        out.loss.backward()
        if step % 50 == 0 or step == steps - 1:
            # Measure grad norm BEFORE the step zeroes it (max_norm=1e9 => no clip).
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            logger.info(f"  step {step:4d}  loss={out.loss.item():.4f}  grad_norm={gn:.4f}")
        opt.step()
        opt.zero_grad()

    final = out.loss.item()
    learned_context = final < unigram_floor - 1.0
    verdict = (
        "CAN fit real data (beats unigram) -> MLM is learnable on this tokenisation"
        if learned_context
        else "STALLS at the unigram floor -> learns only marginal token freq, no context "
        "(tokenisation carries no MLM signal, or a representation bottleneck)"
    )
    logger.info(
        f"[verdict] final loss={final:.3f} (ppl={math.exp(final):.1f}) vs "
        f"unigram_floor={unigram_floor:.3f} (ppl={n_unique}) / ln(vocab)={uniform:.3f} => {verdict}"
    )


if __name__ == "__main__":
    app()
