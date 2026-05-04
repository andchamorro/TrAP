"""Diagnose why MLM pretraining is frozen at the uniform baseline (ln(vocab)).

Runs two decisive checks on the REAL tokenizer + dataset + model:

  1. Input-dependence: forward two different real sequences and measure how much
     the logits change. If the output barely depends on the input, the encoder
     signal is being washed out (a wiring bug) and the model can only ever
     predict the marginal token distribution -> loss pinned at ln(vocab).

  2. Overfit capacity: try to memorise a handful of real sequences with a high
     LR for a few hundred steps. If the loss cannot drop well below ln(vocab)
     even on 16 examples, the model literally cannot fit the data (structural
     bug). If it CAN overfit 16 but full training stays at ln(vocab), the task
     lacks generalisable signal (tokenisation destroys local predictability).

Usage (single GPU recommended, CPU works for a tiny --n-seqs):
    srun --gres=gpu:a100:1 --mem=64G --time=00:20:00 --pty bash
    conda activate trap
    python scripts/diagnose_mlm_learnability.py \
        --preprocessing-name gencode.v48.k17.salmon \
        --albert-config-path config/albert_config_k17_v48.json \
        --tokenizer-path models/<salmon-tokenizer>
"""

import math
import os
from pathlib import Path

from datasets import load_from_disk
from loguru import logger
import torch
import typer
from transformers import AlbertConfig, AlbertForMaskedLM

from trap.config.config import PROCESSED_DATA_DIR
from trap.loaders.tokenizer import WholeKmerMaskingDataCollator, load_kmer_tokenizer

app = typer.Typer(add_completion=False)


def _make_features(ds, idxs):
    keep = ("input_ids", "attention_mask", "token_type_ids", "word_ids", "labels")
    feats = []
    for i in idxs:
        row = ds[int(i)]
        feats.append({k: list(row[k]) for k in keep if k in row})
    return feats


@app.command()
def run(
    preprocessing_name: str = typer.Option("gencode.v48.k17.salmon"),
    albert_config_path: Path = typer.Option("config/albert_config_k17_v48.json"),
    tokenizer_path: Path = typer.Option(..., help="Salmon k-mer tokenizer dir"),
    n_seqs: int = typer.Option(16, help="Sequences to attempt to overfit"),
    steps: int = typer.Option(400),
    lr: float = typer.Option(1e-3),
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
    logger.info(f"device={device} vocab={cfg.vocab_size} ln(vocab)={uniform:.3f}")

    ds = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "masking", "grouped")
    )["train"]
    collator = WholeKmerMaskingDataCollator(tokenizer)

    model = AlbertForMaskedLM(cfg).to(device)

    # --- Check 1: does the output depend on the input? -----------------------
    model.eval()
    a = collator(_make_features(ds, [0]))
    b = collator(_make_features(ds, [1]))
    with torch.no_grad():
        la = model(**{k: v.to(device) for k, v in a.items() if k != "labels"}).logits
        lb = model(**{k: v.to(device) for k, v in b.items() if k != "labels"}).logits
    n = min(la.shape[1], lb.shape[1])
    delta = (la[0, :n] - lb[0, :n]).abs().mean().item()
    logits_std = la.std().item()
    logger.info(
        f"[input-dependence] mean|logits(A)-logits(B)|={delta:.4e}  "
        f"logits_std={logits_std:.4e}  (≈0 => output ignores the input)"
    )

    # --- data sanity: how diverse are the masked labels? ---------------------
    masked = collator(_make_features(ds, list(range(min(n_seqs, len(ds))))))
    lbl = masked["labels"]
    n_masked = int((lbl != -100).sum())
    n_unique = int(torch.unique(lbl[lbl != -100]).numel()) if n_masked else 0
    logger.info(
        f"[data] over {min(n_seqs, len(ds))} seqs: masked_positions={n_masked} "
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
    idxs = list(range(min(n_seqs, len(ds))))
    fixed_batch = None
    if fixed_mask:
        fixed_batch = {k: v.to(device) for k, v in collator(_make_features(ds, idxs)).items()}
    logger.info(
        f"[overfit] memorising {len(idxs)} real seqs, lr={lr}, steps={steps}, "
        f"fixed_mask={fixed_mask}; unigram_floor=ln({n_unique})={unigram_floor:.3f}"
    )
    for step in range(steps):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            batch = {k: v.to(device) for k, v in collator(_make_features(ds, idxs)).items()}
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
        "CAN fit real data (beats unigram) -> full-run failure is signal/scale/optimisation"
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
