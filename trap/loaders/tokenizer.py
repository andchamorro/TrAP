import collections
import itertools
import os
from pathlib import Path
import random
from typing import Optional

from loguru import logger
import numpy as np
from tokenizers import (
    BertWordPieceTokenizer,
    Regex,
    SentencePieceUnigramTokenizer,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast, default_data_collator
import typer

from trap.config import manifest as manifest_mod
from trap.config.schemas import TokenizerConfigSchema, load_config
from trap.config.verbosity import set_verbosity
from trap.loaders.dataset import GenomeDataset
from trap.utils.canonical_kmer import canonical_code
from trap.utils.kmer import kmer_split, kmer_split_batch
from trap.utils.seeding import set_global_seed

app = typer.Typer(help="Train a SentencePiece/WordPiece k-mer tokenizer for TrAP.")


@app.callback()
def _cli() -> None:
    """k-mer tokenizer commands (forces the named ``train`` sub-command)."""


# esaxx (the suffix-array library inside tokenizers' Unigram trainer) indexes
# the concatenated training corpus with i32, so the flat string must stay under
# i32::MAX (2,147,483,647).  k-mer splitting inflates the corpus ~(k+1)x (each
# base becomes a k-char k-mer + a space), so even a moderate genome overflows.
# We cap the *post-k-mer* character footprint well below the i32 limit; this
# also bounds peak memory (esaxx allocates ~3x the flat-string size).
DEFAULT_MAX_TRAINING_CHARS = 500_000_000

# Pinned special tokens for the SPM path. The order fixes the ids
# ([CLS]=0, <pad>=1, [SEP]=2, <unk>=3, [MASK]=4) so they match
# config/albert_config_k17_v48.json (bos=0, pad=1, eos=2) and the Salmon
# tokenizer's pinning (trap/loaders/salmon_tokenizer.py).
_SPM_SPECIALS = ["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"]

# Sidecar written next to an SPM tokenizer recording how it tokenizes input:
# raw_read=True  → feed the raw read (no k-mer pre-split);
# raw_read=False → feed space-joined overlapping k-mers (the default path).
# load_kmer_tokenizer reads it and tags the tokenizer so the diagnostic and
# preprocessing know whether to k-mer-split before tokenizing.
_SPM_MARKER = "trap_spm_config.json"


def _spm_experimental_enabled() -> bool:
    """True when the SPM path is explicitly opted into.

    The SPM tokenizer is re-opened as an exploration on the
    ``explore/spm-metaspace-tokenizer`` branch (the metaspace pre-tokenizer
    mismatch is fixed; see ``train_google_sentencepiece``). It stays gated
    behind ``TRAP_SPM_EXPERIMENTAL=1`` so the default pipeline keeps its
    foot-gun protection and Salmon remains the default tokenizer.
    """
    return os.environ.get("TRAP_SPM_EXPERIMENTAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _write_spm_marker(tok_dir, raw_read, k):
    """Record the SPM input mode next to the tokenizer (see ``_SPM_MARKER``)."""
    import json

    with open(os.path.join(tok_dir, _SPM_MARKER), "w") as fh:
        json.dump({"raw_read": bool(raw_read), "k": int(k)}, fh)


def spm_input_mode(tokenizer):
    """Return ``(raw_read, k)`` for a loaded SPM tokenizer.

    Falls back to ``(False, 17)`` when the tokenizer carries no TrAP SPM marker
    (e.g. a legacy BPE/Unigram tokenizer), i.e. assume the k-mer-split path.
    """
    return (
        bool(getattr(tokenizer, "trap_raw_read", False)),
        int(getattr(tokenizer, "trap_k", 17)),
    )


def _select_training_indices(raw_datasets, k, max_chars, seed, n_pinned=0):
    """Pick a seeded subset of sequence indices for tokenizer training.

    The Unigram trainer builds one suffix array over *all* sentences
    concatenated; with overlapping k-mers the corpus inflates ~(k+1)x and
    overflows esaxx's i32 index on a full transcriptome (panic:
    ``called Result::unwrap() on an Err value: Internal``).  We shuffle
    deterministically and greedily accept sequences until the estimated
    post-k-mer character footprint reaches ``max_chars``.

    The first ``n_pinned`` sequences represent the extra corpus (e.g. L1
    RepeatMasker fragments) whose k-mers should be guaranteed in the
    vocabulary.  They are given at most half the total budget
    (``max_chars // 2``); if they collectively exceed that half, they are
    seeded-randomly subsampled so the main corpus always receives the
    remaining budget.  When they fit within the half-budget, all are kept.

    Sequences shorter than ``k`` are skipped (they k-mer-split to ``""``).

    Args:
        raw_datasets: Indexable sequence of DNA strings.
        k: K-mer length.
        max_chars: Character budget for the post-k-mer corpus.
        seed: RNG seed for reproducible subsampling.
        n_pinned: Number of leading sequences from the extra corpus.

    Returns:
        ``(indices, est_chars)`` — selected indices and the estimated total
        post-k-mer character count.
    """
    selected = []
    total = 0

    # Pinned sequences get at most half the budget so the main corpus is
    # never starved.  Estimate their total cost first; subsample if needed.
    pinned_budget = max_chars // 2
    if n_pinned:
        pinned_cost = sum(
            (len(raw_datasets[i]) - k + 1) * (k + 1)
            for i in range(n_pinned)
            if len(raw_datasets[i]) >= k
        )
        if pinned_cost > pinned_budget:
            logger.warning(
                f"Pinned corpus ({n_pinned:,} seqs, ~{pinned_cost / 1e6:.0f}M chars) "
                f"exceeds half the budget ({pinned_budget / 1e6:.0f}M); "
                f"subsampling pinned seqs with seed={seed}."
            )
            pinned_idx = [i for i in range(n_pinned) if len(raw_datasets[i]) >= k]
            random.Random(seed).shuffle(pinned_idx)
            for i in pinned_idx:
                cost = (len(raw_datasets[i]) - k + 1) * (k + 1)
                if selected and total + cost > pinned_budget:
                    break
                selected.append(i)
                total += cost
        else:
            for i in range(n_pinned):
                seq_len = len(raw_datasets[i])
                if seq_len < k:
                    continue
                selected.append(i)
                total += (seq_len - k + 1) * (k + 1)
        logger.info(
            f"Pinned corpus: {len(selected):,} seqs included "
            f"(~{total / 1e6:.0f}M post-k-mer chars)"
        )

    # Fill remaining budget from the main corpus with a seeded random shuffle.
    idx = list(range(n_pinned, len(raw_datasets)))
    random.Random(seed).shuffle(idx)
    for i in idx:
        seq_len = len(raw_datasets[i])
        if seq_len < k:
            continue  # k-mer split yields an empty string
        # (seq_len - k + 1) k-mers, each rendered as k chars + 1 separator
        cost = (seq_len - k + 1) * (k + 1)
        if selected and total + cost > max_chars:
            break
        selected.append(i)
        total += cost
    return selected, total


def train_sentencepiece(
    raw_datasets,
    out="./",
    name="sequencepiece_unigram",
    vocab_size=10000,
    batch_size=1024,
    k=17,
    fast=False,
    max_training_chars=DEFAULT_MAX_TRAINING_CHARS,
    seed=3469,
    n_pinned=0,
):
    """Train a HuggingFace SentencePiece Unigram tokenizer on k-mer-split sequences.

    Uses the Rust/esaxx suffix-array backend via ``tokenizers``.  For large
    transcriptomes prefer ``train_google_sentencepiece`` (multi-threaded C++)
    or ``train_bpe`` (no suffix array).

    Several non-obvious fixes are baked in for ``tokenizers >=0.19``:

    - ``kmer_split_batch`` yields ``List[str]``; flattened via
      ``itertools.chain.from_iterable`` so each item is one sentence, not a
      pre-tokenised word containing spaces.
    - ``SentencePieceUnigramTokenizer`` default MetaSpace pre-tokenizer
      prepends '▁' (U+2581) which is not in the DNA alphabet; replaced with
      ``Whitespace``.
    - The ``train_from_iterator`` wrapper hard-codes ``max_piece_length=16``;
      overridden to ``k`` so k-mers of length ≥17 are not silently dropped.
    - esaxx indexes the concatenated corpus with i32 (max 2.1B chars);
      k-mer splitting inflates corpus ~(k+1)×, so the full transcriptome
      overflows.  Seeded subsampling caps the post-k-mer footprint at
      ``max_training_chars``.

    Args:
        raw_datasets: Indexable sequence of DNA strings.
        out: Output directory; tokenizer saved under ``<out>/<name>/``.
        name: Tokenizer directory name.
        vocab_size: Target vocabulary size.
        batch_size: Sequences per k-mer-splitting batch.
        k: K-mer length.
        fast: Wrap as ``PreTrainedTokenizerFast`` and save in HF JSON format.
        max_training_chars: Post-k-mer character budget; sequences are
            subsampled (seeded) to keep esaxx under its i32 limit.
        seed: RNG seed for reproducible subsampling.
        n_pinned: Leading sequences in ``raw_datasets`` to include
            unconditionally (see ``_select_training_indices``).
    """
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers import trainers as _hf_trainers

    os.makedirs(os.path.join(out, name), exist_ok=True)

    indices, est_chars = _select_training_indices(
        raw_datasets, k, max_training_chars, seed, n_pinned=n_pinned
    )
    if len(indices) < len(raw_datasets):
        logger.warning(
            f"Subsampling: {len(indices):,}/{len(raw_datasets):,} seqs "
            f"(~{est_chars / 1e6:.0f}M post-k-mer chars, cap "
            f"{max_training_chars / 1e6:.0f}M) — esaxx i32 guard active."
        )
    else:
        logger.info(
            f"Using all {len(indices):,} seqs (~{est_chars / 1e6:.0f}M "
            f"post-k-mer chars, under the {max_training_chars / 1e6:.0f}M cap)."
        )
    subset = [raw_datasets[i] for i in indices]

    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer._tokenizer.pre_tokenizer = _pre_tokenizers.Whitespace()

    _trainer = _hf_trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
        unk_token="<unk>",
        initial_alphabet=list("ACGTN"),
        max_piece_length=k,
        show_progress=True,
    )

    logger.log("STAGE", f"[tokenizer:train] HF Unigram training started — {len(subset):,} seqs")
    logger.info(f"Training on {len(subset):,} seqs (batch_size={batch_size})")
    tokenizer._tokenizer.train_from_iterator(
        itertools.chain.from_iterable(kmer_split_batch(subset, batch_size, k)),
        trainer=_trainer,
        length=len(subset),
    )
    logger.info(f"Training complete. Vocab size: {tokenizer.get_vocab_size():,}")

    logger.info("Post processing ...")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    if fast:
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="<pad>",
            mask_token="[MASK]",
            truncation_side="right",
        )
        # Bake the paired template into the saved tokenizer so it is
        # restored on load and does not need to be re-applied at every call
        # site (QW-10).
        fast_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", fast_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", fast_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        logger.info("Save Tokenizer as fast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        logger.success("Tokenizer training complete.")
    else:
        logger.info("Save Tokenizer ...")
        tokenizer.save_model(out, name)
        logger.success("Tokenizer training complete.")


def train_wordpiece(
    raw_datasets,
    out="./",
    name="wordpiece",
    vocab_size=10000,
    batch_size=1024,
    k=17,
    fast=False,
):
    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Nmt(), normalizers.Lowercase(), normalizers.Replace(Regex(r"[^actg\s]"), "")]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            tokenizer.pre_tokenizer,
        ]
    )

    # And then train
    logger.info("Training tokenizer...")
    tokenizer.train_from_iterator(
        kmer_split_batch(raw_datasets, batch_size, k),
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
        unk_token="<unk>",
    )

    logger.info("Post processing ...")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    if fast:
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="<pad>",
            mask_token="[MASK]",
            truncation_side="right",
        )
        fast_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", fast_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", fast_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        logger.info("Save Tokenizer as fast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        logger.success("Tokenizer training complete.")
    else:
        logger.info("Save Tokenizer ...")
        tokenizer.save_model(out, name)
        logger.success("Tokenizer training complete.")


def train_bpe(
    raw_datasets,
    out="./",
    name="bpe",
    vocab_size=10000,
    batch_size=1024,
    k=17,
    fast=True,
    max_training_chars=DEFAULT_MAX_TRAINING_CHARS,
    seed=3469,
    n_pinned=0,
):
    """Train a BPE tokenizer on k-mer-split genomic sequences.

    BPE is significantly faster than Unigram (no suffix array) and completes
    in minutes even on a full transcriptome. Tokens range from single
    nucleotides up to full k-mers; rare k-mers are split into sub-sequences
    rather than mapped to ``<unk>``.

    The BpeTrainer builds a word-frequency HashMap over all unique k-mers
    before merging; for large transcriptomes this can exceed tens of GB.
    ``max_training_chars`` caps the post-k-mer corpus size via seeded
    subsampling — identical to the Unigram memory guard — so peak RSS stays
    predictable regardless of corpus size.

    Args:
        raw_datasets: Indexable sequence of DNA strings.
        out: Output directory; tokenizer saved under ``<out>/<name>/``.
        name: Tokenizer directory name.
        vocab_size: Target vocabulary size.
        batch_size: Sequences per k-mer-splitting batch.
        k: K-mer length.
        fast: Wrap as ``PreTrainedTokenizerFast`` and save in HF JSON format.
        max_training_chars: Post-k-mer character budget; sequences are
            subsampled (seeded) to fit.  Keeps BPE trainer RSS predictable.
        seed: RNG seed for reproducible subsampling.
        n_pinned: Leading sequences in ``raw_datasets`` to include
            unconditionally (see ``_select_training_indices``).
    """
    from tokenizers import Tokenizer
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers import trainers as _hf_trainers
    from tokenizers.models import BPE

    os.makedirs(os.path.join(out, name), exist_ok=True)

    indices, est_chars = _select_training_indices(
        raw_datasets, k, max_training_chars, seed, n_pinned=n_pinned
    )
    if len(indices) < len(raw_datasets):
        logger.warning(
            f"Subsampling: {len(indices):,}/{len(raw_datasets):,} seqs "
            f"(~{est_chars / 1e6:.0f}M post-k-mer chars, cap "
            f"{max_training_chars / 1e6:.0f}M) — BPE memory guard active."
        )
    else:
        logger.info(
            f"Using all {len(indices):,} seqs (~{est_chars / 1e6:.0f}M "
            f"post-k-mer chars, under the {max_training_chars / 1e6:.0f}M cap)."
        )
    subset = [raw_datasets[i] for i in indices]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = _pre_tokenizers.Whitespace()

    trainer = _hf_trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
        initial_alphabet=list("ACGTN"),
        show_progress=True,
    )

    logger.log("STAGE", f"[tokenizer:train] BPE training started — {len(subset):,} seqs")
    logger.info(
        f"Training BPE on {len(subset):,} seqs "
        f"(k={k}, vocab={vocab_size}, batch_size={batch_size})"
    )
    tokenizer.train_from_iterator(
        itertools.chain.from_iterable(kmer_split_batch(subset, batch_size, k)),
        trainer=trainer,
        length=len(subset),
    )
    logger.info(f"Training complete. Vocab size: {tokenizer.get_vocab_size():,}")

    logger.info("Post processing ...")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    if fast:
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="<pad>",
            mask_token="[MASK]",
            truncation_side="right",
        )
        fast_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", fast_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", fast_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        logger.info("Saving as PreTrainedTokenizerFast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        logger.success("BPE tokenizer training complete.")
    else:
        out_path = os.path.join(out, name, "tokenizer.json")
        logger.info(f"Saving tokenizer to {out_path}")
        tokenizer.save(out_path)
        logger.success("BPE tokenizer training complete.")


def train_google_sentencepiece(
    raw_datasets,
    out="./",
    name="spm_unigram",
    vocab_size=32000,
    k=17,
    fast=True,
    input_sentence_size=1_000_000,
    num_threads=16,
    seed=3469,
    max_training_chars=DEFAULT_MAX_TRAINING_CHARS,
    n_pinned=0,
    raw_read=False,
    raw_max_piece_length=32,
    pin_kmers=None,
):
    """Train a Unigram tokenizer via the Google SentencePiece C++ library.

    Streams k-mer sentences directly via ``sentence_iterator`` (no temp file).
    SPM trains with ``num_threads`` in parallel — significantly faster than the
    HF Rust/esaxx Unigram path for large transcriptomes.

    ``input_sentence_size`` is SPM's built-in reservoir-sampling cap, but it
    only fires when the corpus exceeds that count.  For GENCODE v48 (~410k
    sentences after k-mer expansion), the cap is never reached, so SPM loads
    the full 10B-character corpus and uses ~100 GB RAM.  ``max_training_chars``
    applies seeded subsampling *before* the iterator is passed to SPM, bounding
    peak RSS regardless of corpus size.

    The trained ``.model`` is loaded back and wrapped as a
    ``PreTrainedTokenizerFast`` when ``fast=True``.

    Args:
        raw_datasets: Indexable sequence of DNA strings.
        out: Output directory; tokenizer saved under ``<out>/<name>/``.
        name: Tokenizer directory name.
        vocab_size: Target vocabulary size.
        k: K-mer length; also sets ``max_sentencepiece_length`` to ``k + 1``
            (the extra char is the leading ▁ metaspace marker).
        fast: Wrap as ``PreTrainedTokenizerFast`` and save in HF JSON format.
        input_sentence_size: SPM built-in reservoir-sampling cap (sentences).
            Acts as a secondary guard; the primary cap is ``max_training_chars``.
        num_threads: Parallelism for SPM's EM training step.
        seed: RNG seed for subsampling and SPM's internal shuffle.
        max_training_chars: Post-k-mer character budget; sequences are
            subsampled (seeded) to fit.  Keeps SPM RSS predictable.
        n_pinned: Leading sequences in ``raw_datasets`` to include
            unconditionally (see ``_select_training_indices``).
        raw_read: When ``True``, train on **raw reads** (DNABERT-2 style): each
            sequence is one SPM sentence with no k-mer pre-splitting, so SPM
            learns variable-length subword pieces over the nucleotide stream.
            A 150 bp read then tokenizes to far fewer tokens (≤ read length even
            at char-level) than the ~134 overlapping k-mers of the default path,
            which is what keeps paired reads inside ``max_position_embeddings``.
        raw_max_piece_length: ``max_sentencepiece_length`` for ``raw_read`` mode
            (the default path uses ``k + 1``). Longer pieces compress better and
            let conserved motifs survive at the ≥16 bp length the k-mer entropy
            argument calls for (k ≳ log4(genome) ≈ 16 for per-token uniqueness).
        pin_kmers: Optional list of k-mer strings forced into the vocab as SPM
            ``user_defined_symbols`` (atomic, never split). Seed these with the
            conserved L1 k-mers that feed the Salmon target index so the
            entropy-justified ≥16 bp k-mers become guaranteed single tokens,
            with subword fallback for everything else.
    """
    import sentencepiece as spm

    os.makedirs(os.path.join(out, name), exist_ok=True)
    model_prefix = os.path.join(out, name, "spm")

    indices, est_chars = _select_training_indices(
        raw_datasets, k, max_training_chars, seed, n_pinned=n_pinned
    )
    if len(indices) < len(raw_datasets):
        logger.warning(
            f"Subsampling: {len(indices):,}/{len(raw_datasets):,} seqs "
            f"(~{est_chars / 1e6:.0f}M post-k-mer chars, cap "
            f"{max_training_chars / 1e6:.0f}M) — SPM memory guard active."
        )
    else:
        logger.info(
            f"Using all {len(indices):,} seqs (~{est_chars / 1e6:.0f}M "
            f"post-k-mer chars, under the {max_training_chars / 1e6:.0f}M cap)."
        )

    # The training sentence for each read: raw nucleotides (raw_read) or the
    # space-joined overlapping k-mers (default). Defined once so the length
    # estimate and the SPM iterator stay consistent.
    def _sentence(i):
        return raw_datasets[i] if raw_read else kmer_split(k, raw_datasets[i])

    # Estimate max_sentence_length from a random sample of the selected subset;
    # for pure ASCII DNA, byte length == char length. SPM rejects
    # max_sentence_length < 10, so clamp the floor (short reads / k-mers).
    sample_size = min(1000, len(indices))
    sample_indices = random.Random(seed).sample(indices, sample_size)
    max_sentence_length = max(
        int(max(len(_sentence(i).encode()) for i in sample_indices) * 1.2),
        16,
    )
    max_piece_length = raw_max_piece_length if raw_read else k + 1

    # Force conserved k-mers into the vocab as atomic user-defined symbols so the
    # entropy-justified ≥16 bp k-mers become guaranteed single tokens (the Salmon
    # target index, expressed as SPM pieces). Deduped; capped with a warning so a
    # runaway pin list cannot crowd out the learned subword vocab.
    user_symbols = ["[CLS]", "[SEP]", "[MASK]"]
    if pin_kmers:
        pinned = list(dict.fromkeys(pin_kmers))
        cap = max(vocab_size // 2, 1)
        if len(pinned) > cap:
            logger.warning(
                f"pin_kmers ({len(pinned):,}) exceeds half the vocab ({cap:,}); "
                f"keeping the first {cap:,} (pre-rank by count upstream)."
            )
            pinned = pinned[:cap]
        long_pins = sum(1 for km in pinned if len(km) >= 16)
        logger.log(
            "STAGE",
            f"[tokenizer:train] pinning {len(pinned):,} k-mers as atomic tokens "
            f"({long_pins:,} are ≥16 bp)",
        )
        user_symbols += pinned
    logger.info(
        f"Estimated max_sentence_length: {max_sentence_length:,} bytes (from {sample_size} samples)"
    )

    logger.log(
        "STAGE",
        f"[tokenizer:train] SPM training started — {len(indices):,} seqs "
        f"(threads={num_threads}, input_sentence_size={input_sentence_size:,})",
    )
    logger.info(
        f"Training Google SentencePiece Unigram on {len(indices):,} seqs "
        f"(k={k}, vocab={vocab_size}, threads={num_threads})"
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=(_sentence(i) for i in indices),
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        # Default path: +1 for the leading metaspace marker (▁), so a whole
        # k-mer piece (▁ + k chars) is reachable. raw_read path: a larger cap
        # so SPM can learn longer motif pieces over the raw nucleotide stream.
        max_sentencepiece_length=max_piece_length,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        split_by_unicode_script=False,
        split_by_number=False,
        normalization_rule_name="identity",
        user_defined_symbols=user_symbols,
        pad_id=3,
        train_extremely_large_corpus=True,
        num_threads=num_threads,
    )

    model_path = model_prefix + ".model"
    vocab_path = model_prefix + ".vocab"
    logger.info(f"SPM model written to {model_path}")

    # Load vocab from the SPM .vocab file (avoids sentencepiece_model_pb2
    # protobuf dependency that tokenizers.SentencePieceUnigramTokenizer.from_spm()
    # requires in tokenizers >=0.20).  The .vocab format is one "piece\tscore"
    # per line; scores are natural-log probabilities, matching HF Unigram exactly.
    from tokenizers import Tokenizer
    from tokenizers import decoders as _decoders
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers.models import Unigram

    raw_pieces = []
    with open(vocab_path) as fv:
        for line in fv:
            piece, score = line.rstrip("\n").split("\t")
            raw_pieces.append((piece, float(score)))

    # Pin the specials to ids 0..4 (albert_config order). Drop SPM's own
    # control symbols (<s>, </s>) and any learned duplicate of a pinned special,
    # then prepend the pinned five. Specials never match DNA substrings, so a
    # score of 0.0 is inert during Unigram segmentation of reads.
    _drop = set(_SPM_SPECIALS) | {"<s>", "</s>"}
    # Conserved k-mers are pinned as user-defined symbols, but the .vocab → HF
    # Unigram rebuild loses SPM's "atomic" flag, so they would just compete in
    # Viterbi (≈68% emitted whole). Boost their score to 0.0 (the max, like the
    # specials) so the segmentation always prefers the full pinned k-mer wherever
    # it matches — the ≥16 bp entropy-consistent token. Covers both the bare and
    # the ▁-prefixed (read-initial) form.
    _pin_set = set(pin_kmers) if pin_kmers else set()
    learned = [
        (p, 0.0 if (_pin_set and p.lstrip("▁") in _pin_set) else s)
        for (p, s) in raw_pieces
        if p not in _drop
    ]
    vocab_pieces = [(tok, 0.0) for tok in _SPM_SPECIALS] + learned
    unk_id = _SPM_SPECIALS.index("<unk>")

    hf_tokenizer = Tokenizer(Unigram(vocab_pieces, unk_id=unk_id))
    # Metaspace (not Whitespace): SPM stores word-initial pieces with a leading
    # ▁ marker, so the pre-tokenizer must reproduce that marker or every k-mer
    # falls back to char-level pieces (~k pieces/k-mer — the old corruption).
    # prepend_scheme="always" + split=True turns "ACGTA CGTAC" into the
    # ▁ACGTA / ▁CGTAC pre-tokens the vocab was trained on. The matching decoder
    # makes decode() reconstruct the sequence (content-preserving round-trip).
    hf_tokenizer.pre_tokenizer = _pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )
    hf_tokenizer.decoder = _decoders.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )

    logger.info("Post processing ...")
    hf_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", hf_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", hf_tokenizer.token_to_id("[SEP]")),
        ],
    )

    if fast:
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=hf_tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="<pad>",
            mask_token="[MASK]",
            truncation_side="right",
        )
        fast_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", fast_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", fast_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        logger.info("Saving as PreTrainedTokenizerFast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        _write_spm_marker(os.path.join(out, name), raw_read=raw_read, k=k)
        logger.success(
            f"Google SentencePiece tokenizer training complete "
            f"(mode={'raw-read' if raw_read else 'kmer'})."
        )
    else:
        _write_spm_marker(os.path.join(out, name), raw_read=raw_read, k=k)
        logger.success(f"SPM model saved to {model_prefix}.model / .vocab")


_ALGO_MANIFEST = {
    "unigram": "sentencepiece-unigram",
    "wordpiece": "wordpiece",
    "bpe": "bpe",
    "spm": "sentencepiece-unigram-google",
    "salmon": "salmon-canonical-kmer",
}


# ---------------------------------------------------------------------------
# Salmon-consistent canonical k-mer index (Design C: target + hashed decoy)
# ---------------------------------------------------------------------------


def _iter_jellyfish_dump(path):
    """Yield ``(kmer, count)`` from a ``jellyfish dump -c`` file.

    The columnar ``-c`` format is one ``"<KMER> <COUNT>"`` per line.  Lines that
    are not in that form are skipped, so a plain one-k-mer-per-line file also
    works (count defaults to 1).
    """
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            kmer = parts[0]
            count = int(parts[1]) if len(parts) > 1 else 1
            yield kmer, count


def _select_pin_kmers(counts_iter, pin_k, min_count, max_n):
    """Top-``max_n`` ``pin_k``-mers (by count) from a ``(kmer, count)`` stream.

    Aggregates duplicate k-mers, drops those of the wrong length or below
    ``min_count``, and returns the most frequent — the conserved k-mers worth
    pinning as atomic SPM tokens.
    """
    agg = collections.Counter()
    for kmer, count in counts_iter:
        if len(kmer) == pin_k:
            agg[kmer] += count
    ranked = sorted(
        ((km, c) for km, c in agg.items() if c >= min_count),
        key=lambda x: x[1],
        reverse=True,
    )
    return [km for km, _ in ranked[:max_n]]


def _kmer_counts_from_sequences(sequences, pin_k):
    """Yield ``(kmer, 1)`` for every overlapping ``pin_k``-mer in *sequences*."""
    for seq in sequences:
        for j in range(len(seq) - pin_k + 1):
            yield seq[j : j + pin_k], 1


def build_salmon_index(
    kmer_counts,
    out,
    name,
    k=17,
    n_hash=65536,
    min_count=1,
    max_target=None,
):
    """Build a :class:`SalmonKmerTokenizer` from canonical L1 k-mer counts.

    Reads ``(kmer, count)`` pairs (e.g. ``jellyfish dump -c`` of the L1
    reference), keeps the *frequent* (conserved) k-mers as the collision-free
    target index, and lets every other k-mer fall through to a hashed decoy
    bucket at runtime.

    Args:
        kmer_counts: Path to a ``jellyfish dump -c`` (or one-k-mer-per-line) file.
        out: Output directory; tokenizer saved under ``<out>/<name>``.
        name: Tokenizer directory name.
        k: K-mer length (must match the dump's k).
        n_hash: Number of decoy hash buckets.
        min_count: Drop k-mers occurring fewer than this many times.
        max_target: Cap the target index to the ``max_target`` most frequent
            k-mers (``None`` → keep all above ``min_count``).

    Returns:
        ``(tokenizer, num_target)``.
    """
    from trap.loaders.salmon_tokenizer import SalmonKmerTokenizer

    if kmer_counts is None:
        # Design B: no target index — every canonical k-mer is feature-hashed.
        target_codes = np.empty(0, dtype=np.int64)
        logger.info("No k-mer counts provided → pure canonical feature-hashing (empty target)")
    else:
        kmers, counts = [], []
        for kmer, count in _iter_jellyfish_dump(kmer_counts):
            if count >= min_count and len(kmer) == k:
                kmers.append(kmer)
                counts.append(count)
        logger.info(f"Read {len(kmers):,} k-mers with count >= {min_count} (k={k})")

        if max_target is not None and len(kmers) > max_target:
            top = np.argpartition(np.asarray(counts), -max_target)[-max_target:]
            kmers = [kmers[i] for i in top]
            logger.info(f"Capped target to the {max_target:,} most frequent k-mers")

        target_codes = np.unique(np.fromiter((canonical_code(km) for km in kmers), dtype=np.int64))
    logger.log(
        "STAGE",
        f"[tokenizer:salmon] target={target_codes.shape[0]:,} canonical k-mers  "
        f"decoy_buckets={n_hash:,}  vocab_size={5 + target_codes.shape[0] + n_hash:,}",
    )

    tok_dir = os.path.join(out, name)
    os.makedirs(tok_dir, exist_ok=True)
    codes_path = os.path.join(tok_dir, "target_codes.npy")
    np.save(codes_path, target_codes)
    tokenizer = SalmonKmerTokenizer(
        target_codes_file=codes_path, k=k, n_hash=n_hash, model_max_length=1280
    )
    tokenizer.save_pretrained(tok_dir)
    return tokenizer, int(target_codes.shape[0])


def load_kmer_tokenizer(tokenizer_path, max_position=None):
    """Load a TrAP tokenizer, transparently handling the Salmon k-mer kind.

    If ``tokenizer_path`` holds a :class:`SalmonKmerTokenizer` (detected by its
    ``salmon_kmer_config.json``) it is loaded directly; otherwise the legacy
    fast tokenizer is loaded and the paired ``[CLS]/[SEP]`` template applied.
    ``model_max_length`` is set when ``max_position`` is given.
    """
    from trap.loaders.salmon_tokenizer import CONFIG_FILE, SalmonKmerTokenizer

    path = str(tokenizer_path)
    # SentencePiece was disabled because a metaspace/Whitespace mismatch
    # fragmented each k-mer into ~16 char-level pieces. That wiring is fixed on
    # the SPM-exploration branch (Metaspace pre-tokenizer + decoder); the path is
    # re-opened only under TRAP_SPM_EXPERIMENTAL=1. Otherwise reject loudly.
    is_spm = path.rstrip("/").endswith(".spm") or os.path.exists(os.path.join(path, "spm.model"))
    if is_spm and not _spm_experimental_enabled():
        raise ValueError(
            f"SentencePiece (.spm) tokenizer is DEPRECATED and disabled: {path}. "
            "Its metaspace pre-tokenizer historically fragmented k-mers (~16 char "
            "pieces each), silently corrupting tokenization. The wiring is fixed on "
            "the SPM-exploration branch; set TRAP_SPM_EXPERIMENTAL=1 to opt in, or "
            "use a Salmon canonical k-mer tokenizer "
            "(`python -m trap.loaders.tokenizer salmon_index`)."
        )
    if os.path.exists(os.path.join(path, CONFIG_FILE)):
        tokenizer = SalmonKmerTokenizer.from_pretrained(path, local_files_only=True)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
        # Tag SPM tokenizers with their input mode so the diagnostic and
        # preprocessing know whether to k-mer-split before tokenizing.
        marker = os.path.join(path, _SPM_MARKER)
        if os.path.exists(marker):
            import json

            with open(marker) as fh:
                cfg = json.load(fh)
            tokenizer.trap_raw_read = bool(cfg.get("raw_read", False))
            tokenizer.trap_k = int(cfg.get("k", 17))
    if max_position is not None:
        tokenizer.model_max_length = max_position
    return tokenizer


@app.command()
def salmon_index(
    kmer_counts: Optional[Path] = typer.Option(
        None,
        "--kmer-counts",
        help="jellyfish 'dump -c' file of the L1 reference (KMER COUNT per line). "
        "Pre-filter with 'jellyfish dump -c -L <T>' to keep only conserved k-mers. "
        "Omit for Design B (no target index — pure canonical feature-hashing).",
    ),
    out: Path = typer.Option(..., help="Output dir (tokenizer saved under <out>/<name>)"),
    name: str = typer.Option("tokenizer.l1.k17.salmon", help="Tokenizer directory name"),
    k: int = typer.Option(17, help="K-mer length (must match the dump)"),
    n_hash: int = typer.Option(65536, help="Decoy/hash buckets"),
    min_count: int = typer.Option(1, help="Drop k-mers below this count"),
    max_target: Optional[int] = typer.Option(
        None, help="Cap target to the N most frequent k-mers"
    ),
    seed: int = typer.Option(3469, help="Global RNG seed"),
    verbosity: str = typer.Option("normal", "--verbosity", envvar="TRAP_VERBOSITY"),
):
    """Build a Salmon-consistent canonical k-mer tokenizer (index-build phase)."""
    set_verbosity(verbosity)
    set_global_seed(seed)
    logger.log(
        "STAGE",
        f"[tokenizer:salmon_index] k={k} n_hash={n_hash} min_count={min_count} "
        f"max_target={max_target} counts={kmer_counts} out={out}/{name}",
    )
    _, num_target = build_salmon_index(
        str(kmer_counts) if kmer_counts is not None else None,
        out=str(out),
        name=name,
        k=k,
        n_hash=n_hash,
        min_count=min_count,
        max_target=max_target,
    )
    tok_dir = Path(out) / name
    manifest_mod.write(
        tok_dir,
        seed=seed,
        k=k,
        tokenizer={
            "algorithm": _ALGO_MANIFEST["salmon"],
            "vocab_size": 5 + num_target + n_hash,
            "num_target": num_target,
            "n_hash": n_hash,
            "min_count": min_count,
            "max_target": max_target,
            "kmer_counts": str(kmer_counts) if kmer_counts is not None else None,
        },
    )
    logger.success(f"Salmon k-mer tokenizer + manifest written to {tok_dir}")


@app.command()
def train(
    corpus: Optional[Path] = typer.Option(
        None, help="FASTA/FASTQ corpus to train the tokenizer on"
    ),
    extra_corpus: Optional[Path] = typer.Option(
        None,
        "--extra-corpus",
        help=(
            "Additional FASTA/FASTQ whose sequences are always included in the "
            "vocabulary (not subject to random subsampling). Use to pin L1HS/L1PA "
            "RepeatMasker fragments so their k-mers are guaranteed in the vocab."
        ),
        envvar="L1_CORPUS",
    ),
    out: Path = typer.Option(..., help="Output dir (tokenizer saved under <out>/<name>)"),
    name: str = typer.Option("tokenizer.gencode.v48.k17.32k", help="Tokenizer directory name"),
    algorithm: str = typer.Option(
        "bpe",
        help="'bpe' (fast, default), 'spm' (Google SentencePiece C++, multi-threaded Unigram), "
        "'unigram' (HF Unigram, slow), or 'wordpiece'",
    ),
    k: int = typer.Option(17, help="K-mer size"),
    vocab_size: int = typer.Option(32000, help="Target vocabulary size"),
    batch_size: int = typer.Option(1024, help="Sequences per training batch"),
    file_format: str = typer.Option("fasta", help="BioPython format string"),
    fast: bool = typer.Option(True, help="Wrap as a HuggingFace PreTrainedTokenizerFast"),
    max_training_chars: int = typer.Option(
        DEFAULT_MAX_TRAINING_CHARS,
        help="Cap on post-k-mer corpus chars (Unigram/HF only); keeps the esaxx "
        "suffix array under its i32 limit. Sequences are subsampled (seeded) to fit.",
    ),
    num_threads: int = typer.Option(
        16,
        help="Parallel threads for training (spm only; ignored by bpe/unigram/wordpiece). "
        "Defaults to 16; set to $SLURM_CPUS_PER_TASK for full node utilisation.",
        envvar="TOKENIZER_NUM_THREADS",
    ),
    seed: int = typer.Option(3469, help="Global RNG seed"),
    raw_read: bool = typer.Option(
        False,
        "--raw-read/--kmer-split",
        help="spm only: train on raw reads (no overlapping k-mer pre-split) so "
        "SPM learns subword pieces over the nucleotide stream. Keeps paired "
        "reads inside max_position_embeddings. Default is the k-mer-split path.",
        envvar="TOKENIZER_RAW_READ",
    ),
    pin_kmers: Optional[Path] = typer.Option(
        None,
        "--pin-kmers",
        help="spm only: jellyfish 'dump -c' (or one-k-mer-per-line) file of "
        "conserved L1 k-mers to force into the vocab as atomic tokens, so the "
        "entropy-justified ≥16 bp k-mers stay whole (Salmon target index as SPM "
        "pieces). Mutually informative with --raw-read.",
    ),
    pin_from_extra: bool = typer.Option(
        False,
        "--pin-from-extra",
        help="spm only: derive the pinned k-mers from --extra-corpus sequences "
        "(top --pin-max by frequency) instead of a --pin-kmers dump.",
    ),
    pin_k: Optional[int] = typer.Option(None, help="Length of pinned k-mers (defaults to --k)."),
    pin_min_count: int = typer.Option(1, help="Drop pinned k-mers below this count."),
    pin_max: int = typer.Option(
        8192, help="Keep at most this many (most frequent) pinned k-mers."
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help=(
            "Debug mode: subsample to 500 sequences, cap vocab at 200, "
            "and append '.debug' to the tokenizer name. Completes in seconds "
            "on any machine. Activate via TOKENIZER_DEBUG=1 or --debug."
        ),
        envvar="TOKENIZER_DEBUG",
    ),
    config: Optional[Path] = typer.Option(
        None, help="YAML/JSON TokenizerConfigSchema; explicit flags override it"
    ),
    verbosity: str = typer.Option(
        "normal",
        "--verbosity",
        help="Log verbosity: off, normal (default), detailed.",
        envvar="TRAP_VERBOSITY",
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
):
    """Train a k-mer tokenizer (SentencePiece Unigram or WordPiece) — plan §6.1.

    Loads the corpus via ``GenomeDataset`` (sequences standardized to pure ACTG,
    uppercase), trains on space-separated k-mers, bakes the paired
    ``[CLS] A [SEP] B [SEP]`` template into the saved tokenizer (QW-10), and
    writes a ``manifest.json`` next to it.
    """
    set_verbosity(verbosity)

    # --config provides defaults; flags passed on the command line override them.
    if config is not None:
        from click.core import ParameterSource

        cfg = load_config(config, TokenizerConfigSchema)

        def _pick(flag_name, flag_value, cfg_value):
            on_cli = ctx is not None and ctx.get_parameter_source(flag_name) == (
                ParameterSource.COMMANDLINE
            )
            return flag_value if on_cli else cfg_value

        algorithm = _pick("algorithm", algorithm, cfg.algorithm)
        k = _pick("k", k, cfg.k)
        vocab_size = _pick("vocab_size", vocab_size, cfg.vocab_size)
        batch_size = _pick("batch_size", batch_size, cfg.batch_size)
        name = _pick("name", name, cfg.name)
        out = Path(_pick("out", str(out), cfg.out))
        corpus = _pick("corpus", corpus, Path(cfg.corpus) if cfg.corpus else None)
        extra_corpus = _pick(
            "extra_corpus",
            extra_corpus,
            Path(cfg.extra_corpus) if cfg.extra_corpus else None,
        )
        if cfg.max_training_chars is not None:
            max_training_chars = _pick(
                "max_training_chars", max_training_chars, cfg.max_training_chars
            )

    if corpus is None:
        raise typer.BadParameter("Provide --corpus (or a config with a 'corpus' field).")
    if algorithm == "spm" and not _spm_experimental_enabled():
        raise typer.BadParameter(
            "SentencePiece (spm) is DEPRECATED and disabled by default: its "
            "metaspace pre-tokenizer historically fragmented k-mers (~16 char "
            "pieces each). The wiring is fixed on the SPM-exploration branch; set "
            "TRAP_SPM_EXPERIMENTAL=1 to opt in, or use the Salmon canonical k-mer "
            "tokenizer (`python -m trap.loaders.tokenizer salmon_index`)."
        )
    if algorithm not in ("bpe", "unigram", "wordpiece", "spm"):
        raise typer.BadParameter(
            f"Unknown algorithm {algorithm!r}; expected 'bpe', 'unigram', "
            "'wordpiece', or 'spm' (with TRAP_SPM_EXPERIMENTAL=1)."
        )

    # ------------------------------------------------------------------
    # Debug mode: fast smoke-test that validates each of the four fixes
    # that previously caused the Unigram trainer Rust panic on Grace.
    # Activate with:  TOKENIZER_DEBUG=1 sbatch 10_tokenizer.slurm
    #              or --debug flag
    # ------------------------------------------------------------------
    _DEBUG_MAX_SEQS = 500
    _DEBUG_VOCAB = 200
    # Budget sized for ~500 sequences × a typical 50 bp read length
    _DEBUG_MAX_CHARS = _DEBUG_MAX_SEQS * max(50 - k + 1, 1) * (k + 1)
    if debug:
        vocab_size = _DEBUG_VOCAB
        max_training_chars = _DEBUG_MAX_CHARS
        if not name.endswith(".debug"):
            name = name + ".debug"
        logger.warning("=" * 60)
        logger.warning("TOKENIZER DEBUG MODE")
        logger.warning(f"  max sequences   : {_DEBUG_MAX_SEQS}")
        logger.warning(f"  vocab size      : {vocab_size}")
        logger.warning(f"  max_training_chars: {max_training_chars:,}")
        logger.warning(f"  output name     : {name}")
        logger.warning(f"  k               : {k}")
        logger.warning(f"  seed            : {seed}")
        logger.warning("Fixes being validated:")
        logger.warning("  [1] Iterator flatten   (chain.from_iterable — no spaces as words)")
        logger.warning("  [2] Whitespace pre-tok (no MetaSpace ▁ prefix)")
        logger.warning(f"  [3] max_piece_length={k} (wrapper default 16 skipped k>=17)")
        logger.warning(f"  [4] Corpus budget={max_training_chars:,} (esaxx i32 overflow guard)")
        logger.warning("=" * 60)

    import time as _time

    _stage_t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[tokenizer:train] algorithm={algorithm} k={k} vocab_size={vocab_size:,} "
        f"corpus={corpus} out={out}/{name}",
    )

    set_global_seed(seed)
    out = Path(out)
    corpus = Path(corpus)

    logger.info(f"Loading corpus {corpus} (format={file_format})")
    sequences = GenomeDataset(corpus, file_format).sequences
    logger.log(
        "STAGE",
        f"[tokenizer:train] corpus loaded — {len(sequences):,} sequences",
    )
    logger.info(
        f"Loaded {len(sequences):,} sequences; training {algorithm} tokenizer "
        f"(k={k}, vocab={vocab_size})"
    )

    if debug:
        rng = random.Random(seed)
        sequences = rng.sample(sequences, min(_DEBUG_MAX_SEQS, len(sequences)))
        logger.warning(f"DEBUG MODE: subsampled to {len(sequences)} sequences")

    # Prepend extra corpus (e.g. L1 RepeatMasker fragments) so their k-mers
    # are always represented in the vocabulary regardless of budget subsampling.
    n_pinned = 0
    if extra_corpus is not None and not debug:
        logger.info(f"Loading extra (pinned) corpus {extra_corpus} (format={file_format})")
        extra_sequences = GenomeDataset(extra_corpus, file_format).sequences
        n_pinned = len(extra_sequences)
        sequences = extra_sequences + sequences
        logger.info(
            f"Pinned {n_pinned:,} extra sequences — combined corpus: "
            f"{len(sequences):,} sequences"
        )
    elif extra_corpus is not None and debug:
        logger.warning("DEBUG MODE: --extra-corpus ignored")

    # Collect conserved k-mers to pin as atomic SPM tokens (entropy consistency):
    # from a jellyfish dump, or derived from the prepended --extra-corpus.
    _pin_list = None
    if not debug and (pin_kmers is not None or pin_from_extra):
        _pin_k = pin_k or k
        if pin_kmers is not None:
            _pin_list = _select_pin_kmers(
                _iter_jellyfish_dump(str(pin_kmers)), _pin_k, pin_min_count, pin_max
            )
            logger.info(f"Pinning {len(_pin_list):,} k-mers from dump {pin_kmers}")
        elif pin_from_extra and n_pinned:
            _pin_list = _select_pin_kmers(
                _kmer_counts_from_sequences(sequences[:n_pinned], _pin_k),
                _pin_k,
                pin_min_count,
                pin_max,
            )
            logger.info(f"Pinning {len(_pin_list):,} k-mers derived from --extra-corpus")
        elif pin_from_extra:
            logger.warning("--pin-from-extra set but no --extra-corpus; skipping pin.")
        if _pin_list is not None and algorithm != "spm":
            logger.warning("k-mer pinning only applies to --algorithm spm; ignored.")

    if algorithm == "bpe":
        train_bpe(
            sequences,
            out=str(out),
            name=name,
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            fast=fast,
            max_training_chars=max_training_chars,
            seed=seed,
            n_pinned=n_pinned,
        )
    elif algorithm == "spm":
        train_google_sentencepiece(
            sequences,
            out=str(out),
            name=name,
            vocab_size=vocab_size,
            k=k,
            fast=fast,
            num_threads=num_threads,
            seed=seed,
            max_training_chars=max_training_chars,
            n_pinned=n_pinned,
            raw_read=raw_read,
            pin_kmers=_pin_list,
        )
    elif algorithm == "unigram":
        train_sentencepiece(
            sequences,
            out=str(out),
            name=name,
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            fast=fast,
            max_training_chars=max_training_chars,
            seed=seed,
            n_pinned=n_pinned,
        )
    else:
        train_wordpiece(
            sequences,
            out=str(out),
            name=name,
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            fast=fast,
        )

    tok_dir = out / name
    manifest_mod.write(
        tok_dir,
        seed=seed,
        k=k,
        tokenizer={
            "algorithm": _ALGO_MANIFEST[algorithm],
            "vocab_size": vocab_size,
            "corpus": str(corpus),
            "sha256_corpus": manifest_mod.sha256_file(corpus),
        },
    )
    _elapsed = _time.perf_counter() - _stage_t0
    logger.log("STAGE", f"[tokenizer:train] done — elapsed={_elapsed:.1f} s → {tok_dir}")
    logger.success(f"Tokenizer + manifest written to {tok_dir}")


def _summarize(values, label):
    """Return a one-line mean/percentile summary string for *values*."""
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"{label}: mean={arr.mean():.1f}  p50={np.percentile(arr, 50):.1f}  "
        f"p95={np.percentile(arr, 95):.1f}  p99={np.percentile(arr, 99):.1f}  "
        f"max={arr.max():.1f}"
    )


@app.command()
def spm_fragmentation(
    tokenizer_path: Path = typer.Option(..., help="Trained tokenizer directory"),
    builder: Path = typer.Option(..., help="FASTA/FASTQ reads (R1 or single)"),
    pair: Optional[Path] = typer.Option(None, help="Paired-end R2 file"),
    file_format: str = typer.Option("fastq", help="BioPython format string"),
    k: int = typer.Option(17, help="K-mer size (must match the tokenizer)"),
    max_position_embeddings: int = typer.Option(
        1280, help="Model position budget; the pass/fail reference for tokens-per-pair"
    ),
    sample: int = typer.Option(5000, help="Number of reads (pairs) to sample"),
    seed: int = typer.Option(3469, help="Sampling seed"),
    verbosity: str = typer.Option("normal", "--verbosity", envvar="TRAP_VERBOSITY"),
):
    """Hard Gate 1 — quantify SPM subword fragmentation on real reads.

    Because SentencePiece is a *subword* tokenizer, a read no longer maps to a
    fixed token count: each overlapping k-mer may split into several pieces. This
    reports the distribution that decides whether the SPM vocab fits the model:

    * **pieces-per-k-mer** — tokens(read) / (len(read) − k + 1). 1.0 means every
      k-mer stayed whole; the broken char-level fallback gives ≈ k. This is the
      regression check that the Metaspace fix landed.
    * **tokens-per-read / tokens-per-pair** — mean/p50/p95/p99/max. The pair
      figure (with [CLS]/[SEP]) is the real model input; p99 must clear
      ``max_position_embeddings`` with margin, or the run is truncating signal.
    * **<unk> coverage** — fraction of real tokens that fell back to ``<unk>``.

    Compare against the Salmon baseline of 271 tokens/pair for a 2×150 bp pair.
    """
    set_verbosity(verbosity)
    tokenizer = load_kmer_tokenizer(tokenizer_path, max_position_embeddings)
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            "Tokenizer is not a fast subword tokenizer; fragmentation is 1.0 by "
            "construction (e.g. the Salmon canonical tokenizer). Gate is trivial."
        )

    raw_read, tok_k = spm_input_mode(tokenizer)
    if getattr(tokenizer, "trap_k", None) is not None:
        k = tok_k  # trust the tokenizer's own k over the CLI default

    # raw_read: feed the raw read (no overlapping k-mer pre-split); else feed
    # the space-joined k-mers the k-mer tokenizer expects.
    def _text(seq):
        return seq if raw_read else kmer_split(k, seq)

    rng = random.Random(seed)
    r1 = GenomeDataset(builder, file_format).sequences
    reads2 = GenomeDataset(pair, file_format).sequences if pair is not None else None
    # Sample over the common range when paired: a sample's mates need not be 1:1
    # (e.g. R2 filtered separately), and indexing R2 past its end would crash.
    n = len(r1) if reads2 is None else min(len(r1), len(reads2))
    idx = rng.sample(range(n), min(sample, n))
    logger.log(
        "STAGE",
        f"[tokenizer:spm_fragmentation] tokenizer={tokenizer_path} "
        f"mode={'raw-read' if raw_read else 'kmer'} reads={n:,} "
        f"sampled={len(idx):,} k={k} max_pos={max_position_embeddings}",
    )

    unk_id = tokenizer.unk_token_id
    ratios, toks_read, toks_pair = [], [], []
    total_real, total_unk = 0, 0
    for i in idx:
        seq0 = r1[i]
        ids0 = tokenizer(_text(seq0), add_special_tokens=False)["input_ids"]
        # kmer mode: tokens per k-mer (1.0 = whole k-mers); raw mode: bases per
        # token (compression — higher is better, 1.0 = char-level worst case).
        denom = (len(seq0) if raw_read else max(len(seq0) - k + 1, 0)) or None
        if denom:
            ratios.append((len(seq0) / len(ids0)) if raw_read else (len(ids0) / denom))
        toks_read.append(len(ids0))
        if reads2 is not None:
            seq1 = reads2[i]
            paired = tokenizer(_text(seq0), _text(seq1), add_special_tokens=True)["input_ids"]
            toks_pair.append(len(paired))
            real_ids = paired
        else:
            real_ids = ids0
        total_real += len(real_ids)
        if unk_id is not None:
            total_unk += sum(1 for t in real_ids if t == unk_id)

    coverage = 100.0 * (1.0 - total_unk / total_real) if total_real else 0.0
    ratio_label = "bases/token" if raw_read else "pieces/kmer"
    logger.log("STAGE", f"[tokenizer:spm_fragmentation] {_summarize(ratios, ratio_label)}")
    logger.log("STAGE", f"[tokenizer:spm_fragmentation] {_summarize(toks_read, 'tokens/read')}")
    if toks_pair:
        logger.log(
            "STAGE", f"[tokenizer:spm_fragmentation] {_summarize(toks_pair, 'tokens/pair')}"
        )
    logger.log(
        "STAGE",
        f"[tokenizer:spm_fragmentation] coverage={coverage:.2f}% "
        f"(unk={total_unk:,}/{total_real:,})",
    )

    budget = toks_pair if toks_pair else toks_read
    p99 = float(np.percentile(np.asarray(budget, dtype=np.float64), 99))
    unit = "pair" if toks_pair else "read"
    if p99 >= max_position_embeddings:
        logger.error(
            f"GATE 1 FAIL: p99 tokens/{unit}={p99:.0f} >= max_position_embeddings="
            f"{max_position_embeddings}; reads are truncating. Lower k, raise "
            f"max_position_embeddings, or drop overlapping-k-mer pre-splitting."
        )
    else:
        logger.success(
            f"GATE 1 PASS: p99 tokens/{unit}={p99:.0f} < max_position_embeddings="
            f"{max_position_embeddings} (margin {max_position_embeddings - p99:.0f})."
        )


class WholeKmerMaskingDataCollator:
    """
    Data collator that applies whole kmer masking for masked language modeling.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        wkm_probability (float): The probability of masking a whole kmer.
    """

    def __init__(self, tokenizer, wwm_probability=0.2):
        self.tokenizer = tokenizer
        self.wwm_probability = wwm_probability

    def __call__(self, features):
        for feature in features:
            # Extract word_ids from the feature
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, self.wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)


if __name__ == "__main__":
    app()
