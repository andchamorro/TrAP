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
        k: K-mer length; also sets ``max_sentencepiece_length``.
        fast: Wrap as ``PreTrainedTokenizerFast`` and save in HF JSON format.
        input_sentence_size: SPM built-in reservoir-sampling cap (sentences).
            Acts as a secondary guard; the primary cap is ``max_training_chars``.
        num_threads: Parallelism for SPM's EM training step.
        seed: RNG seed for subsampling and SPM's internal shuffle.
        max_training_chars: Post-k-mer character budget; sequences are
            subsampled (seeded) to fit.  Keeps SPM RSS predictable.
        n_pinned: Leading sequences in ``raw_datasets`` to include
            unconditionally (see ``_select_training_indices``).
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

    # Estimate max_sentence_length from a random sample of the selected subset.
    # kmer_split(k, seq) produces (len(seq)-k+1) k-mers separated by spaces;
    # for pure ASCII DNA, byte length == char length.
    sample_size = min(1000, len(indices))
    sample_indices = random.Random(seed).sample(indices, sample_size)
    max_sentence_length = int(
        max(len(kmer_split(k, raw_datasets[i]).encode()) for i in sample_indices) * 1.2
    )
    logger.info(f"Estimated max_sentence_length: {max_sentence_length:,} bytes (from {sample_size} samples)")

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
        sentence_iterator=(kmer_split(k, raw_datasets[i]) for i in indices),
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        max_sentencepiece_length=k,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        split_by_unicode_script=False,
        split_by_number=False,
        normalization_rule_name="identity",
        user_defined_symbols="[CLS],[SEP],[MASK]",
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
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers.models import Unigram

    vocab_pieces = []
    with open(vocab_path) as fv:
        for line in fv:
            piece, score = line.rstrip("\n").split("\t")
            vocab_pieces.append((piece, float(score)))

    unk_id = next(
        (i for i, (piece, _) in enumerate(vocab_pieces) if piece == "<unk>"),
        0,
    )
    hf_tokenizer = Tokenizer(Unigram(vocab_pieces, unk_id=unk_id))
    hf_tokenizer.pre_tokenizer = _pre_tokenizers.Whitespace()

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
        logger.success("Google SentencePiece tokenizer training complete.")
    else:
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
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
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
    if algorithm not in ("bpe", "spm", "unigram", "wordpiece"):
        raise typer.BadParameter(
            f"Unknown algorithm {algorithm!r}; expected 'bpe', 'spm', 'unigram', or 'wordpiece'."
        )

    # ------------------------------------------------------------------
    # Debug mode: fast smoke-test that validates each of the four fixes
    # that previously caused the Unigram trainer Rust panic on Grace.
    # Activate with:  TOKENIZER_DEBUG=1 sbatch 10_tokenizer.slurm
    #              or --debug flag
    # ------------------------------------------------------------------
    _DEBUG_MAX_SEQS  = 500
    _DEBUG_VOCAB     = 200
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
