"""
Generate TrAP CI test fixtures.

Run once (or re-run to regenerate):
    python tests/fixtures/generate.py

Creates
-------
tests/fixtures/tiny_pair_R1.fq.gz    1 000 × 150-bp read-1 sequences
tests/fixtures/tiny_pair_R2.fq.gz    1 000 × 150-bp read-2 sequences
tests/fixtures/mini_model/            1-layer ALBERT (k=3, vocab=128, hidden=64)
    config.json
    model.safetensors (or pytorch_model.bin)
    tokenizer.json  tokenizer_config.json  special_tokens_map.json  vocab.json

All outputs are deterministic (fixed seed = 42).
"""
from __future__ import annotations

import gzip
import itertools
import random
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent
SEED = 42
N_READS = 1_000
READ_LEN = 150
K = 3                  # k-mer size used by the mini model
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_HEADS = 4
EMBED_SIZE = 32
INTERMEDIATE = 256
MAX_POS = 320          # fits 2 × 148 tokens + 3 special tokens

# A short L1HS-like seed (165 bp); sliced/mutated to produce synthetic reads.
_L1HS_SEED = (
    "ATGGGGAAAGAGTTAAATTTTATTTGATAGTGGTTTATATATTTATAGAGTTATTTATTTTTATCAATAAATTTTTTT"
    "ATTTTAAAATCAACAATTTTTTTTTTTTTGGGGTGGCTATTTTTTTTTTTGTTTTCAAAGTTTTAAGAACTGAAATAG"
    "ATGGGGAAAGAGTTAAATTTTATTTGA"
)  # 165 bp — slice to READ_LEN below

_NEGATIVE_SEED = (
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    "TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    "CTAGCTAGCTAGCTAGCTAGCTAGCTA"
)  # 165 bp synthetic transcript


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACTG") for _ in range(length))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    bases = "ACTG"
    return "".join(
        rng.choice([b for b in bases if b != c]) if rng.random() < rate else c
        for c in seq
    )


# ---------------------------------------------------------------------------
# 1. FASTQ fixtures
# ---------------------------------------------------------------------------

def generate_fastqs(
    r1_path: Path,
    r2_path: Path,
    n: int = N_READS,
    seed: int = SEED,
) -> None:
    rng = random.Random(seed)

    # 30 % L1HS, 10 % L1PA, 60 % NEGATIVE
    n_l1hs = int(n * 0.30)
    n_l1pa = int(n * 0.10)
    n_neg = n - n_l1hs - n_l1pa
    labels = ["L1HS"] * n_l1hs + ["L1PA"] * n_l1pa + ["NEGATIVE"] * n_neg
    rng.shuffle(labels)

    seed_l1 = _L1HS_SEED[:READ_LEN]
    seed_neg = _NEGATIVE_SEED[:READ_LEN]

    with gzip.open(r1_path, "wt") as f1, gzip.open(r2_path, "wt") as f2:
        for i, label in enumerate(labels):
            rid = f"read_{i:04d}|{label}"
            qual = "I" * READ_LEN

            if label == "L1HS":
                r1 = seed_l1
                r2 = _mutate(seed_l1, 0.02, rng)
            elif label == "L1PA":
                r1 = _mutate(seed_l1, 0.05, rng)
                r2 = _mutate(seed_l1, 0.08, rng)
            else:
                r1 = _random_dna(READ_LEN, rng)
                r2 = _random_dna(READ_LEN, rng)

            f1.write(f"@{rid}/1\n{r1}\n+\n{qual}\n")
            f2.write(f"@{rid}/2\n{r2}\n+\n{qual}\n")

    print(f"  wrote {n} read pairs → {r1_path.name}, {r2_path.name}")


# ---------------------------------------------------------------------------
# 2. Mini ALBERT + tokenizer
# ---------------------------------------------------------------------------

def _build_vocab() -> dict[str, int]:
    """Return {token: id} for the mini 3-mer tokenizer."""
    special = ["<pad>", "[CLS]", "[SEP]", "<unk>", "[MASK]"]
    kmers = ["".join(p) for p in itertools.product("ACTG", repeat=K)]
    vocab: dict[str, int] = {}
    for tok in special + kmers:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def generate_mini_model(out_dir: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import AlbertConfig, AlbertForSequenceClassification, PreTrainedTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenizer ---
    vocab = _build_vocab()
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
            ("[SEP]", vocab["[SEP]"]),
        ],
    )

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="<pad>",
        mask_token="[MASK]",
    )
    fast_tok.model_max_length = MAX_POS

    # --- model ---
    id2label = {0: "L1HS", 1: "L1PA", 2: "NEGATIVE"}
    label2id = {v: k for k, v in id2label.items()}

    config = AlbertConfig(
        vocab_size=len(vocab),
        embedding_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POS,
        type_vocab_size=2,
        pad_token_id=vocab["<pad>"],
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )

    import torch
    torch.manual_seed(SEED)
    model = AlbertForSequenceClassification(config)
    model.save_pretrained(str(out_dir))

    # save tokenizer after model so tokenizer_config.json gets model_max_length
    fast_tok.save_pretrained(str(out_dir))

    print(f"  wrote mini model ({model.num_parameters():,} params) → {out_dir}/")


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating TrAP CI fixtures …")

    r1 = FIXTURES / "tiny_pair_R1.fq.gz"
    r2 = FIXTURES / "tiny_pair_R2.fq.gz"
    model_dir = FIXTURES / "mini_model"

    generate_fastqs(r1, r2)
    generate_mini_model(model_dir)

    print("Done.")


if __name__ == "__main__":
    main()
