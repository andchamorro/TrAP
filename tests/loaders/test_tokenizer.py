"""Tests for trap.loaders.tokenizer — WordPiece tokenizer construction."""

import json
import random

import pytest
from typer.testing import CliRunner


def _write_corpus(path, n=40, length=120, seed=0):
    """Write a small random-ACGT FASTA suitable for tokenizer training."""
    rng = random.Random(seed)
    with open(path, "w") as fh:
        for i in range(n):
            seq = "".join(rng.choice("ACGT") for _ in range(length))
            fh.write(f">tx{i}|L1HS-{i}\n{seq}\n")
    return path


class TestTrainWordpieceSignature:
    """Regression tests for the WordPiece fast-tokenizer branch."""

    @pytest.mark.unit
    def test_pad_token_kwarg_is_valid(self):
        """
        tokenizer.py:124 previously had a stray `_token=` instead of
        `pad_token=`. Constructing a PreTrainedTokenizerFast with `pad_token`
        must succeed; the stray key would raise TypeError.
        """
        from tokenizers import BertWordPieceTokenizer
        from transformers import PreTrainedTokenizerFast

        # Minimal vocab for the fast-tokenizer constructor test.
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )
        # The constructor must not raise with `pad_token=`.
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            pad_token="<pad>",
            mask_token="[MASK]",
            truncation_side="right",
        )
        assert fast.pad_token == "<pad>"


class TestTokenizerTrainCLI:
    """The `train` sub-command (plan §6.1) end-to-end on a tiny corpus."""

    @pytest.mark.integration
    def test_train_writes_tokenizer_and_manifest(self, tmp_path):
        from transformers import PreTrainedTokenizerFast

        from trap.loaders.tokenizer import app
        from trap.utils.kmer import kmer_split

        corpus = _write_corpus(tmp_path / "corpus.fasta")
        out = tmp_path / "tok"

        result = CliRunner().invoke(
            app,
            [
                "train",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--name",
                "t",
                "--algorithm",
                "unigram",
                "--k",
                "6",
                "--vocab-size",
                "200",
                "--batch-size",
                "16",
                "--seed",
                "3469",
            ],
        )
        assert result.exit_code == 0, result.output

        tok_dir = out / "t"
        assert (tok_dir / "tokenizer.json").exists()

        # Manifest is written next to the tokenizer with corpus provenance.
        manifest = json.loads((tok_dir / "manifest.json").read_text())
        assert manifest["k"] == 6
        assert manifest["seed"] == 3469
        assert manifest["tokenizer"]["algorithm"] == "sentencepiece-unigram"
        assert manifest["tokenizer"]["vocab_size"] == 200
        assert manifest["tokenizer"]["sha256_corpus"]

        # The saved tokenizer loads and encodes (paired template baked in).
        tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir), local_files_only=True)
        enc = tok(kmer_split(6, "ACGTACGTACGTACGTACGT"))
        assert len(enc["input_ids"]) > 2  # [CLS] ... [SEP]

    @pytest.mark.unit
    def test_train_unigram_k17_no_rust_panic(self, tmp_path):
        """Unigram training with k=17 must not trigger the tokenizers Rust panic.

        Root cause: SentencePieceUnigramTokenizer uses a MetaSpace pre-tokenizer
        that prepends '▁' (U+2581) to every token.  '▁' is not in the DNA
        alphabet (ACGTN), so the Unigram Viterbi forward pass cannot cover the
        first character of any k-mer → log_sum_exp returns None → Rust unwrap()
        panics with "called Result::unwrap() on an Err value: Internal".

        The fix replaces MetaSpace with Whitespace before training: k-mers are
        already space-separated and need no word-boundary prefix.
        max_piece_length is set to k (exact, no ▁ overhead).
        """
        import random

        from tokenizers import SentencePieceUnigramTokenizer
        from tokenizers import pre_tokenizers as _pre
        from tokenizers import trainers as hf_trainers

        from trap.utils.kmer import kmer_split_batch

        rng = random.Random(0)
        seqs = ["".join(rng.choice("ACGT") for _ in range(80)) for _ in range(30)]

        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer._tokenizer.pre_tokenizer = _pre.Whitespace()

        trainer = hf_trainers.UnigramTrainer(
            vocab_size=200,
            special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
            unk_token="<unk>",
            initial_alphabet=list("ACGTN"),
            max_piece_length=17,  # exactly k — no ▁ prefix with Whitespace
            show_progress=False,
        )
        # Must not raise pyo3_runtime.PanicException
        tokenizer._tokenizer.train_from_iterator(
            kmer_split_batch(seqs, batch_size=16, k=17),
            trainer=trainer,
            length=len(seqs),
        )
        assert tokenizer.get_vocab_size() > 0
        # Vocabulary tokens must not contain the MetaSpace ▁ prefix
        vocab = tokenizer.get_vocab()
        assert not any(tok.startswith("▁") for tok in vocab if not tok.startswith("["))

    @pytest.mark.slow
    def test_train_cli_k17(self, tmp_path):
        """CLI train with k=17 must succeed end-to-end (regression for the Rust panic)."""
        from trap.loaders.tokenizer import app

        # Tiny corpus: enough k-mers to prove the CLI path works without
        # triggering hours of SentencePiece Unigram EM at vocab_size=200.
        corpus = _write_corpus(tmp_path / "corpus.fasta", n=8, length=30)
        result = CliRunner().invoke(
            app,
            [
                "train",
                "--corpus", str(corpus),
                "--out", str(tmp_path / "tok"),
                "--name", "t17",
                "--algorithm", "unigram",
                "--k", "17",
                "--vocab-size", "20",
                "--batch-size", "8",
                "--seed", "3469",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "tok" / "t17" / "tokenizer.json").exists()

    @pytest.mark.unit
    def test_train_rejects_unknown_algorithm(self, tmp_path):
        from trap.loaders.tokenizer import app

        corpus = _write_corpus(tmp_path / "corpus.fasta", n=4, length=40)
        result = CliRunner().invoke(
            app,
            [
                "train",
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "o"),
                "--algorithm",
                "bogus",
                "--k",
                "4",
                "--vocab-size",
                "50",
            ],
        )
        assert result.exit_code != 0
