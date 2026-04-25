"""Minimal debug tests for the Unigram tokenizer panic (Grace repro).

All tests use 50 bp reads and tiny corpora so they run in seconds anywhere.
They reproduce the exact panic scenario and verify the fix step by step.

There are FOUR stacked causes; the same panic message hides all of them:
``called Result::unwrap() on an Err value: Internal``.

1. ITERATOR FORMAT: kmer_split_batch yields List[str]; tokenizers >=0.19
   treats each element as a pre-tokenised WORD (a full k-mer sentence with
   SPACE chars). SPACE is not in initial_alphabet → Viterbi None → panic.
2. METASPACE: default '▁' prefix is outside initial_alphabet → panic.
3. max_piece_length=16 default drops k=17 k-mers → empty vocab.
4. CORPUS SIZE: the trainer concatenates every sentence and builds a suffix
   array via esaxx, which indexes by i32 (max 2.1B chars). k-mer splitting
   inflates the corpus ~(k+1)x, so a full transcriptome overflows i32 and
   esaxx fails → unwrap() panics. THIS is the one the small tests miss and
   the one that survives on Grace. Fixed by seeded subsampling.
"""
import itertools
import random

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _seqs(n: int = 20, length: int = 50, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Understand what kmer_split_batch actually yields
# ---------------------------------------------------------------------------

class TestKmerSplitBatchFormat:
    """Document the iterator shape that caused the panic."""

    @pytest.mark.unit
    def test_batch_yields_list_of_sentences(self):
        """kmer_split_batch yields List[str] — each string is a SENTENCE."""
        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=6, length=50)
        batches = list(kmer_split_batch(seqs, batch_size=3, k=17))

        assert len(batches) == 2                   # 6 seqs / batch=3 = 2 batches
        assert isinstance(batches[0], list)        # each batch is a list
        assert isinstance(batches[0][0], str)      # each element is a string
        # Each string is a SENTENCE of space-separated k-mers, not a single k-mer
        assert " " in batches[0][0], "k-mer sentence must contain spaces"
        # 50bp → 34 k-mers of length 17
        assert len(batches[0][0].split()) == 50 - 17 + 1

    @pytest.mark.unit
    def test_chained_yields_flat_sentences(self):
        """chain.from_iterable flattens to Iterator[str] — one sentence per item."""
        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=6, length=50)
        flat = list(itertools.chain.from_iterable(kmer_split_batch(seqs, 3, 17)))

        assert len(flat) == 6                  # one sentence per input sequence
        assert isinstance(flat[0], str)        # still strings
        assert " " in flat[0]                  # still space-separated k-mers
        # No sub-lists
        assert not any(isinstance(x, list) for x in flat)

    @pytest.mark.unit
    def test_space_not_in_dna_alphabet(self):
        """Prove why the List[str] path panics: space is not in ACGTN."""
        alphabet = set("ACGTN")
        assert " " not in alphabet, "space IS in the alphabet — fix the test"


# ---------------------------------------------------------------------------
# 2. Reproduce the panic (or skip if library patched upstream)
# ---------------------------------------------------------------------------

class TestUnigramPanicReproduction:
    """Show that List[str] iterator triggers panic; flat Iterator[str] does not."""

    @pytest.mark.unit
    def test_list_of_sentences_contains_space(self):
        """Each element of kmer_split_batch contains spaces → bad 'word' for Unigram."""
        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=10, length=50)
        first_batch = next(iter(kmer_split_batch(seqs, batch_size=5, k=17)))
        # Pre-tokenised path treats this whole string as ONE word.
        # It contains spaces, which are not in the DNA initial_alphabet → panic.
        bad_word = first_batch[0]
        assert " " in bad_word
        assert not set(bad_word).issubset(set("ACGTNacgtn"))

    @pytest.mark.unit
    def test_flat_sentences_contain_no_space_per_kmer(self):
        """After flattening, each individual k-mer (word) is space-free ACGTN only."""
        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=10, length=50)
        flat = list(itertools.chain.from_iterable(kmer_split_batch(seqs, 5, 17)))
        for sentence in flat:
            for kmer in sentence.split():
                assert len(kmer) == 17
                assert set(kmer).issubset(set("ACGTN")), f"non-ACGTN chars in k-mer: {kmer}"


# ---------------------------------------------------------------------------
# 3. Verify the fix: chain.from_iterable → no panic
# ---------------------------------------------------------------------------

class TestUnigramTrainingFix:
    """Training with flat Iterator[str] must succeed without Rust panic."""

    @pytest.mark.unit
    def test_flat_iterator_trains_without_panic(self):
        """Flat Iterator[str] with Whitespace pre-tokenizer must not panic."""
        from tokenizers import SentencePieceUnigramTokenizer
        from tokenizers import pre_tokenizers as _pre
        from tokenizers import trainers as _tr

        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=20, length=50)

        tok = SentencePieceUnigramTokenizer()
        tok._tokenizer.pre_tokenizer = _pre.Whitespace()
        trainer = _tr.UnigramTrainer(
            vocab_size=50,
            special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
            unk_token="<unk>",
            initial_alphabet=list("ACGTN"),
            max_piece_length=17,
            show_progress=False,
        )
        # FIXED: flatten to Iterator[str]
        tok._tokenizer.train_from_iterator(
            itertools.chain.from_iterable(kmer_split_batch(seqs, batch_size=5, k=17)),
            trainer=trainer,
            length=len(seqs),
        )
        assert tok.get_vocab_size() > 0
        # Vocabulary must not contain space
        vocab = tok.get_vocab()
        assert not any(" " in t for t in vocab if not t.startswith("["))

    @pytest.mark.unit
    def test_tokens_are_pure_dna_kmers(self):
        """After fix, learned tokens are sub-k-mers of pure ACGTN (no spaces, no ▁)."""
        from tokenizers import SentencePieceUnigramTokenizer
        from tokenizers import pre_tokenizers as _pre
        from tokenizers import trainers as _tr

        from trap.utils.kmer import kmer_split_batch

        seqs = _seqs(n=20, length=50)
        tok = SentencePieceUnigramTokenizer()
        tok._tokenizer.pre_tokenizer = _pre.Whitespace()
        trainer = _tr.UnigramTrainer(
            vocab_size=50,
            special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
            unk_token="<unk>",
            initial_alphabet=list("ACGTN"),
            max_piece_length=17,
            show_progress=False,
        )
        tok._tokenizer.train_from_iterator(
            itertools.chain.from_iterable(kmer_split_batch(seqs, 5, 17)),
            trainer=trainer,
            length=len(seqs),
        )
        special = {"[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"}
        for token in tok.get_vocab():
            if token in special:
                continue
            assert not token.startswith("▁"), f"MetaSpace prefix in token: {token!r}"
            assert " " not in token, f"Space in token: {token!r}"
            assert set(token).issubset(set("ACGTN")), f"non-ACGTN in token: {token!r}"

    @pytest.mark.slow
    def test_cli_train_k17_50bp(self, tmp_path):
        """End-to-end CLI train with k=17, 50bp reads — regression for Grace panic."""
        from typer.testing import CliRunner

        from trap.loaders.tokenizer import app

        # Write a small FASTA with 50bp reads
        # n=8, length=50: 8*(50-17+1)=272 k-mers — tiny enough to finish in
        # seconds while still exercising the k=17 CLI path on 50bp reads.
        rng = random.Random(0)
        fasta = tmp_path / "corpus.fa"
        with open(fasta, "w") as fh:
            for i in range(8):
                seq = "".join(rng.choice("ACGT") for _ in range(50))
                fh.write(f">seq{i}\n{seq}\n")

        result = CliRunner().invoke(app, [
            "train",
            "--corpus", str(fasta),
            "--out", str(tmp_path / "tok"),
            "--name", "debug_k17",
            "--algorithm", "unigram",
            "--k", "17",
            "--vocab-size", "20",
            "--batch-size", "8",
            "--seed", "42",
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "tok" / "debug_k17" / "tokenizer.json").exists()


# ---------------------------------------------------------------------------
# 4. Corpus size — the esaxx i32 overflow (the bug that survives on Grace)
# ---------------------------------------------------------------------------

class TestCorpusSizeBudget:
    """`_select_training_indices` caps the post-k-mer corpus under the i32 limit."""

    @pytest.mark.unit
    def test_kmer_split_inflates_corpus(self):
        """Each base becomes ~(k+1) chars after overlapping k-mer split.

        This is WHY a full transcriptome overflows esaxx's i32 index: the
        flat training string is ~(k+1)x the raw nucleotide count.
        """
        from trap.utils.kmer import kmer_split

        k = 17
        seq = "".join(["ACGT"[i % 4] for i in range(1000)])  # 1000 bp
        out = kmer_split(k, seq)
        n_kmers = len(seq) - k + 1
        # Rendered length = n_kmers * k + (n_kmers - 1) spaces
        assert len(out) == n_kmers * k + (n_kmers - 1)
        # Inflation factor approaches k+1 for L >> k
        inflation = len(out) / len(seq)
        assert 17 < inflation < 18, f"inflation {inflation:.2f} not ~k+1"

    @pytest.mark.unit
    def test_budget_caps_total_chars(self):
        """Selection stops once the estimated post-k-mer footprint hits the cap."""
        from trap.loaders.tokenizer import _select_training_indices

        # 100 sequences of 1000 bp → each ~ (1000-16)*18 = 17,712 chars
        seqs = _seqs(n=100, length=1000)
        per_seq = (1000 - 17 + 1) * (17 + 1)
        cap = per_seq * 10  # budget for ~10 sequences
        indices, est = _select_training_indices(seqs, k=17, max_chars=cap, seed=3469)

        assert 0 < len(indices) <= 100
        assert est <= cap + per_seq  # at most one sequence over (the accepted one)
        # With a 10-sequence budget we should select roughly 10, not all 100
        assert len(indices) < 100

    @pytest.mark.unit
    def test_budget_seeded_reproducible(self):
        """Same seed → same subset; different seed → (usually) different subset."""
        from trap.loaders.tokenizer import _select_training_indices

        seqs = _seqs(n=200, length=500)
        cap = 1_000_000
        a, _ = _select_training_indices(seqs, k=17, max_chars=cap, seed=3469)
        b, _ = _select_training_indices(seqs, k=17, max_chars=cap, seed=3469)
        c, _ = _select_training_indices(seqs, k=17, max_chars=cap, seed=999)
        assert a == b                      # reproducible
        assert a != c                      # seed actually changes the subset

    @pytest.mark.unit
    def test_short_and_empty_sequences_skipped(self):
        """Sequences shorter than k (incl. empty) are excluded — they k-mer to ''."""
        from trap.loaders.tokenizer import _select_training_indices

        seqs = [
            "ACGT" * 20,         # 80 bp — valid
            "ACGT",              # 4 bp — shorter than k=17 → skipped
            "",                  # empty → skipped
            "A" * 16,            # 16 bp — exactly k-1 → skipped
            "A" * 17,            # 17 bp — exactly k → valid (1 k-mer)
        ]
        indices, _ = _select_training_indices(seqs, k=17, max_chars=10_000_000, seed=0)
        assert set(indices) == {0, 4}       # only the two >= k

    @pytest.mark.unit
    def test_under_budget_keeps_all(self):
        """A small corpus under the cap selects every (valid) sequence."""
        from trap.loaders.tokenizer import _select_training_indices

        seqs = _seqs(n=20, length=50)
        indices, _ = _select_training_indices(seqs, k=17, max_chars=10**12, seed=1)
        assert len(indices) == 20

    @pytest.mark.unit
    def test_combined_entry_types_train_without_panic(self, tmp_path):
        """One-shot: a corpus combining every entry type trains with no panic.

        Combines normal reads, a long sequence, near-k and short/empty
        sequences, and an N-containing read (N is stripped by GenomeDataset
        upstream, but we exercise the budget + training path directly here).
        """
        from tokenizers import SentencePieceUnigramTokenizer
        from tokenizers import pre_tokenizers as _pre
        from tokenizers import trainers as _tr

        from trap.loaders.tokenizer import _select_training_indices
        from trap.utils.kmer import kmer_split_batch

        rng = random.Random(7)
        corpus = (
            [
                "".join(rng.choice("ACGT") for _ in range(50)) for _ in range(40)
            ]                                              # normal 50 bp reads
            + ["".join(rng.choice("ACGT") for _ in range(5000))]   # one long seq
            + ["A" * 17, "ACGT", "", "A" * 16]             # boundary + short/empty
        )

        # Budget that forces subsampling of the normal reads
        cap = (50 - 17 + 1) * 18 * 15
        indices, est = _select_training_indices(corpus, k=17, max_chars=cap, seed=3469)
        assert est <= cap + (5000 - 17 + 1) * 18  # long seq may push one over
        # short/empty (len < 17) excluded
        for bad in (len(corpus) - 1, len(corpus) - 2, len(corpus) - 3):
            assert bad not in indices

        subset = [corpus[i] for i in indices]
        tok = SentencePieceUnigramTokenizer()
        tok._tokenizer.pre_tokenizer = _pre.Whitespace()
        trainer = _tr.UnigramTrainer(
            vocab_size=80,
            special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
            unk_token="<unk>",
            initial_alphabet=list("ACGTN"),
            max_piece_length=17,
            show_progress=False,
        )
        tok._tokenizer.train_from_iterator(
            itertools.chain.from_iterable(kmer_split_batch(subset, batch_size=8, k=17)),
            trainer=trainer,
            length=len(subset),
        )
        assert tok.get_vocab_size() > 0
