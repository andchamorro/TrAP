"""Integration tests for the metaspace-fixed Google SentencePiece tokenizer.

These cover the wiring fix on the ``explore/spm-metaspace-tokenizer`` branch:
the Metaspace pre-tokenizer/decoder pairing (so tokenization is content-
preserving instead of collapsing k-mers to char-level pieces), the pinned
special-token ids, and the ``TRAP_SPM_EXPERIMENTAL`` opt-in guard. Training a
real SPM model is a few seconds of C++ work, so the build is a module-scoped
fixture and the cases are marked ``integration``.

Note on what is *not* tested here: SPM is a subword tokenizer and legitimately
splits k-mers into reusable sub-pieces — even a hyper-frequent k-mer is not
guaranteed to survive whole (that is a Salmon property, not an SPM one). The
fragmentation *magnitude* is therefore data-dependent and is measured by the
``spm-fragmentation`` gate on real reads, not asserted as a unit invariant. What
is invariant is that encode→decode preserves the sequence (DNA has full char
coverage, so there is no lossy ``<unk>`` fallback).
"""

import random

import numpy as np
import pytest

from trap.loaders.tokenizer import (
    _SPM_MARKER,
    _SPM_SPECIALS,
    load_kmer_tokenizer,
    spm_input_mode,
    train_google_sentencepiece,
)
from trap.utils.kmer import kmer_split

K = 8


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (the built-in fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def reads():
    # Overlapping 150 bp reads from a small genome so k-mers recur (realistic).
    rng = random.Random(13)
    genome = "".join(rng.choice("ACGT") for _ in range(3000))
    return [genome[i : i + 150] for i in range(0, len(genome) - 150, 7)] * 5


@pytest.fixture(scope="module")
def spm_dir(tmp_path_factory, reads):
    out = tmp_path_factory.mktemp("spm")
    train_google_sentencepiece(
        reads, out=str(out), name="spm8", vocab_size=500, k=K, fast=True, num_threads=2
    )
    return str(out / "spm8")


@pytest.fixture(scope="module")
def tokenizer(spm_dir, monkeypatch_module):
    monkeypatch_module.setenv("TRAP_SPM_EXPERIMENTAL", "1")
    return load_kmer_tokenizer(spm_dir)


@pytest.mark.integration
class TestSpecialTokens:
    def test_pinned_ids(self, tokenizer):
        assert tokenizer.cls_token_id == 0
        assert tokenizer.pad_token_id == 1
        assert tokenizer.sep_token_id == 2
        assert tokenizer.unk_token_id == 3
        assert tokenizer.mask_token_id == 4

    def test_pinned_order_constant(self):
        assert _SPM_SPECIALS == ["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"]


@pytest.mark.integration
class TestContentPreserving:
    def test_decode_round_trips(self, tokenizer, reads):
        # The Metaspace pre-tokenizer and decoder must agree, and DNA's full
        # char coverage means no lossy <unk> — so decode reconstructs the input.
        for r in reads[:50]:
            ids = tokenizer(kmer_split(K, r), add_special_tokens=False)["input_ids"]
            decoded = tokenizer.decode(ids).replace(" ", "")
            assert decoded == kmer_split(K, r).replace(" ", "")

    def test_no_unk_on_dna(self, tokenizer, reads):
        ids = tokenizer(kmer_split(K, reads[0]), add_special_tokens=False)["input_ids"]
        assert tokenizer.unk_token_id not in ids

    def test_not_char_level_collapse(self, tokenizer, reads):
        # The broken Whitespace path falls back to ~k single-char pieces per
        # k-mer. The Metaspace fix must do strictly better than full char-level.
        ratios = []
        for r in reads[:100]:
            ids = tokenizer(kmer_split(K, r), add_special_tokens=False)["input_ids"]
            ratios.append(len(ids) / (len(r) - K + 1))
        assert np.mean(ratios) < K


@pytest.mark.integration
class TestRoundTrip:
    def test_ids_in_range(self, tokenizer, reads):
        ids = tokenizer(kmer_split(K, reads[3]))["input_ids"]
        assert ids
        assert all(0 <= i < len(tokenizer) for i in ids)

    def test_save_load_reproduces_ids(self, tokenizer, spm_dir, reads):
        text = kmer_split(K, reads[5])
        first = tokenizer(text)["input_ids"]
        second = load_kmer_tokenizer(spm_dir)(text)["input_ids"]
        assert first == second


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory, reads):
    """Train a raw-read SPM tokenizer (no k-mer pre-split) on the same reads."""
    out = tmp_path_factory.mktemp("spmraw")
    train_google_sentencepiece(
        reads,
        out=str(out),
        name="raw",
        vocab_size=500,
        k=K,
        fast=True,
        num_threads=2,
        raw_read=True,
    )
    return str(out / "raw")


@pytest.mark.integration
class TestRawRead:
    @pytest.fixture(autouse=True)
    def _optin(self, monkeypatch):
        monkeypatch.setenv("TRAP_SPM_EXPERIMENTAL", "1")

    def test_marker_and_mode(self, raw_dir):
        import os

        assert os.path.exists(os.path.join(raw_dir, _SPM_MARKER))
        tok = load_kmer_tokenizer(raw_dir)
        assert spm_input_mode(tok) == (True, K)

    def test_raw_uses_far_fewer_tokens_than_kmer(self, raw_dir, spm_dir, reads):
        # The whole point: raw-read tokenizes a read to far fewer tokens than the
        # overlapping-k-mer path, which is what keeps pairs inside the budget.
        raw = load_kmer_tokenizer(raw_dir)
        kmer = load_kmer_tokenizer(spm_dir)
        read = reads[0]
        n_raw = len(raw(read, add_special_tokens=False)["input_ids"])
        n_kmer = len(kmer(kmer_split(K, read), add_special_tokens=False)["input_ids"])
        assert n_raw < n_kmer
        assert n_raw <= len(read)  # structural bound: char-level is the worst case

    def test_raw_decode_round_trips(self, raw_dir, reads):
        tok = load_kmer_tokenizer(raw_dir)
        for r in reads[:30]:
            ids = tok(r, add_special_tokens=False)["input_ids"]
            assert tok.decode(ids).replace(" ", "") == r

    def test_raw_specials_pinned(self, raw_dir):
        tok = load_kmer_tokenizer(raw_dir)
        assert (tok.cls_token_id, tok.pad_token_id, tok.sep_token_id) == (0, 1, 2)


@pytest.fixture(scope="module")
def pin_dir(tmp_path_factory, reads):
    """Raw-read tokenizer with two conserved 17-mers pinned as atomic tokens."""
    out = tmp_path_factory.mktemp("spmpin")
    pins = [reads[0][20:37], reads[1][30:47]]  # 17-mers present in the corpus
    train_google_sentencepiece(
        reads,
        out=str(out),
        name="pin",
        vocab_size=500,
        k=17,
        fast=True,
        num_threads=2,
        raw_read=True,
        pin_kmers=pins,
    )
    return str(out / "pin"), pins


@pytest.mark.integration
class TestPinning:
    @pytest.fixture(autouse=True)
    def _optin(self, monkeypatch):
        monkeypatch.setenv("TRAP_SPM_EXPERIMENTAL", "1")

    def test_pinned_kmer_is_a_single_vocab_piece(self, pin_dir):
        # Option 2: the conserved >=16 bp k-mer is forced whole into the vocab,
        # restoring per-token specificity where the entropy argument needs it.
        path, pins = pin_dir
        tok = load_kmer_tokenizer(path)
        for km in pins:
            assert len(km) >= 16
            tid = tok.convert_tokens_to_ids(km)
            assert tid is not None and tid != tok.unk_token_id

    def test_pinning_compresses_a_pinned_kmer(self, pin_dir):
        # A read built around a pinned k-mer tokenizes to fewer pieces than the
        # k-mer's length (it is not shattered to char-level there).
        path, pins = pin_dir
        tok = load_kmer_tokenizer(path)
        read = "AAA" + pins[0] + "TTT"
        ids = tok(read, add_special_tokens=False)["input_ids"]
        assert len(ids) < len(read)


@pytest.mark.integration
class TestOptInGuard:
    def test_load_rejected_without_optin(self, spm_dir, monkeypatch):
        monkeypatch.delenv("TRAP_SPM_EXPERIMENTAL", raising=False)
        with pytest.raises(ValueError, match="DEPRECATED"):
            load_kmer_tokenizer(spm_dir)

    def test_load_allowed_with_optin(self, spm_dir, monkeypatch):
        monkeypatch.setenv("TRAP_SPM_EXPERIMENTAL", "1")
        assert load_kmer_tokenizer(spm_dir) is not None
