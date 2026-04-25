"""Unit tests for the Salmon-consistent canonical k-mer tokenizer."""

import numpy as np
import pytest

from trap.loaders.salmon_tokenizer import SalmonKmerTokenizer
from trap.utils.canonical_kmer import canonical_codes

K = 17


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@pytest.fixture
def read():
    # 40 bp -> 24 overlapping 17-mers
    return "ACGTTGCAACGTACGTTGCAACGTACGTTGCAACGTACGT"


@pytest.fixture
def tokenizer():
    return SalmonKmerTokenizer(k=K, n_hash=4096)


@pytest.mark.unit
class TestSpecialTokens:
    def test_pinned_ids(self, tokenizer):
        assert tokenizer.cls_token_id == 0
        assert tokenizer.pad_token_id == 1
        assert tokenizer.sep_token_id == 2
        assert tokenizer.unk_token_id == 3
        assert tokenizer.mask_token_id == 4

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 5 + tokenizer.num_target + tokenizer.n_hash

    def test_len_matches_vocab_size(self, tokenizer):
        # Embedding tables are sized from len(tokenizer); it must cover the full
        # id range, not just the enumerable special tokens (regression: len()==5
        # built nn.Embedding(5, …) → CUDA gather OOB on the first k-mer id).
        assert len(tokenizer) == tokenizer.vocab_size

    def test_emitted_ids_within_len(self, tokenizer, read):
        ids = tokenizer.batch_encode_sequences([read])["input_ids"][0]
        assert max(ids) < len(tokenizer)


@pytest.mark.unit
class TestEncoding:
    def test_one_token_per_kmer(self, tokenizer, read):
        enc = tokenizer.batch_encode_sequences([read])
        n_kmers = len(read) - K + 1
        assert len(enc["input_ids"][0]) == n_kmers + 2  # + [CLS] [SEP]

    def test_cls_sep_wrap(self, tokenizer, read):
        ids = tokenizer.batch_encode_sequences([read])["input_ids"][0]
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id

    def test_forward_and_revcomp_same_token_multiset(self, tokenizer, read):
        fwd = tokenizer.batch_encode_sequences([read])["input_ids"][0][1:-1]
        rev = tokenizer.batch_encode_sequences([_revcomp(read)])["input_ids"][0][1:-1]
        assert sorted(fwd) == sorted(rev)

    def test_slow_path_matches_fast_path(self, tokenizer, read):
        from trap.utils.kmer import kmer_split

        fast = tokenizer.batch_encode_sequences([read])["input_ids"][0]
        slow = tokenizer(kmer_split(K, read))["input_ids"]
        assert fast == slow

    def test_pair_token_type_ids(self, tokenizer, read):
        enc = tokenizer.batch_encode_sequences([read], [read])
        tti = enc["token_type_ids"][0]
        assert set(tti[: tti.index(1)]) == {0}
        assert tti[-1] == 1


@pytest.mark.unit
class TestTargetVsDecoy:
    def test_target_kmers_get_target_ids(self, read):
        target = canonical_codes(read, K)  # exactly this read's k-mers
        np.save("/tmp/_tgt.npy", target)
        tok = SalmonKmerTokenizer(target_codes_file="/tmp/_tgt.npy", k=K, n_hash=4096)
        m = tok.num_target
        ids = tok.batch_encode_sequences([read])["input_ids"][0][1:-1]
        # Every k-mer of the indexed read must land in the target range.
        assert all(5 <= i < 5 + m for i in ids)

    def test_unindexed_kmers_hash_to_decoy(self, read):
        target = canonical_codes(read, K)
        np.save("/tmp/_tgt.npy", target)
        tok = SalmonKmerTokenizer(target_codes_file="/tmp/_tgt.npy", k=K, n_hash=4096)
        m = tok.num_target
        other = "TTTTTTTTTTTTTTTTTTTTGGGGGGGGGGGGGGGGGGGG"
        ids = tok.batch_encode_sequences([other])["input_ids"][0][1:-1]
        assert all(i >= 5 + m for i in ids)


@pytest.mark.unit
class TestPadding:
    def test_pad_to_max_length_multiple_of_8(self, tokenizer, read):
        enc = tokenizer.batch_encode_sequences(
            [read, read[:30]],
            max_length=296,
            padding="max_length",
            pad_to_multiple_of=8,
        )
        assert all(len(x) == 296 for x in enc["input_ids"])

    def test_truncation_respects_max_length(self, tokenizer, read):
        enc = tokenizer.batch_encode_sequences([read], max_length=10, truncation=True)
        assert len(enc["input_ids"][0]) <= 10


@pytest.mark.unit
class TestRoundTrip:
    def test_save_and_load(self, tmp_path, read):
        target = canonical_codes(read, K)
        np.save("/tmp/_tgt.npy", target)
        tok = SalmonKmerTokenizer(target_codes_file="/tmp/_tgt.npy", k=K, n_hash=2048)
        tok.save_pretrained(str(tmp_path))
        loaded = SalmonKmerTokenizer.from_pretrained(str(tmp_path))
        assert loaded.k == K
        assert loaded.n_hash == 2048
        assert loaded.num_target == tok.num_target
        assert (
            loaded.batch_encode_sequences([read])["input_ids"][0]
            == tok.batch_encode_sequences([read])["input_ids"][0]
        )
