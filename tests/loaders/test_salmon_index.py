"""Tests for the Salmon k-mer index-build phase (tokenizer.py)."""

import pytest

from trap.loaders.salmon_tokenizer import SalmonKmerTokenizer
from trap.loaders.tokenizer import build_salmon_index
from trap.utils.canonical_kmer import canonical_code

K = 17


def _write_dump(path, kmer_counts):
    path.write_text("".join(f"{km} {c}\n" for km, c in kmer_counts))
    return str(path)


@pytest.fixture
def dump(tmp_path):
    # Three "conserved" L1 k-mers (high count) + one rare one (count 1).
    kmers = [
        ("ACGTACGTACGTACGTA", 500),
        ("TTTTGGGGCCCCAAAGT", 120),
        ("CAGGAGAGGAGGTGCCA", 40),
        ("AAAAAAAAAAAAAAAAC", 1),
    ]
    return _write_dump(tmp_path / "l1.dump", kmers), kmers


@pytest.mark.unit
class TestBuildSalmonIndex:
    def test_min_count_filters_rare(self, dump, tmp_path):
        path, _ = dump
        _, num_target = build_salmon_index(
            path, out=str(tmp_path), name="tok", k=K, n_hash=256, min_count=10
        )
        assert num_target == 3  # the count-1 k-mer is dropped

    def test_max_target_keeps_most_frequent(self, dump, tmp_path):
        path, _ = dump
        tok, num_target = build_salmon_index(
            path, out=str(tmp_path), name="tok", k=K, n_hash=256, max_target=2
        )
        assert num_target == 2
        # The two most frequent k-mers must be in the target range.
        for km in ("ACGTACGTACGTACGTA", "TTTTGGGGCCCCAAAGT"):
            ids = tok.batch_encode_sequences([km])["input_ids"][0][1:-1]
            assert all(5 <= i < 5 + num_target for i in ids)

    def test_built_tokenizer_loads_and_encodes(self, dump, tmp_path):
        path, _ = dump
        _, _ = build_salmon_index(path, out=str(tmp_path), name="tok", k=K, n_hash=256)
        loaded = SalmonKmerTokenizer.from_pretrained(str(tmp_path / "tok"))
        assert loaded.k == K
        assert loaded.n_hash == 256
        enc = loaded.batch_encode_sequences(["ACGTACGTACGTACGTA"])
        assert enc["input_ids"][0][0] == loaded.cls_token_id

    def test_canonical_consistency_with_revcomp_dump(self, tmp_path):
        # A dump that lists the reverse complement must yield the same target code.
        km = "ACGTACGTACGTACGTA"
        rc = km.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        path = _write_dump(tmp_path / "rc.dump", [(rc, 99)])
        tok, num_target = build_salmon_index(path, out=str(tmp_path), name="tok", k=K, n_hash=256)
        assert num_target == 1
        ids = tok.batch_encode_sequences([km])["input_ids"][0][1:-1]
        assert all(5 <= i < 6 for i in ids)  # forward read hits the rc-derived target
