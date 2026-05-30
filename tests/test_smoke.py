"""
Phase-0 smoke tests.

Validate that the CI fixtures exist and that the core pipeline components
(FASTQ parsing, k-mer splitting, tokenization, model forward pass, and a
full Dataset.from_generator→map integration run) work correctly on them.

Marks
-----
unit        Fast, no I/O beyond fixture reads.
integration Needs BioPython / HF datasets / transformers; still fast (<5 s).
slow        Full 1 000-read Dataset pipeline; allowed up to 60 s.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MINI_MODEL_DIR = FIXTURES / "mini_model"
R1_PATH = FIXTURES / "tiny_pair_R1.fq.gz"
R2_PATH = FIXTURES / "tiny_pair_R2.fq.gz"

N_READS = 1_000
READ_LEN = 150
K = 3  # k used by the mini model
EXPECTED_TOKENS = READ_LEN - K + 1  # 148 per read


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------

class TestFixturesExist:

    @pytest.mark.unit
    def test_r1_fastq_exists(self):
        assert R1_PATH.exists(), f"Missing fixture: {R1_PATH}"

    @pytest.mark.unit
    def test_r2_fastq_exists(self):
        assert R2_PATH.exists(), f"Missing fixture: {R2_PATH}"

    @pytest.mark.unit
    def test_mini_model_config_exists(self):
        assert (MINI_MODEL_DIR / "config.json").exists()

    @pytest.mark.unit
    def test_mini_model_weights_exist(self):
        has_weights = (MINI_MODEL_DIR / "model.safetensors").exists() or \
                      (MINI_MODEL_DIR / "pytorch_model.bin").exists()
        assert has_weights

    @pytest.mark.unit
    def test_mini_tokenizer_exists(self):
        assert (MINI_MODEL_DIR / "tokenizer.json").exists()


# ---------------------------------------------------------------------------
# FASTQ parsing
# ---------------------------------------------------------------------------

class TestFASTQParsing:

    @pytest.fixture(scope="class")
    def r1_records(self):
        from Bio import SeqIO
        from trap.utils.io import genome_file_handle
        with genome_file_handle(R1_PATH) as fh:
            return list(SeqIO.parse(fh, "fastq"))

    @pytest.fixture(scope="class")
    def r2_records(self):
        from Bio import SeqIO
        from trap.utils.io import genome_file_handle
        with genome_file_handle(R2_PATH) as fh:
            return list(SeqIO.parse(fh, "fastq"))

    @pytest.mark.integration
    def test_r1_has_1000_reads(self, r1_records):
        assert len(r1_records) == N_READS

    @pytest.mark.integration
    def test_r2_has_1000_reads(self, r2_records):
        assert len(r2_records) == N_READS

    @pytest.mark.integration
    def test_all_r1_reads_are_150bp(self, r1_records):
        lengths = {len(r.seq) for r in r1_records}
        assert lengths == {READ_LEN}, f"Unexpected lengths: {lengths}"

    @pytest.mark.integration
    def test_all_r2_reads_are_150bp(self, r2_records):
        lengths = {len(r.seq) for r in r2_records}
        assert lengths == {READ_LEN}, f"Unexpected lengths: {lengths}"

    @pytest.mark.integration
    def test_paired_ids_correspond(self, r1_records, r2_records):
        """R1 id base (strip /1 or /2 suffix) must match R2 id base."""
        r1_bases = [r.id.rsplit("/", 1)[0] for r in r1_records]
        r2_bases = [r.id.rsplit("/", 1)[0] for r in r2_records]
        assert r1_bases == r2_bases

    @pytest.mark.integration
    def test_reads_contain_only_actg(self, r1_records):
        """Standardization should produce pure ACTG (no N in fixture)."""
        from trap.loaders.dataset import GenomeDataset
        for rec in r1_records:
            std = GenomeDataset._standardization(str(rec.seq))
            assert set(std) <= set("ACTG"), f"Unexpected bases in {rec.id}: {set(std)}"


# ---------------------------------------------------------------------------
# K-mer splitting on 150-bp reads
# ---------------------------------------------------------------------------

class TestKmerSplitOn150bp:

    @pytest.mark.unit
    def test_token_count_k3_150bp(self):
        from trap.utils.kmer import kmer_split
        seq = "A" * READ_LEN
        tokens = kmer_split(K, seq).split()
        assert len(tokens) == EXPECTED_TOKENS

    @pytest.mark.unit
    def test_kmer_split_returns_string(self):
        from trap.utils.kmer import kmer_split
        result = kmer_split(K, "ACTGACTGACT")
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_first_token_correct(self):
        from trap.utils.kmer import kmer_split
        tokens = kmer_split(K, "ACTGAC").split()
        assert tokens[0] == "ACT"

    @pytest.mark.unit
    def test_last_token_correct(self):
        from trap.utils.kmer import kmer_split
        tokens = kmer_split(K, "ACTGAC").split()
        assert tokens[-1] == "GAC"

    @pytest.mark.unit
    @pytest.mark.parametrize("k", [3, 17, 18])
    def test_token_count_formula(self, k):
        from trap.utils.kmer import kmer_split
        seq = "ACTG" * 40  # 160 bp
        tokens = kmer_split(k, seq).split()
        expected = len(seq) - k + 1
        assert len(tokens) == expected


# ---------------------------------------------------------------------------
# Mini tokenizer
# ---------------------------------------------------------------------------

class TestMiniTokenizer:

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast.from_pretrained(
            str(MINI_MODEL_DIR), local_files_only=True
        )

    @pytest.mark.integration
    def test_loads_without_error(self, tokenizer):
        assert tokenizer is not None

    @pytest.mark.integration
    def test_special_tokens_present(self, tokenizer):
        assert tokenizer.cls_token == "[CLS]"
        assert tokenizer.sep_token == "[SEP]"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.mask_token == "[MASK]"

    @pytest.mark.integration
    def test_encode_single_kmer_string(self, tokenizer):
        from trap.utils.kmer import kmer_split
        seq = "ACTG" * 5  # 20 bp → 18 3-mers
        kmer_str = kmer_split(K, seq)
        enc = tokenizer(kmer_str)
        # [CLS] + 18 tokens + [SEP] = 20
        assert len(enc["input_ids"]) == (20 - K + 1) + 2

    @pytest.mark.integration
    def test_encode_pair_adds_separator(self, tokenizer):
        from trap.utils.kmer import kmer_split
        r1 = kmer_split(K, "ACTG" * 5)
        r2 = kmer_split(K, "TGCA" * 5)
        enc = tokenizer(r1, r2)
        ids = enc["input_ids"]
        # Should contain two [SEP] tokens (one after A, one after B)
        sep_id = tokenizer.sep_token_id
        assert ids.count(sep_id) == 2

    @pytest.mark.integration
    def test_no_unk_tokens_on_pure_actg_kmers(self, tokenizer):
        from trap.utils.kmer import kmer_split
        seq = "ACTG" * 37 + "AC"  # 150 bp
        kmer_str = kmer_split(K, seq)
        enc = tokenizer(kmer_str)
        assert tokenizer.unk_token_id not in enc["input_ids"]

    @pytest.mark.integration
    def test_max_length_respected(self, tokenizer):
        assert tokenizer.model_max_length == 320

    @pytest.mark.integration
    def test_type_vocab_size_pair(self, tokenizer):
        """Pair encoding must produce token_type_ids when explicitly requested."""
        from trap.utils.kmer import kmer_split
        r1 = kmer_split(K, "ACTG" * 5)
        r2 = kmer_split(K, "TGCA" * 5)
        enc = tokenizer(r1, r2, return_token_type_ids=True)
        token_types = enc["token_type_ids"]
        assert 0 in token_types
        assert 1 in token_types


# ---------------------------------------------------------------------------
# Mini model forward pass
# ---------------------------------------------------------------------------

class TestMiniModelForward:

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        import torch
        from transformers import AlbertForSequenceClassification, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(
            str(MINI_MODEL_DIR), local_files_only=True
        )
        model = AlbertForSequenceClassification.from_pretrained(
            str(MINI_MODEL_DIR), local_files_only=True
        )
        model.eval()
        return model, tok

    @pytest.mark.integration
    def test_loads_without_error(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert model is not None

    @pytest.mark.integration
    def test_id2label_correct(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert model.config.id2label == {0: "L1HS", 1: "L1PA", 2: "NEGATIVE"}

    @pytest.mark.integration
    def test_forward_pass_shape(self, model_and_tokenizer):
        import torch
        from trap.utils.kmer import kmer_split
        model, tok = model_and_tokenizer
        r1 = kmer_split(K, "ACTG" * 37 + "AC")
        r2 = kmer_split(K, "TGCA" * 37 + "TG")
        batch = tok(r1, r2, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**batch)
        # logits shape: (1, 3)
        assert out.logits.shape == (1, 3)

    @pytest.mark.integration
    def test_softmax_sums_to_one(self, model_and_tokenizer):
        import torch
        from trap.utils.kmer import kmer_split
        model, tok = model_and_tokenizer
        r1 = kmer_split(K, "ACTG" * 37 + "AC")
        r2 = kmer_split(K, "TGCA" * 37 + "TG")
        batch = tok(r1, r2, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**batch)
        probs = torch.softmax(out.logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    @pytest.mark.integration
    def test_batch_forward_pass(self, model_and_tokenizer):
        import torch
        from transformers import DataCollatorWithPadding
        from trap.utils.kmer import kmer_split
        model, tok = model_and_tokenizer
        pairs = [
            (kmer_split(K, "ACTG" * 37 + "AC"), kmer_split(K, "TGCA" * 37 + "TG")),
            (kmer_split(K, "TTTT" * 37 + "TT"), kmer_split(K, "AAAA" * 37 + "AA")),
            (kmer_split(K, "GCGC" * 37 + "GC"), kmer_split(K, "CGCG" * 37 + "CG")),
        ]
        encoded = [tok(r1, r2, truncation=True) for r1, r2 in pairs]
        collator = DataCollatorWithPadding(tok)
        batch = collator(encoded)
        with torch.no_grad():
            out = model(**batch)
        # logits shape: (3, 3)
        assert out.logits.shape == (3, 3)


# ---------------------------------------------------------------------------
# Full pipeline integration: FASTQ → Dataset → tokenized
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """
    Mimics the core of predict.processing_dataset (the paired-FASTQ branch)
    without going through the Typer CLI or the MODELS_DIR coupling.
    This is the regression target for Phase 1 refactoring.
    """

    @pytest.fixture(scope="class")
    def tokenized_dataset(self):
        """Build the tokenized dataset using the Phase-1 pipeline architecture.

        The generator now yields raw sequences {r1_seq, r2_seq, id} and
        kmer_split runs *inside* the batched map (QW-1), exercising the same
        code path as predict.processing_dataset.
        """
        from Bio import SeqIO
        from datasets import Dataset
        from transformers import PreTrainedTokenizerFast
        from tokenizers import processors

        from trap.loaders.dataset import GenomeDataset
        from trap.utils.io import genome_file_handle
        from trap.utils.kmer import kmer_split

        tok = PreTrainedTokenizerFast.from_pretrained(
            str(MINI_MODEL_DIR), local_files_only=True
        )
        # Safety net: ensure template is present (QW-10 bakes it in at save time).
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tok.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tok.convert_tokens_to_ids("[SEP]")),
            ],
        )
        tok.model_max_length = 320

        standardize = GenomeDataset._standardization
        _k = K

        # Phase-1 generator: raw sequences only (no kmer_split here)
        def generator():
            with genome_file_handle(R1_PATH) as h1, genome_file_handle(R2_PATH) as h2:
                for r1, r2 in zip(
                    SeqIO.parse(h1, "fastq"), SeqIO.parse(h2, "fastq")
                ):
                    yield {
                        "r1_seq": standardize(str(r1.seq)),
                        "r2_seq": standardize(str(r2.seq)),
                        "id": str(r1.id),
                    }

        raw = Dataset.from_generator(generator)

        # Phase-1 tokenise map: kmer_split + tokenise, batched=True (QW-1, QW-6)
        def tokenize(examples):
            r1_kmers = [kmer_split(_k, s) for s in examples["r1_seq"]]
            r2_kmers = [kmer_split(_k, s) for s in examples["r2_seq"]]
            return tok(r1_kmers, r2_kmers, padding=False, truncation=True, verbose=False)

        return raw.map(
            tokenize,
            batched=True,
            batch_size=128,
            remove_columns=["r1_seq", "r2_seq"],  # keep 'id'
        )

    @pytest.mark.slow
    def test_dataset_has_1000_rows(self, tokenized_dataset):
        assert len(tokenized_dataset) == N_READS

    @pytest.mark.slow
    def test_required_columns_present(self, tokenized_dataset):
        cols = set(tokenized_dataset.column_names)
        # token_type_ids may or may not be present depending on tokenizer version;
        # ALBERT handles missing type IDs by defaulting to zeros.
        assert {"input_ids", "attention_mask", "id"} <= cols

    @pytest.mark.slow
    def test_no_sequence_exceeds_max_length(self, tokenized_dataset):
        lengths = [len(ids) for ids in tokenized_dataset["input_ids"]]
        assert max(lengths) <= 320, f"Longest sequence: {max(lengths)}"

    @pytest.mark.slow
    def test_all_sequences_have_cls_and_sep(self, tokenized_dataset):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(
            str(MINI_MODEL_DIR), local_files_only=True
        )
        cls_id = tok.cls_token_id
        sep_id = tok.sep_token_id
        for ids in tokenized_dataset["input_ids"]:
            assert ids[0] == cls_id, "First token is not [CLS]"
            assert ids[-1] == sep_id, "Last token is not [SEP]"

    @pytest.mark.slow
    def test_read_ids_preserved(self, tokenized_dataset):
        """id column must survive the tokenize map."""
        ids = tokenized_dataset["id"]
        assert len(ids) == N_READS
        # All ids should start with 'read_'
        assert all(i.startswith("read_") for i in ids)

    @pytest.mark.slow
    def test_type_vocab_encodes_pair(self, tokenized_dataset):
        """If token_type_ids are present they must contain both 0 and 1."""
        if "token_type_ids" not in tokenized_dataset.column_names:
            pytest.skip("tokenizer version does not emit token_type_ids by default")
        for token_types in tokenized_dataset["token_type_ids"]:
            assert 0 in token_types
            assert 1 in token_types


# ---------------------------------------------------------------------------
# Phase-1 new utilities
# ---------------------------------------------------------------------------

class TestSeeding:

    @pytest.mark.unit
    def test_set_global_seed_no_error(self):
        from trap.utils.seeding import set_global_seed
        set_global_seed(42)  # must not raise

    @pytest.mark.unit
    def test_set_global_seed_reproducible_torch(self):
        import torch
        from trap.utils.seeding import set_global_seed
        set_global_seed(99)
        a = torch.randn(10)
        set_global_seed(99)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    @pytest.mark.unit
    def test_set_global_seed_reproducible_numpy(self):
        import numpy as np
        from trap.utils.seeding import set_global_seed
        set_global_seed(7)
        a = np.random.rand(10)
        set_global_seed(7)
        b = np.random.rand(10)
        assert np.allclose(a, b)


class TestFastqReader:

    @pytest.mark.integration
    def test_paired_iter_count(self):
        from trap.utils.fastq import paired_fastq_iter
        pairs = list(paired_fastq_iter(R1_PATH, R2_PATH))
        assert len(pairs) == N_READS

    @pytest.mark.integration
    def test_paired_iter_tuple_structure(self):
        from trap.utils.fastq import paired_fastq_iter
        first = next(iter(paired_fastq_iter(R1_PATH, R2_PATH)))
        read_id, r1, r2 = first
        assert isinstance(read_id, str)
        assert len(r1) == READ_LEN
        assert len(r2) == READ_LEN

    @pytest.mark.integration
    def test_paired_iter_ids_match_biopython(self):
        """Both readers must yield the same read base IDs (strip trailing /1)."""
        import re
        from Bio import SeqIO
        from trap.utils.fastq import paired_fastq_iter
        from trap.utils.io import genome_file_handle

        def base_id(s: str) -> str:
            return re.sub(r"/[12]$", "", s)

        fastq_bases = []
        with genome_file_handle(R1_PATH) as h:
            for r in SeqIO.parse(h, "fastq"):
                fastq_bases.append(base_id(str(r.id)))

        reader_bases = [base_id(rid) for rid, _, _ in paired_fastq_iter(R1_PATH, R2_PATH)]
        assert fastq_bases == reader_bases


class TestBreakLongRead:

    @pytest.mark.unit
    def test_seeded_is_reproducible(self):
        import numpy as np
        from trap.modeling.predict import break_long_read

        seq = "ACTG" * 500  # 2 000 bp
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        result_a = break_long_read(seq, rng=rng_a)
        result_b = break_long_read(seq, rng=rng_b)
        assert result_a == result_b

    @pytest.mark.unit
    def test_different_seeds_differ(self):
        import numpy as np
        from trap.modeling.predict import break_long_read

        seq = "ACTG" * 500
        r1 = break_long_read(seq, rng=np.random.default_rng(1))
        r2 = break_long_read(seq, rng=np.random.default_rng(2))
        assert r1 != r2

    @pytest.mark.unit
    def test_short_read_returns_single_pair(self):
        from trap.modeling.predict import break_long_read
        result = break_long_read("ACTG" * 10)  # 40 bp < mean_fragment_size
        assert len(result) == 1
        assert "forward" in result[0]
        assert "reverse" in result[0]


class TestNewQuantifyInferencePath:
    """Smoke-test the Phase-1 DataLoader + autocast inference path on the
    mini model without going through the Typer CLI."""

    @pytest.fixture(scope="class")
    def mini_ds(self, tmp_path_factory):
        """Build a tiny tokenized HF Dataset for inference testing."""
        from Bio import SeqIO
        from datasets import Dataset
        from tokenizers import processors
        from transformers import PreTrainedTokenizerFast

        from trap.loaders.dataset import GenomeDataset
        from trap.utils.io import genome_file_handle
        from trap.utils.kmer import kmer_split

        tmp = tmp_path_factory.mktemp("quantify_ds")
        tok = PreTrainedTokenizerFast.from_pretrained(str(MINI_MODEL_DIR), local_files_only=True)
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tok.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tok.convert_tokens_to_ids("[SEP]")),
            ],
        )
        tok.model_max_length = 320
        standardize = GenomeDataset._standardization

        def gen():
            with genome_file_handle(R1_PATH) as h1, genome_file_handle(R2_PATH) as h2:
                for r1, r2 in zip(SeqIO.parse(h1, "fastq"), SeqIO.parse(h2, "fastq")):
                    yield {
                        "r1_seq": standardize(str(r1.seq)),
                        "r2_seq": standardize(str(r2.seq)),
                        "id": str(r1.id),
                    }

        raw = Dataset.from_generator(gen)

        def tokenize(examples):
            r1_kmers = [kmer_split(K, s) for s in examples["r1_seq"]]
            r2_kmers = [kmer_split(K, s) for s in examples["r2_seq"]]
            return tok(r1_kmers, r2_kmers, padding=False, truncation=True, verbose=False)

        return raw.map(tokenize, batched=True, batch_size=64, remove_columns=["r1_seq", "r2_seq"])

    @pytest.mark.slow
    def test_dataloader_inference_produces_parquet(self, mini_ds, tmp_path):
        """End-to-end: DataLoader → autocast → Parquet with id + 3 label columns."""
        import pyarrow.parquet as pq
        import torch
        from torch.utils.data import DataLoader
        from transformers import AlbertForSequenceClassification, DataCollatorWithPadding, PreTrainedTokenizerFast

        from trap.modeling.predict import _autocast_ctx, _resolve_device

        model = AlbertForSequenceClassification.from_pretrained(str(MINI_MODEL_DIR), local_files_only=True)
        tok = PreTrainedTokenizerFast.from_pretrained(str(MINI_MODEL_DIR), local_files_only=True)
        model.eval()

        device = _resolve_device()
        model = model.to(device)

        collator = DataCollatorWithPadding(tok, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        shard_ids = list(mini_ds["id"]) if "id" in mini_ds.column_names else [str(i) for i in range(len(mini_ds))]
        tensor_cols = [c for c in mini_ds.column_names if c != "id"]
        mini_ds.set_format("torch", columns=tensor_cols)
        loader = DataLoader(mini_ds, batch_size=32, collate_fn=collator, num_workers=0)

        import pyarrow as pa
        schema = pa.schema([("id", pa.string())] + [(lbl, pa.float32()) for lbl in labels])
        out_file = tmp_path / "scores.parquet"

        with pq.ParquetWriter(str(out_file), schema) as writer:
            id_cursor = 0
            with torch.inference_mode(), _autocast_ctx(device):
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    probs = torch.softmax(model(**batch).logits, dim=-1).float().cpu().numpy()
                    n = len(probs)
                    writer.write_table(
                        pa.table(
                            {"id": pa.array(shard_ids[id_cursor: id_cursor + n])}
                            | {lbl: pa.array(probs[:, i], type=pa.float32()) for i, lbl in enumerate(labels)}
                        )
                    )
                    id_cursor += n

        assert out_file.exists()
        df = pq.read_table(str(out_file)).to_pandas()
        assert len(df) == N_READS
        assert list(df.columns) == ["id"] + labels
        assert df[labels].sum(axis=1).between(0.999, 1.001).all(), "Probabilities do not sum to 1"
