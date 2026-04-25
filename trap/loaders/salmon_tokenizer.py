"""Salmon-consistent canonical k-mer tokenizer (Design C: target + decoy).

Each overlapping k-mer in a read maps to exactly one token id:

* **target** — canonical k-mers present in the L1 reference index get a
  dedicated, collision-free id in ``[n_special, n_special + M)`` (the analog of
  Salmon's target-transcript index entries);
* **decoy** — every other canonical k-mer is feature-hashed (splitmix64) into
  one of ``n_hash`` background buckets in ``[n_special + M, n_special + M + n_hash)``
  (the analog of Salmon's decoy sequence).

K-mers are canonicalised exactly like Salmon / jellyfish (``A=0,C=1,G=2,T=3``,
``canonical = min(forward, revcomp)``), so a read and its reverse complement
produce the same token multiset.  Because every k-mer is one token, a read of
length ``L`` yields ``L - k + 1`` tokens — making the classification padding
length deterministic again.

Special-token ids are pinned to match ``albert_config`` (``[CLS]=0``, ``<pad>=1``,
``[SEP]=2``, ``<unk>=3``, ``[MASK]=4``).  The hot path is
:meth:`SalmonKmerTokenizer.batch_encode_sequences`, which vectorises the
canonical-code → id mapping over a whole batch; the inherited slow ``__call__``
path is kept for compatibility and tests.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from trap.utils.canonical_kmer import canonical_code, canonical_codes, splitmix64

# Pinned special tokens (order fixes the ids; matches albert_config).
_SPECIALS: List[Tuple[str, int]] = [
    ("[CLS]", 0),
    ("<pad>", 1),
    ("[SEP]", 2),
    ("<unk>", 3),
    ("[MASK]", 4),
]
_N_SPECIAL = len(_SPECIALS)

CONFIG_FILE = "salmon_kmer_config.json"
TARGET_CODES_FILE = "target_codes.npy"


class SalmonKmerTokenizer(PreTrainedTokenizer):
    """Canonical k-mer tokenizer with a target index + hashed decoy buckets.

    Args:
        target_codes_file: Path to a ``.npy`` array of canonical target codes
            (``int64``).  ``None`` → empty target (pure feature hashing).
        k: K-mer length.
        n_hash: Number of decoy hash buckets.
    """

    vocab_files_names = {"target_codes_file": TARGET_CODES_FILE}
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        target_codes_file: Optional[str] = None,
        k: int = 17,
        n_hash: int = 65536,
        cls_token: str = "[CLS]",
        pad_token: str = "<pad>",
        sep_token: str = "[SEP]",
        unk_token: str = "<unk>",
        mask_token: str = "[MASK]",
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        **kwargs,
    ):
        if target_codes_file is not None and os.path.exists(str(target_codes_file)):
            codes = np.load(str(target_codes_file)).astype(np.int64)
        else:
            codes = np.empty(0, dtype=np.int64)
        self._target_codes = np.unique(codes)  # sorted + deduped for searchsorted
        self.k = int(k)
        self.n_hash = int(n_hash)
        self._special_to_id = {tok: i for tok, i in _SPECIALS}
        self._id_to_special = {i: tok for tok, i in _SPECIALS}

        # Forward k / n_hash so they are written to tokenizer_config.json and
        # passed back to __init__ by from_pretrained.
        # bos/eos mirror the pinned special-token ids so the model config stays
        # aligned: bos=[CLS]=0, eos=[SEP]=2 (matches albert_config_k17_v48.json).
        # They are explicit params (not hardcoded) so from_pretrained — which
        # re-passes the saved tokens via kwargs — binds them here instead of
        # raising "multiple values for keyword argument 'bos_token'".
        super().__init__(
            k=self.k,
            n_hash=self.n_hash,
            cls_token=cls_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    # -- size / vocab -------------------------------------------------------
    @property
    def num_target(self) -> int:
        return int(self._target_codes.shape[0])

    @property
    def vocab_size(self) -> int:
        return _N_SPECIAL + self.num_target + self.n_hash

    def __len__(self) -> int:
        # The hashed k-mer space is not enumerable via get_vocab(), so the base
        # class's len() (specials only) would undersize any embedding table built
        # from len(tokenizer). All ids live in [0, vocab_size); report that.
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        # Only the special tokens are enumerable (the k-mer space is hashed);
        # this is enough for the base class's special-token bookkeeping.
        vocab = dict(self._special_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    # -- core mapping -------------------------------------------------------
    def _codes_to_ids(self, codes: np.ndarray) -> np.ndarray:
        """Vectorised canonical-code → token-id mapping."""
        m = self.num_target
        if m:
            idx = np.searchsorted(self._target_codes, codes)
            idx_clamped = np.minimum(idx, m - 1)
            hit = self._target_codes[idx_clamped] == codes
        else:
            idx_clamped = np.zeros_like(codes)
            hit = np.zeros(codes.shape, dtype=bool)
        decoy = _N_SPECIAL + m + (splitmix64(codes) % np.uint64(self.n_hash)).astype(np.int64)
        target = _N_SPECIAL + idx_clamped
        return np.where(hit, target, decoy)

    def kmer_ids(self, sequence: str) -> List[int]:
        """Token ids of every overlapping k-mer in *sequence* (no special tokens)."""
        return self._codes_to_ids(canonical_codes(sequence, self.k)).tolist()

    # -- slow-path primitives (compatibility / tests) ----------------------
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        special = self._special_to_id.get(token)
        if special is not None:
            return special
        code = np.array([canonical_code(token)], dtype=np.int64)
        return int(self._codes_to_ids(code)[0])

    def _convert_id_to_token(self, index: int) -> str:
        if index in self._id_to_special:
            return self._id_to_special[index]
        if index < _N_SPECIAL + self.num_target:
            return f"<target:{int(self._target_codes[index - _N_SPECIAL])}>"
        return "<decoy>"

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls, sep = self._special_to_id["[CLS]"], self._special_to_id["[SEP]"]
        if token_ids_1 is None:
            return [cls] + token_ids_0 + [sep]
        return [cls] + token_ids_0 + [sep] + token_ids_1 + [sep]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls, sep = 1, 1
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + cls + sep) + [1] * (len(token_ids_1) + 1)

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    # -- fast path ----------------------------------------------------------
    def batch_encode_sequences(
        self,
        sequences: List[str],
        sequence_pairs: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding=False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_special_tokens_mask: bool = False,
    ) -> BatchEncoding:
        """Vectorised encode of raw read sequences (the stage-20/40 hot path)."""
        max_length = max_length or self.model_max_length
        n_added = 3 if sequence_pairs is not None else 2
        budget = max_length - n_added

        features = []
        for i, seq in enumerate(sequences):
            ids0 = self._codes_to_ids(canonical_codes(seq, self.k)).tolist()
            ids1 = (
                self._codes_to_ids(canonical_codes(sequence_pairs[i], self.k)).tolist()
                if sequence_pairs is not None
                else None
            )
            if truncation:
                ids0, ids1 = _truncate_longest_first(ids0, ids1, budget)
            input_ids = self.build_inputs_with_special_tokens(ids0, ids1)
            feature = {
                "input_ids": input_ids,
                "token_type_ids": self.create_token_type_ids_from_sequences(ids0, ids1),
                "attention_mask": [1] * len(input_ids),
            }
            if return_special_tokens_mask:
                feature["special_tokens_mask"] = self.get_special_tokens_mask(ids0, ids1)
            features.append(feature)

        return self.pad(
            features,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    # -- persistence --------------------------------------------------------
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = (filename_prefix + "-") if filename_prefix else ""
        codes_path = os.path.join(save_directory, prefix + TARGET_CODES_FILE)
        np.save(codes_path, self._target_codes)
        cfg_path = os.path.join(save_directory, prefix + CONFIG_FILE)
        with open(cfg_path, "w") as fh:
            json.dump({"k": self.k, "n_hash": self.n_hash, "num_target": self.num_target}, fh)
        return (codes_path,)

    @property
    def is_fast(self) -> bool:
        return False


def _truncate_longest_first(
    ids0: List[int], ids1: Optional[List[int]], budget: int
) -> Tuple[List[int], Optional[List[int]]]:
    """Drop tokens from the longer member until both fit ``budget`` (HF-style)."""
    if ids1 is None:
        return ids0[:budget], None
    while len(ids0) + len(ids1) > budget:
        if len(ids0) >= len(ids1):
            ids0.pop()
        else:
            ids1.pop()
    return ids0, ids1
