import os
import io
import gzip
import random
from loguru import logger
from Bio import bgzf, SeqIO
from typing import List, Iterator

from tokenizers import normalizers, pre_tokenizers, processors, BertWordPieceTokenizer, SentencePieceUnigramTokenizer, Regex 
from transformers import PreTrainedTokenizerFast

class GenomeIterator:
    def __init__(self, file_path, file_format, fixed_size = None):
        self.file_path = file_path
        self.file_format = file_format
        self.fixed_size = fixed_size
        self.sequences, self.ids = self._load_sequences()
        self._index = 0  # Initialize the index for iteration
    
    def _load_sequences(self):
        sequences = []
        ids = []
        with self._file_handle() as handle:
            for record in SeqIO.parse(handle, self.file_format):
                if self.fixed_size is None:
                    sequences.append(str(record.seq))
                    ids.append(str(record.id))
                    pass
                else:
                    seq = str(record.seq)
                    for i in range(0, len(seq), self.fixed_size):
                        chunk = seq[i:i + self.fixed_size]
                        if len(chunk) == self.fixed_size:
                            sequences.append(chunk)
                            ids.append(str(record.id) + '/' + str(i))
        return sequences, ids
    
    def _file_handle(self):
        if self.file_path.endswith('.gz'):
            return gzip.open(self.file_path, 'rt')
        elif self.file_path.endswith('.bgz'):
            return bgzf.open(self.file_path, 'rt')
        else :
            return open(self.file_path, 'rt')
    
    def __len__(self):
        return len(self.sequences)
    
    def shuffle_sequences(self):
        random.shuffle(self.sequences)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.sequences[index]
        elif isinstance(index, int):
            if index < 0:
                index += len(self.sequences)
            if index >= len(self.sequences) or index < 0:
                raise IndexError("The index is out of range.")
            return self.sequences[index]
        else:
            raise TypeError("Invalid argument type.")
    
    def __iter__(self):
        self._index = 0  # Reset the index for a new iteration
        return self
    
    def __next__(self):
        if self._index < len(self.sequences):
            result = self.sequences[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
    
    def get_batch(self, start, batch_size):
        end = start + batch_size
        return self.sequences[start:end]

def create_lambda_with_globals(s):
    return eval(s, globals())

# Kmer compression
def kmer_to_encoded(kmer, encoding=None):
    # Do nothing
    if encoding is None:
        return kmer
        pass
    # Define a mapping from nucleotides to 2-bit binary representation
    nucleotide_to_bits = {
        'A': '00',
        'C': '01',
        'G': '10',
        'T': '11'
    }
    
    # Convert the kmer to a binary string
    binary_string = ''.join(nucleotide_to_bits[nuc] for nuc in kmer)
    
    # Pad the binary string to make its length a multiple of 8
    padding_length = (8 - len(binary_string) % 8) % 8
    binary_string = binary_string + '0' * padding_length
    
    # Convert the binary string to bytes
    byte_array = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
    
    # Convert the byte array to the specified encoding
    encoded_string = byte_array.decode(encoding, errors='ignore')
    
    return encoded_string

def _kmer_split(k: int, sequence: str, encoding: str=None) -> List[str]:
    return " ".join([kmer_to_encoded(sequence[j: j + k], encoding=encoding) for j in range(len(sequence) - k + 1)])

def _dataset_batch(raw_datasets: Iterator[str], batch_size: int, k: int, encoding: str=None) -> Iterator[str]:
    for i in range(0, len(raw_datasets), batch_size):
        yield [_kmer_split(k, seq, encoding=encoding) for seq in raw_datasets.get_batch(i, batch_size)]

def train_sentencepiece(raw_datasets, google=False, out="./", name="sequencepiece_unigram", 
                        dataset_filter=lambda e: e, vocab_size=10000, batch_size=1024, k=17, max_sentence_length=500000, fast=False):
    
    try:
        os.makedirs(os.path.join(out, name))
    except FileExistsError:
        # directory already exists
        pass
    
    if google:
        import sentencepiece as spm

        # Byte length to determine the max_sentence_length on sentencepiece
        # TODO: Find a mathematic function to calculate the max sentence length of the kmer profile
        sample_from_datasets = [raw_datasets[i] for i in [random.randint(0, len(raw_datasets)) for _ in range(10000)]]
        max_sentence_length = max(map(lambda seq: len(_kmer_split(k, seq).encode('utf-8')), sample_from_datasets))

        # Initialize an empty tokenizer
        spm.SentencePieceTrainer.train(
            sentence_iterator=map(lambda seq: _kmer_split(k, seq), raw_datasets), 
            model_writer=os.path.join(out, f'{name}.google'), model_type='unigram', vocab_size=vocab_size,
            # https://github.com/google/sentencepiece/issues/341#issuecomment-505471561
            # Try --input_sentence_size=1000000 (or smaller) which allows to sample sentences before training
            max_sentencepiece_length=k, max_sentence_length=max_sentence_length, input_sentence_size=1000000,
            user_defined_symbols="[CLS],[SEP],[MASK]", pad_id=3,
            train_extremely_large_corpus=True, num_threads=16)
        pass
    else:
        # Initialize an empty tokenizer
        tokenizer = SentencePieceUnigramTokenizer()
        # And then train
        logger.info("Training tokenizer...")
        tokenizer.train_from_iterator(
            _dataset_batch(raw_datasets, batch_size, k),
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]",],
            unk_token="<unk>",
            length = len(raw_datasets)
        )
        pass
    logger.info("Post processing ...")
    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ]
            )
    if fast:
        fast_tokenizer =  PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token='[CLS]', eos_token='[SEP]', 
                unk_token='<unk>', sep_token='[SEP]', 
                cls_token='[CLS]', pad_token='<pad>', mask_token='[MASK]',
                truncation_side='right')
        
        logger.info("Save Tokenizer as fast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        logger.success("Tokenizer training complete.")
        pass
    # Save the files
    else:
        logger.info("Save Tokenizer ...")
        tokenizer.save_model(out, name)
        logger.success("Tokenizer training complete.")
        pass

def train_wordpiece(raw_datasets, out="./", name="wordpiece", 
                    dataset_filter=lambda e: e, vocab_size=10000, batch_size=1024, k=17, fast=False):

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
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        tokenizer.pre_tokenizer,
    ])

    # And then train
    logger.info("Training tokenizer...")
    tokenizer.train_from_iterator(
        _dataset_batch(raw_datasets, batch_size, k),
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]"],
        unk_token="<unk>",
    )

    logger.info("Post processing ...")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ]
    )
    if fast:
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='[CLS]', eos_token='[SEP]', 
            unk_token='<unk>', sep_token='[SEP]', 
           _token='<pad>', mask_token='[MASK]',
            truncation_side='right'
        )
        logger.info("Save Tokenizer as fast ...")
        fast_tokenizer.save_pretrained(os.path.join(out, name))
        logger.success("Tokenizer training complete.")
    else:
        logger.info("Save Tokenizer ...")
        tokenizer.save_model(out, name)
        logger.success("Tokenizer training complete.")