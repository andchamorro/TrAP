import re
import os
import argparse
import gzip
from itertools import product
from Bio import bgzf, SeqIO
from loguru import logger
from typing import List

from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import processors

def try_mkdir(dir_name):
    # Save the tokenizer
    try:
        os.makedirs(dir_name)
    except FileExistsError:
            # directory already exists
            pass

class GenomeDataset(Dataset):

    def __init__(self, file_path, file_format, transform=None, target_transform=None):
        self.file_path = file_path
        self.file_format = file_format
        self.transform = transform
        self.target_transform = target_transform
        self.sequences, self.complement, self.ids = self._load_sequences()
        self._index = 0  # Initialize the index for iteration

    def _load_sequences(self):
        sequences = []
        complement = []
        ids = []
        with self._file_handle() as handle:
            for record in SeqIO.parse(handle, self.file_format):
                sequences.append(self._standardization(str(record.seq)))
                complement.append(self._standardization(str(record.seq.reverse_complement())))
                ids.append(str(record.id))
                pass
        return sequences, complement, ids
    
    def _standardization(self, sequence):
        return re.sub(r'[^ACTGN]', '', sequence.upper())
    
    def _file_handle(self):
        if self.file_path.endswith('.gz'):
            return gzip.open(self.file_path, 'rt')
        elif self.file_path.endswith('.bgz'):
            return bgzf.open(self.file_path, 'rt')
        else :
            return open(self.file_path, 'rt')
    
    def __len__(self):
        return len(self.sequences)
    
    def __iter__(self):
        self._index = 0  # Reset the index for a new iteration
        return self
    
    def __next__(self):
        if self._index < len(self.sequences):
            seq = self.sequences[self._index]
            rev = self.complement[self._index]
            self._index += 1
            return seq, rev
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.sequences[index], self.complement[index]
        elif isinstance(index, int):
            if index < 0:
                index += len(self.sequences)
            if index >= len(self.sequences) or index < 0:
                raise IndexError("The index is out of range.")
            seq = self.sequences[index]
            rev = self.complement[index]
            if self.transform:
                seq = self.transform(seq)
                rev = self.transform(rev)
            if self.target_transform:
                index = self.target_transform(index)
            return seq, rev, index
        else:
            raise TypeError("Invalid argument type.")
        
# Seq compression
def seq_to_encoded(seq, encoding=None):
    # Define the nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    # TODO: Handle the padding if the seq is not div by 2
    if encoding in ("pairs", "2-mers", 2):
        # Generate all possible 2-mers
        pairs = [''.join(p) for p in product(nucleotides, repeat=2)]
        
        # Create a dictionary with 2-mers as keys and Unicode characters as values
        pairs_to_unicode = {pair: chr(65 + i) for i, pair in enumerate(pairs)}
        return ''.join(pairs_to_unicode[seq[i:i+2]] for i in range(0, len(seq)-1, 2))
    
    # TODO: Handle the padding if the seq is not div by 3
    if encoding in ("codons", "3-mers", 3):
        # Generate all possible 3-mers (codons)
        codons = [''.join(p) for p in product(nucleotides, repeat=3)]
        # Create a dictionary with 3-mers as keys and Unicode characters as values
        codons_to_unicode = {codon: chr(65 + i) for i, codon in enumerate(codons)}
        return ''.join(codons_to_unicode[seq[i:i+3]] for i in range(0, len(seq)-2, 3))
    # Do nothing
    return seq

def create_lambda_with_globals(s):
    return eval(s, globals())

def _kmer_split(k: int, sequence: str, encoding: str=None) -> List[str]:
    return " ".join([seq_to_encoded(sequence[j: j + k], encoding=encoding) for j in range(len(sequence) - k + 1)])

def dataset_loader(builder='gencode.v47.transcripts.fa.gz', k=18, batch_size=1000, test_split=0.1, num_workers=4):
    logger.info(f"Loading transcripts {builder}.")
    raw_datasets = GenomeDataset(builder, 'fasta')

    def generator_from_iterator(raw_datasets):
        for seq, rev in raw_datasets:
            yield {'sequence': seq, 'kmers': _kmer_split(k, seq)}

    logger.info("Create datasets from GenomeIterator")
    dataset = Dataset.from_generator(generator_from_iterator, gen_kwargs={"raw_datasets": raw_datasets})

    logger.info("Creating train-eval split")
    dataset = dataset.train_test_split(test_size=test_split)

    return dataset

def tokenize_function(examples, tokenizer):
    result = tokenizer(text=examples["kmers"], return_special_tokens_mask=False, truncation=False, verbose=False)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples, chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    parser = argparse.ArgumentParser(description="Tokenize and group genome sequences dataset")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained tokenizer")
    parser.add_argument("--max_position_embeddings", type=int, default=512, help="Maximum position embeddings")
    parser.add_argument("--builder", type=str, required=True, help="Path to the genome dataset")
    parser.add_argument("--k", type=int, default=18, help="K-mer size")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for grouping texts")
    parser.add_argument("--preprocessing_dataset", type=str, required=True, help="Path to save the processed dataset")
    parser.add_argument("--num_workers", default=32, type=int)

    args = parser.parse_args()

    logger.info("Loading pretrained tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path,
                                                        local_files_only=True)
    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
    
    tokenizer.model_max_length = args.max_position_embeddings

    transcripts_dataset = dataset_loader(builder=args.builder, k=args.k, test_split=args.test_split)

    tokenized_datasets = transcripts_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=['sequence', 'kmers'], num_proc=args.num_workers
    )
    try_mkdir(os.path.join(args.preprocessing_dataset, 'tokenized'))
    tokenized_datasets.save_to_disk(os.path.join(args.preprocessing_dataset, 'tokenized'), num_proc=args.num_workers)

    lm_datasets = tokenized_datasets.map(lambda examples: group_texts(examples, args.chunk_size), batched=True, num_proc=args.num_workers)

    try_mkdir(os.path.join(args.preprocessing_dataset, 'grouped'))
    lm_datasets.save_to_disk(os.path.join(args.preprocessing_dataset, 'grouped'), num_proc=args.num_workers)

if __name__ == "__main__":
    main()