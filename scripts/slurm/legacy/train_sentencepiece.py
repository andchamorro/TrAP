import argparse
import os
import re
import gzip
from memory_profiler import profile
from typing import List
from Bio import bgzf, SeqIO
from itertools import product

from datasets import Dataset
from tokenizers import normalizers, pre_tokenizers, processors, SentencePieceUnigramTokenizer, Regex 
from transformers import PreTrainedTokenizerFast

# Seq compression
def seq_to_encoded(seq, encoding=None):
    # Define the nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    # TODO: Handle the padding if the seq is not div by 2
    if encoding in ("pairs", "2-mers"):
        # Generate all possible 2-mers
        pairs = [''.join(p) for p in product(nucleotides, repeat=2)]
        
        # Create a dictionary with 2-mers as keys and Unicode characters as values
        pairs_to_unicode = {pair: chr(65 + i) for i, pair in enumerate(pairs)}
        return ''.join(pairs_to_unicode[seq[i:i+2]] for i in range(0, len(seq)-1, 2))
    
    # TODO: Handle the padding if the seq is not div by 3
    if encoding in ("codons", "3-mers"):
        # Generate all possible 3-mers (codons)
        codons = [''.join(p) for p in product(nucleotides, repeat=3)]
        # Create a dictionary with 3-mers as keys and Unicode characters as values
        codons_to_unicode = {codon: chr(65 + i) for i, codon in enumerate(codons)}
        return ''.join(codons_to_unicode[seq[i:i+3]] for i in range(0, len(seq)-2, 3))
    # Do nothing
    return seq

def encoded_to_seq(encoded_string, encoding=None):
    # Define the nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    # TODO: Handle the padding if the seq is not div by 2
    if encoding in ("pairs", "2-mers"):
        # Generate all possible 2-mers
        pairs = [''.join(p) for p in product(nucleotides, repeat=2)]
        
        # Create a dictionary with 2-mers as values and Unicode characters as keys
        unicode_to_pairs = {chr(65 + i): pair for i, pair in enumerate(pairs)}
        return ''.join(unicode_to_pairs[e] for e in encoded_string)
    
    # TODO: Handle the padding if the seq is not div by 3
    if encoding in ("codos", "3-mers"):
        # Generate all possible 3-mers (codons)
        codons = [''.join(p) for p in product(nucleotides, repeat=3)]
        # Create a dictionary with 3-mers as values and Unicode characters as keys
        unicode_to_codons = {chr(65 + i): codon for i, codon in enumerate(codons)}
        return ''.join(unicode_to_codons[e] for e in encoded_string)
    # Do nothing
    return encoded_string

class GenomeDataset():

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
                sequences.append(str(record.seq))
                complement.append(str(record.seq.reverse_complement()))
                ids.append(str(record.id))
                pass
        return sequences, complement, ids
    
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

def create_lambda_with_globals(s):
    return eval(s, globals())

def generator_from_iterator(raw_datasets):
    for seq, rev in raw_datasets:
        yield {'sequence': re.sub(r'[^ACTG]', '', seq, flags=re.IGNORECASE).upper()}

def _kmer_split(k: int, sequence: str, encoding: str=None) -> List[str]:
    return " ".join([seq_to_encoded(sequence[j: j + k], encoding=encoding) for j in range(len(sequence) - k + 1)])

def get_training_corpus(raw_datasets, features_names, batch_size, k, encoding: str=None):
    for i in range(0, len(raw_datasets), batch_size):
        for feature in features_names:
            yield [_kmer_split(k, seq, encoding) for seq in raw_datasets[i : i + batch_size][feature]]

mem_logs = open('mem_profile.log','a')
@profile(stream=mem_logs)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--name", default="sequencepiece_unigram", type=str, help="The name of the output vocab files"
    )
    parser.add_argument(
        "--file_path", default="", type=str, help="The path of the input dataset"
    )
    parser.add_argument(
        "--dataset_filter", default="lambda e: e", type=create_lambda_with_globals, help="A lambda filter of the input dataset"
    )
    parser.add_argument(
        "--vocab_size", default=10000, type=int, help="vocab size"
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int
    )
    parser.add_argument(
        "--k", default=18, type=int
    )
    parser.add_argument(
        "--encoding", default="codons", type=str, help="Kmer compression"
    )
    parser.add_argument('--fast', action='store_true')
    parser.add_argument(
        "--num_workers", default=32, type=int
    )
    args = parser.parse_args()

    raw_datasets = GenomeDataset(args.file_path, 'fasta')

    raw_datasets = Dataset.from_generator(generator_from_iterator, gen_kwargs={"raw_datasets": raw_datasets})
    raw_datasets = raw_datasets.shuffle(seed=42)
    raw_datasets = raw_datasets.filter(args.dataset_filter, num_proc=args.num_workers)

    features_names = ["sequence"]

    # Initialize an empty tokenizer
    tokenizer = SentencePieceUnigramTokenizer()

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        tokenizer.pre_tokenizer,])

    # And then train
    tokenizer.train_from_iterator(
        get_training_corpus(raw_datasets, features_names, args.batch_size, args.k, args.encoding),
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]",],
        unk_token="<unk>",
    )

    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
    if args.fast:
        try:
            os.makedirs(os.path.join(args.out, args.name))
        except FileExistsError:
            # directory already exists
            pass
        fast_tokenizer =  PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token='[CLS]', eos_token='[SEP]', 
                unk_token='<unk>', sep_token='[SEP]', 
                cls_token='[CLS]', pad_token='<pad>', mask_token='[MASK]',
                truncation_side='right')
        fast_tokenizer.save_pretrained(os.path.join(args.out, args.name))
    # Save the files
    else:
        try:
            os.makedirs(os.path.join(args.out, args.name))
        except FileExistsError:
            # directory already exists
            pass
        tokenizer.save_model(args.out, args.name)

if __name__ == "__main__":
    main()
