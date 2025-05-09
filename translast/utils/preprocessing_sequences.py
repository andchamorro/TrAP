import re
import os
import argparse
import gzip
from itertools import product
import numpy as np
from Bio import bgzf, SeqIO
from loguru import logger
from typing import List
from collections import Counter

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
            i = self.ids[self._index]
            self._index += 1
            return seq, rev, i
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
            i = self.ids[index]
            if self.transform:
                seq, rev, i = self.transform(seq, rev, i)
            if self.target_transform:
                index = self.target_transform(index)
            return seq, rev, i, index
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

def get_label_list(raw_dataset, split="train") -> list[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list

def analyze_labels(transcripts_dataset):
    # Example usage
    # # analyze_labels(transcripts_dataset)
    # Extract labels
    int2str = transcripts_dataset['train'].features['label'].int2str
    labels = np.array([int2str(i) for i in transcripts_dataset['train']['label']])
    
    for split in ["eval", "test"]:
        int2str = transcripts_dataset[split].features['label'].int2str
        val_or_test_labels = [int2str(i) for i in transcripts_dataset[split]['label']]
        diff = set(val_or_test_labels).difference(set(labels))
        if len(diff) > 0:
            # Add the labels that appear in val/test but not in train, throw a warning
            logger.warning(
                f"Labels {diff} in {split} set but not in training set, adding them to the label list"
            )
            labels += list(diff)

    # Count all labels
    label_counts = Counter(labels)
    quartiles = np.quantile(list(label_counts.values()), [0.25, 0.5, 0.75])
    print(f"1st Quartile (Q1): {quartiles[0]}")
    print(f"2nd Quartile (Q2)/Median: {quartiles[1]}")
    print(f"3rd Quartile (Q3): {quartiles[2]}")

    # Count NEGATIVE vs the rest
    negative_count = label_counts['NEGATIVE']
    rest_count = sum(count for label, count in label_counts.items() if label != 'NEGATIVE')
    
    # ASCII-based histogram
    max_label_length = max(len(label) for label in label_counts.keys())
    max_count = max(label_counts.values())
    scale_factor = 50 / max_count  # Scale to fit within 50 characters width

    print("\nLabels Histogram:")
    for label, count in label_counts.items():
        bar = "#" * int(count * scale_factor)
        print(f"{label.ljust(max_label_length)}: {bar} ({count})")
    
    print("\nNEGATIVE vs Rest:")
    print(f"NEGATIVE: {negative_count}")
    print(f"Rest: {rest_count}")

def dataset_loader(builder='gencode.v47.transcripts.fa.gz', pair=None, file_format='fasta', k=18, batch_size=1000, test_split=0.1, val_split=None, class_threshold = None, num_proc=4):
    logger.info(f"Loading sequence {builder}.")
    raw_datasets = GenomeDataset(builder, file_format)
    if pair is not None:
        pair_datasets = GenomeDataset(pair, file_format)
        def generator_from_iterator():
            for (r1, rv1, i1), (r2, rv2, i2) in zip(raw_datasets, pair_datasets):
                yield {'read_1': r1, 
                       'kmers_1': _kmer_split(k, r1),
                       'read_2': r2, 
                       'kmers_2': _kmer_split(k, r2), 
                       'label': i1.split('|')[1].split('-')[0]}
    else:
        def generator_from_iterator():
            for seq, rev, id in raw_datasets:
                yield {'sequence': seq, 'kmers': _kmer_split(k, seq), 'label': id.split('|')[1].split('-')[0]}
        pass
    

    logger.info("Create datasets from GenomeIterator")
    dataset = Dataset.from_generator(generator_from_iterator)
    dataset = dataset.class_encode_column('label')
    
    if class_threshold is not None:
        # Count the occurrences of each label
        label_counts = Counter(dataset['label'])
        # Filter the dataset based on the threshold
        dataset = dataset.filter(lambda example: label_counts[example['label']] >= class_threshold, num_proc=num_proc)
        pass

    logger.info("Creating train-test split")
    dataset = dataset.train_test_split(test_size=(test_split + val_split), stratify_by_column = 'label')

    if val_split is not None:
        logger.info("Creating train-eval split")
        test_eval_split = dataset['test'].train_test_split(test_size=test_split/(test_split + val_split), stratify_by_column = 'label')
        dataset['test'] = test_eval_split['train']
        dataset['eval'] = test_eval_split['test']
        pass

    return dataset

def masking(args):
    logger.info("Loading pretrained tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, local_files_only=True)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
            ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
        ],
    )
    
    tokenizer.model_max_length = args.max_position_embeddings

    transcripts_dataset = dataset_loader(builder=args.builder, file_format=args.file_format, k=args.k, test_split=args.test_split, val_split=args.val_split, num_proc=args.num_proc)

    def tokenize_function(examples):
        result = tokenizer(text=examples["kmers"], return_special_tokens_mask=False, truncation=False, verbose=False)
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_datasets = transcripts_dataset.map(
        tokenize_function, batched=True, remove_columns=['sequence', 'kmers', 'id'], num_proc=args.num_proc
    )
    try_mkdir(os.path.join(args.preprocessing_dataset, 'masking', 'tokenized'))
    tokenized_datasets.save_to_disk(os.path.join(args.preprocessing_dataset, 'masking', 'tokenized'))

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

    lm_datasets = tokenized_datasets.map(lambda examples: group_texts(examples, args.chunk_size), batched=True, num_proc=args.num_proc)

    try_mkdir(os.path.join(args.preprocessing_dataset, 'masking', 'grouped'))
    lm_datasets.save_to_disk(os.path.join(args.preprocessing_dataset, 'masking', 'grouped'))

def classification(args):
    logger.info("Loading pretrained tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, local_files_only=True)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
            ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
        ],
    )
    
    tokenizer.model_max_length = args.max_position_embeddings
    
    transcripts_dataset = dataset_loader(builder=args.builder, pair=args.pair, file_format=args.file_format, k=args.k, test_split=args.test_split, val_split=args.val_split, class_threshold = args.class_threshold, num_proc=args.num_proc)

    label_list = transcripts_dataset['train'].features['label'].names
    for split in ["eval", "test"]:
        val_or_test_labels = transcripts_dataset[split].features['label'].names
        diff = set(val_or_test_labels).difference(set(label_list))
        if len(diff) > 0:
            # add the labels that appear in val/test but not in train, throw a warning
            logger.warning(
                f"Labels {diff} in {split} set but not in training set, adding them to the label list"
            )
            label_list += list(diff)
    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == -1:
            logger.warning("Label -1 found in label list, removing it.")
            label_list.remove(label)
    
    label_list.sort()
    num_labels = len(label_list)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")
    
    analyze_labels(transcripts_dataset)

    if args.pair is not None:
        remove_columns=['read_1', 'read_2', 'kmers_1', 'kmers_2']
        def tokenize_function(examples):
            result = tokenizer(examples["kmers_1"], examples["kmers_2"], padding=args.padding, pad_to_multiple_of=8, truncation=True, verbose=False)
            return result
    else:
        remove_columns=['sequence', 'kmers']
        def tokenize_function(examples):
            result = tokenizer(examples["kmers"], padding=args.padding, pad_to_multiple_of=8, truncation=True, verbose=False)
            return result
        pass

    tokenized_datasets = transcripts_dataset.map(
        tokenize_function, batched=True, remove_columns=remove_columns, load_from_cache_file=False, num_proc=args.num_proc,
        desc="Running tokenizer on dataset"
    )
    try_mkdir(os.path.join(args.preprocessing_dataset, 'classification', 'tokenized'))
    tokenized_datasets.save_to_disk(os.path.join(args.preprocessing_dataset, 'classification', 'tokenized'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and group genome sequences dataset")
    subparsers = parser.add_subparsers()

    masking_parser = subparsers.add_parser("masking", help="Masking task")
    masking_parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained tokenizer")
    masking_parser.add_argument("--max_position_embeddings", type=int, default=512, help="Maximum position embeddings")
    masking_parser.add_argument("--builder", type=str, required=True, help="Path to the genome dataset")
    masking_parser.add_argument("--file_format", type=str, default="fasta", help="Format to the genome dataset")
    masking_parser.add_argument("--k", type=int, default=18, help="K-mer size")
    masking_parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    masking_parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio")
    masking_parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for grouping texts")
    masking_parser.add_argument("--preprocessing_dataset", type=str, required=True, help="Path to save the processed dataset")
    masking_parser.add_argument("--num_proc", type=int, default=32, help="Number of workers")
    masking_parser.set_defaults(func=masking)

    classification_parser = subparsers.add_parser("classification", help="Classification task")
    classification_parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained tokenizer")
    classification_parser.add_argument("--max_position_embeddings", type=int, default=512, help="Maximum position embeddings")
    classification_parser.add_argument("--builder", type=str, required=True, help="Path to the genome dataset")
    classification_parser.add_argument("--pair", type=str, default=None, help="Path to the pair read dataset(R2)")
    classification_parser.add_argument("--file_format", type=str, default="fastq", help="Format to the genome dataset")
    classification_parser.add_argument("--k", type=int, default=18, help="K-mer size")
    classification_parser.add_argument("--test_split", type=float, default=0.20, help="Test split ratio")
    classification_parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio")
    classification_parser.add_argument("--class_threshold", type=int, default=10, help="Class filter threshold")
    classification_parser.add_argument("--padding", type=str, default="max_length", help="Validation split ratio")
    classification_parser.add_argument("--preprocessing_dataset", type=str, required=True, help="Path to save the processed dataset")
    classification_parser.add_argument("--num_proc", type=int, default=32, help="Number of workers")
    classification_parser.set_defaults(func=classification)

    args = parser.parse_args()
    args.func(args)