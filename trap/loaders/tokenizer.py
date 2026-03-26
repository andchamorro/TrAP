import os
import random
import collections

import numpy as np
from loguru import logger

from tokenizers import normalizers, pre_tokenizers, processors
from tokenizers import BertWordPieceTokenizer, SentencePieceUnigramTokenizer, Regex
from transformers import PreTrainedTokenizerFast, default_data_collator

from trap.utils.kmer import kmer_split, kmer_split_batch


def train_sentencepiece(raw_datasets, google=False, out="./", name="sequencepiece_unigram", 
                        dataset_filter=lambda e: e, vocab_size=10000, batch_size=1024, k=17, max_sentence_length=500000, fast=False):
    
    os.makedirs(os.path.join(out, name), exist_ok=True)
    
    if google:
        import sentencepiece as spm

        # Byte length to determine the max_sentence_length on sentencepiece
        # TODO: Find a mathematic function to calculate the max sentence length of the kmer profile
        sample_from_datasets = [raw_datasets[i] for i in [random.randrange(len(raw_datasets)) for _ in range(10000)]]
        max_sentence_length = max(map(lambda seq: len(kmer_split(k, seq).encode('utf-8')), sample_from_datasets))

        # Initialize an empty tokenizer
        spm.SentencePieceTrainer.train(
            sentence_iterator=map(lambda seq: kmer_split(k, seq), raw_datasets), 
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
            kmer_split_batch(raw_datasets, batch_size, k),
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
        kmer_split_batch(raw_datasets, batch_size, k),
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

class WholeKmerMaskingDataCollator:
    """
    Data collator that applies whole kmer masking for masked language modeling.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        wkm_probability (float): The probability of masking a whole kmer.
    """
    def __init__(self, tokenizer, wwm_probability=0.2):
        self.tokenizer = tokenizer
        self.wwm_probability = wwm_probability

    def __call__(self, features):
        for feature in features:
            # Extract word_ids from the feature
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, self.wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)