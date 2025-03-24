import os
import argparse
from transformers import PreTrainedTokenizerFast, convert_slow_tokenizer
import sentencepiece as spm
from typing import Union, List

def try_mkdir(dir_name):
    # Save the tokenizer
    try:
        os.makedirs(dir_name)
    except FileExistsError:
            # directory already exists
            pass

def convert_tokens_to_ids(spm_tokenizer, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
    if tokens is None:
        return None
    if isinstance(tokens, str):
        return spm_tokenizer.piece_to_id(tokens)
    ids = []
    for token in tokens:
        ids.append(spm_tokenizer.piece_to_id(token))
    return ids

def convert_sentencepiece_to_huggingface(model_dir, model_name, model_max_length=1024):
    # Define special tokens
    special_tokens = {
        'unk_token': '<unk>',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '<pad>',
        'mask_token': '[MASK]',
        'ambiguous base': '[N]'
    }

    # Load SentencePiece model
    spm_tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, f'{model_name}.model'))
    spm_tokenizer.vocab_file = os.path.join(model_dir, f'{model_name}.model')
    spm_tokenizer.keep_accents = True
    spm_tokenizer.do_lower_case = False
    spm_tokenizer.convert_tokens_to_ids = lambda t: convert_tokens_to_ids(spm_tokenizer, t)

    # Create a custom tokenizer using the SentencePiece model
    hugging_tokenizer = convert_slow_tokenizer.SpmConverter(spm_tokenizer)
    hugging_tokenizer = hugging_tokenizer.converted()

    try_mkdir(os.path.join(model_dir, 'huggingface', 'slow'))
    hugging_tokenizer.save(os.path.join(model_dir, 'huggingface', 'slow', f'{model_name}.json'))
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, 'huggingface', 'slow', f'{model_name}.json'),
                                             model_max_length=model_max_length,
                                             local_files_only=True, **special_tokens)
    fast_tokenizer.save_pretrained(os.path.join(model_dir, 'huggingface', 'fast'))

def main():
    parser = argparse.ArgumentParser(description="Convert Google SentencePiece tokenizer to Hugging Face tokenizer")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the SentencePiece model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the SentencePiece model")
    parser.add_argument("--model_max_length", type=int, default=1024, help="Maximum length of the model")

    args = parser.parse_args()

    convert_sentencepiece_to_huggingface(args.model_dir, args.model_name, args.model_max_length)

if __name__ == "__main__":
    main()