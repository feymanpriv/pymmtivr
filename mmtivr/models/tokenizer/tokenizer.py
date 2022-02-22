#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizer."""


import transformers
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class Tokenizer():
	"""Tokenizer class."""
    def __init__(self, model):
        self.model = model
        if self.model == 'CLIP4Clip':
            self.tokenizer = tokenize
        elif self.model == 'FrozenInTime':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
				        "distilbert-base-uncased", TOKENIZERS_PARALLELISM=False)
        else:
            raise ValueError('Model not recognized.') 

    def tokenize(self, captions):
    	"""tokenize"""
        if self.model == 'CLIP4Clip':
            return self.tokenizer(captions, truncate=True)
        elif self.model  == 'FrozenInTime':
            return self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
