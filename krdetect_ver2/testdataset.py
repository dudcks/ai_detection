import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


def load_texts(data_file, expected_size=None):
    texts = []

    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line)['text'])

    return texts


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False):
        self.name = name
        self.test = load_texts(f'{data_dir}/{name}.test.jsonl')


class EncodedDataset(Dataset):
    def __init__(self, human_texts: List[str], gpt_texts: List[str], gemini_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.human_texts = human_texts
        self.gpt_texts = gpt_texts
        self.gemini_texts = gemini_texts
        self.tokenizer = tokenizer
        self.effective_max_len = max_sequence_length - 2 if max_sequence_length is not None else None
        self.max_sequence_length = max_sequence_length

        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

        self.all_texts = (
            [(text, 0) for text in self.human_texts] +
            [(text, 1) for text in self.gpt_texts] +
            [(text, 2) for text in self.gemini_texts]
        )

    def __len__(self):
        return self.epoch_size or len(self.human_texts) + len(self.gpt_texts) + len(self.gemini_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            text, label = self.all_texts[self.random.randint(len(self.all_texts))]
        else:
            text, label = self.all_texts[index]

        tokens = self.tokenizer.encode(text)

        if self.effective_max_len is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:

            output_length = min(len(tokens), self.effective_max_len)

            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        num_padding = self.max_sequence_length - (len(tokens) + 2)
        padding = [self.tokenizer.pad_token_id] * num_padding

        # 최종 텐서 생성
        final_tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)

        mask = torch.ones(self.max_sequence_length, dtype=torch.long)
        mask[len(tokens)+2:] = 0 

        return final_tokens, mask, label