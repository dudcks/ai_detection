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
        self.train = load_texts(f'{data_dir}/{name}.train.jsonl', expected_size=7429) if not skip_train else None
        self.test = load_texts(f'{data_dir}/{name}.test.jsonl', expected_size=5000)
        self.valid = load_texts(f'{data_dir}/{name}.valid.jsonl', expected_size=5000)


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        # *** max_sequence_length를 내부적으로 사용할 때 2를 빼줍니다 (BOS/EOS 토큰 공간 확보) ***
        # None 체크는 유지
        self.effective_max_len = max_sequence_length - 2 if max_sequence_length is not None else None
        self.max_sequence_length = max_sequence_length # 원래 값도 유지 (패딩 계산 등에 필요할 수 있음)

        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)


    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 1
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 0

        tokens = self.tokenizer.encode(text) # raw 토큰화

        if self.effective_max_len is None: # max_sequence_length가 주어지지 않은 경우
            # 이 로직은 현재 코드에서 사용되지 않을 수 있으나, 원래 코드 참고하여 유지
            tokens = tokens[:self.tokenizer.model_max_length - 2] # BOS/EOS 고려
        else:
            # effective_max_len (원래 max_len - 2) 기준으로 토큰 슬라이싱
            output_length = min(len(tokens), self.effective_max_len) # 최대 길이를 effective_max_len으로 제한

            if self.min_sequence_length:
                # min_sequence_length도 effective_max_len 넘지 않도록 조정 필요 시 추가 가능
                # 여기서는 일단 원래 로직 유지 (min_len ~ output_length 사이에서 랜덤 선택)
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)

            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index] # 실제 텍스트 토큰 슬라이스 (최대 effective_max_len 길이)

        # Token dropout (기존과 동일)
        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        # *** 최종 텐서 생성 및 패딩 (총 길이를 max_sequence_length로 맞춤) ***
        # 현재 tokens 길이는 최대 effective_max_len (126)

        # 필요한 패딩 계산 (BOS/EOS 포함해서 최종 길이가 max_sequence_length가 되도록)
        # 현재 토큰 수 + BOS(1) + EOS(1) + 패딩 = max_sequence_length(128)
        # 패딩 수 = max_sequence_length - (len(tokens) + 2)
        num_padding = self.max_sequence_length - (len(tokens) + 2)
        padding = [self.tokenizer.pad_token_id] * num_padding

        # 최종 텐서 생성
        final_tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)

        # 마스크 생성 (패딩 부분은 0)
        mask = torch.ones(self.max_sequence_length, dtype=torch.long) # 항상 max_sequence_length 길이
        mask[len(tokens)+2:] = 0 # BOS+tokens+EOS 이후 부분을 0으로

        # 최종 반환되는 텐서 길이는 항상 max_sequence_length (128)가 됨
        return final_tokens, mask, label