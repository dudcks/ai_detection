import nltk
from konlpy.tag import Okt
from collections import Counter
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

okt = Okt()
pos_korean_map = {
    'Noun': '명사',
    'Verb': '동사',
    'Adjective': '형용사',
    'Adverb': '부사',
    'Determiner': '관형사',
    'Exclamation': '감탄사',
    'Josa': '조사',
    'PreEomi': '선어말어미',
    'Eomi': '어미',  
    'Suffix': '접미사',
    'Punctuation': '구두점',
    'Foreign': '외국어',
    'Alpha': '알파벳',
    'Number': '숫자',
    'Unknown': '미상',
    'KoreanParticle': '조사',
    'Modifier': '수식언' ,
    'Conjunction' : '접속사'

}

def cal_avg_sentence_length(text):
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        print("nltk데이터 필요")
        return 0.0
    
    if not sentences:
        return 0.0

    total_morphemes = 0
    valid_sentence_count = 0

    for sentence in sentences:
        morphemes = okt.morphs(sentence)
        morphemes = [m for m in morphemes if re.match(r'[가-힣a-zA-Z0-9]+',m)]

        if morphemes:
            total_morphemes+=len(morphemes)
            valid_sentence_count+=1
        
    if valid_sentence_count==0:
        return 0.0
    
    avg_len = total_morphemes/valid_sentence_count
    return avg_len

def calculate_ttr(text):
    morphemes = okt.morphs(text)

    tokens = [m for m in morphemes if re.match(r'[가-힣a-zA-Z0-9]+', m)]

    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    uniquie_types = len(set(tokens))

    ttr = uniquie_types/total_tokens

    return ttr


def load_texts(file_path, expected_size=None):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    
    if expected_size and len(texts) != expected_size:
        raise ValueError(f"Expected {expected_size} texts, but got {len(texts)}.")
    
    return texts

ai_data = "./data/ai_data.test.jsonl"
human_data = "./data/human_data.test.jsonl"

ai_texts = load_texts(ai_data, expected_size=51000)
human_texts = load_texts(human_data, expected_size=55000)
print("AI texts:", len(ai_texts))
print("Human texts:", len(human_texts))

ai_ttr=[]
human_ttr=[]
ai_avg_len=[]
human_avg_len=[]

for text in tqdm(ai_texts, desc="AI TTR, AVG_LEN Calculation"):
    ai_ttr.append(calculate_ttr(text))
    ai_avg_len.append(cal_avg_sentence_length(text))
for text in tqdm(human_texts, desc="Human TTR, AVG_LEN Calculation"):
    human_ttr.append(calculate_ttr(text))
    human_avg_len.append(cal_avg_sentence_length(text))

# count=1
# for text in ai_texts:
#     count+=1
#     ai_ttr.append(calculate_ttr(text))
#     ai_avg_len.append(cal_avg_sentence_length(text))
#     if count > 100:
#         break
# count=1
# for text in human_texts:
#     count+=1
#     human_ttr.append(calculate_ttr(text))
#     human_avg_len.append(cal_avg_sentence_length(text))
#     if count > 100:
#         break

ai_mean = np.mean(ai_ttr)
human_mean = np.mean(human_ttr)
ai_max= np.max(ai_ttr)
human_max= np.max(human_ttr)
ai_min= np.min(ai_ttr)      
human_min= np.min(human_ttr)
print(f"AI TTR Mean:{ai_mean:.3f} MAX:{ai_max:.3f} MIN:{ai_min:.3f}")
print(f"Human TTR Mean:{human_mean:.3f} MAX:{human_max:.3f} MIN:{human_min:.3f}")

ai_avg_len_mean = np.mean(ai_avg_len)
human_avg_len_mean = np.mean(human_avg_len)
ai_avg_len_max= np.max(ai_avg_len)
human_avg_len_max= np.max(human_avg_len)
ai_avg_len_min= np.min(ai_avg_len)
human_avg_len_min= np.min(human_avg_len)
print(f"AI Avg Len Mean:{ai_avg_len_mean:.3f} MAX:{ai_avg_len_max:.3f} MIN:{ai_avg_len_min:.3f}")
print(f"Human Avg Len Mean:{human_avg_len_mean:.3f} MAX:{human_avg_len_max:.3f} MIN:{human_avg_len_min:.3f}")        


# plt.figure(figsize=(10, 6))
# plt.hist(ai_ttr, bins=30, alpha=0.5, label='AI TTR', color='blue')      
# plt.hist(human_ttr, bins=30, alpha=0.5, label='Human TTR', color='orange')
# plt.axvline(ai_mean, color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(human_mean, color='orange', linestyle='dashed', linewidth=1)
# plt.axvline(ai_max, color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(human_max, color='orange', linestyle='dashed', linewidth=1)
# plt.axvline(ai_min, color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(human_min, color='orange', linestyle='dashed', linewidth=1)
# plt.title('TTR Distribution')
# plt.xlabel('TTR')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid()
# plt.show()