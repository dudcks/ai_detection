import nltk
from konlpy.tag import Okt
from collections import Counter
import re

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

def cal_avg_sentence_length(text): #글 전체에 있는 문장들이 평균적으로 몇 개의 (의미 있는) 형태소로 이루어져 있는지 (총 형태소 개수 ÷ 총 문장 개수)
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

def calculate_ttr(text): #ttr: 텍스트의 어휘 풍부도를 측정하는 지표로, 전체 단어(토큰) 수 대비 고유 단어(타입) 수의 비율을 계산
    morphemes = okt.morphs(text)

    tokens = [m for m in morphemes if re.match(r'[가-힣a-zA-Z0-9]+', m)]

    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    uniquie_types = len(set(tokens))

    ttr = uniquie_types/total_tokens

    return ttr

def calculate_pos_frequency(text):
    pos_tags = okt.pos(text)

    tags = [tag for word, tag in pos_tags if re.match(r'[가-힣a-zA-Z0-9]+', word)]

    if not tags:
        return {}
    
    total_tags = len(tags)
    tag_counts = Counter(tags)

    pos_frequencies = {tag: count / total_tags for tag, count in tag_counts.items()}
    return pos_frequencies

def get_eojeol_ngrams(text, n):
  """주어진 텍스트에서 어절(띄어쓰기) 단위 n-gram을 추출."""
  eojeols = text.split()
  ngrams = []
  if len(eojeols) >= n:
    for i in range(len(eojeols) - n + 1):
      ngrams.append(tuple(eojeols[i:i+n]))
  return ngrams

def get_morpheme_ngrams(text, n):
  """주어진 텍스트에서 형태소 단위 n-gram을 추출."""
  morphemes = okt.morphs(text)
  ngrams = []
  if len(morphemes) >= n:
    for i in range(len(morphemes) - n + 1):
      ngrams.append(tuple(morphemes[i:i+n]))
  return ngrams

n_value = 1 # 바이그램 (bigram)

korean_text_sample = "인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다. 이 기술은 다양한 산업 분야에 적용될 수 있습니다. 하지만 윤리적인 문제도 고려해야 합니다."
avg_len = cal_avg_sentence_length(korean_text_sample)
ttr_score = calculate_ttr(korean_text_sample)
pos_freq = calculate_pos_frequency(korean_text_sample)
eojeol_bigrams = get_eojeol_ngrams(korean_text_sample, n_value)

print(f"평균 문장 길이 (형태소 기준): {avg_len:.2f}")

print(f"어휘 다양성 (TTR): {ttr_score:.3f}")

print("품사(POS) 태그 빈도:")
for tag, freq in pos_freq.items():
    korean_tag = pos_korean_map.get(tag, tag)
    print(f"  {korean_tag}: {freq:.3f}")

# eojeol_bigram_counts = Counter(eojeol_bigrams)
# for ngram, count in eojeol_bigram_counts.items():
#     print(f"{ngram}: {count}")

morpheme_bigrams = get_morpheme_ngrams(korean_text_sample, n_value)
morpheme_bigram_counts = Counter(morpheme_bigrams)
for ngram, count in morpheme_bigram_counts.items():
    print(f"{ngram}: {count}")
    
