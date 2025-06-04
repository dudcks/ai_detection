import nltk
from konlpy.tag import Okt
from collections import Counter
import re
from nltk.tokenize import sent_tokenize

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

def pos_ngram_diversity(text, n=5):
    pos_tags = [tag for word, tag in okt.pos(text) if re.match(r'[가-힣a-zA-Z0-9]+', word)]
    
    if len(pos_tags) < n:
        return 0.0
    
    ngrams = [tuple(pos_tags[i:i+n]) for i in range(len(pos_tags) - n + 1)]
    diversity = len(set(ngrams)) / len(ngrams)
    
    return diversity

# def get_eojeol_ngrams(text, n):
#   """주어진 텍스트에서 어절(띄어쓰기) 단위 n-gram을 추출."""
#   eojeols = text.split()
#   ngrams = []
#   if len(eojeols) >= n:
#     for i in range(len(eojeols) - n + 1):
#       ngrams.append(tuple(eojeols[i:i+n]))
#   return ngrams

# def get_morpheme_ngrams(text, n):
#   """주어진 텍스트에서 형태소 단위 n-gram을 추출."""
#   morphemes = okt.morphs(text)
#   ngrams = []
#   if len(morphemes) >= n:
#     for i in range(len(morphemes) - n + 1):
#       ngrams.append(tuple(morphemes[i:i+n]))
#   return ngrams

def comma_inclusion_rate(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    count = sum(1 for s in sentences if ',' in s)
    return count / len(sentences)

def avg_comma_usage_rate(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    rates = []
    for s in sentences:
        morphemes = okt.morphs(s)
        if len(morphemes) == 0:
            continue
        comma_count = s.count(',')
        rates.append(comma_count / len(morphemes))

    return sum(rates) / len(rates) if rates else 0.0

def avg_comma_relative_position(text):
    sentences = sent_tokenize(text)
    positions = []

    for s in sentences:
        morphemes = okt.morphs(s)
        if not morphemes or ',' not in s:
            continue

        # 쉼표 위치 찾기
        char_idx = [i for i, c in enumerate(s) if c == ',']
        for idx in char_idx:
            prior = len(okt.morphs(s[:idx]))
            total = len(morphemes)
            if total > 0:
                positions.append(prior / total)

    return sum(positions) / len(positions) if positions else 0.0

def pos_diversity_around_commas(text):
    pos_pairs = []

    for s in sent_tokenize(text):
        tokens = okt.pos(s)
        for i in range(1, len(tokens)-1):
            if tokens[i][0] == ',':
                prev_tag = tokens[i-1][1]
                next_tag = tokens[i+1][1]
                pos_pairs.append((prev_tag, next_tag))

    total = len(pos_pairs)
    unique = len(set(pos_pairs))
    return unique / total if total > 0 else 0.0

def avg_segment_length(text):
    sentences = sent_tokenize(text)
    lengths = []

    for s in sentences:
        segments = s.split(',')
        for seg in segments:
            length = len(okt.morphs(seg))
            if length > 0:
                lengths.append(length)

    return sum(lengths) / len(lengths) if lengths else 0.0

def repeated_morpheme_ratio(text):
    morphs = okt.morphs(text)
    total = len(morphs)
    if total == 0:
        return 0.0
    counts = Counter(morphs)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / total

def avg_commas_per_sentence(text):
    sents = sent_tokenize(text)
    if not sents:
        return 0.0
    return sum(s.count(',') for s in sents) / len(sents)