import nltk
from konlpy.tag import Okt
import re

okt = Okt()

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

korean_text_sample = "인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다.인공지능 모델은 매우 빠르게 발전하고 있습니다. 이 기술은 다양한 산업 분야에 적용될 수 있습니다. 하지만 윤리적인 문제도 고려해야 합니다."
avg_len = cal_avg_sentence_length(korean_text_sample)
print(f"평균 문장 길이 (형태소 기준): {avg_len:.2f}")
