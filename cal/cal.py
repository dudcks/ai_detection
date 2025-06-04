import nltk
from konlpy.tag import Okt
from collections import Counter
import re
from .utils import *
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def process_jsonl_to_csv(input_filename, output_filename, label_field="label"):
    input_path = Path("data") / input_filename
    output_path = Path("cal") / output_filename

    data = []

    # 먼저 라인 수 미리 로딩 (tqdm에 사용하기 위해)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"🔍 {input_filename} 처리 중"):
        item = json.loads(line)
        text = item.get("text") or item.get("content") or item.get("body")
        label = item.get(label_field)

        if not text:
            continue

        features = extract_features(text)
        features["label"] = label
        data.append(features)

    df = pd.DataFrame(data)

    if output_path.suffix == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 저장 완료: {output_path}")

def extract_features(text):
    return {
        "avg_sentence_length": cal_avg_sentence_length(text),
        "ttr": calculate_ttr(text),
        "pos_ngram_diversity": pos_ngram_diversity(text, n=4),
        "comma_inclusion_rate": comma_inclusion_rate(text),
        "avg_comma_usage_rate": avg_comma_usage_rate(text),
        "avg_comma_relative_position": avg_comma_relative_position(text),
        "avg_segment_length": avg_segment_length(text),
        "pos_diversity_around_commas": pos_diversity_around_commas(text),
        "repeated_morpheme_ratio": repeated_morpheme_ratio(text),
        "avg_commas_per_sentence": avg_commas_per_sentence(text),
    }


# korean_text_sample = "쥐포는 쥐치라는 작은 물고기를 잡아 젓갈로 만든 후, 조미간단조리건조해서 만든 음식입니다. 쥐치회는 잡어먹는 것으로 자연산회이며, 매우 맛있는 음식입니다. 쥐포의 껍데기를 벗긴 후에는, 양이 적어 구이로 먹을만한 양이 아닙니다. 그러나 쥐치를 음식으로 먹을 수는 있습니다. 쥐포의 껍데기를 벗기면 지느러미 부분만 남게 되며, 이 부분은 회나 조림으로 먹을 수 있습니다."

# data = extract_features(korean_text_sample)
    
# print(data)

#process_jsonl_to_csv("ai_data.train.jsonl", "ai.train.xlsx")
process_jsonl_to_csv("ai_data.valid.jsonl", "ai.valid.xlsx")
process_jsonl_to_csv("ai_data.test.jsonl", "ai.test.xlsx")

process_jsonl_to_csv("human_data.train.jsonl", "human.train.xlsx")
process_jsonl_to_csv("human_data.valid.jsonl", "human.valid.xlsx")
process_jsonl_to_csv("human_data.test.jsonl", "human.test.xlsx")