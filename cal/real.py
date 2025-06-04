from .utils import *
import joblib
import pandas as pd

feature_columns = [
    "avg_sentence_length",
    "ttr",
    "pos_ngram_diversity",
    "comma_inclusion_rate",
    "avg_comma_usage_rate",
    "avg_comma_relative_position",
    "avg_segment_length",
    "pos_diversity_around_commas",
    "repeated_morpheme_ratio",
    "avg_commas_per_sentence",
]

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


def extract_features_df(text, feature_columns):
    feat_dict = extract_features(text)
    df = pd.DataFrame([feat_dict])
    # 학습에 사용한 feature 컬럼명과 순서에 맞게 정렬
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def predict_text_label(text, model, feature_columns):
    features_df = extract_features_df(text, feature_columns)
    pred = model.predict(features_df)
    return pred[0]


model = joblib.load("rf_model.pkl")


file_path = 'text/text8.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

predicted_label = predict_text_label(text, model, feature_columns)
print(f"예측된 라벨: {predicted_label} (0=AI, 1=Human)")