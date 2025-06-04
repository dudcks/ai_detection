import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_featureset(file_path, label):
    df = pd.read_excel(file_path)
    df["label"] = label
    return df

def load_all_feature_data():
    base = Path("cal")

    dataset = {
        "train": [
            (base / "ai.train.xlsx", 0),
            (base / "human.train.xlsx", 1)
        ],
        "valid": [
            (base / "ai.valid.xlsx", 0),
            (base / "human.valid.xlsx", 1)
        ],
        "test": [
            (base / "ai.test.xlsx", 0),
            (base / "human.test.xlsx", 1)
        ],
    }

    result = {}
    for split, file_label_list in dataset.items():
        frames = [load_featureset(f, label) for f, label in file_label_list]
        result[split] = pd.concat(frames, ignore_index=True)

    return result["train"], result["valid"], result["test"]

train_df, valid_df, test_df = load_all_feature_data()


X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_valid = valid_df.drop(columns=["label"])
y_valid = valid_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# 학습
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=4,
    min_samples_split=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 검증
print("📊 [Validation Set]")
print(classification_report(y_valid, clf.predict(X_valid)))

# 테스트
print("📊 [Test Set]")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, "rf_model.pkl")

# import joblib

# model = joblib.load("rf_model.pkl")
# print("✅ 모델 불러오기 완료")

# # 예측
# preds = model.predict(X_test)