import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd
from pathlib import Path

def load_featureset(file_path, label):
    df = pd.read_excel(file_path)
    df["label"] = label
    return df

def load_test_data():
    base = Path("cal")
    test_files = [
        (base / "ai.test.xlsx", 0),
        (base / "human.test.xlsx", 1)
    ]
    frames = [load_featureset(f, label) for f, label in test_files]
    test_df = pd.concat(frames, ignore_index=True)
    return test_df

# 모델 불러오기
model = joblib.load("rf_model.pkl")
print("✅ 모델 불러오기 완료")

# 테스트 데이터 로드
test_df = load_test_data()

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# 예측
y_pred = model.predict(X_test)

# 평가 지표 출력
print("📊 [Test Set] Classification Report")
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,          # 셀 안에 숫자 표시
    fmt="d",             # 정수 형태로 표시
    cmap="Blues",        # 색상 테마
    xticklabels=["AI", "Human"],  # x축 레이블
    yticklabels=["AI", "Human"]   # y축 레이블
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()