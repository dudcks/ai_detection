
# AI 글 분류기 모델 설계 및 구현 문서

본 문서는 **Transformer 기반의 AI 작성 글 판별 모델**의 구조, 동작 원리, 그리고 학습 과정에 대한 전체적인 설명을 담고 있습니다. 이 모델은 **KoBERT 토크나이저**와 PyTorch의 **Transformer 인코더**를 활용하여, AI가 쓴 글과 사람이 쓴 글을 구분합니다.

---

## 📁 프로젝트 구성

- `model.py` : Transformer 인코더 및 분류기 정의
- `train2.py` : 전체 학습 및 검증 루프
- `dataset2.py` : 데이터셋 구성 및 전처리

---

## 📌 전체 파이프라인 요약
```
[text] → [Tokenizer] → [Transformer] → [Classifier] → [Label]
```

```
1. JSONL 데이터 불러오기
→ {"id":1 "text": "이것은 AI가 쓴 글입니다." }

2. Tokenizer (예: KoBERT)
→ 텍스트를 토큰 ID 시퀀스로 변환 (숫자 벡터)

3. Transformer 인코더 입력
→ 이 숫자 시퀀스를 Transformer에 넣어서
→ 문장 전체의 특징 벡터 추출 ([CLS] 위치 벡터)

4. Classifier (Linear Layer)
→ 특징 벡터를 입력으로 받아
→ 사람/AI 중 하나로 분류 (라벨 예측)
```

```
JSONL 텍스트 데이터
     ↓
KoBERT Tokenizer로 토크나이즈
     ↓
Transformer 인코더 (Positional Encoding 포함)
     ↓
[CLS] 토큰 벡터 추출 → Classifier Linear Layer
     ↓
사람이 썼는지 / AI가 썼는지 분류
```

---

## 🧾 1. 데이터 구성 및 전처리 (`dataset2.py`)

### 🔹 JSONL 데이터
각 데이터는 다음과 같은 형식의 `.jsonl` 파일로 구성:
```json
{"text": "이것은 사람이 쓴 글입니다."}
{"text": "이것은 AI가 쓴 글입니다."}
```

### 🔹 `Corpus` 클래스
- `.train.jsonl`, `.valid.jsonl`, `.test.jsonl` 파일을 로딩
- 각 텍스트 리스트를 반환

### 🔹 `EncodedDataset` 클래스 주요 기능
- Tokenizer로 텍스트 토크나이징
- 랜덤 슬라이스 및 token dropout 적용 가능
- BOS, EOS 토큰 추가
- `max_sequence_length` 기준으로 padding
- Attention mask 생성 (패딩: 0)

---

## 🧠 2. 모델 구조 (`model.py`)

### 🔹 `TransformerClassifier` 구성
```python
class TransformerClassifier(nn.Module):
    def __init__(...):
        self.embedding = nn.Embedding(...)
        self.pos_encoder = PositionalEncoding(...)
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.cls_head = nn.Linear(...)
```

- **Embedding**: 입력 토큰을 고차원 벡터로
- **PositionalEncoding**: 위치 정보 추가
- **TransformerEncoder**: Self-Attention 기반 문장 인코딩
- **[CLS] 위치 벡터 추출** 후 Linear Layer를 통해 분류

---

## 🏋️‍♂️ 3. 학습 루틴 (`train2.py`)

### 🔹 학습 함수 `train()`
- Forward: `model(input_ids, attention_mask)`
- Loss: `CrossEntropyLoss`
- Accuracy: softmax 최대값과 정답 비교
- Optimizer: `Adam`

### 🔹 검증 함수 `validate()`
- 여러 번 투표(vote) → 예측 평균
- 더 안정적인 결과 확보

---

## ⚙️ 4. 학습 실행 (`run` 함수)

- 주요 하이퍼파라미터:
  - `max_epochs`, `batch_size`, `max_sequence_length`
  - `d_model`, `nhead`, `num_layers`, `learning_rate`
  - `token_dropout`, `patience`

- **EarlyStopping** 구현
- **TensorBoard** 로그 작성
- **Best model 자동 저장**
- **`.xlsx` 학습 로그 저장**

---

## 📐 모델 설정 기본값

| 항목 | 값 |
|------|-----|
| 토크나이저 | `klue/roberta-base` 또는 KoBERT |
| d_model | 768 |
| nhead | 12 |
| num_layers | 4 |
| max_sequence_length | 128 |
| batch_size | 24 |
| token_dropout | 선택적 |
| optimizer | Adam |
| loss | CrossEntropyLoss |

---

## ✅ 핵심 정리

- 글 데이터를 Transformer 인코더에 입력하여 문장 벡터 추출
- 해당 벡터를 분류기로 전달하여 AI/사람 여부 판단
- 토크나이저를 통해 입력을 처리하고, 길이 제한, 마스크, 드롭아웃 등의 전처리 수행
- 학습과 검증을 체계적으로 수행하며, 자동 저장 및 조기 종료 기능 포함

---

## 💡 향후 발전 방향

- 다양한 AI 생성 글 스타일을 추가하여 성능 강화
- FastText, CNN 등 경량 모델과 비교 실험
- 모델 앙상블, adversarial training 적용 가능성 검토

---
