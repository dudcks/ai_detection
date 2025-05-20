import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from .model import TransformerClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "klue/roberta-base"
#model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

checkpoint = torch.load("logs/best-model.pt", map_location=device, weights_only=True)

saved_args = checkpoint.get('args')

if saved_args:
    config = vars(saved_args) if not isinstance(saved_args, dict) else saved_args
    d_model = config.get('d_model', 768)
    nhead = config.get('nhead', 12)
    num_layers = config.get('num_layers', 4)
    num_classes = config.get('num_classes', 2) 
    max_sequence_length = config.get('max_len', 128)
else:
    print("_____Warning: Model config not found in checkpoint, using hardcoded values._____")
    d_model = 768
    nhead = 12
    num_layers = 4
    num_classes = 2

model = TransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    num_classes=num_classes,
    max_len=max_sequence_length
)
print(num_layers)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

def detect_ai_generated_text(text):
    try:
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(x=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        logits = outputs
        probabilities = F.softmax(logits, dim=1)
        ai_probability = probabilities[:, 0].item()

        return round(ai_probability, 4)

    except Exception as e:
        print(f"AI 판별 오류: {e}")
        return None

file_path = 'text8.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

ai_prob = detect_ai_generated_text(text)

print(ai_prob)

try:
    import kss
    KSS_AVAILABLE = True
except ImportError:
    print("경고: 'kss' 라이브러리를 찾을 수 없습니다. 한국어 문장 분리 정확도가 떨어질 수 있습니다.")
    print("개행 문자 또는 마침표 기준으로 문장을 분리합니다.")
    print("정확한 분리를 위해 'pip install kss'를 실행하세요.")
    KSS_AVAILABLE = False   

def get_ai_prob_for_chunk(text_chunk, tokenizer, model, device, max_len):
    """단일 텍스트 청크를 처리하여 AI 생성 확률을 반환합니다."""
    if not text_chunk or not text_chunk.strip(): # 비어있거나 공백만 있는 청크 처리
        return None
    try:
        inputs = tokenizer(
            text_chunk,
            padding='max_length', # 항상 max_length로 패딩 (배치 처리 시 유리하지만, 단일 처리도 가능)
            truncation=True,      # max_len 초과 시 잘라냄
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # 모델의 forward 정의에 맞게 호출 (x=..., attention_mask=...)
            outputs = model(x=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        logits = outputs # 모델 반환값 자체가 로짓
        probabilities = F.softmax(logits, dim=1)
        
        # 중요: 모델 학습 시 클래스 0이 AI인지, 1이 AI인지 확인 필요
        # 현재 코드는 클래스 0을 AI 확률로 가정합니다.
        ai_probability = probabilities[:, 0].item() 

        return round(ai_probability, 4)

    except Exception as e:
        print(f"AI 판별 오류 (청크 처리 중): {e}")
        # 오류 발생 시 어떤 청크에서 문제 생겼는지 확인 위해 일부 출력
        print(f"오류 발생 청크 (앞 50자): {text_chunk[:50]}...") 
        return None

# --- 메인 실행 로직 ---
file_path = 'text8.txt'
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        full_text = file.read()
except FileNotFoundError:
    print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"파일 읽기 오류: {e}")
    exit()

# 1. 텍스트를 문장으로 분리
if KSS_AVAILABLE:
    try:
        sentences = kss.split_sentences(full_text)
    except Exception as e:
        print(f"kss 문장 분리 중 오류 발생: {e}")
        print("대체 방법으로 개행/마침표 기준으로 분리합니다.")
        sentences = [s for s in full_text.split('\n') if s.strip()] # 개행 기준 분리 (1차)
        if not sentences or len(sentences) <= 1: # 개행 분리 실패 시 마침표 기준
             sentences = [s.strip() + '.' for s in full_text.split('.') if s.strip()]
else:
    # kss 미사용 시 대체 분리 (개행 -> 마침표)
    sentences = [s for s in full_text.split('\n') if s.strip()]
    if not sentences or len(sentences) <= 1: 
        sentences = [s.strip() + '.' for s in full_text.split('.') if s.strip()]

print(f"총 {len(sentences)}개의 문장으로 분리됨.")

# 2. 문장들을 max_sequence_length를 넘지 않는 청크로 그룹화
chunks = []
current_chunk_sentences = []
max_len_for_chunking = max_sequence_length - 2 

current_length = 0
for sentence in sentences:
    # 현재 문장의 토큰 수 (특수 토큰 제외하고 계산)
    sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    sentence_length = len(sentence_token_ids)

    # 현재 문장 하나만으로도 max_len을 초과하는 경우
    if sentence_length > max_len_for_chunking:
        print(f"경고: 단일 문장이 너무 깁니다 (토큰 {sentence_length}개). 이 문장은 {max_sequence_length} 토큰으로 잘려서 처리됩니다.")
        # 긴 문장도 일단 청크로 추가 (get_ai_prob_for_chunk 함수에서 잘라낼 것임)
        if current_chunk_sentences: # 이전까지 모은 청크가 있다면 먼저 추가
            chunks.append(" ".join(current_chunk_sentences))
        chunks.append(sentence) # 긴 문장 자체를 하나의 청크로
        current_chunk_sentences = [] # 현재 청크 초기화
        current_length = 0
        continue # 다음 문장으로

    # 현재 문장을 추가하면 max_len을 초과하는 경우
    if current_length + sentence_length > max_len_for_chunking:
        # 이전까지 모아둔 문장들을 하나의 청크로 추가
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        # 현재 문장으로 새 청크 시작
        current_chunk_sentences = [sentence]
        current_length = sentence_length
    # 현재 문장을 추가해도 괜찮은 경우
    else:
        current_chunk_sentences.append(sentence)
        current_length += sentence_length

# 마지막 남은 청크 추가
if current_chunk_sentences:
    chunks.append(" ".join(current_chunk_sentences))

print(f"총 {len(chunks)}개의 청크로 분할됨.")

# 3. 각 청크별 AI 생성 확률 계산
chunk_probabilities = []
print("\n--- 청크별 AI 생성 확률 ---")
for i, chunk_text in enumerate(chunks):
    prob = get_ai_prob_for_chunk(chunk_text, tokenizer, model, device, max_sequence_length)
    if prob is not None:
        chunk_probabilities.append(prob)
        # 청크 내용이 너무 길면 일부만 출력
        chunk_preview = chunk_text[:80].replace("\n", " ") + ("..." if len(chunk_text) > 80 else "")
        print(f"  청크 {i+1:>{len(str(len(chunks)))}}: {prob:.4f}  (내용: \"{chunk_preview}\")")
    else:
        chunk_preview = chunk_text[:80].replace("\n", " ") + ("..." if len(chunk_text) > 80 else "")
        print(f"  청크 {i+1:>{len(str(len(chunks)))}}: 처리 오류 (내용: \"{chunk_preview}\")")

# 4. 최종 확률 계산 (평균) 및 출력
if chunk_probabilities:
    final_avg_prob = sum(chunk_probabilities) / len(chunk_probabilities)
    print("\n--- 최종 집계 결과 ---")
    print(f"평균 AI 생성 확률: {final_avg_prob:.4f}")
    # 필요하다면 다른 집계 방식 추가 가능 (e.g., 최대값)
    final_max_prob = max(chunk_probabilities)
    print(f"최대 AI 생성 확률: {final_max_prob:.4f}")
else:
    print("\n분석할 유효한 텍스트 청크가 없습니다.")

