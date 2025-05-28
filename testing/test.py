from readability import Document
import requests
from bs4 import BeautifulSoup
import time
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from model import TransformerClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "klue/roberta-base"
#model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

checkpoint = torch.load("../logs/best-model.pt", map_location=device, weights_only=True)

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
#print(num_layers)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

def detect_ai_generated_text_kor(full_text, tokenizer, model, device, max_sequence_length=128):
    try:
        import kss
        KSS_AVAILABLE = True
    except ImportError:
        print("경고: 'kss' 라이브러리를 찾을 수 없습니다. 한국어 문장 분리 정확도가 떨어질 수 있습니다.")
        print("개행 문자 또는 마침표 기준으로 문장을 분리합니다.")
        print("정확한 분리를 위해 'pip install kss'를 실행하세요.")
        KSS_AVAILABLE = False  
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

    chunks = []
    current_chunk_sentences = []
    max_len_for_chunking = max_sequence_length - 2 

    current_length = 0
    for sentence in sentences:
        sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_token_ids)

        # 현재 문장 하나만으로도 max_len을 초과하는 경우
        if sentence_length > max_len_for_chunking:
            if current_chunk_sentences: # 이전까지 모은 청크가 있다면 먼저 추가
                chunks.append(" ".join(current_chunk_sentences))
            chunks.append(sentence) # 긴 문장 자체를 하나의 청크로
            current_chunk_sentences = [] # 현재 청크 초기화
            current_length = 0
            continue # 다음 문장으로

        if current_length + sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_length = sentence_length
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_length

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    chunk_probabilities = []
    print("\n--- 청크별 AI 생성 확률 ---")
    for i, chunk_text in enumerate(chunks):
        prob = get_ai_prob_for_chunk(chunk_text, tokenizer, model, device, max_sequence_length)
        if prob is not None:
            chunk_probabilities.append(prob)
            chunk_preview = chunk_text[:80].replace("\n", " ") + ("..." if len(chunk_text) > 80 else "")
            print(f"  청크 {i+1:>{len(str(len(chunks)))}}({len(chunk_text)}):  {prob:.4f}  (내용: \"{chunk_preview}\")")
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

    if final_avg_prob is None:
        return {
            "max_ai_probability": "분석 실패",
            "avg_ai_probability": "분석 실패",
        }
    else:
        return {
            "max_ai_probability": final_max_prob,
            "avg_ai_probability": final_avg_prob,
        }

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

def get_text_from_url(url):
    """ 주어진 URL에서 본문 텍스트 크롤링 """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # 봇 차단 우회
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None

        doc = Document(response.text)
        doc_content = doc.summary()

        soup = BeautifulSoup(doc_content, "html.parser")

        return soup.get_text()
    
    except Exception as e:
        print(f"크롤링 오류: {e}")
        log_file = open("log/log"+time.ctime()+".txt","w")
        log_file.write(url+"\n",e)
        log_file.close()
        return None

def main():

    links = [
        "https://m.blog.naver.com/junkigi11/20173492987",
        # "https://ai3886.tistory.com/1",
        # "https://ai3886.tistory.com/2",
        # "https://ai3886.tistory.com/3",
        # "https://ai3886.tistory.com/4",
        # "https://ai3886.tistory.com/5",
        # "https://ai3886.tistory.com/6",
        # "https://ai3886.tistory.com/7",
        # "https://ai3886.tistory.com/8",
        # "https://ai3886.tistory.com/9",
        # "https://ai3886.tistory.com/10",
        # "https://ai3886.tistory.com/11",
        # "https://ai3886.tistory.com/12",
        # "https://ai3886.tistory.com/13",
        # "https://ai3886.tistory.com/14",
        # "https://ai3886.tistory.com/15",
        # "https://ai3886.tistory.com/16",
        # "https://hrhobby.tistory.com/39",

    ]
    results = []
    for url in links:
        full_text = get_text_from_url(url)
        if not full_text:
            results.append({"url": url, "ai_probability": "크롤링 실패"})
            continue
        result = detect_ai_generated_text_kor(full_text, tokenizer, model, device, max_sequence_length)
        results.append({
            "url": url,
            "max_ai_probability": result["max_ai_probability"],
            "avg_ai_probability": result["avg_ai_probability"]
        })
    for res in results:
        print(f"URL: {res['url']}")
        print(f"최대 AI 생성 확률: {res['max_ai_probability']}")
        print(f"평균 AI 생성 확률: {res['avg_ai_probability']}")
        print("-" * 50)
        time.sleep(1)
        # 각 URL 사이에 1초 대기 (크롤링 서버 부하 방지)

main()