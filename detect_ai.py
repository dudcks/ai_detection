# etc module
from bs4 import BeautifulSoup

import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from model import TransformerClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "klue/roberta-base"
#model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

checkpoint = torch.load("best-model.pt", map_location=device, weights_only=True)

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
        #print(ai_probability)

        return round(ai_probability, 4)

    except Exception as e:
        print(f"AI 판별 오류: {e}")
        return None