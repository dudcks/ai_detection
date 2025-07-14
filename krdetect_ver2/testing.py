import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from .testdataset import Corpus,EncodedDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .model import TransformerClassifier


def load_datasets(data_dir, human_dataset, gpt_dataset, gemini_dataset, tokenizer,
                  max_sequence_length):

    human_corpus = Corpus(human_dataset, data_dir=data_dir)
    gpt_corpus = Corpus(gpt_dataset, data_dir=data_dir)
    gemini_corpus = Corpus(gemini_dataset, data_dir=data_dir)

    Sampler = RandomSampler 

    test_dataset = EncodedDataset(human_corpus.test, gpt_corpus.test, gemini_corpus.test, tokenizer,  max_sequence_length=max_sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=Sampler(test_dataset))

    return test_loader

def test(model: nn.Module, device: str, loader: DataLoader, desc='test'):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_preds = []
    all_labels = []

    test_loss = 0
    test_epoch_size = 0
    test_accuracy = 0

    with tqdm(loader, desc=desc, disable=False) as loop, torch.no_grad():
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            logits = model(texts, attention_mask=masks)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1).cpu().numpy()  # 예측값
            labels = labels.cpu().numpy()  # 실제 정답

            batch_accuracy = (preds == labels).sum()
            test_accuracy += batch_accuracy
            test_epoch_size += batch_size
            test_loss += loss.item() * batch_size

            # F1-score를 위해 전체 예측값 저장
            all_preds.extend(preds)
            all_labels.extend(labels)

            loop.set_postfix(loss=loss.item(), acc=test_accuracy / test_epoch_size)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)
    labels_names = ['Human', 'GPT', 'Gemini'] 
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    save_filename = 'confusion_matrix.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "test/accuracy": test_accuracy / test_epoch_size,
        "test/loss": test_loss / test_epoch_size,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1_score": f1
    }


def run(model_path,
        data_dir,
        human_dataset,
        gpt_dataset,
        gemini_dataset,
        max_sequence_length,
        device):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    #model_name = "klue/roberta-base"
    model_name = "skt/kobert-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint = torch.load(model_path, map_location=device)

    saved_args = checkpoint.get('args')

    if saved_args:
        config = vars(saved_args) if not isinstance(saved_args, dict) else saved_args
        d_model = config.get('d_model', 768)
        nhead = config.get('nhead', 12)
        num_layers = config.get('num_layers', 4)
        num_classes = config.get('num_classes', 3) 
        max_sequence_length = config.get('max_sequence_length', max_sequence_length)
    else:
        print("_____Warning: Model config not found in checkpoint, using hardcoded values._____")
        d_model = 768
        nhead = 12
        num_layers = 4
        num_classes = 3
    
    model = TransformerClassifier(
        vocab_size=len(tokenizer.get_vocab()),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=max_sequence_length # 테스트 시 인자로 받은 값 사용
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 테스트 데이터 로드
    test_loader = load_datasets(data_dir, human_dataset, gpt_dataset, gemini_dataset, tokenizer, max_sequence_length)

    # 테스트 실행
    test_metrics = test(model, device, test_loader)
    print("Test Results:", test_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./logs/best-model.pt')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--human-dataset', type=str, default='human_data')
    parser.add_argument('--gpt-dataset', type=str, default='gpt_data')
    parser.add_argument('--gemini-dataset', type=str, default='gemini_data')
    args = parser.parse_args()

    run(**vars(args))
