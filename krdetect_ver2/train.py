"""Training code for the detector model"""

import argparse
import os
from itertools import count

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from .model import TransformerClassifier

from .dataset2 import Corpus,EncodedDataset


def load_datasets(data_dir, human_dataset, gpt_dataset, gemini_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):

    human_corpus = Corpus(human_dataset, data_dir=data_dir)

    gpt_corpus = Corpus(gpt_dataset, data_dir=data_dir)

    gemini_corpus = Corpus(gemini_dataset, data_dir=data_dir)

    real_train, real_valid = human_corpus.train, human_corpus.valid
    gpt_train, gpt_valid = gpt_corpus.train, gpt_corpus.valid
    gemini_train, gemini_valid = gemini_corpus.train, gemini_corpus.valid

    # real_valid = real_valid[:300]
    # gpt_valid = gpt_valid[:300]
    # gemini_valid = gemini_valid[:300]

    Sampler = RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, gpt_train, gemini_train, tokenizer, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, gpt_valid , gemini_valid, tokenizer, max_sequence_length=max_sequence_length)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    return train_loader, validation_loader


def accuracy_sum(logits, labels):
    classification = logits.argmax(dim=1)
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(model: nn.Module, optimizer, criterion, device: str, loader: DataLoader, desc='Train'):
    model.train() 

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    with tqdm(loader, desc=desc) as loop:
        for texts, masks, labels in loop:

            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()

            logits = model(texts, attention_mask=masks)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            #정확도 계산
            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, criterion, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}', disable=False)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=False) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                logits = model(texts, attention_mask=masks) # labels 인자 제거
                loss = criterion(logits, labels)

                losses.append(loss)
                logit_votes.append(logits)


            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            labels = example[0][2].to(device)
            batch_size = labels.shape[0]

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

def run(max_epochs,
        device,
        batch_size,
        max_sequence_length,
        random_sequence_length,
        epoch_size,
        seed,
        data_dir,
        human_dataset,
        gpt_dataset,
        gemini_dataset,
        token_dropout,
        learning_rate,
        weight_decay,
        patience,
        d_model,
        nhead,
        num_layers,
        **kwargs):
    args = locals()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('device:', device)

    #model_name = "klue/roberta-base"
    model_name = "skt/kobert-base-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_classes = 3

    model = TransformerClassifier(
        vocab_size=len(tokenizer.get_vocab()),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=max_sequence_length
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, validation_loader = load_datasets(data_dir, human_dataset, gpt_dataset,gemini_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length, epoch_size,
                                                    token_dropout, seed)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logdir) if device == 'cuda' else None

    best_validation_accuracy = 0
    patience_counter = 0
    previous_validation_loss = float('inf')

    log_data = []

    for epoch in epoch_loop:
        train_metrics = train(model, optimizer, criterion, device, train_loader, f'Epoch {epoch}')
        validation_metrics = validate(model, criterion, device, validation_loader)

        combined_metrics = {**validation_metrics, **train_metrics}

        combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        log_data.append({
        "epoch": epoch,
        "train/accuracy": combined_metrics["train/accuracy"],
        "train/loss": combined_metrics["train/loss"],
        "validation/accuracy": combined_metrics["validation/accuracy"],
        "validation/loss": combined_metrics["validation/loss"],
        })

        print(f"Epoch {epoch}: Train Loss={combined_metrics['train/loss']:.4f}, Train Acc={combined_metrics['train/accuracy']:.4f}, "
            f"Val Loss={combined_metrics['validation/loss']:.4f}, Val Acc={combined_metrics['validation/accuracy']:.4f}")

        if device == 'cuda':
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

        current_validation_loss = combined_metrics["validation/loss"]
        if current_validation_loss < previous_validation_loss:
            print(f"📉 Validation loss improved from {previous_validation_loss:.4f} to {current_validation_loss:.4f}")
            previous_validation_loss = current_validation_loss
            patience_counter = 0

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                 best_validation_accuracy = combined_metrics["validation/accuracy"]
                 print(f"🚀 New best validation accuracy: {best_validation_accuracy:.4f}. Saving model...")
                 torch.save(dict(
                     epoch=epoch,
                     model_state_dict=model.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(),
                     args=args
                 ), os.path.join(logdir, "best-model.pt"))

        else:
            patience_counter += 1
            print(f"📉 No improvement in validation loss for {patience_counter} epoch(s). Current loss: {current_validation_loss:.4f}")


        if epoch % 10 == 0:
            torch.save(dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                args=args
            ), os.path.join(logdir, f"checkpoint-epoch-{epoch}.pt"))

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    args=args
                ),
                os.path.join(logdir, "early-"+epoch+"-epoch.pt")
            ) 
            break   
    df = pd.DataFrame(log_data)
    excel_path = os.path.join(logdir, "training_logs.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"✅ Training logs saved to {excel_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-sequence-length', type=int, default=256)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--human-dataset', type=str, default='human_data')
    parser.add_argument('--gpt-dataset', type=str, default='gpt_data')
    parser.add_argument('--gemini-dataset', type=str, default='gemini_data')
    parser.add_argument('--token-dropout', type=float, default=None)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=768, help="Dimension of model embeddings and hidden states")
    parser.add_argument('--nhead', type=int, default=12, help="Number of attention heads")
    parser.add_argument('--num-layers', type=int, default=8, help="Number of transformer encoder layers")

    args = parser.parse_args()

    run(**vars(args))
