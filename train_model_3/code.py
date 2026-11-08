"""
training mental/mental-bert-base-uncased on dataset 3, model available on Hugging Face Hub at https://huggingface.co/arshjaved/Model3-FL
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, TrainingArguments, Trainer
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import os
import json
import pickle
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    DATA_PATH = '/content/dataset3.csv'
    CACHE_DIR = './cache3'
    MODEL_SAVE_DIR = './models3'
    MODEL_NAME = "mental/mental-bert-base-uncased"
    HF_TOKEN = ""
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 128
    GRADIENT_ACCUMULATION_STEPS = 2
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    FP16 = True
    DATALOADER_NUM_WORKERS = 0 # for colab since multiprocessing is kind of tricky
    EARLY_STOPPING_PATIENCE = 3
    SAVE_STEPS = 500
    EVAL_STEPS = 500

def setup_directories():
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(f"{Config.MODEL_SAVE_DIR}/checkpoints", exist_ok=True)

def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class OptimizedTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item


def load_and_preprocess_data():
    cache_file = os.path.join(Config.CACHE_DIR, 'preprocessed_data.pkl')

    if os.path.exists(cache_file):
        print("Loading cached preprocessed data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Loading and preprocessing raw data...")
    data = pd.read_csv(Config.DATA_PATH)
    print(f"Loaded {len(data)} samples")
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    data = data.dropna(subset=['cleaned_text', 'label'])
    data = data[data['cleaned_text'].str.len() > 0]
    data = data[data['label'].isin([0, 1])]
    data['label'] = data['label'].astype(int)

    print(f"After preprocessing: {len(data)} samples")
    print(f"Class distribution: {data['label'].value_counts().to_dict()}")

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        data['cleaned_text'].tolist(), data['label'].tolist(),
        train_size=Config.TRAIN_SPLIT, random_state=42, stratify=data['label']
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5, random_state=42, stratify=temp_labels
    )

    print("Saving data splits to CSV files...")
    base_name = os.path.splitext(os.path.basename(Config.DATA_PATH))[0]
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    train_save_path = os.path.join(Config.CACHE_DIR, f"{base_name}_train.csv")
    val_save_path = os.path.join(Config.CACHE_DIR, f"{base_name}_val.csv")
    test_save_path = os.path.join(Config.CACHE_DIR, f"{base_name}_test.csv")
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f"Successfully saved splits to: {train_save_path}, {val_save_path}, and {test_save_path}")

    result = {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels}
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    return result

def create_tokenized_datasets(data_splits, tokenizer):
    cache_file = os.path.join(Config.CACHE_DIR, 'tokenized_datasets.pkl')

    if os.path.exists(cache_file):
        print("Loading cached tokenized datasets...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Creating tokenized datasets...")

    datasets = {}
    for split_name, split_data in data_splits.items():
        print(f"Tokenizing {split_name} set...")

        encodings = tokenizer(
            split_data['texts'],
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        )

        encodings = {
            'input_ids': encodings['input_ids'].tolist(),
            'attention_mask': encodings['attention_mask'].tolist()
        }

        datasets[split_name] = OptimizedTextDataset(encodings, split_data['labels'])

    with open(cache_file, 'wb') as f:
        pickle.dump(datasets, f)

    return datasets

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

def evaluate_model_comprehensive(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    metrics = {
        'test_loss': total_loss / len(test_loader),
        'test_accuracy': accuracy_score(true_labels, predictions),
        'test_f1': f1_score(true_labels, predictions, average='weighted'),
        'test_precision': precision_score(true_labels, predictions, average='weighted'),
        'test_recall': recall_score(true_labels, predictions, average='weighted')
    }

    print("=" * 50)
    print("FINAL TEST SET EVALUATION")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=["Class 0", "Class 1"]))

    return metrics

def main():
    setup_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME, token=Config.HF_TOKEN)
    model = BertForSequenceClassification.from_pretrained(Config.MODEL_NAME, token=Config.HF_TOKEN)
    model.to(device)
    data_splits = load_and_preprocess_data()
    datasets = create_tokenized_datasets(data_splits, tokenizer)
    train_loader = DataLoader(
        datasets['train'],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.DATALOADER_NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.DATALOADER_NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.DATALOADER_NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    all_labels = data_splits['train']['labels']
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eps=1e-8
    )

    total_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler() if Config.FP16 and torch.cuda.is_available() else None
    early_stopping = EarlyStoppingCallback(patience=Config.EARLY_STOPPING_PATIENCE)

    best_val_loss = float('inf')
    best_model_path = None
    training_history = []

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if Config.FP16 and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_predictions, val_true_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if Config.FP16 and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                val_predictions.extend(preds.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'learning_rate': scheduler.get_last_lr()[0]
        }

        training_history.append(epoch_metrics)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"{Config.MODEL_SAVE_DIR}/best_model_epoch_{epoch + 1}"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")

        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    print(f"Loading best model from {best_model_path}")
    model = BertForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)

    test_metrics = evaluate_model_comprehensive(model, test_loader, device)
    config_to_save = {k: v for k, v in vars(Config).items() if not k.startswith('__')}

    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'config': config_to_save,
        'timestamp': datetime.now().isoformat()
    }

    with open(f"{Config.MODEL_SAVE_DIR}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    final_model_path = f"{Config.MODEL_SAVE_DIR}/Model3-FL"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    from huggingface_hub import HfApi
    api = HfApi(token=Config.HF_TOKEN)
    api.upload_folder(
        folder_path=f"{Config.MODEL_SAVE_DIR}/Model3-FL",
        repo_id="arshjaved/Model3-FL",
        commit_message="Initial Model Upload",
        repo_type="model",
    )

    print(f"Training completed! Final model saved to {final_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test accuracy: {test_metrics['test_accuracy']:.4f}")

    return model, test_metrics

if __name__ == "__main__":
    main()