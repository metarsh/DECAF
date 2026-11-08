import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Cleans text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(file_paths):
    """Loads, merges, and cleans the datasets."""
    print("Merging and cleaning datasets...")
    df_list = [pd.read_csv(fp) for fp in file_paths]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Total samples before cleaning: {len(merged_df)}")
    
    merged_df['cleaned_text'] = merged_df['text'].apply(preprocess_text)
    merged_df.dropna(subset=['cleaned_text', 'label'], inplace=True)
    merged_df = merged_df[merged_df['cleaned_text'].str.len() > 0]
    merged_df['label'] = merged_df['label'].astype(int)

    print(f"Total samples after cleaning: {len(merged_df)}")
    print(f"Final class distribution:\n{merged_df['label'].value_counts()}\n")
    
    return merged_df


class InferenceDataset(Dataset):
    """Custom PyTorch dataset for inference."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model_name, test_df, device):
    """Loads model and evaluates on dataframe."""
    print(f"--- Evaluating model: {model_name} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    test_dataset = InferenceDataset(
        texts=test_df.cleaned_text.to_numpy(),
        labels=test_df.label.to_numpy(),
        tokenizer=tokenizer
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting with {model_name}"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
            
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(true_labels, predictions, target_names=["Class 0", "Class 1"]))
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def predict_single_text(text, model, tokenizer, device, max_len=128):
    """Predicts the label for a single piece of text."""
    model.eval()
    cleaned_text = preprocess_text(text)
    
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)
        
    label_map = {0: 'Class 0', 1: 'Class 1'}
    
    print(f"\n--- Single Prediction ---")
    print(f"Text: '{text}'")
    print(f"Predicted Label: {label_map[prediction.item()]}")
    return label_map[prediction.item()]

MODELS_TO_COMPARE = ["arshjaved/FedAVG-FL", "arshjaved/Final-FL"]
DATASET_FILES = [
    'dataset1_test.csv', 'dataset2_test.csv', 
    'dataset3_test.csv', 'dataset4_test.csv'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
try:
    combined_test_df = load_and_prepare_data(DATASET_FILES)
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Please make sure all test CSV files are uploaded to your Colab session.")
else:
    results = []
    for model_name in MODELS_TO_COMPARE:
        try:
            metrics = evaluate_model(model_name, combined_test_df, DEVICE)
            results.append(metrics)
        except Exception as e:
            print(f"\nCould not evaluate model {model_name}. Error: {e}")
    if results:
        print("\n======================================")
        print("   FINAL MODEL COMPARISON RESULTS")
        print("======================================")
        results_df = pd.DataFrame(results).set_index("model")
        print(results_df.round(4))
        print("======================================\n")
    try:
        print("Loading arshjaved/Final-FL for a single inference example...")
        inference_model = AutoModelForSequenceClassification.from_pretrained("arshjaved/Final-FL").to(DEVICE)
        inference_tokenizer = AutoTokenizer.from_pretrained("arshjaved/Final-FL")
        
        sample_text = "I've been feeling overwhelmed and anxious for weeks."
        predict_single_text(sample_text, inference_model, inference_tokenizer, DEVICE)

    except Exception as e:
        print(f"Could not perform single-text inference. Error: {e}")