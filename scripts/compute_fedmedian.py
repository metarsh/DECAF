"""
FedMedian Baseline Computation
-------------------------------
Computes the coordinate-wise median of 4 client BERT models and evaluates
the resulting model on the combined test set.

Based on: Yin et al., "Byzantine-Robust Distributed Learning: Towards
Optimal Statistical Rates," ICML 2018.
"""

import torch
import pandas as pd
import re
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
CLIENT_MODELS = [
    "arshjaved/Model1-FL",
    "arshjaved/Model2-FL",
    "arshjaved/Model3-FL",
    "arshjaved/Model4-FL",
]

DATASET_FILES = [
    "dataset1_test.csv",
    "dataset2_test.csv",
    "dataset3_test.csv",
    "dataset4_test.csv",
]

DATASET_SEARCH_DIRS = [
    ".",
    "/content",
    "/content",
    "/content",
    "/content",
]

OUTPUT_DIR = "./fedmedian_results"
FEDMEDIAN_SAVE_DIR = "./fedmedian_model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==================== DATA ====================
def preprocess_text(text):
    """Cleans text data (same preprocessing as original experiments)."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_dataset_file(filename):
    """Search for a dataset file in multiple directories."""
    for d in DATASET_SEARCH_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def load_and_prepare_data():
    """Loads, merges, and cleans the test datasets."""
    print("Loading and merging test datasets...")
    dfs = []
    for f in DATASET_FILES:
        path = find_dataset_file(f)
        if path:
            print(f"  Found: {path}")
            dfs.append(pd.read_csv(path))
        else:
            alt_name = f
            alt_path = find_dataset_file(alt_name)
            if alt_path:
                print(f"  Found: {alt_path}")
                dfs.append(pd.read_csv(alt_path))
            else:
                print(f"  WARNING: Could not find {f}")

    if not dfs:
        raise FileNotFoundError("No test dataset files found!")

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples before cleaning: {len(merged_df)}")

    merged_df['cleaned_text'] = merged_df['text'].apply(preprocess_text)
    merged_df.dropna(subset=['cleaned_text', 'label'], inplace=True)
    merged_df = merged_df[merged_df['cleaned_text'].str.len() > 0]
    merged_df['label'] = merged_df['label'].astype(int)

    print(f"Total samples after cleaning: {len(merged_df)}")
    print(f"Class distribution:\n{merged_df['label'].value_counts()}\n")

    return merged_df


class InferenceDataset(Dataset):
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
      encoding = self.tokenizer(          # ← was self.tokenizer.encode_plus(
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

      


# ==================== FEDMEDIAN ====================
def compute_fedmedian(client_model_names):
    """
    Compute coordinate-wise median of client model weights.

    FedMedian takes the element-wise median across all client state dicts.
    This is a post-hoc aggregation that does not require retraining.

    Reference: Yin et al., ICML 2018
    """
    print("\n=== Computing FedMedian (Coordinate-wise Median) ===")

    # Load all client state dicts
    client_state_dicts = []
    for name in client_model_names:
        print(f"  Loading {name}...")
        model = AutoModelForSequenceClassification.from_pretrained(name)
        client_state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})
        del model

    # Use the first client as template
    fedmedian_state = {}
    keys = client_state_dicts[0].keys()

    for key in tqdm(keys, desc="Computing coordinate-wise median"):
        # Stack all client tensors for this key
        stacked = torch.stack([sd[key].float() for sd in client_state_dicts], dim=0)
        # Take element-wise median along the client dimension
        median_vals, _ = torch.median(stacked, dim=0)
        fedmedian_state[key] = median_vals

    # Load into a fresh model
    print("  Loading FedMedian weights into model...")
    fedmedian_model = AutoModelForSequenceClassification.from_pretrained(client_model_names[0])
    fedmedian_model.load_state_dict(fedmedian_state)

    return fedmedian_model


def evaluate_model(model, model_name, test_df, tokenizer, device):
    """Evaluate a model on the test dataframe."""
    print(f"\n--- Evaluating: {model_name} ---")
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
        for batch in tqdm(test_loader, desc=f"Predicting"):
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
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEDMEDIAN_SAVE_DIR, exist_ok=True)

# 1. Compute FedMedian model
fedmedian_model = compute_fedmedian(CLIENT_MODELS)

# Save the FedMedian model
print(f"\nSaving FedMedian model to {FEDMEDIAN_SAVE_DIR}...")
fedmedian_model.save_pretrained(FEDMEDIAN_SAVE_DIR)

tokenizer = AutoTokenizer.from_pretrained(CLIENT_MODELS[0])
tokenizer.save_pretrained(FEDMEDIAN_SAVE_DIR)

# 2. Load test data
test_df = load_and_prepare_data()

# 3. Evaluate FedMedian
results = []

fedmedian_metrics = evaluate_model(
    fedmedian_model, "FedMedian", test_df, tokenizer, DEVICE
)
results.append(fedmedian_metrics)

# 4. Also evaluate FedAvg and Final-FL for direct comparison
for model_name in ["arshjaved/FedAVG-FL", "arshjaved/Final-FL"]:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metrics = evaluate_model(model, model_name, test_df, tokenizer, DEVICE)
        results.append(metrics)
        del model
    except Exception as e:
        print(f"Could not evaluate {model_name}: {e}")

# 5. Print final comparison
print("\n" + "=" * 60)
print("   FINAL COMPARISON: FedMedian vs FedAvg vs Reconstructed")
print("=" * 60)

for r in results:
    print(f"  {r['model']:30s}  Acc: {r['accuracy']:.4f}  F1: {r['f1_score']:.4f}")

print("=" * 60)

# 6. Save results
results_path = os.path.join(OUTPUT_DIR, "fedmedian_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}")