# code for federated averaging (FedAvg) of BERT models with equal weights, available on Hugging Face Hub at https://huggingface.co/arshjaved/FedAvg-FL

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

HF_TOKEN = ""

CLIENT_MODELS = [
    "arshjaved/Model1-Fl",
    "arshjaved/Model2-Fl",
    "arshjaved/Model3-Fl",
    "arshjaved/Model4-Fl"
]

FEDAVG_OUTPUT_DIR = "fedavg_model"


def load_models():
    """Load client models."""
    models = []
    for model_name in CLIENT_MODELS:
        model = BertForSequenceClassification.from_pretrained(
            model_name, token=HF_TOKEN
        )
        models.append(model)
    return models


def fedavg(models):
    """Perform federated averaging of model weights with equal weighting."""
    num_clients = len(models)
    fedavg_model = BertForSequenceClassification.from_pretrained(
        CLIENT_MODELS[0], token=HF_TOKEN
    )
    fedavg_state = fedavg_model.state_dict()

    # Initialize with zeros
    for key in fedavg_state.keys():
        fedavg_state[key] = torch.zeros_like(fedavg_state[key])

    # Equal averaging
    for model in models:
        state_dict = model.state_dict()
        for key in fedavg_state.keys():
            fedavg_state[key] += (1.0 / num_clients) * state_dict[key]

    # Load averaged state back into model
    fedavg_model.load_state_dict(fedavg_state)
    return fedavg_model


def main():
    print("Loading models...")
    models = load_models()

    print("Performing FedAvg (equal weights)...")
    fedavg_model = fedavg(models)

    print("Saving FedAvg model...")
    os.makedirs(FEDAVG_OUTPUT_DIR, exist_ok=True)
    fedavg_model.save_pretrained(FEDAVG_OUTPUT_DIR)
    tokenizer = BertTokenizer.from_pretrained(CLIENT_MODELS[0], token=HF_TOKEN)
    tokenizer.save_pretrained(FEDAVG_OUTPUT_DIR)

    print(f"FedAvg model saved at: {FEDAVG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
