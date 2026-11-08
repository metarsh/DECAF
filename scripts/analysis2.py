import torch
import numpy as np
from transformers import BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, device):
    try:
        model = BertForSequenceClassification.from_pretrained(model_name).to(device)
        return model
    except Exception as e:
        print(f"Failed to load {model_name}. Error: {e}")
        return None

@torch.no_grad()
def calculate_sparsity(model, threshold=1e-3):
    """Calculates the percentage of weights below a certain threshold."""
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(torch.abs(param) < threshold).item()
    return (zero_params / total_params) * 100

fedavg_model = load_model("arshjaved/FedAVG-FL", DEVICE)
final_model = load_model("arshjaved/Final-FL", DEVICE)

if fedavg_model and final_model:
    fedavg_sparsity = calculate_sparsity(fedavg_model)
    final_sparsity = calculate_sparsity(final_model)
    
    print("\n--- Model Sparsity Comparison ---")
    print(f"FedAVG Model Sparsity: {fedavg_sparsity}%")
    print(f"Final-FL Model Sparsity: {final_sparsity}%")
    print("\nA higher sparsity score indicates a more efficient model that has learned to ignore irrelevant information.")


@torch.no_grad()
def calculate_norms(model):
    """Calculates the L1 and L2 norms of all model parameters."""
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters()).sqrt()
    return l1_norm.item(), l2_norm.item()

if fedavg_model and final_model:
    fedavg_l1, fedavg_l2 = calculate_norms(fedavg_model)
    final_l1, final_l2 = calculate_norms(final_model)

    print("\n--- Parameter Norm Comparison ---")
    print(f"                |   L1 Norm    |   L2 Norm")
    print(f"----------------|--------------|--------------")
    print(f"FedAVG Model    | {fedavg_l1} | {fedavg_l2}")
    print(f"Final-FL Model  | {final_l1} | {final_l2}")
    print("\nLower norms suggest a better-regularized model that is less likely to overfit and more likely to generalize well.")