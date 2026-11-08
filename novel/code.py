import os
import json
import re
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertForSequenceClassification
from tqdm.auto import tqdm

# ------------------ Config ------------------
INDIVIDUAL_MODELS = [
    "arshjaved/Model1-FL", "arshjaved/Model2-FL",
    "arshjaved/Model3-FL", "arshjaved/Model4-FL"
]
FEDAVG_MODEL = "arshjaved/FedAVG-FL"

TRAIN_RATIO = 0.7
EPOCHS = 20
SEED = 42

OUTPUT_DIR = "./fl_results"
MODEL_SAVE_DIR = "./fl_model"
CACHE_DIR = "./model_cache"

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    _ = torch.cuda.current_device()
print(f"Using device: {DEVICE}")


# ------------------ Model ------------------
class WeightPredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in [h_dim for h_dim in hidden_dims if h_dim > 0]:
            layers.extend([
                nn.Linear(prev_dim, h), nn.BatchNorm1d(h),
                nn.GELU(), nn.Dropout(dropout_rate)
            ])
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, 1)

    def forward(self, x):
        return self.out(self.mlp(x)).squeeze(-1)


# ------------------ Utils ------------------
def get_adaptive_params(num_train_points, thresholds):
    small_thresh, large_thresh = thresholds
    if num_train_points < small_thresh:
        return {"lr": 5e-5, "h1": 64, "h2": 32, "h3": 0, "dropout": 0.5, "wd": 1e-4}
    elif num_train_points > large_thresh:
        return {"lr": 2e-4, "h1": 256, "h2": 128, "h3": 64, "dropout": 0.15, "wd": 1e-6}
    else:
        return {"lr": 1e-4, "h1": 128, "h2": 64, "h3": 32, "dropout": 0.3, "wd": 1e-5}


def analyze_and_get_thresholds(state_dicts):
    layer_sizes = [
        int(p.numel() * TRAIN_RATIO) for p in state_dicts["fedavg"].values() if p.ndim > 1
    ]
    if not layer_sizes:
        raise ValueError("Could not find any layers to analyze.")
    sizes_arr = np.array(layer_sizes)
    q25 = int(np.quantile(sizes_arr, 0.25))
    q75 = int(np.quantile(sizes_arr, 0.75))
    print("\n--- Layer Size Distribution Analysis ---")
    print(f"25th Percentile: {q25} (SMALL_LAYER_THRESHOLD)")
    print(f"75th Percentile: {q75} (LARGE_LAYER_THRESHOLD)")
    print("----------------------------------------\n")
    return q25, q75


def get_cached_state_dict(model_name: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")
    print(f"Downloading '{model_name}'...")
    model = BertForSequenceClassification.from_pretrained(model_name)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, cache_path)
    return state_dict


def prepare_data_for_layer(state_dicts, layer_name):
    fed_param = state_dicts["fedavg"].get(layer_name)
    if fed_param is None or fed_param.ndim <= 1: return None
    client_params = [sd.get(layer_name) for sd in state_dicts["clients"]]
    if any(p is None for p in client_params): return None
    client_stack = np.stack([p.flatten().numpy() for p in client_params], axis=1)
    fedavg_flat = fed_param.flatten().numpy()
    indices = np.arange(fed_param.numel())
    X_train, X_infer, y_train, y_infer, _, idx_infer = train_test_split(
        client_stack, fedavg_flat, indices, train_size=TRAIN_RATIO, random_state=SEED
    )
    return {
        "X_train": X_train, "y_train": y_train, "X_infer": X_infer, "y_infer": y_infer,
        "idx_infer": idx_infer, "original_shape": fed_param.shape, "original_fedavg": fedavg_flat
    }


# ------------------ Training ------------------
def train_weight_predictor(layer_data, params):
    X_train = layer_data["X_train"]
    y_train = layer_data["y_train"]
    if len(X_train) < 2:
        return None

    # Scaling
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Move full dataset to GPU once
    X_train_t = torch.from_numpy(X_train_scaled).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train_scaled).float().to(DEVICE)

    model = WeightPredictionNetwork(
        input_dim=X_train_t.shape[1],
        hidden_dims=[params["h1"], params["h2"], params["h3"]],
        dropout_rate=params["dropout"]
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    batch_size = min(params.get("batch_size", 2048), len(X_train_t))

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_train_t), device=DEVICE)  # shuffle on GPU
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    return model, scaler_x, scaler_y



# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("--- Loading All Models ---")
    state_dicts = {
        "fedavg": get_cached_state_dict(FEDAVG_MODEL),
        "clients": [get_cached_state_dict(name) for name in INDIVIDUAL_MODELS]
    }

    thresholds = analyze_and_get_thresholds(state_dicts)

    print("\n--- Training Predictors for Each Layer ---")
    results = {}
    r2_layers = []

    for layer_name in tqdm(state_dicts["fedavg"].keys(), desc="Processing Layers"):
        layer_data = prepare_data_for_layer(state_dicts, layer_name)
        if not layer_data: continue
        params_to_use = get_adaptive_params(len(layer_data["X_train"]), thresholds)
        training_result = train_weight_predictor(layer_data, params_to_use)
        if not training_result: continue
        model, scaler_x, scaler_y = training_result

        model.eval()
        with torch.no_grad():
            X_infer_scaled = scaler_x.transform(layer_data["X_infer"])
            X_infer_t = torch.from_numpy(X_infer_scaled).float().to(DEVICE)
            preds_scaled = model(X_infer_t).cpu().numpy().reshape(-1, 1)
            preds = scaler_y.inverse_transform(preds_scaled).flatten()

        r2 = r2_score(layer_data["y_infer"], preds)
        mae = mean_absolute_error(layer_data["y_infer"], preds)

        r2_layers.append({"layer_name": layer_name, "r2_score": r2, "num_samples": len(layer_data["X_train"])})
        reconstructed = layer_data["original_fedavg"].copy()
        reconstructed[layer_data["idx_infer"]] = preds
        results[layer_name] = {"r2": r2, "mae": mae, "reconstructed_weights": reconstructed}

    print("\n--- Reconstructing and Saving Final Model ---")
    final_model = BertForSequenceClassification.from_pretrained(FEDAVG_MODEL)
    new_state_dict = final_model.state_dict()
    for name, res in results.items():
        if "reconstructed_weights" in res:
            original_tensor = new_state_dict[name]
            weights = torch.from_numpy(res["reconstructed_weights"].reshape(original_tensor.shape))
            new_state_dict[name] = weights.to(original_tensor.device, original_tensor.dtype)
    final_model.load_state_dict(new_state_dict)
    final_model.save_pretrained(MODEL_SAVE_DIR)

    summary = {
        "avg_r2": float(np.nanmean([r.get("r2") for r in results.values()])),
        "avg_mae": float(np.nanmean([r.get("mae") for r in results.values()])),
        "thresholds_used": {"small": thresholds[0], "large": thresholds[1]},
        "r2_layers": r2_layers,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n--- Analysis of Layers with R² ---")
    r2_layers.sort(key=lambda x: x["r2_score"])
    for layer in r2_layers:
        print(f"  - {layer['layer_name']}: R²={layer['r2_score']:.4f}, Train Samples={layer['num_samples']}")
    print(f"\n=== Done ===\nAvg R2 Score: {summary['avg_r2']:.4f}")
    print(f"Model saved to: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
