import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

FEDAVG_MODEL_NAME = "arshjaved/FedAVG-FL"
FINAL_MODEL_NAME = "arshjaved/Final-FL"

CLIENT_MODELS = [
    "arshjaved/Model1-FL",
    "arshjaved/Model2-FL",
    "arshjaved/Model3-FL",
    "arshjaved/Model4-FL"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_model(model_name, device):
    try:
        print(f"Loading {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}. Error: {e}")
        return None

print("\n--- Analyzing Parameter Distributions ---")
fedavg_model = load_model(FEDAVG_MODEL_NAME, DEVICE)
final_model = load_model(FINAL_MODEL_NAME, DEVICE)

if fedavg_model and final_model:
    # Extract weights from the final classifier layer
    with torch.no_grad():
        fedavg_weights = fedavg_model.classifier.weight.cpu().numpy().flatten()
        final_weights = final_model.classifier.weight.cpu().numpy().flatten()

    # Plotting the distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(fedavg_weights, color="skyblue", label="FedAVG Model Weights", kde=True, stat="density", linewidth=0)
    sns.histplot(final_weights, color="red", label="Final-FL Model Weights", kde=True, stat="density", linewidth=0, alpha=0.6)
    
    plt.title("Comparison of Classifier Weight Distributions", fontsize=16)
    plt.xlabel("Weight Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.show()

    print("The plot above shows the distribution of weights in the final layer.")
    print("A more spread-out or differently shaped distribution for 'Final-FL' suggests it learned a more complex representation than the simple average.")


print("\n--- Part 2: Analyzing Cosine Similarity to Client Models ---")

@torch.no_grad()
def get_flat_params(model):
    """Extracts all model parameters and flattens them into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters()]).cpu().numpy()

if len(CLIENT_MODELS) < 2:
    print("Skipping Cosine Similarity Analysis: Please provide at least two client models in the `CLIENT_MODELS` list.")
else:
    # Get flattened parameters for the central models
    flat_fedavg = get_flat_params(fedavg_model)
    flat_final = get_flat_params(final_model)

    similarities = {
        "client_model": [],
        "similarity_to_fedavg": [],
        "similarity_to_final": []
    }

    # Calculate similarity between each client and the central models
    for client_name in CLIENT_MODELS:
        client_model = load_model(client_name, DEVICE)
        if client_model:
            flat_client = get_flat_params(client_model)
            
            # Reshape for sklearn's cosine_similarity function
            sim_fedavg = cosine_similarity(flat_client.reshape(1, -1), flat_fedavg.reshape(1, -1))[0][0]
            sim_final = cosine_similarity(flat_client.reshape(1, -1), flat_final.reshape(1, -1))[0][0]
            
            similarities["client_model"].append(client_name.split('/')[-1])
            similarities["similarity_to_fedavg"].append(sim_fedavg)
            similarities["similarity_to_final"].append(sim_final)

    if similarities["client_model"]:
        # Calculate statistics
        avg_sim_fedavg = np.mean(similarities["similarity_to_fedavg"])
        std_sim_fedavg = np.std(similarities["similarity_to_fedavg"])
        
        avg_sim_final = np.mean(similarities["similarity_to_final"])
        std_sim_final = np.std(similarities["similarity_to_final"])

        print("\n--- Cosine Similarity Results ---")
        print(f"FedAVG Model Average Similarity to Clients: {avg_sim_fedavg} (Std: {std_sim_fedavg})")
        print(f"Final-FL Model Average Similarity to Clients: {avg_sim_final} (Std: {std_sim_final})")
        
        print("\nInterpretation:")
        print("- A **high average similarity** (close to 1.0) shows a model is a 'central point'. The FedAVG model should be very high.")
        print("- A **lower average similarity** for the Final-FL model suggests it's not just an average but a more generalized solution.")
        print("- A **higher standard deviation (Std)** for Final-FL indicates its relationship with each client is more varied and less uniform, a sign of a more nuanced model.")

        # Plotting the results
        x = np.arange(len(similarities["client_model"]))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, similarities["similarity_to_fedavg"], width, label='Similarity to FedAVG', color='skyblue')
        rects2 = ax.bar(x + width/2, similarities["similarity_to_final"], width, label='Similarity to Final-FL', color='red', alpha=0.7)

        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity of Central Models to Client Models')
        ax.set_xticks(x)
        ax.set_xticklabels(similarities["client_model"])
        ax.legend()
        ax.set_ylim(0.8, 1.0)

        fig.tight_layout()
        plt.show()