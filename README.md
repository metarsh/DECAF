# My first research paper

I started working on it in my 3rd semester as part of my course's mini project. Presented it at IC2E3 2026, NIT Uttarakhand in my 6th semester of B.Tech CSE.

# Overview

We fine-tuned [MentalBERT](https://huggingface.co/mental/mental-bert-base-uncased) on four different datasets collected from different parts of the internet. The task was to classify whether a text was depressive or not.

After fine-tuning the same base model on each dataset, we obtained four different models.

Federated learning allows a model to be trained on a client's own device or server without sending the client's data to a central server. For example, a hospital can train a model on its own servers because of privacy requirements. Instead of sharing the data, the hospital sends the trained model's weights to a central server. These weights are then aggregated using a mathematical function to create a single global model.

This approach improves data privacy, but it can involve trade-offs in accuracy, generalizability, and communication costs. Traditional methods such as FedAvg and FedMedian use a fixed mathematical function to aggregate the model weights.

In this project, we tried a neural-network-based approach. Instead of aggregating the four models with one fixed function, we trained neural networks to learn the aggregation function. The input consists of 70% of the weights from the four client models, and the output consists of 70% of the weights from a central model fine-tuned on all four datasets combined. This central model acts as the ideal or ground-truth model.

We validated the approach using the remaining 30% of the model weights. The resulting model achieved accuracy close to that of the ideal model.

It includes:

- Training one central model on combined data
- Training four client models on separate datasets
- Combining client models with Federated Averaging (FedAvg) and FedMedian
- Our novel neural network approach
- Evaluation and comparison scripts

## Project structure

```text
datasets/              all 4 datasets
train_central_model/   Central-model training code and outputs
train_model_1..4/      Client-model training code and outputs
train_model_fedavg/    FedAvg code
novel/                 code for our novel method
scripts/               Classification, comparison, and utilities
docs/                  certificate, research paper and presentation
```