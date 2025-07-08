import torch
import os

def load_graph_embedding(data_dir):
    dataset_path = os.path.join(data_dir, "training", "dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    dataset = torch.load(dataset_path)
    return dataset
