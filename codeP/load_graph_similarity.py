import torch
import os

def load_graph_similarity(data_dir):
    pairs_path = os.path.join(data_dir, "training", "pairs.pt")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Similarity pairs not found at {pairs_path}")
    pairs = torch.load(pairs_path)
    return pairs
