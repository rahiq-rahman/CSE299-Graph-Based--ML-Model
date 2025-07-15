import os
import torch
import torch.serialization
from torch_geometric.data.data import Data

torch.serialization.add_safe_globals([Data])

def load_graph_matching_pairs(data_dir):
    pairs = []
    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            g1, g2, label = torch.load(os.path.join(data_dir, file), weights_only=False)
            pairs.append((g1, g2, label))
    return pairs
