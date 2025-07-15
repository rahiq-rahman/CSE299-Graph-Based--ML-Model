import os
import torch
from torch_geometric.loader import DataLoader
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

add_safe_globals([Data])

def load_synthetic_regression(train_path, test_path):
    train_graphs = [torch.load(os.path.join(train_path, f), weights_only=False) for f in os.listdir(train_path) if f.endswith(".pt")]
    test_graphs = [torch.load(os.path.join(test_path, f), weights_only=False) for f in os.listdir(test_path) if f.endswith(".pt")]

    input_dim = train_graphs[0].x.shape[1]
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    return train_loader, test_loader, input_dim
