import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.serialization

torch.serialization.add_safe_globals({Data})

def load_mutag(train_path, test_path):
    train_graphs = []
    test_graphs = []

    for file in sorted(os.listdir(train_path)):
        if file.endswith(".pt"):
            graph = torch.load(os.path.join(train_path, file), weights_only=False)
            train_graphs.append(graph)

    for file in sorted(os.listdir(test_path)):
        if file.endswith(".pt"):
            graph = torch.load(os.path.join(test_path, file), weights_only=False)
            test_graphs.append(graph)

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    input_dim = train_graphs[0].num_node_features
    num_classes = len(set([g.y.item() for g in train_graphs + test_graphs]))

    return train_loader, test_loader, input_dim, num_classes
