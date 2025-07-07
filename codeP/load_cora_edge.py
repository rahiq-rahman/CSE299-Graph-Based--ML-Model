import os
import torch

def load_cora_edge(train_path, test_path):
    x = torch.load(os.path.join(train_path, "features.pt"))
    edge_index = torch.load(os.path.join(train_path, "edge_index.pt"))
    edge_labels = torch.load(os.path.join(train_path, "edge_labels.pt"))
    train_mask = torch.load(os.path.join(train_path, "edge_label_mask.pt"))
    test_mask = torch.load(os.path.join(test_path, "edge_label_mask.pt"))
    return x, edge_index, edge_labels, train_mask, test_mask
