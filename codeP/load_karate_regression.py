import torch
import os

def load_karate_regression(train_path, test_path):
    x = torch.load(os.path.join(train_path, "features.pt"))
    edge_index = torch.load(os.path.join(train_path, "edge_index.pt"))
    targets = torch.load(os.path.join(train_path, "targets.pt"))
    train_mask = torch.load(os.path.join(train_path, "train_mask.pt"))
    test_mask = torch.load(os.path.join(test_path, "test_mask.pt"))
    return x, edge_index, targets, train_mask, test_mask
