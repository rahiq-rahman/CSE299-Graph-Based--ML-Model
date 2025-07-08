import torch
import os

def load_graph_regression(root_dir):
    train_data = torch.load(os.path.join(root_dir, "training", "dataset.pt"))
    test_data = torch.load(os.path.join(root_dir, "testing", "dataset.pt"))
    return train_data, test_data
