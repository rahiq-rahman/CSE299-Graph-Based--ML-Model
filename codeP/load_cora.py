import torch
import os

def load_cora(training_dir, testing_dir):
    x = torch.load(os.path.join(training_dir, "features.pt"))
    edge_index = torch.load(os.path.join(training_dir, "edge_index.pt"))
    labels = torch.load(os.path.join(training_dir, "labels.pt"))
    train_mask = torch.load(os.path.join(training_dir, "train_mask.pt"))
    test_mask = torch.load(os.path.join(testing_dir, "test_mask.pt"))

    return x, edge_index, labels, train_mask, test_mask
