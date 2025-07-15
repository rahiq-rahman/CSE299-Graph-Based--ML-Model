import os
import torch

def load_cora_reconstruction(training_dir, testing_dir):
    train_data = torch.load(os.path.join(training_dir, "graph.pt"))
    test_data = torch.load(os.path.join(testing_dir, "graph.pt"))

    x = train_data["x"]
    train_edge_index = train_data["train_edge_index"]
    test_edge_index = test_data["test_edge_index"]

    return x, train_edge_index, test_edge_index
