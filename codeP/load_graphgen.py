import os
import torch

def load_graph_generation_dataset(data_dir):
    dataset = []

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".pt"):
            data = torch.load(os.path.join(data_dir, fname), weights_only=False)
            dataset.append(data)

    return dataset
