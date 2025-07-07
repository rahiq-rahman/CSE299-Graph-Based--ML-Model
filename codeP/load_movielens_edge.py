import os
import torch

def load_movielens_edge(train_path, test_path):
    x = torch.load(os.path.join(train_path, "features.pt"))
    edge_index = torch.load(os.path.join(train_path, "edge_index.pt"))
    edge_ratings = torch.load(os.path.join(train_path, "edge_ratings.pt"))
    train_mask = torch.load(os.path.join(train_path, "edge_rating_mask.pt"))
    test_mask = torch.load(os.path.join(test_path, "edge_rating_mask.pt"))
    return x, edge_index, edge_ratings, train_mask, test_mask
