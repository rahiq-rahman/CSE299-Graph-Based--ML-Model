import torch
from loader_config import load_named_files, safe_tensor

def is_feature_matrix(t):
    return t.ndim == 2 and t.shape[1] > 1 and t.dtype in [torch.float32, torch.float64]

def is_edge_index(t):
    return t.ndim == 2 and ((t.shape[0] == 2) or (t.shape[1] == 2)) and t.dtype in [torch.int32, torch.int64]

def is_edge_rating(t):
    return t.ndim == 1 and t.dtype in [torch.float32, torch.float64]

def is_mask(t):
    return t.ndim == 1 and t.dtype == torch.bool

def load_movielens_edge(train_path, test_path):
    train_data = load_named_files(train_path)
    test_data = load_named_files(test_path)

    x = edge_index = edge_ratings = train_mask = test_mask = None

    for fname, data in train_data.items():
        t = safe_tensor(data)
        if x is None and is_feature_matrix(t):
            x = t.float(); print(f"[Match] Feature matrix from: {fname}")
        elif edge_index is None and is_edge_index(t):
            edge_index = t.long(); print(f"[Match] edge_index from: {fname}")
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
        elif edge_ratings is None and is_edge_rating(t):
            edge_ratings = t.float(); print(f"[Match] edge_ratings from: {fname}")
        elif train_mask is None and is_mask(t):
            train_mask = t; print(f"[Match] train_mask from: {fname}")

    for fname, data in test_data.items():
        t = safe_tensor(data)
        if test_mask is None and is_mask(t):
            test_mask = t; print(f"[Match] test_mask from: {fname}")

    if None in [x, edge_index, edge_ratings, train_mask, test_mask]:
        raise ValueError("Missing required components")

    return x, edge_index, edge_ratings, train_mask, test_mask
