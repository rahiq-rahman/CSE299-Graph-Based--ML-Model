import torch
from loader_config import load_named_files, safe_tensor

def is_edge_index(t):
    return (
        isinstance(t, torch.Tensor) and
        t.ndim == 2 and
        ((t.shape[0] == 2) or (t.shape[1] == 2)) and
        t.dtype in [torch.int32, torch.int64] and
        t.max().item() < 100_000
    )

def is_feature_matrix(t):
    return (
        isinstance(t, torch.Tensor) and
        t.ndim == 2 and
        t.shape[1] > 1 and
        t.dtype in [torch.float32, torch.float64]
    )

def load_movielens(training_path, testing_path):
    train_data = load_named_files(training_path)
    test_data = load_named_files(testing_path)

    x = train_edge_index = test_edge_index = None

    for name, data in train_data.items():
        t = safe_tensor(data)
        if x is None and is_feature_matrix(t):
            x = t.float()
            print(f"[Match] Feature matrix from: {name}")
        elif train_edge_index is None and is_edge_index(t):
            train_edge_index = t.long()
            if train_edge_index.shape[0] != 2:
                train_edge_index = train_edge_index.T
            print(f"[Match] Train edge_index from: {name}")

    for name, data in test_data.items():
        t = safe_tensor(data)
        if test_edge_index is None and is_edge_index(t):
            test_edge_index = t.long()
            if test_edge_index.shape[0] != 2:
                test_edge_index = test_edge_index.T
            print(f"[Match] Test edge_index from: {name}")

    if x is None or train_edge_index is None or test_edge_index is None:
        raise ValueError("Missing required data")

    num_users = train_edge_index[0].max().item() + 1
    num_items = x.size(0) - num_users

    return x, train_edge_index, test_edge_index, num_users, num_items