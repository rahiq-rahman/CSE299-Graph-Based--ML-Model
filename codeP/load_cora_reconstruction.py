import torch
from loader_config import load_named_files, safe_tensor

def is_feature_matrix(t):
    return t.ndim == 2 and t.shape[1] > 1 and t.dtype in [torch.float32, torch.float64]

def is_edge_index(t):
    return t.ndim == 2 and ((t.shape[0] == 2) or (t.shape[1] == 2)) and t.dtype in [torch.int32, torch.int64]

def load_cora_reconstruction(training_dir, testing_dir):
    train_data = load_named_files(training_dir)
    test_data = load_named_files(testing_dir)

    x = train_edge_index = test_edge_index = None

    for fname, data in train_data.items():
        t = safe_tensor(data)
        if x is None and is_feature_matrix(t):
            x = t.float()
            print(f"[Match] Feature matrix from: {fname}")
        elif train_edge_index is None and is_edge_index(t):
            train_edge_index = t.long()
            if train_edge_index.shape[0] != 2:
                train_edge_index = train_edge_index.T
            print(f"[Match] Train edge_index from: {fname}")

    for fname, data in test_data.items():
        t = safe_tensor(data)
        if test_edge_index is None and is_edge_index(t):
            test_edge_index = t.long()
            if test_edge_index.shape[0] != 2:
                test_edge_index = test_edge_index.T
            print(f"[Match] Test edge_index from: {fname}")

    if None in [x, train_edge_index, test_edge_index]:
        raise ValueError("Missing required components")

    return x, train_edge_index, test_edge_index
