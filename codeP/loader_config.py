import os
import json
import torch
import pickle
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr

torch.serialization.add_safe_globals([Data, DataEdgeAttr])

def try_loaders(path):
    loaders = [
        lambda p: torch.load(p, weights_only=False),
        lambda p: pickle.load(open(p, 'rb')),
        lambda p: np.load(p, allow_pickle=True),
        lambda p: pd.read_csv(p, header=None).values,
        lambda p: json.load(open(p, 'r', encoding='utf-8')),
        lambda p: [json.loads(line) for line in open(p, 'r', encoding='utf-8')],
        lambda p: open(p, 'r', encoding='utf-8').read(),
    ]
    for loader in loaders:
        try:
            return loader(path)
        except Exception:
            continue
    print(f"[Warning] Could not load file: {path}")
    return None


def load_file(path):
    try:
        return try_loaders(path)
    except Exception as e:
        print(f"[Error] Exception while loading {path}: {e}")
        return None


def load_named_files(folder_path):
    data_dict = {}
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        name, _ = os.path.splitext(file)

        if not os.path.isfile(path):
            continue

        data = load_file(path)
        if data is None:
            continue

        if isinstance(data, dict):
            for key, value in data.items():
                if key in data_dict:
                    print(f"[Warning] Key '{key}' from '{file}' already exists. Overwriting.\n")
                data_dict[key] = value
        else:
            if name in data_dict:
                print(f"[Warning] '{file}' produces key '{name}' that already exists. Overwriting.\n")
            data_dict[name] = data

    return data_dict


def safe_tensor(data, dtype=None, bool_flag=False):
    if isinstance(data, torch.Tensor):
        t = data.detach().clone()
        if bool_flag:
            return t.bool()
        return t.to(dtype) if dtype else t
    else:
        try:
            return torch.tensor(data, dtype=torch.bool if bool_flag else dtype)
        except Exception as e:
            print(f"[Warning] Could not convert to tensor: {e}")
            return None
