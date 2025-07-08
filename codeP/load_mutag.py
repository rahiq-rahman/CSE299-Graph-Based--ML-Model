from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_mutag(data_path):
    dataset = TUDataset(data_path, name='MUTAG')
    return dataset
