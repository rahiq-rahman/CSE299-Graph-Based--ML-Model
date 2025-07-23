from torch_geometric.loader import DataLoader
from loader_config import load_named_files
from torch_geometric.data import Data

def load_synthetic_regression(train_path, test_path):
    def load_graphs(path):
        data_dict = load_named_files(path)
        graphs = []
        for _, value in data_dict.items():
            if isinstance(value, list):
                graphs.extend([g for g in value if isinstance(g, Data)])
            elif isinstance(value, Data):
                graphs.append(value)
        return graphs

    train_graphs = load_graphs(train_path)
    test_graphs = load_graphs(test_path)

    if not train_graphs or not test_graphs:
        raise ValueError("No valid graph data found")

    input_dim = train_graphs[0].x.shape[1]

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    return train_loader, test_loader, input_dim
