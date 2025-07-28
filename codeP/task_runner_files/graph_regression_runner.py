import torch
import os
import random
from load_synthetic import load_synthetic_regression

def run_graph_regression(model_fn, training_path, testing_path, model_path):
    train_loader, test_loader, input_dim = load_synthetic_regression(training_path, testing_path)

    if not os.path.exists(model_path):
        model = model_fn(train_loader, test_loader, input_dim)
        torch.save({'model': model}, model_path)
    else:
        model = torch.load(model_path, weights_only=False)['model']

    test_graphs = list(test_loader)
    print("\nSample Graph Regression:")
    for idx in random.sample(range(len(test_graphs)), 3):
        graph = test_graphs[idx]
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))
            pred = out.item() if out.numel() == 1 else out.squeeze().item()
        print(f"Graph {idx} → Predicted: {pred:.4f} | True: {graph.y.item():.4f}")
