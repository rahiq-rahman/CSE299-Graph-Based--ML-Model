import torch
import os
import random
from load_mutag import load_mutag

def run_graph_classification(model_fn, training_path, testing_path, model_path):
    train_loader, test_loader, input_dim, num_classes = load_mutag(training_path, testing_path)

    if not os.path.exists(model_path):
        model = model_fn(train_loader, test_loader, input_dim, num_classes)
        torch.save({'model': model}, model_path)
    else:
        model = torch.load(model_path, weights_only=False)['model']

    test_graphs = list(test_loader)
    print("\nSample Graph Classification:")
    for idx in random.sample(range(len(test_graphs)), 3):
        graph = test_graphs[idx]
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph.batch)
            pred = out.argmax(dim=1).item()
        print(f"Graph {idx} → Predicted: {pred} | True: {graph.y.item()}")
