import torch
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from load_mutag import load_mutag

def run_graph_classification(model_fn, training_path, testing_path, model_path):
    train_loader, test_loader, input_dim, num_classes = load_mutag(training_path, testing_path)

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(train_loader, test_loader, input_dim, num_classes)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        model = torch.load(model_path, weights_only=False)['model']

    y_true, y_pred = [], []
    for batch in test_loader:
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())

    print("\nEvaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("\nSample Graph Classification:")
    test_graphs = list(test_loader)
    for idx in random.sample(range(len(test_graphs)), min(5, len(test_graphs))):
        graph = test_graphs[idx]
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph.batch)
            pred = out.argmax(dim=1).item()
        print(f"Graph {idx} → Predicted: {pred} | Actual: {graph.y.item()} {'✅' if pred == graph.y.item() else '❌'}")
