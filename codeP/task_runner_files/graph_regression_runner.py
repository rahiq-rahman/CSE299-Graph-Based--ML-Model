import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_synthetic import load_synthetic_regression

def run_graph_regression(model_fn, training_path, testing_path, model_path):
    train_loader, test_loader, input_dim = load_synthetic_regression(training_path, testing_path)

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(train_loader, test_loader, input_dim)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        model = torch.load(model_path, weights_only=False)['model']

    y_true, y_pred = [], []
    for graph in test_loader:
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))
            pred = out.item() if out.numel() == 1 else out.squeeze().item()
        y_true.append(graph.y.item())
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nEvaluation on Test Set:")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

    print("\nSample Graph Regression:")
    for idx in random.sample(range(len(y_true)), min(5, len(y_true))):
        print(f"Graph {idx} → Predicted: {y_pred[idx]:.4f} | True: {y_true[idx]:.4f} | Abs Error: {abs(y_pred[idx]-y_true[idx]):.4f}")

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Graph Regression: Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(np.abs(y_pred - y_true), bins=20, color='orange', edgecolor='black')
    plt.title("Absolute Error Distribution")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
