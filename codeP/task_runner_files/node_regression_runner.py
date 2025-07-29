import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from load_karate_regression import load_karate_regression

def run_node_regression(model_fn, training_path, testing_path, model_path):
    x, edge_index, targets, train_mask, test_mask = load_karate_regression(training_path, testing_path)

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(x, edge_index, targets, train_mask, test_mask)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        model = torch.load(model_path, weights_only=False)['model']

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index).squeeze()
    
    y_true = targets[test_mask].cpu().numpy()
    y_pred = out[test_mask].cpu().numpy()

    # Evaluation Metrics
    print("\nEvaluation on Test Set:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

    # Per-node error (absolute)
    print("\nSample Node Regression Results (Test Set):")
    test_indices = torch.where(test_mask)[0].tolist()
    samples = random.sample(test_indices, min(5, len(test_indices)))
    for nid in samples:
        pred = out[nid].item()
        true = targets[nid].item()
        error = abs(pred - true)
        print(f"Node {nid} → Predicted: {pred:.4f} | Actual: {true:.4f} | Absolute Error: {error:.4f}")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # y=x line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Node Regression: Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Error Distribution
    errors = np.abs(y_pred - y_true)
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=20, color='orange', edgecolor='black')
    plt.title("Distribution of Absolute Errors")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
