import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

from load_cora import load_cora


def run_node_classification(model_fn, training_path, testing_path, model_path):
    x, edge_index, labels, train_mask, test_mask = load_cora(training_path, testing_path)

    model = None

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(x, edge_index, labels, train_mask, test_mask)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)

    y_true = labels[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()

    # Evaluation Metrics
    print("\nEvaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Sample Prediction Comparison
    print("\nSample Node Predictions (Test Set):")
    test_indices = torch.where(test_mask)[0].tolist()
    samples = random.sample(test_indices, min(5, len(test_indices)))
    for nid in samples:
        pred = preds[nid].item()
        true = labels[nid].item()
        print(f"Node {nid} → Predicted: {pred}, Actual: {true} {'✅' if pred == true else '❌'}")

    # Per-Class Accuracy
    print("\nPer-Class Accuracy:")
    for cls in set(y_true):
        cls_mask = (y_true == cls)
        acc = np.mean(y_pred[cls_mask] == y_true[cls_mask])
        print(f"Class {cls}: Accuracy = {acc:.4f}")

    # Class Probabilities for a Random Node
    sample_node = random.choice(test_indices)
    print(f"\nNode {sample_node} Class Probabilities:")
    for cls, p in enumerate(probs[sample_node]):
        print(f"  Class {cls}: {p.item():.4f}")
    print(f"  → Predicted Class: {preds[sample_node].item()} with Confidence: {probs[sample_node].max().item():.4f}")

    # Visualize Embeddings with t-SNE
    print("\nVisualizing node embeddings with t-SNE (Test Set)...")
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(logits[test_mask].cpu().numpy())
    labels_test = labels[test_mask].cpu().numpy()

    plt.figure(figsize=(8, 6))
    for lbl in set(labels_test):
        idx = labels_test == lbl
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Class {lbl}", alpha=0.7)
    plt.legend()
    plt.title("t-SNE of Node Embeddings (Test Set)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
