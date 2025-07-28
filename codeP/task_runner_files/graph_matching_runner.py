import torch
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from load_graph_matching import load_graph_matching_pairs

def run_graph_matching(model_fn, training_path, testing_path, model_path):
    train_pairs = load_graph_matching_pairs(training_path)
    test_pairs = load_graph_matching_pairs(testing_path)

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(train_pairs, test_pairs)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        model = torch.load(model_path, weights_only=False)['model']

    y_true, y_pred = [], []
    for g1, g2, label in test_pairs:
        pred = model.predict(g1, g2)
        y_true.append(label.item())
        y_pred.append(pred)

    print("\nEvaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("\nSample Graph Matching Results:")
    for i in random.sample(range(len(test_pairs)), min(5, len(test_pairs))):
        g1, g2, label = test_pairs[i]
        pred = model.predict(g1, g2)
        print(f"Pair {i} → Predicted: {pred} | Actual: {label.item()} {'✅' if pred == label.item() else '❌'}")
