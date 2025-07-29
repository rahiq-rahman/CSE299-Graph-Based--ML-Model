import torch
import os
import random
from sklearn.metrics import classification_report, accuracy_score

from load_cora_edge import load_cora_edge

def run_edge_classification(model_fn, training_path, testing_path, model_path):
    x, edge_index, edge_labels, train_mask, test_mask = load_cora_edge(training_path, testing_path)

    if not os.path.exists(model_path):
        model, classifier = model_fn(x, edge_index, edge_labels, train_mask, test_mask)
        torch.save({'model': model, 'classifier': classifier}, model_path)
    else:
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        classifier = loaded['classifier']

    model.eval()
    classifier.eval()
    with torch.no_grad():
        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        edge_logits = classifier(edge_emb)
        edge_preds = edge_logits.argmax(dim=1)
        edge_probs = torch.softmax(edge_logits, dim=1)

    # Evaluation
    y_true = edge_labels[test_mask].cpu().numpy()
    y_pred = edge_preds[test_mask].cpu().numpy()
    acc = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nSample Predictions on Test Edges:")
    test_indices = torch.where(test_mask)[0].tolist()
    samples = random.sample(test_indices, min(5, len(test_indices)))
    for eid in samples:
        u, v = edge_index[0, eid].item(), edge_index[1, eid].item()
        pred = edge_preds[eid].item()
        true = edge_labels[eid].item()
        conf = edge_probs[eid][pred].item()
        print(f"Edge ({u}, {v}) → Predicted: {pred}, True: {true}, Confidence: {conf:.4f} {'✅' if pred == true else '❌'}")

