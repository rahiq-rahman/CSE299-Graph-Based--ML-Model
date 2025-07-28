import torch
import os
import random
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
    node_embeddings = model(x, edge_index)
    edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
    edge_logits = classifier(edge_emb)
    probs = torch.softmax(edge_logits, dim=1)

    print("\nSample Edge Classifications:")
    for _ in range(3):
        eid = random.randint(0, edge_index.shape[1] - 1)
        prob = probs[eid]
        print(f"Edge ({edge_index[0,eid].item()}, {edge_index[1,eid].item()}) → "
              f"Pred: {prob.argmax().item()} | True: {edge_labels[eid].item()} | "
              f"Confidence: {prob.max().item():.4f}")
