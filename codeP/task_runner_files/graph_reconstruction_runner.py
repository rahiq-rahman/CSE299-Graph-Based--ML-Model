import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

# Loader
from load_cora_reconstruction import load_cora_reconstruction

def run_graph_reconstruction(model_fn, training_path, testing_path, model_path):
    x, train_edge_index, test_edge_index = load_cora_reconstruction(training_path, testing_path)
        
    if not os.path.exists(model_path):
        model, z = model_fn(x, train_edge_index, test_edge_index)
        torch.save({'model': model, 'z': z}, model_path)
    else:
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        z = loaded['z']

    neg_test_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=x.size(0),
        num_neg_samples=test_edge_index.size(1)
    )

    def sigmoid(x): return 1 / (1 + torch.exp(-x))

    pos_scores = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_test_edges[0]] * z[neg_test_edges[1]]).sum(dim=1)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    scores = sigmoid(scores).numpy()
    labels = labels.numpy()

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    print("\nGraph Reconstruction Results:")
    print(f"AUC: {auc:.4f} | Average Precision: {ap:.4f}")