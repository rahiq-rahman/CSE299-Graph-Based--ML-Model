import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)
from torch_geometric.utils import negative_sampling, degree
import networkx as nx

# Loader
from load_cora_reconstruction import load_cora_reconstruction

def run_graph_reconstruction(model_fn, training_path, testing_path, model_path):
    x, train_edge_index, test_edge_index = load_cora_reconstruction(training_path, testing_path)

    # Basic graph statistics
    num_nodes = x.size(0)
    num_edges = train_edge_index.size(1)
    avg_degree = num_edges * 2 / num_nodes
    degrees = degree(train_edge_index[0], num_nodes=num_nodes)
    max_degree = degrees.max().item()

    print("\nGraph Statistics:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Max degree: {max_degree}")

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model, z = model_fn(x, train_edge_index, test_edge_index)
        torch.save({'model': model, 'z': z}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        z = loaded['z']

    # Generate negative samples
    neg_test_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=test_edge_index.size(1)
    )

    def sigmoid(x): return 1 / (1 + torch.exp(-x))

    # Compute scores
    pos_scores = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_test_edges[0]] * z[neg_test_edges[1]]).sum(dim=1)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    scores = sigmoid(scores).numpy()
    labels = labels.numpy()

    # Compute metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    pred_labels = (scores >= 0.5).astype(int)
    f1 = f1_score(labels, pred_labels)
    acc = accuracy_score(labels, pred_labels)

    print("\nGraph Reconstruction Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"F1-Score (threshold 0.5): {f1:.4f}")
    print(f"Reconstruction Accuracy (threshold 0.5): {acc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histogram of Scores
    pos_scores_np = scores[:len(pos_scores)]
    neg_scores_np = scores[len(pos_scores):]

    plt.figure(figsize=(6, 4))
    plt.hist(pos_scores_np, bins=30, alpha=0.6, label='Positive', color='blue', density=True)
    plt.axvline(np.mean(pos_scores_np), color='blue', linestyle='--', label='Pos Mean')
    plt.hist(neg_scores_np, bins=30, alpha=0.6, label='Negative', color='orange', density=True)
    plt.axvline(np.mean(neg_scores_np), color='orange', linestyle='--', label='Neg Mean')
    plt.title("Score Distribution (Positive vs Negative Edges)")
    plt.xlabel("Predicted Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Original vs Reconstructed Graphs
    all_possible_edges = torch.cat([test_edge_index, neg_test_edges], dim=1)
    edge_scores_tensor = torch.tensor(scores)
    top_k = test_edge_index.size(1) 

    _, top_k_indices = torch.topk(edge_scores_tensor, top_k)
    top_k_edges = all_possible_edges[:, top_k_indices]

    G_original = nx.Graph()
    G_original.add_edges_from(test_edge_index.t().cpu().numpy())

    G_reconstructed = nx.Graph()
    G_reconstructed.add_edges_from(top_k_edges.t().cpu().numpy())

    all_nodes = set(G_original.nodes()).union(set(G_reconstructed.nodes()))
    G_all = nx.Graph()
    G_all.add_nodes_from(all_nodes)

    # Picks 200 random nodes to draw subgraphs
    sample_nodes = random.sample(list(all_nodes), 200)
    G_original_sub = G_original.subgraph(sample_nodes).copy()
    G_reconstructed_sub = G_reconstructed.subgraph(sample_nodes).copy()

    subgraph_nodes = set(G_original_sub.nodes()).union(set(G_reconstructed_sub.nodes()))
    G_sub_all = nx.Graph()
    G_sub_all.add_nodes_from(subgraph_nodes)

    layout_sub_all = nx.spring_layout(G_sub_all, seed=42)

    # Subgraph of the original graph
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G_sub_all, pos=layout_sub_all, node_size=50, node_color='gray', alpha=0.3)  # background nodes
    nx.draw_networkx_edges(G_original_sub, pos=layout_sub_all, edge_color='blue', alpha=0.5)
    nx.draw_networkx_nodes(G_original_sub, pos=layout_sub_all, node_size=50, node_color='blue', alpha=0.7)
    plt.title("Original Graph Subgraph (Sample)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Subgraph of the reconstructed graph
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G_sub_all, pos=layout_sub_all, node_size=50, node_color='gray', alpha=0.3)  # background nodes
    nx.draw_networkx_edges(G_reconstructed_sub, pos=layout_sub_all, edge_color='green', alpha=0.5)
    nx.draw_networkx_nodes(G_reconstructed_sub, pos=layout_sub_all, node_size=50, node_color='green', alpha=0.7)
    plt.title("Reconstructed Graph Subgraph (Sample)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

