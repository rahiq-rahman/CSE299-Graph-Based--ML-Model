import torch
import os
import random
import numpy as np
from collections import Counter

def run_node_clustering(model_fn, training_path, model_path):
    x = torch.load(os.path.join(training_path, "features.pt"))
    edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model, embeddings, cluster_labels = model_fn(x, edge_index)
        torch.save({
            'model': model,
            'embeddings': embeddings,
            'cluster_labels': cluster_labels
        }, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        embeddings = loaded['embeddings']
        cluster_labels = loaded['cluster_labels']

    cluster_labels = np.array(cluster_labels)

    print(f"\nTotal Clusters: {len(set(cluster_labels))}")
    print("\nCluster Size Distribution:")
    for cluster_id, count in Counter(cluster_labels).items():
        print(f"  Cluster {cluster_id}: {count} nodes")

    print("\nSample Node Clusters:")
    for nid in random.sample(range(len(cluster_labels)), 5):
        print(f"Node {nid} → Cluster {cluster_labels[nid]}")
