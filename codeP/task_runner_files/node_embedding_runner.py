import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def run_node_embedding(model_fn, training_path, model_path):
    x = torch.load(os.path.join(training_path, "features.pt"))
    edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))

    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model, embeddings = model_fn(x, edge_index)
        torch.save({'model': model, 'embeddings': embeddings}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        embeddings = loaded['embeddings']

    embeddings_np = embeddings.cpu().numpy()

    print("\nSample Node Embeddings:")
    for nid in range(min(5, embeddings.size(0))):
        emb_str = ", ".join(f"{v:.4f}" for v in embeddings[nid][:5])
        print(f"Node {nid} → Embedding: [{emb_str}, ...]")

    # Embedding stats
    print("\nEmbedding Statistics:")
    print(f"Mean: {np.mean(embeddings_np):.4f}")
    print(f"Std Dev: {np.std(embeddings_np):.4f}")
    print(f"Min: {np.min(embeddings_np):.4f}")
    print(f"Max: {np.max(embeddings_np):.4f}")

    # Cosine similarity
    print("\nCosine Similarity Between Sample Nodes:")
    sample_ids = list(range(min(5, len(embeddings_np))))
    cos_sim = cosine_similarity(embeddings_np[sample_ids])
    for i in range(len(sample_ids)):
        for j in range(i + 1, len(sample_ids)):
            print(f"Node {sample_ids[i]} vs Node {sample_ids[j]}: {cos_sim[i][j]:.4f}")
