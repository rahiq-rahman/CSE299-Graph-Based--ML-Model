import torch
import os
import random
from load_movielens_edge import load_movielens_edge

def run_edge_regression(model_fn, training_path, testing_path, model_path):
    x, edge_index, edge_ratings, train_mask, test_mask = load_movielens_edge(training_path, testing_path)

    if not os.path.exists(model_path):
        model, regressor = model_fn(x, edge_index, edge_ratings, train_mask, test_mask)
        torch.save({'model': model, 'regressor': regressor}, model_path)
    else:
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        regressor = loaded['regressor']

    model.eval()
    regressor.eval()
    node_embeddings = model(x, edge_index)
    edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
    preds = regressor(edge_emb).squeeze()

    print("\nSample Edge Regression Results:")
    for _ in range(3):
        eid = random.randint(0, edge_index.size(1) - 1)
        u, v = edge_index[0, eid].item(), edge_index[1, eid].item()
        print(f"Edge ({u}, {v}) → Predicted: {preds[eid]:.2f} | Actual: {edge_ratings[eid]:.2f}")
