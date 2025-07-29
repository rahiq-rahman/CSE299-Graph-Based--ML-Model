import torch
import os
import random
from sklearn.metrics import mean_squared_error, r2_score

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
    with torch.no_grad():
        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        preds = regressor(edge_emb).squeeze()

    # Evaluation
    y_true = edge_ratings[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    print("\nSample Predictions:")
    test_indices = torch.where(test_mask)[0].tolist()
    samples = random.sample(test_indices, min(5, len(test_indices)))
    for eid in samples:
        u, v = edge_index[0, eid].item(), edge_index[1, eid].item()
        pred = preds[eid].item()
        true = edge_ratings[eid].item()
        print(f"Edge ({u}, {v}) → Predicted: {pred:.2f}, Actual: {true:.2f}, Error: {abs(pred - true):.2f}")
