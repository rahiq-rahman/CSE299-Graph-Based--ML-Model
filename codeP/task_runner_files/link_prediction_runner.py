import torch
import os
import random
from load_movielens import load_movielens
from link_prediction import recommend

def run_link_prediction(model_fn, training_path, testing_path, model_path):
    x, edge_index, test_edge_index, num_users, num_items = load_movielens(training_path, testing_path)

    if not os.path.exists(model_path):
        model, x, edge_index, num_users, num_items = model_fn(
            num_users, num_items, list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        )
        torch.save({
            'model': model,
            'x': x,
            'edge_index': edge_index,
            'num_users': num_users,
            'num_items': num_items
        }, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']
        x = loaded['x']
        edge_index = loaded['edge_index']
        num_users = loaded['num_users']
        num_items = loaded['num_items']

    print("\nSample Link Predictions:")
    for _ in range(3):
        user_id = random.randint(0, num_users - 1)
        recs = recommend(model, x, edge_index, user_id, num_users, num_items, top_k=5)
        print(f"\nTop recommendations for user {user_id}:")
        for rank, (item_id, score) in enumerate(recs, start=1):
            print(f"{rank}. Item {item_id} → Score: {score:.4f}")
