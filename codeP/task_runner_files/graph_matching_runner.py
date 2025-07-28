import torch
import os
import random
from load_graph_matching import load_graph_matching_pairs

def run_graph_matching(model_fn, training_path, testing_path, model_path):
    train_pairs = load_graph_matching_pairs(training_path)
    test_pairs = load_graph_matching_pairs(testing_path)

    if not os.path.exists(model_path):
        model = model_fn(train_pairs, test_pairs)
        torch.save({'model': model}, model_path)
    else:
        model = torch.load(model_path, weights_only=False)['model']

    print("\nSample Graph Matching Results:")
    for i in random.sample(range(len(test_pairs)), 3):
        g1, g2, label = test_pairs[i]
        pred = model.predict(g1, g2)
        print(f"Pair {i} → Predicted: {pred} | Actual: {label.item()}")
