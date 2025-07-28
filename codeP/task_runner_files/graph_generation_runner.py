import torch
import os

# Loader
from load_graphgen import load_graph_generation_dataset

def run_graph_generation(model_fn, training_path, testing_path, model_path):
    graphs = load_graph_generation_dataset(training_path)
        
    if not os.path.exists(model_path):
        model = model_fn(graphs)
        torch.save({'model': model}, model_path)
    else:
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']

    generated_graphs = model.generate(num_graphs=3)
    print("\nGenerated Graphs:")
    for i, g in enumerate(generated_graphs):
        print(f"Graph {i+1} → Nodes: {g.num_nodes} | Edges: {g.num_edges}")