import torch
import os
import matplotlib.pyplot as plt
import networkx as nx

# Loader
from load_graphgen import load_graph_generation_dataset

def run_graph_generation(model_fn, training_path, testing_path, model_path):
    graphs = load_graph_generation_dataset(training_path)
        
    if not os.path.exists(model_path):
        print("No saved model found. Training from scratch...")
        model = model_fn(graphs)
        torch.save({'model': model}, model_path)
    else:
        print(f"Loading saved model from: {model_path}")
        loaded = torch.load(model_path, weights_only=False)
        model = loaded['model']

    generated_graphs = model.generate(num_graphs=3)
    print("\nGenerated Graphs:")
    
    for i, g in enumerate(generated_graphs):
        num_nodes = g.num_nodes
        edge_index = g.edge_index.cpu().numpy()

        G_nx = nx.Graph()
        G_nx.add_edges_from(list(zip(edge_index[0], edge_index[1])))

        num_edges = G_nx.number_of_edges()
        print(f"Graph {i+1} → Nodes: {num_nodes} | Edges: {num_edges}")

        degrees = [d for _, d in G_nx.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        print(f"Avg Degree: {avg_degree:.2f}\nMax Degree: {max_degree}")

        edge_index = g.edge_index.cpu().numpy()
        G_nx = nx.Graph()
        G_nx.add_edges_from(list(zip(edge_index[0], edge_index[1])))

        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        nx.draw(G_nx, ax=ax, node_size=50, with_labels=False)
        ax.set_title(f"Generated Graph {i+1}")
        plt.show()
