import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
import random
import networkx as nx
import matplotlib.pyplot as plt
from link_prediction import recommend

# Loaders
from load_movielens import load_movielens_from_udata
from load_cora import load_cora
from load_karate_regression import load_karate_regression
from load_cora_edge import load_cora_edge
from load_movielens_edge import load_movielens_edge
from load_mutag import load_mutag
from load_synthetic import load_synthetic_regression
from load_graph_matching import load_graph_matching_pairs
from load_cora_reconstruction import load_cora_reconstruction
from load_graphgen import load_graph_generation_dataset


def run_task(task, model_fn, dataset_key, training_path, testing_path):
    if task == "node_classification":
        x, edge_index, labels, train_mask, test_mask = load_cora(training_path, testing_path)
        model = model_fn(x, edge_index, labels, train_mask, test_mask)
        model.eval()
        preds = model(x, edge_index).argmax(dim=1)
        print("\nSample Node Predictions:")
        for nid in random.sample(range(len(preds)), 3):
            print(f"Node {nid} → Predicted Class: {preds[nid].item()}")

    elif task == "node_regression":
        x, edge_index, targets, train_mask, test_mask = load_karate_regression(training_path, testing_path)
        model = model_fn(x, edge_index, targets, train_mask, test_mask)
        model.eval()
        out = model(x, edge_index).squeeze()
        print("\nSample Node Regression Results:")
        for nid in random.sample(range(len(out)), 3):
            print(f"Node {nid} → Predicted: {out[nid].item():.4f} | Actual: {targets[nid].item():.4f}")

    elif task == "link_prediction":
        train_file = os.path.join(training_path, "u.data")
        interactions, num_users, num_items = load_movielens_from_udata(train_file)
        model, x, edge_index, num_users, num_items = model_fn(num_users, num_items, interactions)
        print("\nSample Link Predictions:")
        for _ in range(3):
            user_id = random.randint(0, num_users - 1)
            recs = recommend(model, x, edge_index, user_id, num_users, num_items, top_k=5)
            print(f"\nTop recommendations for user {user_id}:")
            for rank, (item_id, score) in enumerate(recs, start=1):
                print(f"{rank}. Item {item_id} → Score: {score:.4f}")

    elif task == "edge_classification":
        x, edge_index, edge_labels, train_mask, test_mask = load_cora_edge(training_path, testing_path)
        model, classifier = model_fn(x, edge_index, edge_labels, train_mask, test_mask)
        model.eval()
        classifier.eval()
        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        edge_logits = classifier(edge_emb)
        probs = torch.softmax(edge_logits, dim=1)
        print("\nSample Edge Classifications:")
        for _ in range(3):
            eid = random.randint(0, edge_index.shape[1] - 1)
            prob = probs[eid]
            print(f"Edge ({edge_index[0,eid].item()}, {edge_index[1,eid].item()}) → "
                f"Pred: {prob.argmax().item()} | True: {edge_labels[eid].item()} | "
                f"Confidence: {prob.max().item():.4f}")

    elif task == "edge_regression":
        x, edge_index, edge_ratings, train_mask, test_mask = load_movielens_edge(training_path, testing_path)
        model, regressor = model_fn(x, edge_index, edge_ratings, train_mask, test_mask)
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

    elif task == "node_clustering":
        x = torch.load(os.path.join(training_path, "features.pt"))
        edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))
        model, embeddings, cluster_labels = model_fn(x, edge_index)
        print("\nSample Node Clusters:")
        for nid in random.sample(range(len(cluster_labels)), 3):
            print(f"Node {nid} → Cluster {cluster_labels[nid]}")

    elif task == "node_embedding":
        x = torch.load(os.path.join(training_path, "features.pt"))
        edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))
        model, embeddings = model_fn(x, edge_index)
        print("\nSample Node Embeddings:")
        for nid in range(min(3, embeddings.size(0))):
            emb_str = ", ".join(f"{v:.4f}" for v in embeddings[nid][:5])
            print(f"Node {nid} → Embedding: [{emb_str}, ...]")

    elif task == "graph_classification":
        train_loader, test_loader, input_dim, num_classes = load_mutag(training_path, testing_path)
        model = model_fn(train_loader, test_loader, input_dim, num_classes)
        test_graphs = list(test_loader)
        print("\nSample Graph Classification:")
        sample_indices = random.sample(range(len(test_graphs)), 3)
        for idx in sample_indices:
            graph = test_graphs[idx]
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, graph.batch)
                pred = out.argmax(dim=1).item()
            print(f"Graph {idx} → Predicted: {pred} | True: {graph.y.item()}")

    elif task == "graph_regression":
        train_loader, test_loader, input_dim = load_synthetic_regression(training_path, testing_path)
        model = model_fn(train_loader, test_loader, input_dim)
        test_graphs = list(test_loader)
        print("\nSample Graph Regression:")
        sample_indices = random.sample(range(len(test_graphs)), 3)
        for idx in sample_indices:
            graph = test_graphs[idx]
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))
                pred = out.item() if out.numel() == 1 else out.squeeze().item()
            print(f"Graph {idx} → Predicted: {pred:.4f} | True: {graph.y.item():.4f}")

    elif task == "graph_matching":
        train_pairs = load_graph_matching_pairs(training_path)
        test_pairs = load_graph_matching_pairs(testing_path)
        model = model_fn(train_pairs, test_pairs)
        print("\nSample Graph Matching Results:")
        for i in random.sample(range(len(test_pairs)), 3):
            g1, g2, label = test_pairs[i]
            pred = model.predict(g1, g2)
            print(f"Pair {i} → Predicted: {pred} | True: {label.item()}")

    elif task == "graph_reconstruction":
        x, train_edge_index, test_edge_index = load_cora_reconstruction(training_path, testing_path)
        model, z = model_fn(x, train_edge_index, test_edge_index)
        neg_test_edges = negative_sampling(edge_index=train_edge_index, num_nodes=x.size(0), num_neg_samples=test_edge_index.size(1))

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

        print("\nVisualizing Reconstructed Edges (Green = True Pos, Red = False Pos)...")

        G = nx.Graph()
        G.add_nodes_from(range(x.size(0)))

        topk = 5
        edge_scores = list(zip(test_edge_index[0].tolist(), test_edge_index[1].tolist(), pos_scores.tolist()))
        edge_scores.sort(key=lambda x: -x[2])

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)

        train_edges = list(zip(train_edge_index[0].tolist(), train_edge_index[1].tolist()))
        nx.draw_networkx_edges(G, pos, edgelist=train_edges, edge_color='lightgray', width=0.5, alpha=0.5)

        for i in range(topk):
            u, v, score = edge_scores[i]
            G.add_edge(u, v)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='green', width=2.0, label="Reconstructed Edge")

        neg_edge_scores = list(zip(neg_test_edges[0].tolist(), neg_test_edges[1].tolist(), neg_scores.tolist()))
        neg_edge_scores.sort(key=lambda x: -x[2])
        for i in range(topk):
            u, v, score = neg_edge_scores[i]
            G.add_edge(u, v)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', width=2.0, style='dashed')

        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title("Top Reconstructed Edges (Green=True Positives, Red=False Positives)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    elif task == "graph_generation":
        graphs = load_graph_generation_dataset(training_path)
        model = model_fn(graphs)
        generated_graphs = model.generate(num_graphs=3)
        print("\nGenerated Graphs:")
        for i, g in enumerate(generated_graphs):
            print(f"Graph {i+1} → Nodes: {g.num_nodes} | Edges: {g.num_edges}")

    else:
        print("Task not implemented.")