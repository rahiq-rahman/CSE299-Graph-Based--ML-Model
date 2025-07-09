import os
import sys
import argparse
import torch
from load_movielens import load_movielens_from_udata
from load_cora import load_cora
from load_karate_regression import load_karate_regression
from load_cora_edge import load_cora_edge
from load_movielens_edge import load_movielens_edge

sys.path.append(os.path.dirname(__file__))

import gcn, gat, graphsage

# Available Models
available_models = {
    "gcn": {
        "name": "GCN",
        "module": {
            "link_prediction": gcn.train_link_prediction,
            "node_classification": gcn.train_node_classification,
            "node_regression": gcn.train_node_regression,
            "edge_classification": gcn.train_edge_classification,
            "edge_regression": gcn.train_edge_regression,
            "node_clustering": gcn.train_node_clustering,
            "node_embedding": gcn.train_node_embedding
        }
    },
    "gat": {
        "name": "GAT",
        "module": {
            "link_prediction": gat.train_link_prediction,
            "node_classification": gat.train_node_classification,
            "node_regression": gat.train_node_regression,
            "edge_classification": gat.train_edge_classification,
            "edge_regression": gat.train_edge_regression,
            "node_clustering": gat.train_node_clustering,
            "node_embedding": gat.train_node_embedding
        }
    },
    "graphsage": {
        "name": "GraphSAGE",
        "module": {
            "link_prediction": graphsage.train_link_prediction,
            "node_classification": graphsage.train_node_classification,
            "node_regression": graphsage.train_node_regression,
            "edge_classification": graphsage.train_edge_classification,
            "edge_regression": graphsage.train_edge_regression,
            "node_clustering": graphsage.train_node_clustering,
            "node_embedding": graphsage.train_node_embedding
        }
    },
}

# Implemented Graph Tasks
tasks = {
    "1": {"name": "Link Prediction", "function": "link_prediction"},
    "2": {"name": "Node Classification", "function": "node_classification"},
    "3": {"name": "Node Regression", "function": "node_regression"},
    "4": {"name": "Edge Classification", "function": "edge_classification"},
    "5": {"name": "Edge Regression", "function": "edge_regression"},
    "6": {"name": "Node Clustering", "function": "node_clustering"},
    "7": {"name": "Node Embedding", "function": "node_embedding"}
}

# Task compatibility configuration
task_config = {
    "link_prediction": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["movielens"]
    },
    "node_classification": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["cora"]
    },
    "node_regression": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["karate_regression"]
    },
    "edge_classification": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["cora_edge"]
    },
    "edge_regression": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["movielens_edge"]
    },
    "node_clustering": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["cora"]
    },
    "node_embedding": {
        "models": ["gcn", "gat", "graphsage"],
        "datasets": ["cora", "karate_regression"]
    }
}

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, help='Model name (e.g., gcn, gat)')
parser.add_argument('--dataset', required=False, type=str, help='Dataset folder name inside /data')
args = parser.parse_args()

data_root = os.path.join(os.path.dirname(__file__), "..", "data")

model_key = args.model.lower() if args.model else None
dataset_key = args.dataset.lower() if args.dataset else None

def get_compatible_tasks(model_key, dataset_key):
    compatible = {}
    for task_key, config in task_config.items():
        if model_key in config["models"] and dataset_key in config["datasets"]:
            for task_id, task in tasks.items():
                if task["function"] == task_key:
                    compatible[task_id] = task
                    break
    return compatible

# Task Selection
if model_key and dataset_key:
    if model_key not in available_models:
        print(f"Model '{model_key}' not found.")
        exit()

    dataset_path = os.path.join(data_root, dataset_key)
    if not os.path.isdir(dataset_path):
        print(f"Dataset '{dataset_key}' not found in /data folder.")
        exit()

    compatible_tasks = get_compatible_tasks(model_key, dataset_key)

    if not compatible_tasks:
        print(f"No compatible tasks found for model '{model_key}' and dataset '{dataset_key}'. Exiting.")
        exit()

    print("\nCompatible Tasks for provided model and dataset:")
    for key, task in compatible_tasks.items():
        print(f"{key}. {task['name']}")

    task_choice = input("\nSelect a task by number (e.g., 1): ").strip()
    if task_choice not in compatible_tasks:
        print("Invalid task. Exiting.")
        exit()

    selected_task = compatible_tasks[task_choice]["function"]
else:
    print("Available Graph-Based Tasks:")
    for key, task in tasks.items():
        print(f"{key}. {task['name']}")

    task_choice = input("\nSelect a task by number (e.g., 1): ").strip()
    if task_choice not in tasks:
        print("Invalid task. Exiting.")
        exit()
    selected_task = tasks[task_choice]["function"]

    compatible_models = task_config[selected_task]["models"]
    compatible_datasets = task_config[selected_task]["datasets"]

    # Prompt for model
    while model_key not in compatible_models:
        if model_key is not None:
            print(f"Model '{model_key}' is not compatible with task '{selected_task}'.")
        print("Compatible Models:", ", ".join(compatible_models))
        model_key = input("Enter a valid model name: ").strip().lower()

    # Prompt for dataset
    while dataset_key not in compatible_datasets:
        if dataset_key is not None:
            print(f"Dataset '{dataset_key}' is not compatible with task '{selected_task}'.")
        print("Compatible Datasets:", ", ".join(compatible_datasets))
        dataset_key = input("Enter a valid dataset name: ").strip()

# Final validation & setup
model_name = available_models[model_key]["name"]
train_model_fn = available_models[model_key]["module"][selected_task]
dataset_name = dataset_key
dataset_path = os.path.join(data_root, dataset_name)
training_path = os.path.join(dataset_path, "training")
testing_path = os.path.join(dataset_path, "testing")

# Final Info Display
print(f"\nSelected Task: {tasks[task_choice]['name']}")
print(f"Selected Model: {model_name}")
print(f"Selected Dataset: {dataset_name}\n")

# Perform Task
if selected_task == "link_prediction":
    train_file = os.path.join(training_path, "u.data")
    test_file = os.path.join(testing_path, "u.data")

    interactions, num_users, num_items = load_movielens_from_udata(train_file)
    if not interactions:
        print("No training interactions found. Exiting.")
        exit()

    model, x, edge_index, num_users, num_items = train_model_fn(
        num_users=num_users,
        num_items=num_items,
        interactions=interactions
    )

    while True:
        user_input = input(f"\nEnter user ID (0 to {num_users - 1}) for recommendations, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        if not user_input.isdigit():
            print("Invalid input.")
            continue
        user_id = int(user_input)
        if user_id < 0 or user_id >= num_users:
            print("User ID out of range.")
            continue
        recs = gcn.recommend(model, x, edge_index, user_id, num_users, num_items, top_k=5)
        print(f"\nTop recommendations for user {user_id}:")
        for rank, (item_id, score) in enumerate(recs, start=1):
            print(f"{rank}. Item {item_id} -> Score: {score:.4f}")

elif selected_task == "node_classification":
    training_dir = os.path.join(data_root, dataset_name, "training")
    testing_dir = os.path.join(data_root, dataset_name, "testing")
    x, edge_index, labels, train_mask, test_mask = load_cora(training_dir, testing_dir)

    train_model_fn = available_models[model_key]["module"][selected_task]
    model = train_model_fn(x, edge_index, labels, train_mask, test_mask)

    model.eval()
    out = model(x, edge_index)
    preds = out.argmax(dim=1)

    while True:
        user_input = input(f"\nEnter a node ID (0 to {x.size(0)-1}) to predict its label (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input.isdigit():
            print("Invalid input. Enter a valid node ID.")
            continue

        node_id = int(user_input)
        if node_id < 0 or node_id >= x.size(0):
            print(f"Node ID out of range (0 to {x.size(0)-1})")
            continue

        predicted_label = preds[node_id].item()
        print(f"Node {node_id} is predicted to belong to class: {predicted_label}")

        # show similar nodes
        distances = (out - out[node_id]).norm(dim=1)
        similar_nodes = distances.argsort()[1:6]
        print("Top 5 most similar nodes (by embedding distance):")
        for i, nid in enumerate(similar_nodes.tolist(), start=1):
            print(f"{i}. Node {nid} - Predicted Class: {preds[nid].item()} - Distance: {distances[nid].item():.4f}")

elif selected_task == "node_regression":
    train_path = os.path.join(data_root, dataset_name, "training")
    test_path = os.path.join(data_root, dataset_name, "testing")

    x, edge_index, targets, train_mask, test_mask = load_karate_regression(train_path, test_path)
    train_model_fn = available_models[model_key]["module"][selected_task]
    model = train_model_fn(x, edge_index, targets, train_mask, test_mask)

    model.eval()
    out = model(x, edge_index).squeeze()

    while True:
        user_input = input(f"\nEnter a node ID (0 to {x.size(0)-1}) to predict its value (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input.isdigit():
            print("Invalid input. Enter a valid node ID.")
            continue

        node_id = int(user_input)
        if node_id < 0 or node_id >= x.size(0):
            print(f"Node ID out of range (0 to {x.size(0)-1})")
            continue

        pred_value = out[node_id].item()
        actual_value = targets[node_id].item()
        print(f"Node {node_id} prediction: {pred_value:.4f} | Actual: {actual_value:.4f}")

        # show similar nodes based on embedding distance
        distances = (out - out[node_id]).abs()
        similar_nodes = distances.argsort()[1:6]
        print("\nTop 5 most similar nodes:")
        for i, nid in enumerate(similar_nodes.tolist(), start=1):
            print(f"{i}. Node {nid} - Predicted: {out[nid].item():.4f} | Actual: {targets[nid].item():.4f} | Difference: {abs(out[nid].item() - pred_value):.4f}")


elif selected_task == "edge_classification":
    train_path = os.path.join(data_root, dataset_name, "training")
    test_path = os.path.join(data_root, dataset_name, "testing")

    x, edge_index, edge_labels, train_mask, test_mask = load_cora_edge(train_path, test_path)
    train_model_fn = available_models[model_key]["module"][selected_task]
    model, classifier = train_model_fn(x, edge_index, edge_labels, train_mask, test_mask)

    model.eval()
    classifier.eval()

    with torch.no_grad():
        node_embeddings = model(x, edge_index)
        src = edge_index[0]
        dst = edge_index[1]
        edge_emb = node_embeddings[src] * node_embeddings[dst]
        edge_logits = classifier(edge_emb)
        probs = torch.softmax(edge_logits, dim=1)

        while True:
            edge_id_input = input(f"\nEnter edge index (0 to {edge_index.size(1)-1}) for prediction (or 'q' to quit): ").strip()
            if edge_id_input.lower() == 'q':
                break
            if not edge_id_input.isdigit():
                print("Invalid input.")
                continue

            eid = int(edge_id_input)
            if eid < 0 or eid >= edge_index.size(1):
                print("Out of range.")
                continue

            prob = probs[eid]
            pred_label = torch.argmax(prob).item()
            true_label = edge_labels[eid].item()
            print(f"Edge ({edge_index[0,eid].item()}, {edge_index[1,eid].item()}) → Predicted: {pred_label}, True: {true_label}, Confidence: {prob[pred_label]:.4f}")


elif selected_task == "edge_regression":
    x, edge_index, edge_ratings, train_mask, test_mask = load_movielens_edge(
        training_path, testing_path
    )

    train_model_fn = available_models[model_key]["module"][selected_task]
    model, regressor = train_model_fn(x, edge_index, edge_ratings, train_mask, test_mask)

    model.eval()
    regressor.eval()
    with torch.no_grad():
        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        pred = regressor(edge_emb).squeeze()

        while True:
            edge_id = input(f"\nEnter edge index (0 to {edge_index.size(1)-1}) for predicted rating (or 'q' to quit): ")
            if edge_id.lower() == 'q':
                break
            if not edge_id.isdigit():
                print("Invalid input.")
                continue
            eid = int(edge_id)
            if eid < 0 or eid >= edge_index.size(1):
                print("Out of range.")
                continue

            u = edge_index[0, eid].item()
            v = edge_index[1, eid].item()
            true_rating = edge_ratings[eid].item()
            pred_rating = pred[eid].item()
            print(f"Edge ({u}, {v}) → Predicted Rating: {pred_rating:.2f}, Actual: {true_rating:.2f}")


elif selected_task == "node_embedding":
    x = torch.load(os.path.join(training_path, "features.pt"))
    edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))

    train_model_fn = available_models[model_key]["module"][selected_task]
    model, embeddings = train_model_fn(x, edge_index)

    print("\nNode Embedding (First 10 nodes):")
    for i in range(min(10, embeddings.size(0))):
        emb_str = ", ".join(f"{val:.4f}" for val in embeddings[i][:5])
        print(f"Node {i}: [{emb_str}, ...]")


elif selected_task == "node_clustering":
    x = torch.load(os.path.join(training_path, "features.pt"))
    edge_index = torch.load(os.path.join(training_path, "edge_index.pt"))

    train_model_fn = available_models[model_key]["module"][selected_task]
    model, embeddings, cluster_labels = train_model_fn(x, edge_index)

    print("\nNode Clustering Results (First 10 nodes):")
    for i in range(min(10, len(cluster_labels))):
        print(f"Node {i}: Cluster {cluster_labels[i]}")



else:
    print("Task not implemented yet.")
