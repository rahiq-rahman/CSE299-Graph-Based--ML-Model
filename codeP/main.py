import os
import sys
import argparse
from load_movielens import load_movielens_from_udata

sys.path.append(os.path.dirname(__file__))

import gcn, gat, graphsage

# Available Models

available_models = {
    "gcn": {"name": "GCN", "module": gcn.train_manual_gcn},
    "gat": {"name": "GAT", "module": gat.train_gat},
    "graphsage": {"name": "GraphSAGE", "module": graphsage.train_graphsage}
}

# Implemented Graph Tasks

tasks = {
    "1": {"name": "Link Prediction", "function": "link_prediction"},
}

print("Available Graph-Based Tasks:")
for key, task in tasks.items():
    print(f"{key}. {task['name']}")

# Available Tasks Selection

task_choice = input("\nSelect a task by number (e.g., 1): ").strip()
if task_choice not in tasks:
    print("Invalid task. Exiting.")
    exit()
selected_task = tasks[task_choice]["function"]

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, help='Model name (e.g., gcn, gat)')
parser.add_argument('--dataset', required=False, type=str, help='Dataset folder name inside /data')

args = parser.parse_args()
model_key = args.model
dataset_key = args.dataset

# Model Validation

while model_key not in available_models:
    if model_key is not None:
        print(f"Model '{model_key}' not found.")
    print("Available Models:", ", ".join(available_models.keys()))
    model_key = input("Enter a valid model name: ").strip().lower()

model_name = available_models[model_key]["name"]
train_model_fn = available_models[model_key]["module"]

# Dataset Validation

data_root = os.path.join(os.path.dirname(__file__), "..", "data")
available_datasets = [
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d))
]

while dataset_key not in available_datasets:
    if dataset_key is not None:
        print(f"Dataset '{dataset_key}' not found in /data folder.")
    print("Available Datasets:", ", ".join(available_datasets))
    dataset_key = input("Enter a valid dataset name: ").strip()

dataset_name = dataset_key
dataset_path = os.path.join(data_root, dataset_name)
training_path = os.path.join(dataset_path, "training")
testing_path = os.path.join(dataset_path, "testing")

# Selected Dataset and Model Message

print(f"\nSelected Task: {tasks[task_choice]['name']}")
print(f"Selected Model: {model_name}")
print(f"Selected Dataset: {dataset_name}\n")


# Performing Selected Tasks

if selected_task == "link_prediction":
    interactions = []

    # Load training interactions
    train_file = os.path.join('./data/movielens/training/', "u.data")
    test_file = os.path.join('./data/movielens/testing/', "u.data")

    interactions, num_users, num_items = load_movielens_from_udata(train_file)
    if not interactions:
        print("No training interactions found. Exiting.")
        exit()

    # Train model
    model, x, edge_index, num_users, num_items = train_model_fn(
        num_users=num_users,
        num_items=num_items,
        interactions=interactions
    )

    while True:
        user_input = input(f"\nEnter user ID (0 to {num_users-1}) for recommendations, or 'q' to quit: ")
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
else:
    print("Task not implemented yet.")
