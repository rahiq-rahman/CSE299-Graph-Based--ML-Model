import sys
import os
sys.path.append(os.path.dirname(__file__))

import argparse
from example import exampleFunction
import gcn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--otherOptions', required=False, type=str, default=None)
parser.add_argument('--message', required=False, type=str, default=None, help='Custom Message')

args = parser.parse_args()

if args.model.lower() == "gcn":
    # Step 1: Ask user for interactions manually
    interactions = []
    print("Enter user-item interactions (e.g., '0 3'). Type 'done' to finish.")

    while True:
        line = input("Interaction: ")
        if line.strip().lower() == 'done':
            break
        try:
            user, item = map(int, line.strip().split())
            interactions.append((user, item))
        except:
            print("Invalid format. Please enter two integers separated by space.")

    # Step 2: Infer num_users and num_items automatically
    if interactions:
        max_user = max(u for u, _ in interactions)
        max_item = max(i for _, i in interactions)
        num_users = max_user + 1
        num_items = max_item - max_user
    else:
        print("No interactions given. Exiting.")
        exit()

    # Step 3: Train the model using manual data
    model, x, edge_index, num_users, num_items = gcn.train_manual_gcn(
        num_users=num_users,
        num_items=num_items,
        interactions=interactions
    )


    # You can add some default interaction or skip this step
    edge_index = gcn.add_interaction(edge_index, user=2, item=6)

    # Ask user for input
    while True:
        user_input = input(f"\nEnter a user ID (0 to {num_users-1}) to get top recommendations, or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting.")
            break
        if not user_input.isdigit():
            print("Please enter a valid number or 'q'.")
            continue

        user_id = int(user_input)
        if user_id < 0 or user_id >= num_users:
            print(f"User ID must be between 0 and {num_users-1}.")
            continue

        recs = gcn.recommend(model, x, edge_index, user_id, num_users, num_items, top_k=5)
        print(f"\nTop recommendations for user {user_id}:")
        for rank, (item_id, score) in enumerate(recs, start=1):
            print(f"{rank}. Item {item_id} -> Score: {score:.4f}")


else:
    exampleFunction(args.message)
