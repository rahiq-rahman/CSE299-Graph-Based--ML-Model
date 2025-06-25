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
    # Train the manual GCN
    model, x, edge_index, num_users, num_items = gcn.train_manual_gcn()

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
        for item_id, score in recs:
            print(f"Item {item_id} with score {score:.4f}")

else:
    exampleFunction(args.message)
