import os
import sys
import argparse
from model_config import available_models
from dataset_config import valid_datasets
from task_runner import run_task

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--task', required=True)
args = parser.parse_args()

model_key = args.model.lower()
dataset_key = args.dataset.lower()
selected_task = args.task.lower()

if model_key not in available_models:
    print(f"Invalid model: {model_key}")
    sys.exit(1)

if selected_task not in available_models[model_key]["module"]:
    print(f"Task '{selected_task}' not supported by model '{model_key}'")
    sys.exit(1)

if selected_task not in valid_datasets or dataset_key not in valid_datasets[selected_task]:
    print(f"Dataset '{dataset_key}' is not compatible with task '{selected_task}'")
    sys.exit(1)

# Paths
data_root = os.path.join(os.path.dirname(__file__), "..", "data")
dataset_path = os.path.join(data_root, dataset_key)
training_path = os.path.join(dataset_path, "training")
testing_path = os.path.join(dataset_path, "testing")

train_model_fn = available_models[model_key]["module"][selected_task]
run_task(selected_task, train_model_fn, dataset_key, training_path, testing_path)
