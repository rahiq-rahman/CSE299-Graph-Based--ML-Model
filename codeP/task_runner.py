# Task Runners
from task_runner_files.node_classification_runner import run_node_classification
from task_runner_files.node_regression_runner import run_node_regression
from task_runner_files.link_prediction_runner import run_link_prediction
from task_runner_files.node_embedding_runner import run_node_embedding
from task_runner_files.node_clustering_runner import run_node_clustering
from task_runner_files.edge_classification_runner import run_edge_classification
from task_runner_files.edge_regression_runner import run_edge_regression
from task_runner_files.graph_classification_runner import run_graph_classification
from task_runner_files.graph_regression_runner import run_graph_regression
from task_runner_files.graph_matching_runner import run_graph_matching
from task_runner_files.graph_reconstruction_runner import run_graph_reconstruction
from task_runner_files.graph_generation_runner import run_graph_generation


def run_task(task, model_fn, dataset_key, training_path, testing_path):
    print(f"Running task: {task} using dataset: {dataset_key}")

    # Model save path
    model_name = model_fn.__name__
    model_path = f"codeP/saved_models/{task}_{model_name}_{dataset_key}.pt"


    # Tasks
    if task == "node_classification":
        run_node_classification(model_fn, training_path, testing_path, model_path)
        return

    elif task == "node_regression":
        run_node_regression(model_fn, training_path, testing_path, model_path)
        return

    elif task == "link_prediction":
        run_link_prediction(model_fn, training_path, testing_path, model_path)
        return

    elif task == "edge_classification":
        run_edge_classification(model_fn, training_path, testing_path, model_path)
        return

    elif task == "edge_regression":
        run_edge_regression(model_fn, training_path, testing_path, model_path)
        return

    elif task == "node_clustering":
        run_node_clustering(model_fn, training_path, model_path)
        return

    elif task == "node_embedding":
        run_node_embedding(model_fn, training_path, model_path)
        return

    elif task == "graph_classification":
        run_graph_classification(model_fn, training_path, testing_path, model_path)
        return

    elif task == "graph_regression":
        run_graph_regression(model_fn, training_path, testing_path, model_path)
        return

    elif task == "graph_matching":
        run_graph_matching(model_fn, training_path, testing_path, model_path)
        return

    elif task == "graph_reconstruction":
        run_graph_reconstruction(model_fn, training_path, testing_path, model_path)
        return

    elif task == "graph_generation":
        run_graph_generation(model_fn, training_path, testing_path, model_path)
        return

    else:
        print("Task not implemented.")
