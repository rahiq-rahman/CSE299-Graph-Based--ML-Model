from loader_config import load_named_files

def load_graph_generation_dataset(data_dir):
    dataset = []
    data_dict = load_named_files(data_dir)

    for key, value in data_dict.items():
        if isinstance(value, list):
            dataset.extend(value)
        else:
            dataset.append(value)

    if not dataset:
        raise ValueError("No valid data found for graph generation.")

    return dataset
