from loader_config import load_named_files

def load_graph_matching_pairs(data_dir):
    pairs = []
    data_dict = load_named_files(data_dir)

    for key, value in data_dict.items():
        if isinstance(value, dict):
            g1 = value.get("g1") or value.get("graph1")
            g2 = value.get("g2") or value.get("graph2")
            label = value.get("label")
            if g1 is not None and g2 is not None and label is not None:
                pairs.append((g1, g2, label))
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            pairs.append(tuple(value))

    if not pairs:
        raise ValueError("No valid graph matching pairs found.")

    return pairs
