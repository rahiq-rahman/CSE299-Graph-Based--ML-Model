import os

def load_movielens_from_udata(file_path, min_rating=4.0):
    interactions = []
    user_set = set()
    item_set = set()

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')  # u.data format: user\titem\trating\ttimestamp
            if len(parts) < 3:
                continue
            user, item, rating = int(parts[0]), int(parts[1]), float(parts[2])
            if rating >= min_rating:
                interactions.append((user, item))
                user_set.add(user)
                item_set.add(item)

    # Remap to compact indices
    user_list = sorted(list(user_set))
    item_list = sorted(list(item_set))

    user_map = {u: i for i, u in enumerate(user_list)}
    item_map = {i: j for j, i in enumerate(item_list)}

    bipartite_pairs = [(user_map[u], item_map[i] + len(user_map)) for u, i in interactions]
    return bipartite_pairs, len(user_map), len(item_map)
