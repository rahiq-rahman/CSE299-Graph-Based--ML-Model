import os

def load_movielens_from_udata(filepath, return_ratings=False):
    interactions = []
    user_ids = set()
    item_ids = set()

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            user = int(parts[0])
            item = int(parts[1])
            rating = float(parts[2])

            user_ids.add(user)
            item_ids.add(item)

            if return_ratings:
                interactions.append((user, item, rating))
            else:
                interactions.append((user, item))

    user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    item_id_map = {iid: idx + len(user_id_map) for idx, iid in enumerate(sorted(item_ids))}

    if return_ratings:
        mapped_interactions = [(user_id_map[u], item_id_map[i], r) for u, i, r in interactions]
    else:
        mapped_interactions = [(user_id_map[u], item_id_map[i]) for u, i in interactions]

    num_users = len(user_id_map)
    num_items = len(item_id_map)

    return mapped_interactions, num_users, num_items

