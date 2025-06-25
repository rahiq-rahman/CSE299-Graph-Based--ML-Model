import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_manual_graph(num_users=3, num_items=4, interactions=None):
    num_nodes = num_users + num_items
    x = torch.zeros((num_nodes, 2))
    x[:num_users, 0] = 1  # users
    x[num_users:, 1] = 1  # items

    if interactions is None:
        interactions = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 6)]

    user_nodes = []
    item_nodes = []
    for u, i in interactions:
        user_nodes.append(u)
        item_nodes.append(i)

    edge_index = torch.tensor([user_nodes, item_nodes], dtype=torch.long)
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    pos_edges = edge_index[:, :len(interactions)].t()
    pos_set = set(interactions)
    neg_edges = []
    for u in range(num_users):
        for i in range(num_users, num_nodes):
            if (u, i) not in pos_set:
                neg_edges.append([u, i])
    neg_edges = torch.tensor(neg_edges, dtype=torch.long)

    return x, edge_index, pos_edges, neg_edges

def train_manual_gcn(num_users=3, num_items=4, interactions=None, epochs=100):
    x, edge_index, pos_edges, neg_edges = create_manual_graph(num_users, num_items, interactions)

    model = GCN(input_dim=2, hidden_dim=8, output_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x, edge_index)

        pos_scores = (embeddings[pos_edges[:,0]] * embeddings[pos_edges[:,1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edges[:,0]] * embeddings[neg_edges[:,1]]).sum(dim=1)

        loss = - (torch.log(torch.sigmoid(pos_scores)).mean() + torch.log(1 - torch.sigmoid(neg_scores)).mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, x, edge_index, num_users, num_items

def add_interaction(edge_index, user, item):
    new_edges = torch.tensor([[user, item], [item, user]], dtype=torch.long)
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    return edge_index

def recommend(model, x, edge_index, user_id, num_users, num_items, top_k=3):
    model.eval()
    embeddings = model(x, edge_index)

    item_ids = list(range(num_users, num_users + num_items))
    scores = []
    for item_id in item_ids:
        score = (embeddings[user_id] * embeddings[item_id]).sum().item()
        scores.append((item_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]