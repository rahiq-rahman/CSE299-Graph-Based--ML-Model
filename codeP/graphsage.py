import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_graph(num_users, num_items, interactions):
    num_nodes = num_users + num_items
    x = torch.eye(num_nodes)

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
        for i in range(num_users, num_users + num_items):
            if (u, i) not in pos_set:
                neg_edges.append([u, i])
    neg_edges = torch.tensor(neg_edges, dtype=torch.long)

    return x, edge_index, pos_edges, neg_edges

def train_graphsage(num_users, num_items, interactions, epochs=100):
    x, edge_index, pos_edges, neg_edges = create_graph(num_users, num_items, interactions)

    model = GraphSAGE(input_dim=x.shape[1], hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x, edge_index)

        pos_scores = (embeddings[pos_edges[:, 0]] * embeddings[pos_edges[:, 1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edges[:, 0]] * embeddings[neg_edges[:, 1]]).sum(dim=1)

        loss = - (torch.log(torch.sigmoid(pos_scores)).mean() + torch.log(1 - torch.sigmoid(neg_scores)).mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[GraphSAGE] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, x, edge_index, num_users, num_items
