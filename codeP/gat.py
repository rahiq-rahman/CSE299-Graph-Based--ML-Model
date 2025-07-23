import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

def create_graph_gat(num_users, num_items, interactions):
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