import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphEmbeddingNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class GMN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.embedder = GraphEmbeddingNet(in_channels, hidden_channels)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data1, data2):
        emb1 = self.embedder(data1.x, data1.edge_index, data1.batch)
        emb2 = self.embedder(data2.x, data2.edge_index, data2.batch)

        interaction = torch.cat([emb1, emb2], dim=1)
        attention = self.attn(interaction)
        attended = attention * emb1 + (1 - attention) * emb2
        return self.classifier(attended)

    def predict(self, g1, g2):
        self.eval()
        g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
        g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)
        with torch.no_grad():
            out = self.forward(g1, g2)
            return torch.sigmoid(out).item() > 0.5
