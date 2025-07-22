import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GRAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z


class GRAE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, latent_dim=32):
        super().__init__()
        self.encoder = GRAEEncoder(in_channels, hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_pred = torch.sigmoid(z @ z.T)
        return z, adj_pred
