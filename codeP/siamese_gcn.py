import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class SiameseGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data1, data2):
        x1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        x2 = self.encoder(data2.x, data2.edge_index, data2.batch)
        x = torch.cat([x1, x2], dim=1)
        return self.classifier(x)

    def predict(self, g1, g2):
        self.eval()
        g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
        g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)
        with torch.no_grad():
            out = self.forward(g1, g2)
            return torch.sigmoid(out).item() > 0.5

def train_graph_matching(train_pairs, test_pairs, in_channels=None, epochs=20, lr=0.001):
    in_channels = in_channels or train_pairs[0][0].x.size(1)

    model = SiameseGCN(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for g1, g2, label in train_pairs:
            g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
            g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)
            optimizer.zero_grad()
            out = model(g1, g2).squeeze()
            loss = loss_fn(out, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    return model
