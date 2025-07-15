import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logstd = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        return mu, logstd

class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, in_channels)
        )

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)

        # Decode to dense adjacency
        adj = torch.sigmoid(z @ z.T)
        return adj, mu, logstd

    def generate(self, num_graphs=5, num_nodes=10):
        generated_graphs = []
        for _ in range(num_graphs):
            z = torch.randn((num_nodes, self.encoder.conv_mu.out_channels))
            adj = torch.sigmoid(z @ z.T)
            edge_index, _ = dense_to_sparse((adj > 0.5).float())
            x = self.decoder(z)
            generated_graphs.append(Data(x=x, edge_index=edge_index))
        return generated_graphs

def train_graph_generation(graphs, epochs=100, lr=0.01):
    in_channels = graphs[0].x.size(1)
    model = GraphVAE(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in graphs:
            x, edge_index = data.x, data.edge_index
            adj_true = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]

            optimizer.zero_grad()
            adj_pred, mu, logstd = model(x, edge_index)

            # Loss = reconstruction + KL
            recon_loss = F.binary_cross_entropy(adj_pred, adj_true)
            kl_loss = -0.5 * torch.mean(1 + 2 * logstd - mu**2 - (2 * logstd).exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model
