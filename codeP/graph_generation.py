import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from graphvae import GraphVAE

def graph_generation_graphvae(graphs, epochs=100, lr=0.01):
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

            recon_loss = F.binary_cross_entropy(adj_pred, adj_true)
            kl_loss = -0.5 * torch.mean(1 + 2 * logstd - mu**2 - (2 * logstd).exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model