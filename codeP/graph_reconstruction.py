import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from gae import GCNEncoder
from grae import GRAE

def graph_reconstruction_gae(x, train_edge_index, test_edge_index, hidden_dim=64, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)

    encoder = GCNEncoder(x.size(1), hidden_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        encoder.train()
        optimizer.zero_grad()
        z = encoder(x, train_edge_index)

        pos_edge = train_edge_index
        neg_edge = negative_sampling(
            edge_index=train_edge_index, num_nodes=x.size(0),
            num_neg_samples=pos_edge.size(1)
        )

        pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=1)
        neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} - Loss: {loss.item():.4f}")

    return encoder, z.detach().cpu()

def graph_reconstruction_grae(x, train_edge_index, test_edge_index, hidden_dim=64, latent_dim=32, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)

    model = GRAE(x.size(1), hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z, adj_pred = model(x, train_edge_index)

        adj_true = torch.zeros_like(adj_pred)
        adj_true[train_edge_index[0], train_edge_index[1]] = 1.0

        loss = F.binary_cross_entropy(adj_pred, adj_true)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[GRAE] Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    return model.encoder, z.detach().cpu()