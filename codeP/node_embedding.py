import torch
import torch.nn.functional as F
from gcn import GCN
from gat import GAT
from graphsage import GraphSAGE


#GCN
def node_embedding_gcn(x, edge_index, epochs=100):
    model = GCN(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = (out ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[GCN-NodeEmbed] Epoch {epoch} - Dummy Loss: {loss.item():.4f}")

    return model, out.detach()


#GAT
def node_embedding_gat(x, edge_index, epochs=100):
    model = GAT(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = (out ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[GAT-NodeEmbed] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, out.detach()


#GraphSAGE
def node_embedding_graphsage(x, edge_index, epochs=100):
    model = GraphSAGE(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = (out ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[SAGE-NodeEmbed] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, out.detach()