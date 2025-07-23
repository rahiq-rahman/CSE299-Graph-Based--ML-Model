import torch
import torch.nn.functional as F
from gcn import GCN
from gat import GAT
from graphsage import GraphSAGE


#GCN
def node_regression_gcn(x, edge_index, targets, train_mask, test_mask, epochs=200):
    model = GCN(input_dim=x.shape[1], hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index).squeeze()
        loss = F.mse_loss(out[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            test_preds = out[test_mask]
            test_labels = targets[test_mask]
            mse = F.mse_loss(test_preds, test_labels).item()
            print(f"[GCN-NodeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model


#GAT
def node_regression_gat(x, edge_index, targets, train_mask, test_mask, epochs=200):
    model = GAT(input_dim=x.shape[1], hidden_dim=8, output_dim=1, heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index).squeeze()
        loss = F.mse_loss(out[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            test_preds = out[test_mask]
            test_labels = targets[test_mask]
            mse = F.mse_loss(test_preds, test_labels).item()
            print(f"[GAT-NodeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model


#GraphSAGE
def node_regression_graphsage(x, edge_index, targets, train_mask, test_mask, epochs=200):
    model = GraphSAGE(input_dim=x.shape[1], hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index).squeeze()
        loss = F.mse_loss(out[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            test_preds = out[test_mask]
            test_labels = targets[test_mask]
            mse = F.mse_loss(test_preds, test_labels).item()
            print(f"[GraphSAGE-NodeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model