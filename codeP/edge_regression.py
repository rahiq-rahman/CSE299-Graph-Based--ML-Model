import torch
import torch.nn.functional as F
from gcn import GCN
from gat import GAT
from graphsage import GraphSAGE


#GCN
def edge_regression_gcn(x, edge_index, edge_targets, train_mask, test_mask, epochs=200):
    model = GCN(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    regressor = torch.nn.Linear(16, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=0.01)

    for epoch in range(epochs):
        model.train()
        regressor.train()
        optimizer.zero_grad()

        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        pred = regressor(edge_emb).squeeze()

        loss = F.mse_loss(pred[train_mask], edge_targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            regressor.eval()
            with torch.no_grad():
                test_pred = pred[test_mask]
                test_true = edge_targets[test_mask]
                mse = F.mse_loss(test_pred, test_true).item()
                print(f"[GCN-EdgeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model, regressor


#GAT
def edge_regression_gat(x, edge_index, edge_targets, train_mask, test_mask, epochs=200):
    model = GAT(input_dim=x.size(1), hidden_dim=8, output_dim=16, heads=2)
    regressor = torch.nn.Linear(16, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=0.005)

    for epoch in range(epochs):
        model.train()
        regressor.train()
        optimizer.zero_grad()

        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        pred = regressor(edge_emb).squeeze()

        loss = F.mse_loss(pred[train_mask], edge_targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            regressor.eval()
            with torch.no_grad():
                test_pred = pred[test_mask]
                test_true = edge_targets[test_mask]
                mse = F.mse_loss(test_pred, test_true).item()
                print(f"[GAT-EdgeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model, regressor


#GraphSAGE
def edge_regression_graphsage(x, edge_index, edge_targets, train_mask, test_mask, epochs=200):
    model = GraphSAGE(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    regressor = torch.nn.Linear(16, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=0.005)

    for epoch in range(epochs):
        model.train()
        regressor.train()
        optimizer.zero_grad()

        node_embeddings = model(x, edge_index)
        edge_emb = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
        pred = regressor(edge_emb).squeeze()

        loss = F.mse_loss(pred[train_mask], edge_targets[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            regressor.eval()
            with torch.no_grad():
                test_pred = pred[test_mask]
                test_true = edge_targets[test_mask]
                mse = F.mse_loss(test_pred, test_true).item()
                print(f"[GraphSAGE-EdgeReg] Epoch {epoch} - Loss: {loss.item():.4f} - Test MSE: {mse:.4f}")

    return model, regressor