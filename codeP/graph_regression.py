import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


#GCN
class GCNReg(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.head = torch.nn.Linear(64, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

def graph_regression_gcn(train_loader, test_loader, input_dim, epochs=100):
    model = GCNReg(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[GCN-GraphReg] Epoch {epoch} - Train MSE: {total_loss/total:.6f}")
            
    return model


#GAT
class GATReg(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, 64, heads=4, concat=True)
        self.conv2 = GATConv(64*4, 64, heads=4, concat=False)
        self.head = torch.nn.Linear(64,1)
    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

def graph_regression_gat(train_loader, test_loader, input_dim, epochs=100):
    model = GATReg(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss=0
        total=0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"[GAT-GraphReg] Epoch {epoch} - Train MSE: {total_loss/total:.6f}")
            
    return model


#GraphSAGE
class SAGEReg(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.head = torch.nn.Linear(64, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

def graph_regression_graphsage(train_loader, test_loader, input_dim, epochs=100):   
    model = SAGEReg(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss=0
        total=0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"[SAGE-GraphReg] Epoch {epoch} - Train MSE: {total_loss/total:.6f}")
            
    return model