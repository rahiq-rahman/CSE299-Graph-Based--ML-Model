import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


#GCN
class GCNGraphClassifier(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCNGraphClassifier, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch)

            return self.classifier(x)

def graph_classification_gcn(train_loader, test_loader, input_dim, num_classes, epochs=100):
    model = GCNGraphClassifier(input_dim, hidden_dim=64, output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to('cpu')
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[GCN-GraphCls] Epoch {epoch} - Loss: {avg_loss:.4f} - Train Acc: {acc:.4f}")

    return model


#GAT
class GATGraphClassifier(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GATGraphClassifier, self).__init__()
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False)
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = global_mean_pool(x, batch)
            return self.classifier(x)

def graph_classification_gat(train_loader, test_loader, input_dim, num_classes, epochs=100):
    model = GATGraphClassifier(input_dim, hidden_dim=64, output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to('cpu')
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[GAT-GraphCls] Epoch {epoch} - Loss: {avg_loss:.4f} - Train Acc: {acc:.4f}")

    return model


#GraphSAGE
class SAGEGraphClassifier(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SAGEGraphClassifier, self).__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch)
            return self.classifier(x)

def graph_classification_graphsage(train_loader, test_loader, input_dim, num_classes, epochs=100):
    model = SAGEGraphClassifier(input_dim, hidden_dim=64, output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to('cpu') 
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[SAGE-GraphCls] Epoch {epoch} - Loss: {avg_loss:.4f} - Train Acc: {acc:.4f}")

    return model