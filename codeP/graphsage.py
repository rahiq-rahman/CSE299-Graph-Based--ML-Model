import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.cluster import KMeans

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_graph_graphsage(num_users, num_items, interactions):
    num_nodes = num_users + num_items
    x = torch.eye(num_nodes)

    user_nodes = []
    item_nodes = []
    for u, i in interactions:
        user_nodes.append(u)
        item_nodes.append(i)

    edge_index = torch.tensor([user_nodes, item_nodes], dtype=torch.long)
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    pos_edges = edge_index[:, :len(interactions)].t()
    pos_set = set(interactions)
    neg_edges = []
    for u in range(num_users):
        for i in range(num_users, num_users + num_items):
            if (u, i) not in pos_set:
                neg_edges.append([u, i])
    neg_edges = torch.tensor(neg_edges, dtype=torch.long)

    return x, edge_index, pos_edges, neg_edges

def train_link_prediction(num_users, num_items, interactions, epochs=100):
    x, edge_index, pos_edges, neg_edges = create_graph_graphsage(num_users, num_items, interactions)

    model = GraphSAGE(input_dim=x.shape[1], hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x, edge_index)

        pos_scores = (embeddings[pos_edges[:, 0]] * embeddings[pos_edges[:, 1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edges[:, 0]] * embeddings[neg_edges[:, 1]]).sum(dim=1)

        loss = - (torch.log(torch.sigmoid(pos_scores)).mean() + torch.log(1 - torch.sigmoid(neg_scores)).mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[GraphSAGE] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, x, edge_index, num_users, num_items

def train_node_classification(x, edge_index, labels, train_mask, test_mask, epochs=200):
    model = GraphSAGE(input_dim=x.shape[1], hidden_dim=16, output_dim=labels.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            preds = out[test_mask].argmax(dim=1)
            acc = (preds == labels[test_mask]).sum().item() / test_mask.sum().item()
            print(f"[GraphSAGE-NodeCls] Epoch {epoch} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

    return model

def train_node_regression(x, edge_index, targets, train_mask, test_mask, epochs=200):
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


def train_edge_classification(x, edge_index, edge_labels, train_mask, test_mask, epochs=200):
    model = GraphSAGE(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    classifier = torch.nn.Linear(16, 2)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        node_embeddings = model(x, edge_index)
        src = edge_index[0]
        dst = edge_index[1]
        edge_emb = node_embeddings[src] * node_embeddings[dst]
        edge_logits = classifier(edge_emb)

        loss = F.cross_entropy(edge_logits[train_mask], edge_labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = edge_logits[test_mask].argmax(dim=1)
                acc = (preds == edge_labels[test_mask]).float().mean().item()
                print(f"[GraphSAGE-EdgeCls] Epoch {epoch} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

    return model, classifier


def train_edge_regression(x, edge_index, edge_targets, train_mask, test_mask, epochs=200):
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


def train_node_embedding(x, edge_index, epochs=100):
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

def train_node_clustering(x, edge_index, num_clusters=7):
    model, embeddings = train_node_embedding(x, edge_index)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())
    return model, embeddings, cluster_labels

