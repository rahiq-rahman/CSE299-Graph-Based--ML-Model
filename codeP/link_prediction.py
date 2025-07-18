import torch
import torch.nn.functional as F
from gcn import GCN, create_graph_gcn
from gat import GAT, create_graph_gat
from graphsage import GraphSAGE, create_graph_graphsage

def recommend(model, x, edge_index, user_id, num_users, num_items, top_k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index)

        user_emb = F.normalize(embeddings[user_id], dim=0)
        item_ids = list(range(num_users, num_users + num_items))

        scores = []
        for item_id in item_ids:
            item_emb = F.normalize(embeddings[item_id], dim=0)
            score = (user_emb * item_emb).sum().item()
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# GCN
def link_prediction_gcn(num_users, num_items, interactions, epochs=100):
    x, edge_index, pos_edges, neg_edges = create_graph_gcn(num_users, num_items, interactions)

    model = GCN(input_dim=x.shape[1], hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x, edge_index)

        pos_scores = (embeddings[pos_edges[:,0]] * embeddings[pos_edges[:,1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edges[:,0]] * embeddings[neg_edges[:,1]]).sum(dim=1)

        epsilon = 1e-7
        pos_probs = torch.sigmoid(pos_scores).clamp(min=epsilon, max=1 - epsilon)
        neg_probs = torch.sigmoid(neg_scores).clamp(min=epsilon, max=1 - epsilon)

        loss = - (torch.log(pos_probs).mean() + torch.log(1 - neg_probs).mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[GCN] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, x, edge_index, num_users, num_items


#GAT
def link_prediction_gat(num_users, num_items, interactions, epochs=100):
    x, edge_index, pos_edges, neg_edges = create_graph_gat(num_users, num_items, interactions)

    model = GAT(input_dim=x.shape[1], hidden_dim=16, output_dim=16)
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
            print(f"[GAT] Epoch {epoch} - Loss: {loss.item():.4f}")

    return model, x, edge_index, num_users, num_items


#GraphSAGE
def link_prediction_graphsage(num_users, num_items, interactions, epochs=100):
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