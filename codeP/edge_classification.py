import torch
import torch.nn.functional as F
from gcn import GCN
from gat import GAT
from graphsage import GraphSAGE


#GCN
def edge_classification_gcn(x, edge_index, edge_labels, train_mask, test_mask, epochs=200):
    model = GCN(input_dim=x.size(1), hidden_dim=16, output_dim=16)
    classifier = torch.nn.Linear(16, 2)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.01)

    for epoch in range(epochs):
        model.train()
        classifier.train()
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
            classifier.eval()
            with torch.no_grad():
                preds = edge_logits[test_mask].argmax(dim=1)
                acc = (preds == edge_labels[test_mask]).float().mean().item()
                print(f"[GCN-EdgeCls] Epoch {epoch} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

    return model, classifier


#GAT
def edge_classification_gat(x, edge_index, edge_labels, train_mask, test_mask, epochs=200):
    model = GAT(input_dim=x.size(1), hidden_dim=8, output_dim=16, heads=2)
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
                print(f"[GAT-EdgeCls] Epoch {epoch} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

    return model, classifier


#GraphSAGE
def edge_classification_graphsage(x, edge_index, edge_labels, train_mask, test_mask, epochs=200):
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