import torch
import torch.nn as nn
from siamese_gcn import SiameseGCN

def graph_matching_siamese_gcn(train_pairs, test_pairs, in_channels=None, epochs=20, lr=0.001):
    in_channels = in_channels or train_pairs[0][0].x.size(1)

    model = SiameseGCN(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for g1, g2, label in train_pairs:
            g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
            g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)
            optimizer.zero_grad()
            out = model(g1, g2).squeeze()
            loss = loss_fn(out, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    return model