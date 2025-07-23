import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


class GraphRNN(nn.Module):
    def __init__(self, hidden_size=64, max_nodes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes

        self.node_rnn = nn.GRUCell(1, hidden_size)
        self.edge_rnn = nn.GRUCell(1, hidden_size)
        self.output_edge = nn.Linear(hidden_size, 1)

    def forward(self, graph):
        num_nodes = graph.num_nodes
        device = next(self.parameters()).device

        loss = 0.0
        h_node = torch.zeros(1, self.hidden_size).to(device)

        for i in range(num_nodes):
            inp = torch.tensor([[1.0]]).to(device)
            h_node = self.node_rnn(inp, h_node)

            h_edge = torch.zeros(1, self.hidden_size).to(device)
            for j in range(i):
                edge_inp = torch.tensor([[1.0]]).to(device)
                h_edge = self.edge_rnn(edge_inp, h_edge)
                logit = self.output_edge(h_edge)
                label = graph.edge_index.new_tensor(
                    [[j], [i]]).T in graph.edge_index.T
                label = torch.tensor([[float(label)]]).to(device)
                loss += F.binary_cross_entropy_with_logits(logit, label)

        return loss

    def generate(self, num_graphs=1):
        generated = []
        device = next(self.parameters()).device

        for _ in range(num_graphs):
            adj = torch.zeros(self.max_nodes, self.max_nodes).to(device)
            h_node = torch.zeros(1, self.hidden_size).to(device)

            for i in range(self.max_nodes):
                inp = torch.tensor([[1.0]]).to(device)
                h_node = self.node_rnn(inp, h_node)

                h_edge = torch.zeros(1, self.hidden_size).to(device)
                for j in range(i):
                    edge_inp = torch.tensor([[1.0]]).to(device)
                    h_edge = self.edge_rnn(edge_inp, h_edge)
                    logit = self.output_edge(h_edge)
                    prob = torch.sigmoid(logit)
                    if prob.item() > 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1

            edge_index, _ = dense_to_sparse(adj)
            x = torch.ones((self.max_nodes, 1))
            generated.append(Data(x=x, edge_index=edge_index.cpu()))

        return generated
