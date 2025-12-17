import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import TransformerConv
import pickle
from data import *


class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """

    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads),
                                         heads=num_heads, dropout=dropout, edge_dim=num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V)

        return h_V, h_E


class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """

    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid()
        )

    def forward(self, h_V):
        # 计算全局特征
        c_V = torch.mean(h_V, dim=0, keepdim=True)  # 计算平均值并保持维度
        h_V = h_V * self.V_MLP_g(c_V.squeeze(0))  # 将全局特征通过MLP并广播到每个节点
        return h_V


class Graph_encoder(nn.Module):
    """
    construct the graph encoder module
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
            GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))

    def forward(self, h_V, edge_index, h_E):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))

        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E)

        return h_V


class GraphL(nn.Module):
    """
    construct the GraphL model
    """

    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, augment_eps, task):
        super(GraphL, self).__init__()
        self.augment_eps = augment_eps
        # define the encoder layer
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim,
                                           hidden_dim=hidden_dim, num_layers=num_layers,
                                           drop_rate=dropout)
        self.task = task
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, h_V,edge_index, h_E):
        # Data augmentation
        h_V = h_V.to_dense()
        if self.training and self.augment_eps > 0:
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)
        h_V = self.Graph_encoder(h_V, edge_index, h_E)  # [num_residue, hidden_dim]
        return h_V