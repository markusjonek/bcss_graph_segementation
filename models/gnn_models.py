import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv
from torch_geometric.utils import to_undirected, add_self_loops

from models.layers import EAGNNLayer


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, layers, dropout=0.5):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [input_dim] + layers

        for i in range(1, len(dims)):
            self.layers.append(SAGEConv(dims[i-1], dims[i]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i]))
            
        self.dropout = dropout
        self.out_dim = dims[-1]  # Output dimension

    def forward(self, node_features, edge_index, edge_attr=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=node_features.size(0))
        
        x = node_features
        
        # SAGEConv --> BatchNorm --> ReLU --> Dropout
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index)
            #x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphAttention(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, device='cpu'):
        super(GraphAttention, self).__init__()

        self.gat_convs = nn.ModuleList(
            [GATConv(input_dim, hidden_dims[0])]
        )

        for i in range(1, len(hidden_dims) - 1):
            self.gat_convs.append(GATConv(hidden_dims[i-1], hidden_dims[i]))

        self.gat_convs.append(GATConv(hidden_dims[-1], output_dim))

        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, node_features, edge_index):
        node_features, edge_index = node_features.to(self.device), edge_index.to(self.device)

        x = node_features
        for layer in self.gat_convs[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x) 
        
        x = self.gat_convs[-1](x, edge_index)
        
        return x



class EAGNN(nn.Module):
    def __init__(self, input_dim, channel_dim, layers, dropout=0.5):
        super(EAGNN, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()        
        
        self.layers.append(EAGNNLayer(input_dim, layers[0], channel_dim))
        self.batch_norms.append(nn.BatchNorm1d(layers[0]*channel_dim))

        for i in range(1, len(layers)):
            self.layers.append(EAGNNLayer(layers[i-1]*channel_dim, layers[i], channel_dim))
            #self.batch_norms.append(nn.BatchNorm1d(layers[i]*channel_dim))
        
        self.dropout = dropout
        self.out_dim = layers[-1]  # Output dimension

    def forward(self, node_features, edge_index, edge_attr):
        x = node_features
        # EAGNN --> BatchNorm --> ReLU --> Dropout
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index, edge_attr)
            #x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x  