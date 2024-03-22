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

        dims = [input_dim] + layers

        for i in range(1, len(dims)):
            self.layers.append(SAGEConv(dims[i-1], dims[i]))
            
        self.dropout = dropout
        self.out_dim = dims[-1]  # Output dimension

    def forward(self, node_features, edge_index, edge_attr=None):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        edge_index: torch.Tensor of shape [2, E]
            E is the number of edges
            
        edge_attr: Not used
        
        Returns:
            torch.Tensor of shape [N, out_dim]
        """

        #edge_index, _ = add_self_loops(edge_index, num_nodes=node_features.size(0))
        
        x = node_features
        
        # SAGEConv --> ReLU --> Dropout
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, layers, dropout=0.5):
        super(GAT, self).__init__()

        self.layers = nn.ModuleList()

        dims = [input_dim] + layers

        for i in range(1, len(dims)):
            self.layers.append(GATv2Conv(dims[i-1], dims[i]))
            
        self.dropout = dropout
        self.out_dim = dims[-1]  # Output dimension

    def forward(self, node_features, edge_index, edge_attr=None):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        edge_index: torch.Tensor of shape [2, E]
            E is the number of edges
            
        edge_attr: torch.Tensor of shape [E, P]
            P is the number of edge features
        
        Returns:
            torch.Tensor of shape [N, out_dim]
        """

        #edge_index, _ = add_self_loops(edge_index, num_nodes=node_features.size(0))
        
        x = node_features
        
        # SAGEConv --> ReLU --> Dropout
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class EAGNN(nn.Module):
    def __init__(self, input_dim, channel_dim, layers, dropout=0.5):
        super(EAGNN, self).__init__()

        self.layers = nn.ModuleList()
        
        self.layers.append(EAGNNLayer(input_dim, layers[0], channel_dim))

        for i in range(1, len(layers)):
            self.layers.append(EAGNNLayer(layers[i-1]*channel_dim, layers[i], channel_dim))
        
        self.dropout = dropout
        self.out_dim = layers[-1]  # Output dimension

    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        edge_index: torch.Tensor of shape [2, E]
            E is the number of edges
            
        edge_attr: torch.Tensor of shape [E, P]
            P is the number of edge features
        
        Returns:
            torch.Tensor of shape [N, out_dim]
        """

        x = node_features
        # EAGNN --> ReLU --> Dropout
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x  