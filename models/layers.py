import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EAGNNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, channel_dim):
        super(EAGNNLayer, self).__init__(aggr=None)  # Use sum aggregation.

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim
        
        self.weight0 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        self.bias = nn.Parameter(torch.FloatTensor(output_dim * channel_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: node features [N, node_features]
        # edge_index: edge list [2, E]
        # edge_attr: edge features [E, p], where p is the number of features
        
        support0 = torch.matmul(x, self.weight0)
        support1 = torch.matmul(x, self.weight1)

        out_tensor = torch.zeros((support0.size(0), support0.size(1), edge_attr.shape[1]), dtype=torch.float32, device=self.weight1.device)
        
        for channel in range(edge_attr.shape[1]):
            size = (support1.size(0), support1.size(0))
            edge_weight = edge_attr[:, channel]
            out_tensor[:, :, channel] = self.propagate(edge_index, size=size, x=support1, edge_attr=edge_weight) + support0
        
        # concatenate the output tensor along the channel dimension
        out_shape = (-1, support0.size(1) * edge_attr.shape[1])
        out_tensor = out_tensor.transpose(2, 1).reshape(out_shape)
        
        return out_tensor + self.bias

    def message(self, x_j, edge_attr):
        # x_j: Input features of source nodes [E, in_channels]
        # edge_attr: Edge weights [E, 1] for each edge in edge_index

        # Multiply each message by its corresponding edge weight
        weighted_message = x_j * edge_attr.unsqueeze(1)
        return weighted_message

    def update(self, aggr_out):
        # aggr_out: Output of the aggregation step [N, out_channels]
        return aggr_out




    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, channel_dim={self.channel_dim})"