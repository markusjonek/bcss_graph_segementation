import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv
from torch_geometric.utils import to_undirected, add_self_loops

from models.classifier_models import *
from models.gnn_models import *



class CombinedEdgeModel(nn.Module):
    def __init__(self, gnn, classifier, graphlet_type):
        """
        gnn: GNN model

        classifier: Classifier model

        graphlet_type: str  
            Type of graphlet to classify. Either 'edge' or 'kite'
        """

        super(CombinedEdgeModel, self).__init__()   
        
        self.gnn = gnn
        self.classifier = classifier
    
        if graphlet_type not in ["edge", "kite"]:
            raise ValueError(f"Invalid classifier type: {self.graphlet_type}. Expected 'edge' or 'kite'.")
            
        self.graphlet_type = graphlet_type

        
    def forward(self, node_features, edge_list, kites, edge_attr):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features

        edge_list: torch.Tensor of shape [2, E]
            E is the number of edges

        kites: torch.Tensor of shape [E, 4]
            E is the number of edges

        edge_attr: torch.Tensor of shape [E, P]
            E is the number of edges
            P is the number of edge features
        """

        # Pass graphs through GNN to get node embeddings
        gnn_node_features = self.gnn(node_features, edge_list, edge_attr)

        if self.graphlet_type == "edge":
            edge_embs1 = gnn_node_features[edge_list.t()]
            edge_embs2 = gnn_node_features[edge_list.flip(0).t()]
            out1 = self.classifier(edge_embs1)
            out2 = self.classifier(edge_embs2) 
            return out1 + out2
        elif self.graphlet_type == "kite":
            v1 = torch.cat((edge_list.t(), kites), dim=1)
            v2 = torch.cat((edge_list.flip(0).t(), kites), dim=1)
            v3 = torch.cat((edge_list.t(), kites.flip(1)), dim=1)
            v4 = torch.cat((edge_list.flip(0).t(), kites.flip(1)), dim=1)
            
            out1 = self.classifier(gnn_node_features[v1])
            out2 = self.classifier(gnn_node_features[v2])
            out3 = self.classifier(gnn_node_features[v3])
            out4 = self.classifier(gnn_node_features[v4])
            
            return out1 + out2 + out3 + out4
        else:
            raise ValueError(f"Invalid classifier type: {self.graphlet_type}.")
            
