import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv
from torch_geometric.utils import to_undirected, add_self_loops

from models.classifier_models import *
from models.gnn_models import *



class CombinedEdgeModel(nn.Module):
    def __init__(self, gnn, classifier, graphlet_type, dropout=0.5):
        super(CombinedEdgeModel, self).__init__()
        
        self.gnn = gnn
        self.classifier = classifier
    
        if graphlet_type not in ["edge", "kite"]:
            raise ValueError(f"Invalid classifier type: {self.graphlet_type}. Expected 'edge' or 'kite'.")
            
        self.graphlet_type = graphlet_type

        
    def forward(self, node_features, edge_list, kites, edge_attr):
        # Pass graphs through GNN to get node embeddings
        gnn_node_features = self.gnn(node_features, edge_list, edge_attr)

        if self.graphlet_type == "edge":
            edge_embs1 = gnn_node_features[edge_list.t()]
            edge_embs2 = gnn_node_features[edge_list.flip(0).t()]
            out1 = self.classifier(edge_embs1)
            out2 = self.classifier(edge_embs1) 
            return out1 + out2
        elif self.graphlet_type == "kite":
            edge_embs = gnn_node_features[kites]
            out = self.classifier(edge_embs)
            return out
        else:
            raise ValueError(f"Invalid classifier type: {self.graphlet_type}.")
            
