import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader, ConcatDataset
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.metrics import accuracy_score, classification_report
import sys
from tqdm import tqdm
import pandas as pd

import sys
import time

import argparse

from models.classifier_models import *
from models.gnn_models import *
from models.models import *

import utils
from config import Config


def main():
    # get arguments
    #args = sys.argv
    #classifier_type = args[-1]
    #if classifier_type not in ["edge", "kite"]:
    #    print("Not valid model type")
    #    return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_folder = "datasets/full_graph_data_edge/test"
    bcss_folder = "/home/jovyan/gnn_bcss_img/BCSS"

    test_graphs = utils.generate_random_graphs_multithreaded(bcss_folder, test_folder, device, max_workers=5)
    
    gnn_input_dim = test_graphs[0].x.shape[1]
    gnn_layers = [1024]*4
    mlp_hidden_dim = 1024
    mlp_out_dim = test_graphs[0].y.shape[1]
    channel_dim = test_graphs[0].edge_attr.shape[1]
    
    dropout = 0.5
    
    gnn = EAGNN(
        gnn_input_dim, 
        channel_dim,
        gnn_layers,
        dropout
    )
    
    gnn = GraphSAGE(
        gnn_input_dim, 
        gnn_layers,
        dropout
    )
    
    classifier = EdgeClassifier(
        gnn_layers[-1]*2, 
        mlp_hidden_dim, 
        mlp_out_dim, 
        dropout
    )
    
    combined_model = CombinedEdgeModel(gnn, classifier, "edge")
    combined_model.load_state_dict(torch.load(f'saved_models/random_graph3.pth'))
    combined_model = combined_model.to(device)
    combined_model.eval()

    print("Combined model: ", combined_model)
    print("Number of parameters: ", sum(p.numel() for p in combined_model.parameters() if p.requires_grad))
    print()


    all_true_values = []
    all_predicted_values = []

    for i, graph in tqdm(enumerate(test_graphs), total=len(test_graphs)):
        node_features = graph.x
        edge_index = graph.edge_index
        kites = graph.kites
        dists = graph.edge_attr
        y = graph.y

        with torch.no_grad():
            out = combined_model(node_features, edge_index, kites, dists)
        
        true_classes = np.argmax(y.cpu().numpy(), axis=1)
        pred_classes = np.argmax(out.cpu().numpy(), axis=1)
        
        all_true_values.extend(true_classes.tolist())
        all_predicted_values.extend(pred_classes.tolist())

    all_true_values = np.array(all_true_values)
    all_predicted_values = np.array(all_predicted_values)
    
    class_report = classification_report(all_true_values, all_predicted_values, target_names=["tumor", "stroma", "inflammatory", "necrosis", "other"], digits=4, zero_division=0)

    print(class_report)
            
            
            
if __name__ == '__main__':
    main()







    