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

    train_folder = "datasets/full_graph_data_edge/train"
    val_folder = "datasets/full_graph_data_edge/val"
    bcss_folder = "/home/jovyan/gnn_bcss_img/BCSS"

    num_graphs = 40
    epochs_per_graph = 200
    train_graphs = utils.generate_random_graphs_multithreaded(bcss_folder, train_folder, device, max_workers=5)
    val_graphs = utils.generate_random_graphs_multithreaded(bcss_folder, val_folder, device, max_workers=5)
    
    gnn_input_dim = train_graphs[0].x.shape[1]
    gnn_layers = [1024]*4
    mlp_hidden_dim = 1024
    mlp_out_dim = train_graphs[0].y.shape[1]
    channel_dim = train_graphs[0].edge_attr.shape[1]
    
    dropout = 0.3
    
    #gnn = EAGNN(
    #    gnn_input_dim, 
    #    channel_dim,
    #    gnn_layers,
    #    dropout
    #)
    
    gnn = GraphSAGE(
        gnn_input_dim, 
        gnn_layers,
        dropout
    )
    
    classifier = EdgeClassifier(
        gnn_layers[-1]*2*channel_dim, 
        mlp_hidden_dim, 
        mlp_out_dim, 
        dropout
    )
    
    combined_model = CombinedEdgeModel(gnn, classifier, "edge")
    #combined_model.load_state_dict(torch.load(f'saved_models/random_graph.pth'))
    combined_model = combined_model.to(device)
    combined_model.train()

    print("Combined model: ", combined_model)
    print("Number of parameters: ", sum(p.numel() for p in combined_model.parameters() if p.requires_grad))
    print()
    
    class_weights = torch.zeros(mlp_out_dim, device=device)
    for g in train_graphs:
        unique, counts = torch.unique(torch.argmax(g.y, dim=1), return_counts=True)
        for u, c in zip(unique, counts):
            class_weights[u] += c
    class_weights = 1 / class_weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    learning_rate = 0.0001
    weight_decay = 1e-5
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)
    
    scaler = GradScaler()
    
    num_epochs = 5000
    for epoch in range(num_epochs):
        f_times = []
        b_times = []
        combined_model.train()
        for b_idx, graph in enumerate(train_graphs):
            optimizer.zero_grad()
            
            with autocast():
                s1 = time.time()
                out = combined_model(graph.x, graph.edge_index, graph.kites, graph.edge_attr)
                s2 = time.time()
                loss = criterion(out, graph.y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            s3 = time.time()
            
            f_times.append(s2-s1)
            b_times.append(s3-s2)
            
        scheduler.step()

        if epoch % 10 == 0:
            combined_model.eval()
            tot_val_loss = 0
            tot_val_edges = 0
            for graph in val_graphs:
                with torch.no_grad():
                    out = combined_model(graph.x, graph.edge_index, graph.kites, graph.edge_attr)
                    loss = criterion(out, graph.y)
                    
                    tot_val_loss += loss.item() * len(graph.y)
                    tot_val_edges += len(graph.y)
            
            tot_train_loss = 0
            tot_train_edges = 0
            for graph in train_graphs:
                with torch.no_grad():
                    out = combined_model(graph.x, graph.edge_index, graph.kites, graph.edge_attr)
                    loss = criterion(out, graph.y)
                    
                    tot_train_loss += loss.item() * len(graph.y)
                    tot_train_edges += len(graph.y)
                    
            print(f"Epoch {epoch}/{num_epochs}, Train loss: {tot_train_loss/tot_train_edges:.4f}, Val loss: {tot_val_loss/tot_val_edges:.4f}, Avg f-pass: {np.mean(f_times)*1000:.2f} ms, Avg b-pass: {np.mean(b_times)*1000:.2f} ms")

            torch.save(combined_model.state_dict(), f"saved_models/random_graph3.pth")
        
        if epoch % epochs_per_graph == 0 and epoch > 0:
            train_graphs = utils.generate_random_graphs_multithreaded(bcss_folder, train_folder, device, max_workers=5)
            
            
            
if __name__ == '__main__':
    main()


