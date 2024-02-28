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

random.seed(0)


def train(cfg):
    # get arguments
    parser = argparse.ArgumentParser(description="Train GNN")
    parser.add_argument("--gnn", choices=["graphsage", "eagnn"], required=True)
    parser.add_argument("--graphlet", choices=["edge", "kite"], required=True)
    parser.add_argument("--classifier", choices=["mlp", "gcnn"], required=True)
    parser.add_argument("--log", type=bool, required=False)
    
    args = parser.parse_args()

    print(f"\nGNN Type: {args.gnn} \nGraphlet Type: {args.graphlet} \nClassifier Type: {args.classifier}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {str(device).upper()}\n")

    train_folder = os.path.join(cfg.dataset_dir, "train")
    val_folder = os.path.join(cfg.dataset_dir, "val")

    train_graphs = utils.load_graphs(train_folder, device)
    val_graphs = utils.load_graphs(val_folder, device)
    
    if args.gnn == "eagnn":
        channel_dim = train_graphs[0].edge_attr.shape[1]
    elif args.gnn == "graphsage":
        channel_dim = 1
        
    if args.graphlet == "edge":
        graphlet_size = 2
    elif args.graphlet == "kite":
        graphlet_size = 4
    
    gnn_input_dim = train_graphs[0].x.shape[1]
    out_dim = train_graphs[0].y.shape[1]
    
    dropout = 0.5

    # ------------ Get GNN ------------
    if args.gnn == "graphsage":
        gnn = GraphSAGE(
            gnn_input_dim, 
            cfg.gnn_layers, 
            cfg.dropout
        )
    elif args.gnn == "eagnn":
        gnn = EAGNN(
            gnn_input_dim, 
            channel_dim, 
            cfg.gnn_layers, 
            cfg.dropout
        )
    
    # ------------ Get classifier head ------------
    if args.classifier == "mlp":
        classifier = EdgeClassifier(
            cfg.gnn_layers[-1]*graphlet_size*channel_dim, 
            cfg.classifier_hidden_dim, 
            out_dim, 
            cfg.dropout
        )
    elif args.classifier == "gcnn":
        classifier = gCNN(
            graphlet_size, 
            cfg.gnn_layers[-1]*channel_dim, 
            cfg.classifier_hidden_dim, 
            out_dim, 
            cfg.dropout
        )
    
    model_name = f"{args.gnn}_{args.graphlet}_{args.classifier}"
    
    # ------------ Build full model ------------
    combined_model = CombinedEdgeModel(gnn, classifier, args.graphlet)
    try:
        combined_model.load_state_dict(torch.load(f'{cfg.save_dir}/{model_name}_last.pth'))
    except:
        print("Could not load any previous weights.\n")
    combined_model = combined_model.to(device)
    combined_model.train()

    print("Combined model: ", combined_model)
    num_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}\n")
    
    # ------------ Get num samples of each class ------------
    class_weights = torch.zeros(out_dim, device=device)
    for g in train_graphs:
        unique, counts = torch.unique(torch.argmax(g.y, dim=1), return_counts=True)
        for u, c in zip(unique, counts):
            class_weights[u] += c
    class_weights = 1 / class_weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    if out_dim != len(cfg.class_names):
        print(f"Out dimension ({out_dim}) does not match class names ({len(cfg.class_names)})!")
        return
    
    print(f"Class weights:")
    for c, w in zip(cfg.class_names, class_weights):
        print(f"  {c}: {w:.3f}")
    print("")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(combined_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = GradScaler()
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    lowest_val_loss = np.inf

    if args.log:
        loss_model_path = os.path.join(cfg.loss_log_dir, model_name + '.csv')
        os.makedirs(cfg.loss_log_dir, exist_ok=True)
        log_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    for epoch in range(cfg.num_epochs):
        random.shuffle(train_graphs)
        f_times = []
        b_times = []
        combined_model.eval()
        for b_idx, graph in enumerate(train_graphs):
            node_features = graph.x
            edge_list = graph.edge_index
            kites = graph.kites
            edge_labels = graph.y
            edge_attr = graph.edge_attr
                                    
            optimizer.zero_grad()
            
            with autocast():
                s1 = time.time()
                out = combined_model(node_features, edge_list, kites, edge_attr)
                s2 = time.time()
                loss = criterion(out, edge_labels)
                
                #print(loss.item())
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            s3 = time.time()
            
            f_times.append(s2-s1)
            b_times.append(s3-s2)
            optimizer.zero_grad()
            
        if epoch % 10 == 0:
            tot_val_loss = 0
            tot_val_edges = 0
            combined_model.eval()
            for graph in val_graphs:
                with torch.no_grad() and autocast():
                    node_features = graph.x
                    edge_list = graph.edge_index
                    kites = graph.kites
                    edge_labels = graph.y
                    edge_attr = graph.edge_attr
                    
                    out = combined_model(node_features, edge_list, kites, edge_attr)
                    loss = criterion(out, edge_labels)
                    
                    tot_val_loss += loss.item() #* len(edge_labels)
                    tot_val_edges += 1 #len(edge_labels)
            
            tot_train_loss = 0
            tot_train_edges = 0
            for graph in train_graphs:
                with torch.no_grad() and autocast():
                    node_features = graph.x
                    edge_list = graph.edge_index
                    kites = graph.kites
                    edge_labels = graph.y
                    edge_attr = graph.edge_attr
                    
                    out = combined_model(node_features, edge_list, kites, edge_attr)
                    loss = criterion(out, edge_labels)
                    
                    tot_train_loss += loss.item() #* len(edge_labels)
                    tot_train_edges += 1# len(edge_labels)
            
            avg_val_loss = tot_val_loss/tot_val_edges
            avg_train_loss = tot_train_loss/tot_train_edges
            
            if args.log:
                new_row = {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
                new_row_df = pd.DataFrame([new_row])
                log_df = pd.concat([log_df, new_row_df], ignore_index=True)
                log_df.to_csv(loss_model_path, index=False)

            print(f"Epoch {epoch}/{cfg.num_epochs}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Avg f-pass: {np.mean(f_times)*1000:.2f} ms, Avg b-pass: {np.mean(b_times)*1000:.2f} ms")

            torch.save(combined_model.state_dict(), f"{cfg.save_dir}/{model_name}_last.pth")
            if avg_val_loss < lowest_val_loss:
                torch.save(combined_model.state_dict(), f"{cfg.save_dir}/{model_name}_best.pth")
                lowest_val_loss = avg_val_loss
                

if __name__ == '__main__':
    cfg = Config()
    train(cfg)







    