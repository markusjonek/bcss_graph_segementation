import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader, ConcatDataset
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.metrics import confusion_matrix, classification_report
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

def evaluate(cfg):
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

    test_folder = os.path.join(cfg.dataset_dir, "train")
    test_graphs = utils.load_graphs(test_folder, device)
    
    if args.gnn == "eagnn":
        channel_dim = test_graphs[0].edge_attr.shape[1]
    elif args.gnn == "graphsage":
        channel_dim = 1
        
    if args.graphlet == "edge":
        graphlet_size = 2
    elif args.graphlet == "kite":
        graphlet_size = 4
    
    gnn_input_dim = test_graphs[0].x.shape[1]
    out_dim = test_graphs[0].y.shape[1]
    
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
        print("\nCould not load previous weights. Using untrained model.")
    combined_model = combined_model.to(device)
    combined_model.eval()
    
    result_model_dir = os.path.join(cfg.result_dir, model_name)
    os.makedirs(result_model_dir, exist_ok=True)

    print("Combined model: ", combined_model)
    num_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}\n")
    
    # ------------ Get num samples of each class ------------
    class_weights = torch.zeros(out_dim, device=device)
    for g in test_graphs:
        unique, counts = torch.unique(torch.argmax(g.y, dim=1), return_counts=True)
        for u, c in zip(unique, counts):
            class_weights[u] += c
    
    if out_dim != len(cfg.class_names):
        print(f"Out dimension ({out_dim}) does not match class names ({len(cfg.class_names)})!")
        return
    
    print(f"Class samples:")
    for c, w in zip(cfg.class_names, class_weights):
        print(f"  {c}: {w:.0f}")
    print("")

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
        
        edges_result_path = os.path.join(result_model_dir, graph.tile_name + '_edges.npy')
        nodes_result_path = os.path.join(result_model_dir, graph.tile_name + '_nodes.npy')
        
        yolo_classes = np.argmax(node_features.cpu().numpy(), axis=1)
        true_classes = np.argmax(y.cpu().numpy(), axis=1)
        pred_classes = np.argmax(out.cpu().numpy(), axis=1)
        
        edge_data = np.stack((true_classes, pred_classes), axis=1)
        node_data = np.concatenate((yolo_classes.reshape(-1, 1), graph.coords.cpu().numpy()), axis=1)
        np.save(edges_result_path, edge_data)
        np.save(nodes_result_path, node_data)

        all_true_values.extend(true_classes.tolist())
        all_predicted_values.extend(pred_classes.tolist())

    all_true_values = np.array(all_true_values)
    all_predicted_values = np.array(all_predicted_values)
    
    class_report = classification_report(all_true_values, all_predicted_values, target_names=cfg.class_names, digits=4, zero_division=0)

    print(str(class_report))
    
    cm = confusion_matrix(all_true_values, all_predicted_values, labels=list(range(len(cfg.class_names))))
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, name in enumerate(cfg.class_names):
        print(f"{name}: {class_accuracies[i]*100:.2f}%")
    
    if args.log:
        with open('experiments.txt', 'a') as f:
            f.write('-'*53 + '\n')
            f.write(model_name.replace('_', ' ').upper() + ': \n\n')
            f.write(str(class_report) + '\n')
            f.write('Accuracy:\n')
            for i, name in enumerate(cfg.class_names):
                f.write(f"{name}: {class_accuracies[i]*100:.2f}%\n")
            f.write('\nConfig: \n')
            f.write(str(cfg) + '\n')
            f.write(f'Batch size: full graphs \n')
            f.write(f'Loss function: nn.CrossEntropyLoss() without Softmax \n')
            f.write(f'Num params: {num_params} \n')
            f.write(f'Node input labels: YOLOv8s NuCLS Super-classes one-hot \n')
            f.write(f'Edge labels: mid point mask value \n')
            f.write(f'Training with mixed precision \n')

            f.write(f'\n{combined_model} \n')

            f.write('-'*53 + '\n')

                

if __name__ == '__main__':
    cfg = Config()
    evaluate(cfg)

