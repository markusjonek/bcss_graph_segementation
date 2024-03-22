import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import f1_score

import time
import argparse

from models.classifier_models import *
from models.gnn_models import *
from models.models import *

import utils
from config import Config

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def train(cfg):
    # get arguments
    parser = argparse.ArgumentParser(description="Train GNN")
    parser.add_argument("--gnn", choices=["graphsage", "eagnn"], required=True)
    parser.add_argument("--graphlet", choices=["edge", "kite"], required=True)
    parser.add_argument("--classifier", choices=["mlp", "gcnn"], required=True)
    
    args = parser.parse_args()

    print(f"\nGNN Type: {args.gnn} \nGraphlet Type: {args.graphlet} \nClassifier Type: {args.classifier}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {str(device).upper()}")

    train_folder = os.path.join(cfg.dataset_dir, "train")
    val_folder = os.path.join(cfg.dataset_dir, "val")
    test_folder = os.path.join(cfg.dataset_dir, "test")

    print("\nLoading train, val and test graphs")
    train_graphs = utils.load_bm_graphs_pt(train_folder, device)    
    val_graphs = utils.load_bm_graphs_pt(val_folder, device)
    test_graphs = utils.load_bm_graphs_pt(test_folder, device)

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
        print("\nCould not load any previous weights.")
    combined_model = combined_model.to(device)
    combined_model.train()

    print(f"\nConfig: \n{cfg}")
    #print("Combined model: ", combined_model)
    num_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}\n")

    os.makedirs(cfg.model_save_dir, exist_ok=True)
    
    # ------------ Get training stuff ------------
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(
        combined_model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    all_time_best_f1 = 0
    all_time_best_threshold = 0
    
    # ------------ Train ------------
    for epoch in range(cfg.num_epochs):
        random.shuffle(train_graphs)
        start_epoch_time = time.time()
        combined_model.train()
        tot_train_loss = 0
        for b_idx, graph in enumerate(train_graphs):
            node_features = graph.x
            edge_list = graph.edge_index
            kites = graph.kites
            edge_labels = graph.y
            edge_attr = graph.edge_attr
            
            optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()): 
                out = combined_model(node_features, edge_list, kites, edge_attr)
                loss = criterion(out, edge_labels)
                tot_train_loss += loss.item()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        
        # Validation every val_every epochs
        if epoch % cfg.val_every == 0 or epoch == cfg.num_epochs-1:
            tot_val_loss = 0
            combined_model.eval()
            all_true = []
            all_pred_probs = []
            for graph in val_graphs:
                node_features = graph.x
                edge_list = graph.edge_index
                kites = graph.kites
                edge_labels = graph.y
                edge_attr = graph.edge_attr

                with torch.no_grad():
                    out = combined_model(node_features, edge_list, kites, edge_attr)
                    loss = criterion(out, edge_labels)
                
                tot_val_loss += loss.item() 
                
                true_classes = np.round(edge_labels.cpu().numpy())
                pred_probs = F.sigmoid(out).cpu().numpy()
                
                all_true.extend(true_classes.tolist())
                all_pred_probs.extend(pred_probs.tolist())
        
            all_true = np.array(all_true)
            all_pred_probs = np.array(all_pred_probs)
            
            avg_val_loss = tot_val_loss/len(val_graphs)
            avg_train_loss = tot_train_loss/len(train_graphs)
            
            thresholds = np.linspace(0, 1, 100)
            best_f1 = 0
            best_f1_acc = 0
            best_threshold = 0
            for t in thresholds:
                pred_classes = (all_pred_probs > t).astype(int)
                num_correct = (all_true == pred_classes).sum()
                acc = num_correct / len(all_true)
                f1 = f1_score(all_true, pred_classes, average='binary')
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
                    best_f1_acc = acc
            
            print(f"Epoch {epoch}/{cfg.num_epochs}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Val acc: {best_f1_acc*100:.2f}%, Val f1: {best_f1:.4f}, Threshold: {best_threshold:.2f}, Time/epoch: {epoch_time:.2f} seconds")

            torch.save(combined_model.state_dict(), f"{cfg.model_save_dir}/{model_name}_last.pth")
            if best_f1 > all_time_best_f1:
                all_time_best_f1 = best_f1
                all_time_best_threshold = best_threshold
                torch.save(combined_model.state_dict(), f"{cfg.model_save_dir}/{model_name}_best.pth")

    print(f"Best validation F1: {all_time_best_f1:.4f}, Threshold: {all_time_best_threshold:.2f}")

    print("\nTraining complete. Evaluating best validation model on test set...")

    # ------------ Evaluate on test set ------------
    combined_model.load_state_dict(torch.load(f'{cfg.model_save_dir}/{model_name}_best.pth'))
    combined_model.eval()

    all_true_classes = []
    all_pred_classes = []
    for graph in test_graphs:
        node_features = graph.x
        edge_list = graph.edge_index
        kites = graph.kites
        edge_attr = graph.edge_attr
        edge_labels = graph.y
        with torch.no_grad():
            out = combined_model(node_features, edge_list, kites, edge_attr)
            
        true_classes = np.round(edge_labels.cpu().numpy())
        pred_probs = F.sigmoid(out).cpu().numpy()
        pred_classes = (pred_probs > all_time_best_threshold).astype(int)
        
        all_true_classes.extend(true_classes.tolist())
        all_pred_classes.extend(pred_classes.tolist())

    all_true_classes = np.array(all_true_classes)
    all_pred_classes = np.array(all_pred_classes)

    num_correct = (all_true_classes == all_pred_classes).sum()
    acc = num_correct / len(all_true_classes)
    f1 = f1_score(all_true_classes, all_pred_classes, average='binary')

    print(f"Test accuracy: {acc*100:.2f}%, Test f1: {f1:.4f}")


if __name__ == '__main__':
    cfg = Config()
    train(cfg)







    