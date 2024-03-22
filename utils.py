import os

# import cv2
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
import torch_geometric
from torch_geometric.utils import to_undirected, add_self_loops
from torch_scatter import scatter

from tqdm import tqdm
from pathlib import Path

import random
import pandas as pd

def get_all_kites(edge_list):
    if len(edge_list[0]) == 0:
        return torch.zeros((0, 4), dtype=torch.long)

    # Check that edge_list is a torch tensor
    if not torch.is_tensor(edge_list):
        edge_list = torch.tensor(edge_list, dtype=torch.long)

    num_nodes = torch.max(edge_list).item() + 1
    num_edges = edge_list.shape[1]

    kites = torch.zeros((num_edges, 2), dtype=torch.long)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj[edge_list[0], edge_list[1]] = 1

    for i, edge in enumerate(edge_list.t()):
        n1 = edge[0]
        n2 = edge[1]

        n1_neighbours = torch.nonzero(adj[n1]).squeeze(1)
        n2_neighbours = torch.nonzero(adj[n2]).squeeze(1)

        # Find common neighbours
        common_mask = torch.isin(n1_neighbours, n2_neighbours)
        common_neighbours = n1_neighbours[common_mask]
        
        if len(common_neighbours) == 0:
            kites[i][0] = n1
            kites[i][1] = n2
        elif len(common_neighbours) == 1:
            kites[i][0] = common_neighbours[0]
            kites[i][1] = common_neighbours[0]
        elif len(common_neighbours) == 2:
            kites[i][0] = common_neighbours[0]
            kites[i][1] = common_neighbours[1]

    return kites


def normalize_edge_attributes(edge_index, edge_attr):
    """
    edge_index: [2, E] tensor with source and target nodes of each edge
    edge_attr: [E] tensor with the attribute (weight) of each edge
    """

    src_nodes = edge_index[0, :]
    
    # Sum weights of outgoing edges for each node
    # Effectively sums up the weights of all outgoing edges for each node
    sum_outgoing_weights = scatter(edge_attr, src_nodes, dim=0, reduce='add')
    
    # For normalization, we divide each edge's weight by the sum of its source node's outgoing edge weights
    # We use src_nodes to index into sum_outgoing_weights to get the normalization factor for each edge
    safe_sum_feature_outgoing = sum_outgoing_weights.clone()
    safe_sum_feature_outgoing[safe_sum_feature_outgoing == 0] = 1

    # Normalize feature, temporarily using safe sum
    normalized_feature = edge_attr / safe_sum_feature_outgoing[src_nodes]

    # Correct cases where the sum was originally 0: set normalized feature to 0 in these cases
    correction_mask = sum_outgoing_weights[src_nodes] == 0  # Mask of positions where sum was 0
    normalized_feature[correction_mask] = 0  # Apply correction

    return normalized_feature


def preprocess_bm_csv_to_pt(path_dir, save_dir):
    files = os.listdir(path_dir)

    os.makedirs(save_dir, exist_ok=True)

    node_files = [file for file in files if file.endswith('nodes.csv')]
    edge_files = [file for file in files if file.endswith('edges.csv')]

    node_files.sort()
    edge_files.sort()

    graphs = []
    
    mapping = {
        'inflammatory': 0, 
        'lymphocyte': 1, 
        'fibroblast and endothelial': 2, 
        'epithelial': 3
    }

    for node_file, edge_file in tqdm(zip(node_files, edge_files), total=len(edge_files)):
        node_df = pd.read_csv(os.path.join(path_dir, node_file))
        edge_df = pd.read_csv(os.path.join(path_dir, edge_file))

        # Map ground truth node labels to one-hot encodings
        node_df['gt_id'] = node_df['gt'].map(mapping).fillna(0)
        node_features = np.zeros((node_df['id'].max() + 1), dtype=np.int32)
        node_features[node_df['id']] = node_df['gt_id'].values 
        node_features = np.eye(4)[node_features]
            
        adj_list = []
        edge_attr_list = []
        edge_labels = []
        for i, row in edge_df.iterrows():
            source = row['source']
            target = row['target']
            adj_list.append([source, target])
            adj_list.append([target, source])
            
            distance = row['distance']
            delta_entropy = row['Delta_Entropy']
            sorenson_similarity = row['Sorenson_Similarity']
            edge_attr_list.append([distance, delta_entropy, sorenson_similarity])
            edge_attr_list.append([distance, delta_entropy, sorenson_similarity])
            
            edge_labels.append(row['type'])
            edge_labels.append(row['type'])
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        adj_list = torch.tensor(adj_list, dtype=torch.long).t()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32).view(-1, 1)
        edge_attr_list = torch.tensor(edge_attr_list, dtype=torch.float32)
        
        kites = get_all_kites(adj_list)
        
        edge_attr_list[:, 0] = normalize_edge_attributes(adj_list, edge_attr_list[:, 0])
        edge_attr_list[:, 1] = normalize_edge_attributes(adj_list, edge_attr_list[:, 1])
        edge_attr_list[:, 2] = normalize_edge_attributes(adj_list, edge_attr_list[:, 2])

        torch.save(
            Data(
                x=node_features, 
                y=edge_labels, 
                edge_index=adj_list, 
                kites=kites,
                edge_attr=edge_attr_list,
            ),
            os.path.join(save_dir, f"{node_file.split('_nodes')[0]}.pt")
        )


def load_bm_graphs_pt(path_dir, device):
    files = os.listdir(path_dir)
    files.sort()

    graphs = []
    for file in files:
        graphs.append(torch.load(os.path.join(path_dir, file)).to(device))
    return graphs


if __name__ == "__main__":
    preprocess_bm_csv_to_pt('datasets/bm_dataset/Train', 'datasets/bm_dataset_pt/Train')
    preprocess_bm_csv_to_pt('datasets/bm_dataset/Val', 'datasets/bm_dataset_pt/Val')
    preprocess_bm_csv_to_pt('datasets/bm_dataset/Test', 'datasets/bm_dataset_pt/Test')
        
        

