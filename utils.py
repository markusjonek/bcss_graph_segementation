import sys
import os

# import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
import torch_geometric
from torch_geometric.utils import to_undirected, add_self_loops
from scipy.spatial import Delaunay
from torch_scatter import scatter

from models.img_encoders import ResNet18Encoder

from tqdm import tqdm
from pathlib import Path

import random

import time
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed



BCSS_AGGREGATION = {
    0: 5000,  # outside_roi
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 4,
    8: 4,
    9: 4,
    10: 2,
    11: 2,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 4,
    17: 4,
    18: 4,
    19: 0,
    20: 0,
    21: 4
}


def get_dataset(dataset_folder, device):
    files = os.listdir(dataset_folder)

    node_files = [f for f in files if 'node' in f]
    edge_files = [f for f in files if 'edge' in f]

    node_files = sorted(node_files)
    edge_files = sorted(edge_files)

    pairs = list(zip(node_files, edge_files))
    random.shuffle(pairs)

    print(f"Loading datasets from: {dataset_folder}...")
    datasets = []
    for node_file, edge_file in tqdm(pairs):
        node = os.path.join(dataset_folder, node_file)
        edge = os.path.join(dataset_folder, edge_file)
        ds = BCSSDataset([node, edge], device)
        if len(ds) > 0:
            datasets.append(ds)

        # if len(datasets) == 10:
        #    break

    return datasets



def normalize_matrix_rows(mat):
    """
    Parameters
    ----------
    dist : 2d torch.Tensor
        A 2d tensor
    Returns
    ----------
    dist_norm : 2d torch.Tensor
        A normilized tensor along rows
    """
    mat_sum = torch.sum(mat, dim=1)
    mat_N = torch.reshape(mat_sum, (-1, 1))
    mat_norm = torch.div(mat, mat_N)
    mat_norm = torch.nan_to_num(mat_norm) # fill nan with 0 for given by zero division
    mat_norm = mat_norm.float()
    return mat_norm


def normalize_edge_features_rows(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    deno = torch.sum(torch.abs(edge_features), dim=2, keepdim=True)
    return np.divide(edge_features, deno, where = deno != 0)


def normalize_edge_feature_doubly_stochastic(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    edge_features_deno = np.sum(edge_features, axis=2, keepdims=True)
    edge_features_tilda = np.divide(edge_features, edge_features_deno, where=edge_features_deno!=0)

    channel = edge_features.shape[0]
    size = edge_features.shape[1]
    edge_features_normed = np.zeros((channel, size, size))
    for p in range(channel):
        d = np.sum(edge_features_tilda[p,:,:], axis=0)
        mul = np.matmul(np.divide(edge_features_tilda[p,:,:], d, where = d != 0),
                        edge_features_tilda[p,:,:].T)
        edge_features_normed[p] = mul
    return edge_features_normed


def normalize_edge_features_symmetric(edge_features):
    """
    Parameters
    ----------
    edge_features : numpy array
        3d numpy array (P x N x N).
        edge_features[p, i, j] is the jth feature of node i in pth channel
    Returns
    -------
    edge_features_normed : numpy array
        normalized edge_features.
    """
    deno1 = torch.sum(torch.abs(edge_features), axis=2, keepdims=True)
    deno2 = torch.sum(torch.abs(edge_features), axis=1, keepdims=True)
    sqrt_deno_product = torch.sqrt(deno1) * torch.sqrt(deno2)
    
    # Use torch.where to perform division where both denominators are not zero
    # and set elements to 0 where the condition is False
    normalized_edge_features = torch.where((deno1 != 0) & (deno2 != 0),
                                           torch.divide(edge_features, sqrt_deno_product),
                                           torch.zeros_like(edge_features))
    
    return normalized_edge_features


def get_neighbours(edge_list, node_id):
    start_indices = torch.where(edge_list[0] == node_id)[0]
    end_indices = torch.where(edge_list[1] == node_id)[0]

    # Extract neighbors from both start and end indices
    neighbors_start = edge_list[1, start_indices]
    neighbors_end = edge_list[0, end_indices]

    # Combine and remove duplicates, and ensure node_id is not in the list
    neighbors = torch.cat((neighbors_start, neighbors_end)).unique()
    neighbors = neighbors[neighbors != node_id]
    return neighbors

def get_neighbours_adj(adj, node_id):
    return torch.nonzero(adj[node_id]).squeeze(1)


def get_kites(edge_list, edge_batch_indicies):
    s = time.time()
    kites = torch.zeros((len(edge_batch_indicies), 4), dtype=torch.long)
    for i in range(len(edge_batch_indicies)):
        edge_idx = edge_batch_indicies[i]
        n1 = edge_list[0][edge_idx]
        n2 = edge_list[1][edge_idx]

        n1_neighbours = get_neighbours(edge_list, n1)
        n2_neighbours = get_neighbours(edge_list, n2)

        # Find common neighbours
        common_mask = torch.isin(n1_neighbours, n2_neighbours)
        common_neighbours = n1_neighbours[common_mask]
        
        if len(common_neighbours) == 0:
            common_neighbours = torch.tensor([0, 0], dtype=torch.long)
        elif len(common_neighbours) == 1:
            common_neighbours = torch.tensor([common_neighbours[0], 0], dtype=torch.long)
        
        c1, c2 = common_neighbours[0], common_neighbours[1]
        kite = torch.cat((c1.view(1), n1.view(1), n2.view(1), c2.view(1)))
        kites[i] = kite

    e = time.time()
    return kites


def get_kites_adj(gnn_out_node_features, edge_list, edge_batch_indicies):
    num_nodes = torch.max(edge_list).item() + 1
    gnn_out_dim = gnn_out_node_features.shape[1]

    kites = torch.zeros((len(edge_batch_indicies), gnn_out_dim*4), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj[edge_list[0], edge_list[1]] = 1

    for i in range(len(edge_batch_indicies)):
        edge_idx = edge_batch_indicies[i]
        n1 = edge_list[0][edge_idx]
        n2 = edge_list[1][edge_idx]

        n1_neighbours = get_neighbours_adj(adj, n1)
        n2_neighbours = get_neighbours_adj(adj, n2)

        # Find common neighbours
        common_mask = torch.isin(n1_neighbours, n2_neighbours)
        common_neighbours = n1_neighbours[common_mask]
        
        if len(common_neighbours) == 0:
            kite_indicies = torch.tensor([n1, n2, n1, n2], dtype=torch.long)
        elif len(common_neighbours) == 1:
            cn1 = common_neighbours[0]
            kite_indicies = torch.tensor([n1, n2, cn1, cn1], dtype=torch.long)
        else:
            cn1, cn2 = common_neighbours[0], common_neighbours[1]
            kite_indicies = torch.tensor([n1, n2, cn1, cn2], dtype=torch.long)
        kites[i] = gnn_out_node_features[kite_indicies].view(-1)
    return kites

    

def get_all_kites(edge_list, device):
    if len(edge_list[0]) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    # Check that edge_list is a torch tensor
    if not torch.is_tensor(edge_list):
        edge_list = torch.tensor(edge_list, dtype=torch.long, device=device)

    num_nodes = torch.max(edge_list).item() + 1
    num_edges = edge_list.shape[1]

    kites = torch.zeros((num_edges, 4), dtype=torch.long, device=device)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=device)
    adj[edge_list[0], edge_list[1]] = 1

    for i, edge in enumerate(edge_list.t()):
        n1 = edge[0]
        n2 = edge[1]

        n1_neighbours = get_neighbours_adj(adj, n1)
        n2_neighbours = get_neighbours_adj(adj, n2)

        # Find common neighbours
        common_mask = torch.isin(n1_neighbours, n2_neighbours)
        common_neighbours = n1_neighbours[common_mask]
        
        kites[i][0] = n1
        kites[i][1] = n2
        if len(common_neighbours) == 0:
            n1, n2 = random.choice([[n1, n2], [n2, n1]])
            kites[i][2] = n1
            kites[i][3] = n2
        elif len(common_neighbours) == 1:
            kites[i][2] = common_neighbours[0]
            kites[i][3] = common_neighbours[0]
        elif len(common_neighbours) == 2:
            c1, c2 = common_neighbours
            c1, c2 = random.choice([[c1, c2], [c2, c1]])
            kites[i][2] = c1
            kites[i][3] = c2

    return kites


def normalize_edge_attributes(edge_index, edge_attr):
    # edge_index: [2, E] tensor with source and target nodes of each edge
    # edge_attr: [E] tensor with the attribute (weight) of each edge
    #     
    # Step 1: Identify source nodes for each edge
    src_nodes = edge_index[0, :]
    
    # Step 2: Sum weights of outgoing edges for each node
    # Use scatter_add for this purpose, which accumulates values from src into out at the indices specified in index
    # This effectively sums up the weights of all outgoing edges for each node
    sum_outgoing_weights = scatter(edge_attr, src_nodes, dim=0, reduce='add')
    
    # Step 3: Normalize edge weights
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


def upsample_minority_classes(labels, edge_list):
    # Count the occurrences of each class
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Find the class with the most occurrences
    max_class = max(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]
    
    # Initialize a list to hold the upsampled data
    labels_upsampled_data = [labels[labels == max_class]]  # Start with the majority class
    edge_list_upsampled_data = [edge_list.T[labels == max_class]]  # Start with the majority class
    
    # Upsample the minority classes
    for class_val in class_counts:
        if class_val != max_class:
            class_indices = np.where(labels == class_val)[0]
            # Check if the class is not to be upsampled, skip it
            if len(class_indices) == 0:
                continue
            upsample_indices = np.random.choice(class_indices, size=max_count, replace=True)
            labels_upsampled_class_data = labels[upsample_indices]
            edge_list_upsampled_class_data = edge_list.T[upsample_indices]
            
            labels_upsampled_data.append(labels_upsampled_class_data)
            edge_list_upsampled_data.append(edge_list_upsampled_class_data)
    
    # Concatenate all upsampled classes together
    labels_upsampled_data = np.concatenate(labels_upsampled_data)
    edge_list_upsampled_data = np.concatenate(edge_list_upsampled_data)
    
    return labels_upsampled_data, edge_list_upsampled_data.T


def load_graphs(dataset_folder, device):
    files = os.listdir(dataset_folder)
    node_files = [f for f in files if 'node' in f]
    edge_files = [f for f in files if 'edge' in f]

    node_files = sorted(node_files)
    edge_files = sorted(edge_files)

    graphs = []

    for node_file, edge_file in zip(node_files, edge_files):
        node_path = os.path.join(dataset_folder, node_file)
        edge_path = os.path.join(dataset_folder, edge_file)

        node_data = np.load(node_path)
        edge_data = np.load(edge_path)
        
        # extract the data
        coords = node_data[:, :2]
        yolo_classes = node_data[:, 2].astype(np.int32)
        #cell_embeddings = node_data[:, 6:]
        
        edge_list = edge_data[:, :2].T
        kites = edge_data[:, 2:6]
        edge_attr = edge_data[:, 6:10]
        edge_labels = edge_data[:, 10].astype(np.int32)
        
        edge_mask = edge_labels != 5000
        edge_list = edge_list[:, edge_mask]
        kites = kites[edge_mask]
        edge_attr = edge_attr[edge_mask]
        edge_labels = edge_labels[edge_mask]
        
        #unique, counts = np.unique(edge_labels, return_counts=True)
        #if len(unique) == 5:
        #    edge_labels, edge_list = upsample_minority_classes(edge_labels, edge_list)
        #    unique, counts = np.unique(edge_labels, return_counts=True)
        
        # one hot encode
        yolo_classes = np.eye(5, dtype=np.float32)[yolo_classes]
        edge_labels = np.eye(6, dtype=np.float32)[edge_labels]
        
        # to torch tensors
        coords = torch.tensor(coords, dtype=torch.float32, device=device)
        yolo_classes = torch.tensor(yolo_classes, dtype=torch.float32, device=device)
        #cell_embeddings = torch.tensor(cell_embeddings, dtype=torch.float32, device=device)
        
        edge_list = torch.tensor(edge_list, dtype=torch.long, device=device)
        kites = torch.tensor(kites, dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32, device=device)
        
        edge_dists = edge_attr[:, 0]
        edge_dists_norm = normalize_edge_attributes(edge_list, edge_dists)
        edge_attr[:, 0] = edge_dists_norm
        
        tile_name = Path(node_file).stem.split('_nodes.npy')[0]

        graph = Data(
            x=yolo_classes,
            y=edge_labels,
            edge_index=edge_list,
            edge_attr=edge_attr,
            kites=kites,
            tile_name=tile_name,
            coords=coords
        )
        if len(edge_labels) > 0:
            graphs.append(graph)
    return graphs








def permutate_kites(kites):
    switch_desicion = torch.randint(0, 2, (kites.shape[0], 2))
    for i in range(len(kites)):
        if switch_desicion[i][0]:
            temp_0 = kites[i][0]
            kites[i][0] = kites[i][1]
            kites[i][1] = temp_0
        if switch_desicion[i][1]:
            temp_2 = kites[i][2]
            kites[i][2] = kites[i][3]
            kites[i][3] = temp_2
    return kites




def build_random_graph(bcss_img_path, bcss_mask_path, nodes_path, device, cell_encoder):
    bcss_img = cv2.imread(bcss_img_path)
    bcss_img = torch.tensor(bcss_img, dtype=torch.float32, device=device)
    
    num_nodes = 1.5 * bcss_img.shape[0] * bcss_img.shape[1] * 0.00013373223950858838

    # sample random coordinates in the image
    x_coords = np.random.randint(0, bcss_img.shape[1], int(num_nodes))
    y_coords = np.random.randint(0, bcss_img.shape[0], int(num_nodes))
    coords = np.stack((x_coords, y_coords), axis=1)
    
    #nodes = np.load(nodes_path)
    #coords = nodes[:, :2]
    #num_nodes = len(coords)

    # ------------ Get edges ------------
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            # Ensure the smaller index comes first to avoid duplicates like (1,2) and (2,1)
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges.add(edge)

    # Convert the set of edges to a list for further processing or visualization
    edge_list = list(edges)
    edge_list = np.array(edge_list).T
    edge_list = torch.tensor(edge_list, dtype=torch.long)
    edge_list, _ = to_undirected(edge_list, None)
    #edge_list, _ = add_self_loops(edge_list, None)
    edge_list = edge_list.numpy()

    # ------------ Get edge distances ------------
    edge_distances = np.zeros(len(edge_list[0]))
    for i, edge in enumerate(edge_list.T):
        n1 = edge[0]
        n2 = edge[1]
        dist = np.linalg.norm(coords[n1] - coords[n2])
        edge_distances[i] = dist


    # ------------ Get edge labels (middle point) ------------
    edge_labels = np.zeros(len(edge_list[0]), dtype=np.int32)
    mask = cv2.imread(bcss_mask_path, cv2.IMREAD_GRAYSCALE)
    for i, edge in enumerate(edge_list.T):
        n1 = edge[0]
        n2 = edge[1]
        mid = (coords[n1] + coords[n2]) / 2
        mid = mid.astype(np.int32)
        mask_val = mask[mid[1], mid[0]]
        edge_labels[i] = BCSS_AGGREGATION[mask_val]

    edge_mask = edge_labels != 5000
    edge_list = edge_list[:, edge_mask]
    edge_distances = edge_distances[edge_mask]
    edge_labels = edge_labels[edge_mask]

    edge_labels = np.eye(5, dtype=np.float32)[edge_labels]
    
    # ------------ get BCSS node image embeddings ------------
    cell_images = torch.zeros((len(coords), 64, 64, 3), dtype=torch.float32, device=device)
    for i, coord in enumerate(coords):
        x, y = coord
        xmin = max(int(x - 32), 0)
        xmax = min(int(x + 32), bcss_img.shape[1])
        ymin = max(int(y - 32), 0)
        ymax = min(int(y + 32), bcss_img.shape[0])

        if xmin == 0:
            xmax = 64
        if ymin == 0:
            ymax = 64
        if xmax == bcss_img.shape[1]:
            xmin = xmax - 64
        if ymax == bcss_img.shape[0]:
            ymin = ymax - 64

        cell_img = bcss_img[ymin:ymax, xmin:xmax]
        cell_images[i] = cell_img
    
    cell_images = cell_images.permute(0, 3, 1, 2)
    cell_images = cell_images / 255.0
    batch_size = 2048
    num_batches = int(len(cell_images) / batch_size) + 1
    
    cell_encoder.eval()
    cell_encodings = torch.zeros((len(cell_images), 64), dtype=torch.float32, device=device)
    for b in range(num_batches):
        imgs = cell_images[b*batch_size:(b+1)*batch_size]
        with torch.no_grad():
            cell_encodings[b*batch_size:(b+1)*batch_size, :] = cell_encoder(imgs)
    
    #yolo_classes = nodes[:, 2].astype(int)
    #yolo_classes = np.eye(5)[yolo_classes]
    #yolo_classes = torch.tensor(yolo_classes, dtype=torch.float32, device=device)
    
    #cell_encodings = torch.cat((cell_encodings, yolo_classes), dim=1)
        
    edge_list = torch.tensor(edge_list, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float32, device=device).view(-1, 1)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float32, device=device)

    graph = Data(x=cell_encodings, edge_index=edge_list, edge_attr=edge_attr, y=edge_labels, kites=edge_list)
    return graph


def generate_random_graphs(bcss_dir, device, num_graphs=None):
    bcss_image_files = os.listdir(os.path.join(bcss_dir, 'images'))
    bcss_mask_files = os.listdir(os.path.join(bcss_dir, 'masks'))

    bcss_image_files = sorted(bcss_image_files)
    bcss_mask_files = sorted(bcss_mask_files)
    
    pairs = list(zip(bcss_image_files, bcss_mask_files))
    
    random.shuffle(pairs)
    
    if num_graphs is None:
        num_graphs = len(pairs)
    
    graphs = []
    
    cell_encoder = ResNet18Encoder(64)
    cell_encoder.load_state_dict(torch.load(f'saved_models/resnet18_simclr_encoder.pth', map_location=torch.device(device)))
    cell_encoder.to(device)
    cell_encoder.eval()

    for img_file, mask_file in tqdm(pairs[:num_graphs]):
        bcss_img_path = os.path.join(bcss_dir, 'images', img_file)
        bcss_mask_path = os.path.join(bcss_dir, 'masks', mask_file)

        graph = build_random_graph(bcss_img_path, bcss_mask_path, device, cell_encoder)
        graphs.append(graph)
    return graphs


def load_graph_pair(bcss_dir, split_folder, img_mask_pair, device, cell_encoder):
    img_file, mask_file, tile_name = img_mask_pair
    bcss_img_path = os.path.join(bcss_dir, 'images', img_file)
    bcss_mask_path = os.path.join(bcss_dir, 'masks', mask_file)
    nodes_path = os.path.join(split_folder, tile_name + '_nodes.npy')
    graph = build_random_graph(bcss_img_path, bcss_mask_path, nodes_path, device, cell_encoder)
    return graph


def generate_random_graphs_multithreaded(bcss_dir, split_folder, device, num_graphs=None, max_workers=10):
    bcss_image_files = os.listdir(os.path.join(bcss_dir, 'images'))
    bcss_mask_files = os.listdir(os.path.join(bcss_dir, 'masks'))

    bcss_image_files = sorted(bcss_image_files)
    bcss_mask_files = sorted(bcss_mask_files)
    
    valid_tiles = [f.split('_nodes.npy')[0] for f in os.listdir(split_folder) if 'node' in f]

    pairs = list(zip(bcss_image_files, bcss_mask_files))
    valid_pairs = []
    for bcss, mask in pairs:
        if bcss.split('.png')[0] in valid_tiles:
            valid_pairs.append((bcss, mask, bcss.split('.png')[0]))
    del pairs

    random.shuffle(valid_pairs)

    cell_encoder = ResNet18Encoder(64)
    cell_encoder.load_state_dict(torch.load(f'saved_models/resnet18_simclr_encoder.pth'))
    cell_encoder.to(device)
    cell_encoder.eval()

    if num_graphs is None:
        num_graphs = len(valid_pairs)
    else:
        valid_pairs = valid_pairs[:num_graphs]

    graphs = []

    # Use ThreadPoolExecutor to generate graphs in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each image-mask pair
        future_to_pair = {executor.submit(load_graph_pair, bcss_dir, split_folder, pair, device, cell_encoder): pair for pair in valid_pairs}

        # Process as each future completes
        for future in tqdm(as_completed(future_to_pair), total=len(valid_pairs)):
            try:
                graph = future.result()  # Get the result from the future
                graphs.append(graph)
            except Exception as e:
                print(f"Failed to generate graph for pair {future_to_pair[future]}: {e}")

    return graphs

def get_cell_images(graph_data_dir, bcss_dir, img_size, device):
    node_files = [f for f in os.listdir(graph_data_dir) if 'node' in f]
    
    tot_num_cells = 0
    for node_f in node_files:
        nodes_path = os.path.join(graph_data_dir, node_f)
        nodes = np.load(nodes_path)
        tot_num_cells += len(nodes)
    
    cell_images = torch.zeros((tot_num_cells, img_size, img_size, 3), dtype=torch.float32, device=device)
    mask_labels = torch.zeros(tot_num_cells, dtype=torch.long, device=device)
    
    idx_counter = 0
    
    for node_f in tqdm(node_files):
        tile_name = node_f.split('_nodes.npy')[0]
        
        nodes_path = os.path.join(graph_data_dir, node_f)
        bcss_path = os.path.join(bcss_dir, 'images', tile_name + '.png')
        mask_path = os.path.join(bcss_dir, 'masks', tile_name + '.png')
        
        nodes = np.load(nodes_path)
        bcss_img = cv2.imread(bcss_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        bcss_img = torch.tensor(bcss_img, dtype=torch.float32, device=device)
        
        real_coords = nodes[:, :2]
        
        rnd_x_coords = np.random.randint(img_size, bcss_img.shape[1] - img_size, len(nodes))
        rnd_y_coords = np.random.randint(img_size, bcss_img.shape[0] - img_size, len(nodes))
        rnd_coords = np.stack((rnd_x_coords, rnd_y_coords), axis=1)
        
        all_coords = np.concatenate((real_coords, rnd_coords), axis=0)
        
        for i, coord in enumerate(rnd_coords):
            x, y = coord
            xmin = max(int(x - img_size//2), 0)
            xmax = min(int(x + img_size//2), bcss_img.shape[1])
            ymin = max(int(y - img_size//2), 0)
            ymax = min(int(y + img_size//2), bcss_img.shape[0])

            if xmin == 0:
                xmax = img_size
            if ymin == 0:
                ymax = img_size
            if xmax == bcss_img.shape[1]:
                xmin = xmax - img_size
            if ymax == bcss_img.shape[0]:
                ymin = ymax - img_size

            cell_img = bcss_img[ymin:ymax, xmin:xmax]
            cell_images[idx_counter + i] = cell_img
            mask_labels[idx_counter + i] = BCSS_AGGREGATION[mask[int(y), int(x)]]
            
        idx_counter += len(nodes)
    
    # remove images from oustise_roi
    mask_label_mask = mask_labels != 5000
    mask_labels = mask_labels[mask_label_mask]
    cell_images = cell_images[mask_label_mask]
        
    return cell_images, mask_labels