import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from pytorch_metric_learning import losses

import cv2
from tqdm import tqdm

import utils
from models.img_encoders import ResNet18Encoder


class CellImageDataset(Dataset):
    def __init__(self, bcss_dir, split_dir, img_size, device):
        images, labels = utils.get_cell_images(split_dir, bcss_dir, img_size, device)
        images /= 255
        images = images.permute(0, 3, 2, 1)
        
        labels = F.one_hot(labels, num_classes=5)
    
        self.images = images
        self.labels = labels.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


data_transforms = transforms.Compose([
    transforms.RandomRotation(90, fill=(0.6960, 0.5110, 0.7107)),
    transforms.RandomResizedCrop(size=64, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)
    ], p=0.8),
    #transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
])


class BSCSSClassifier(nn.Module):
    def __init__(self, in_dim, dropout=0.5):
        super(BSCSSClassifier, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(in_dim, 5)
        )
    
    def forward(self, x):
        return self.projection_head(x)


def train():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bcss_dir = '../gnn_bcss_img/BCSS'
    train_data_dir = 'datasets/full_graph_data/train'
    val_data_dir = 'datasets/full_graph_data/val'

    img_size = 64
    batch_size = 256

    train_dataset = CellImageDataset(bcss_dir, train_data_dir, img_size, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #val_cell_images = utils.get_cell_images(val_data_dir, bcss_dir, img_size, device)
    #val_cell_images /= 255
    #val_cell_images = val_cell_images.permute(0, 3, 2, 1)
    #val_loader = torch.utils.data.DataLoader(val_cell_images, batch_size=batch_size*2, shuffle=True)
    
    encoded_dim = 64

    resnet18 = ResNet18Encoder(encoded_dim, dropout=0.5)
    resnet18.load_state_dict(torch.load(f'saved_models/resnet18_simclr_encoder.pth'))
    resnet18.to(device)
    resnet18.train()
    
    bcss_classifier = BSCSSClassifier(encoded_dim, dropout=0.5)
    #bcss_classifier.load_state_dict(torch.load(f'saved_models/resnet18_simclr_encoder.pth'))
    bcss_classifier.to(device)
    bcss_classifier.train()

    print("Num params: ", sum(p.numel() for p in resnet18.parameters() if p.requires_grad))
    print("Num train images: ", len(train_dataset))

    ntxent_criterion = losses.NTXentLoss(temperature=0.5) 
    cross_entropy_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(resnet18.parameters()) + list(bcss_classifier.parameters()),
        lr=0.0005
    )
    
    scaler = GradScaler()
    
    n_epochs = 1000
    simclr_labels = torch.arange(0, batch_size, dtype=torch.long, device=device)
    simclr_labels = torch.cat((simclr_labels, simclr_labels), dim=0)
    
    n_epochs = 1000
    for epoch in range(n_epochs):
        train_losses = []
        simclr_losses = []
        bcss_losses = []
        
        for i, (images, bcss_labels) in enumerate(train_loader):
            if len(images) != batch_size:
                continue
                
            optimizer.zero_grad()
            
            batch_aug1 = data_transforms(images)
            batch_aug2 = data_transforms(images)
            batch = torch.cat((batch_aug1, batch_aug2), dim=0)
            
            bcss_labels = torch.cat((bcss_labels, bcss_labels), dim=0)
            
            with autocast():
                simclr_out = resnet18(batch)
                simclr_loss = ntxent_criterion(simclr_out, simclr_labels)
                bcss_out = bcss_classifier(simclr_out)
                bcss_loss = cross_entropy_criterion(bcss_out, bcss_labels) * 0.1
                loss = simclr_loss + bcss_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            simclr_losses.append(simclr_loss.item())
            bcss_losses.append(bcss_loss.item())
            
            print(f"\rLoss: {np.mean(train_losses):.5f}, simCLR Loss: {np.mean(simclr_losses):.5f}, BCSS Loss: {np.mean(bcss_losses):.5f}, {i/len(train_loader)*100:.2f}%", end="")
        
        print(f"\rEpoch: {epoch}, Loss: {np.mean(train_losses):.5f}")
            
        torch.save(resnet18.state_dict(), f'saved_models/resnet18_simclr_encoder.pth')
        
        train_dataset = CellImageDataset(bcss_dir, train_data_dir, img_size, device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    train()



