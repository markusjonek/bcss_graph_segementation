import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18Encoder(nn.Module):
    def __init__(self, out_dim, dropout=0.5):
        super(ResNet18Encoder, self).__init__()
        # Load pre-trained ResNet model
        base_model = models.resnet18(weights=None)
        
        # (3, 3) kernel instead of (7, 7) for the smaller patches
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add a projection head
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            #nn.Linear(512, 128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Dropout(dropout),

            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.projection_head(x)

        # normalize the feature vectors
        x = F.normalize(x, dim=1)
        return x