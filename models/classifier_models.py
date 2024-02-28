import torch
import torch.nn as nn
import torch.nn.functional as F


class gCNN(nn.Module):
    def __init__(self, num_channels, input_size, hidden_dim, output_dim, dropout=0.5, training=True):
        super(gCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(32 * input_size, hidden_dim)  # Adjust the input size based on your data
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output size is 1  for regression

        self.dropout = dropout
        self.training = training
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    

class EdgeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        
        mom = 0.1
        self.class_seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim, momentum=mom),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim, momentum=mom),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
            # No softmax because we use nn.CrossEntropyLoss
        )
        

    def forward(self, edge_embs):
        edge_embs = edge_embs.view(len(edge_embs), -1)
        return self.class_seq(edge_embs)
    
