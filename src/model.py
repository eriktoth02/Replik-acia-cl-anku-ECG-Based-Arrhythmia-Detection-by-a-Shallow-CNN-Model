import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    def __init__(self, input_length):
        super(ShallowCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)  # novinka – batch norm
        self.pool = nn.MaxPool1d(kernel_size=2)

        conv_output_length = (input_length - 4) // 2  # kernel_size=5, pool=2
        self.fc1 = nn.Linear(32 * conv_output_length, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)               # použijeme BatchNorm
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)               # logity bez sigmoid
        return x
