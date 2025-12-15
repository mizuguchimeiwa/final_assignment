from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    Architecture:
        - 2 convolutional layers
        - ReLU + MaxPooling
        - 3 fully-connected layers
        - Output: 10 classes
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)            # (batch, 3, 32, 32) → (batch, 6, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)             # (batch, 6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, 5)           # (batch, 16, 10, 10)

        # FC part
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Input:
            x: Tensor of shape (batch_size, 3, 32, 32)
        Output:
            Tensor of shape (batch_size, 10)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
