import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Input layer: 784 -> 256
        self.fc1 = nn.Linear(784, 256)

        # Hidden layer: 256 -> 64
        self.fc2 = nn.Linear(256, 64)

        # Output layer: 64 -> 10 classes
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Flatten image: (B, 1, 28, 28) -> (B, 784)
        x = self.flatten(x)

        # First hidden layer
        x = self.relu(self.fc1(x))

        # Second hidden layer: z_MLP
        z_mlp = self.relu(self.fc2(x))

        # Classification logits
        logits = self.fc3(z_mlp)

        return logits, z_mlp
