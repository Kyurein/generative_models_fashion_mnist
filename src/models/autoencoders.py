import torch
from torch import nn


class Autoencoder(nn.Module):
    """Vanilla Autoencoder."""

    def __init__(self, latent_dim=64):
        super().__init__()

        self.flatten = nn.Flatten()

        self.encoder_hidden = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )

        self.encoder_latent = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.flatten(x)
        h = self.encoder_hidden(x)
        z = self.encoder_latent(h)
        return z, h

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, 1, 28, 28)

    def forward(self, x):
        z, h = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z, h


class SparseAutoencoder(Autoencoder):
    """Same architecture as Vanilla AE; sparsity is applied in the loss."""

    def __init__(self, latent_dim=64):
        super().__init__(latent_dim=latent_dim)
