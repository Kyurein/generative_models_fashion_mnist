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


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.flatten = nn.Flatten()

        # Encoder: extract hidden features
        self.encoder_hidden = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: reconstruct the input image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.flatten(x)
        h = self.encoder_hidden(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar
