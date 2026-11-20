import torch
from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
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
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)

        return reconstruction, z

if __name__ == "__main__":
    m = Autoencoder()
    x = torch.rand(8, 1, 28, 28)
    recon, z = m(x)
    print(recon.shape, z.shape, recon.min().item(), recon.max().item())