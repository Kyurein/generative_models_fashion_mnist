import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.network(z)
        return generated.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, images):
        images = images.view(images.size(0), -1)
        return self.network(images)
