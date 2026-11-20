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
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Autoencoder(latent_dim=64).to(device)
    x = torch.rand(8, 1, 28, 28).to(device)

    recon, z = model(x)

    print("Device:", device)
    print("Input:", x.shape)
    print("Reconstruction:", recon.shape)
    print("Latent:", z.shape)
    print("Pixel range:", recon.min().item(),
          recon.max().item())