import torch
from torch import nn


class Generator(nn.Module):
    """
    DCGAN Generator for Fashion-MNIST.

    Input:  (B, 100)
    Output: (B, 1, 28, 28), range [-1, 1]
    """

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            # (B, 100, 1, 1) -> (B, 256, 7, 7)
            nn.ConvTranspose2d(
                latent_dim, 256,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (B, 256, 7, 7) -> (B, 128, 14, 14)
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (B, 128, 14, 14) -> (B, 1, 28, 28)
            nn.ConvTranspose2d(
                128, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.network(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for Fashion-MNIST.

    Input:  (B, 1, 28, 28)
    Output: (B, 1), raw logits
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 64, 14, 14)
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 7, 7) -> (B, 1, 1, 1)
            nn.Conv2d(
                128, 1,
                kernel_size=7,
                stride=1,
                padding=0
            )
        )

    def forward(self, images):
        logits = self.network(images)
        return logits.view(-1, 1)


def weights_init(module):
    """
    DCGAN weight initialization.

    Convolution weights: N(0, 0.02)
    BatchNorm weights:   N(1, 0.02)
    Biases:              0
    """

    if isinstance(
        module,
        (nn.Conv2d, nn.ConvTranspose2d)
    ):
        nn.init.normal_(module.weight, 0.0, 0.02)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)
