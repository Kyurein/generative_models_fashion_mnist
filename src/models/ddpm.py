import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts diffusion timesteps into sinusoidal embeddings.

    Input:  (B,)
    Output: (B, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32
            )
            / max(half_dim - 1, 1)
        )

        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1
        )

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Architecture:
        GroupNorm -> SiLU -> Conv3x3
        Add projected time embedding
        GroupNorm -> SiLU -> Conv3x3
        Residual connection
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_embedding):
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_features = self.time_projection(time_embedding)
        time_features = time_features[:, :, None, None]

        h = h + time_features

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class UNet(nn.Module):
    """
    Small U-Net for DDPM on Fashion-MNIST.

    Input:
        x_t: (B, 1, 28, 28)
        t:   (B,)

    Output:
        Predicted noise: (B, 1, 28, 28)
    """

    def __init__(self, time_dim=128):
        super().__init__()

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial convolution: 1 -> 32
        self.input_conv = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        )

        # Encoder: 28 x 28
        self.down_block1 = ResBlock(
            32, 64, time_dim
        )

        self.downsample1 = nn.Conv2d(
            64,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Encoder: 14 x 14
        self.down_block2 = ResBlock(
            64, 128, time_dim
        )

        self.downsample2 = nn.Conv2d(
            128,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Middle: 7 x 7
        self.middle_block = ResBlock(
            128, 128, time_dim
        )

        # Decoder: 7 x 7 -> 14 x 14
        self.upsample1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            )
        )

        # 128 + 128 skip channels -> 64
        self.up_block1 = ResBlock(
            256, 64, time_dim
        )

        # Decoder: 14 x 14 -> 28 x 28
        self.upsample2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            )
        )

        # 64 + 64 skip channels -> 32
        self.up_block2 = ResBlock(
            128, 32, time_dim
        )

        # Predict noise (no output activation)
        self.output_conv = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t):
        # Time embedding
        time_embedding = self.time_embedding(t)

        # Initial features: (B, 32, 28, 28)
        x = self.input_conv(x)

        # Encoder: 28 x 28
        skip1 = self.down_block1(
            x,
            time_embedding
        )

        x = self.downsample1(skip1)

        # Encoder: 14 x 14
        skip2 = self.down_block2(
            x,
            time_embedding
        )

        x = self.downsample2(skip2)

        # Middle: 7 x 7
        x = self.middle_block(
            x,
            time_embedding
        )

        # Decoder: 7 -> 14
        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)

        x = self.up_block1(
            x,
            time_embedding
        )

        # Decoder: 14 -> 28
        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)

        x = self.up_block2(
            x,
            time_embedding
        )

        # Predicted noise
        return self.output_conv(x)
