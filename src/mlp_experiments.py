
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data import build_dataloaders
from src.models.mlp_classifier import MLPClassifier
from src.utils import get_device, set_seed


# Fashion-MNIST class names, in dataset label order
CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


class FrozenLatentDecoder(nn.Module):
    """Same decoder architecture used in Tahap 4."""

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, 1, 28, 28)


def save_class_vector_grid(images, output_path):
    """Save the 10 class-vector transpose projections."""

    fig, axes = plt.subplots(1, 10, figsize=(18, 3))

    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES[i], fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "Tahap 2: Weight Transpose from Class Vectors",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_test_projection_grid(
    originals,
    projections,
    labels,
    output_path
):
    """Save original images and their weight-transpose projections."""

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))

    for i in range(8):
        axes[0, i].imshow(
            originals[i],
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[0, i].set_title(
            CLASS_NAMES[labels[i]],
            fontsize=8
        )

        axes[1, i].imshow(
            projections[i],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Hide tick marks but preserve row labels
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Projection", fontsize=10)

    fig.suptitle(
        "Tahap 3: Weight Transpose from Test Latents",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_gaussian_sample_grid(images, output_path):
    """Save 64 images generated from Gaussian latent samples."""

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            images[i],
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.axis("off")

    fig.suptitle(
        "Tahap 5: Gaussian Sampling in MLP Latent Space",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(42)

    device = get_device()
    print(f"Device: {device}")

    # Create output directories
    checkpoint_dir = Path("checkpoints")
    figures_dir = Path("figures")
    results_dir = Path("results")

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load trained classifier
    classifier = MLPClassifier().to(device)

    classifier_path = (
        checkpoint_dir / "mlp_classifier_best.pth"
    )

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_path}"
        )

    classifier.load_state_dict(
        torch.load(
            classifier_path,
            map_location=device,
            weights_only=True
        )
    )

    classifier.eval()

    print(f"Loaded classifier: {classifier_path}")

    # Load the Tahap 4 decoder
    decoder = FrozenLatentDecoder().to(device)

    decoder_path = (
        checkpoint_dir / "frozen_decoder_best.pth"
    )

    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Frozen decoder checkpoint not found: {decoder_path}"
        )

    decoder.load_state_dict(
        torch.load(
            decoder_path,
            map_location=device,
            weights_only=True
        )
    )

    decoder.eval()

    print(f"Loaded frozen decoder: {decoder_path}")

    # Load Fashion-MNIST test set
    _, _, test_loader = build_dataloaders()

    metrics = {}

    # ==========================================================
    # TAHAP 2: Weight Transpose from Class Vectors
    # ==========================================================

    print("\nTahap 2: Weight Transpose from Class Vectors")

    with torch.no_grad():

        # Generate all 10 one-hot class vectors
        class_vectors = torch.eye(10, device=device)

        # Reverse projection without biases:
        # (10, 10) @ (10, 64) -> (10, 64)
        z_hat = torch.relu(
            class_vectors @ classifier.fc3.weight
        )

        # (10, 64) @ (64, 256) -> (10, 256)
        h_hat = torch.relu(
            z_hat @ classifier.fc2.weight
        )

        # (10, 256) @ (256, 784) -> (10, 784)
        x_hat = torch.sigmoid(
            h_hat @ classifier.fc1.weight
        )

        class_images = x_hat.view(-1, 28, 28).cpu().numpy()

    tahap2_path = (
        figures_dir / "mlp_transpose_class_vectors.png"
    )

    save_class_vector_grid(class_images, tahap2_path)

    print(f"Saved class vector projections: {tahap2_path}")

    # ==========================================================
    # TAHAP 3: Weight Transpose from Real Test Latents
    # ==========================================================

    print("\nTahap 3: Weight Transpose from Test Latents")

    # Take 8 real test images
    images, labels = next(iter(test_loader))

    images = images[:8].to(device)
    labels = labels[:8]

    with torch.no_grad():

        # Extract real MLP latent representations
        _, z_mlp = classifier(images)

        # Reverse projection without biases:
        # (8, 64) @ (64, 256) -> (8, 256)
        h_hat = torch.relu(
            z_mlp @ classifier.fc2.weight
        )

        # (8, 256) @ (256, 784) -> (8, 784)
        x_hat = torch.sigmoid(
            h_hat @ classifier.fc1.weight
        )

        projections = x_hat.view(-1, 1, 28, 28)

        # Calculate average pixel-wise MSE
        projection_mse = nn.functional.mse_loss(
            projections,
            images
        ).item()

    tahap3_path = (
        figures_dir / "mlp_transpose_test_latents.png"
    )

    save_test_projection_grid(
        images.squeeze(1).cpu().numpy(),
        projections.squeeze(1).cpu().numpy(),
        labels.cpu().tolist(),
        tahap3_path
    )

    metrics["tahap3_projection_mse"] = projection_mse

    print(f"Projection MSE: {projection_mse:.6f}")
    print(f"Saved test latent projections: {tahap3_path}")

    # ==========================================================
    # TAHAP 5: Gaussian Sampling in MLP Latent Space
    # ==========================================================

    print("\nTahap 5: Gaussian Sampling in MLP Latent Space")

    latent_batches = []

    # Collect z_MLP from the entire test set
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            _, z_mlp = classifier(images)

            latent_batches.append(z_mlp)

    all_latents = torch.cat(latent_batches, dim=0)

    # Calculate per-dimension mean and standard deviation
    latent_mu = all_latents.mean(dim=0)
    latent_sigma = all_latents.std(dim=0)

    print(f"Collected latent vectors: {all_latents.shape}")
    print(f"Latent mean shape: {latent_mu.shape}")
    print(f"Latent std shape: {latent_sigma.shape}")

    # Generate 64 Gaussian latent samples
    torch.manual_seed(42)

    z_rand = latent_mu + latent_sigma * torch.randn(
        64,
        64,
        device=device
    )

    # Decode using the Tahap 4 frozen decoder
    with torch.no_grad():
        generated_images = decoder(z_rand)

    generated_images = (
        generated_images.squeeze(1).cpu().numpy()
    )

    tahap5_path = (
        figures_dir / "mlp_latent_gaussian_samples.png"
    )

    save_gaussian_sample_grid(
        generated_images,
        tahap5_path
    )

    print(f"Generated latent samples: {z_rand.shape}")
    print(f"Saved Gaussian sample grid: {tahap5_path}")

    # ==========================================================
    # SAVE EXPERIMENT METRICS
    # ==========================================================

    metrics["tahap5_latent_mean_average"] = (
        latent_mu.mean().item()
    )

    metrics["tahap5_latent_std_average"] = (
        latent_sigma.mean().item()
    )

    metrics["number_of_test_latents"] = (
        all_latents.size(0)
    )

    metrics_path = (
        results_dir / "mlp_experiments_metrics.json"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nAll MLP experiments completed.")
    print(f"Results saved to: {metrics_path}")


if __name__ == "__main__":
    main()
