import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src.data import build_dataloaders
from src.models.autoencoders import (
    Autoencoder,
    SparseAutoencoder,
    VAE,
)
from src.models.gan import Generator as MLPGenerator
from src.models.dcgan import Generator as DCGANGenerator
from src.utils import set_seed, get_device


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
LATENT_DIM_AE = 64
LATENT_DIM_GAN = 100

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CHECKPOINT_DIR = Path("checkpoints")
FIGURE_DIR = Path("figures")

# The script generates 16 candidate images for each GAN.
GAN_PREVIEW_COUNT = 16

# IMPORTANT:
# Inspect the generated GAN preview images first.
# Change these indices to select two visually contrasting samples.
# The default indices (0, 15) are only initial selections.
GAN_PAIRS = {
    "MLP GAN": (9, 2),
    "DCGAN": (9, 0),
}

CLASS_NAMES = {
    1: "Trouser",
    9: "Ankle boot",
}


# --------------------------------------------------
# Checkpoint loading
# --------------------------------------------------

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Support raw state_dict and wrapped checkpoints.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded: {checkpoint_path}")

    return model


# --------------------------------------------------
# Find two contrasting real test images
# --------------------------------------------------

def get_test_image_pair(test_loader, device):
    """
    Find the first Trouser (label 1) and
    the first Ankle boot (label 9) in the test set.
    """

    selected_images = {}

    for images, labels in test_loader:
        for i, label in enumerate(labels.tolist()):
            if label in CLASS_NAMES and label not in selected_images:
                selected_images[label] = images[i:i + 1]

        if len(selected_images) == len(CLASS_NAMES):
            break

    if len(selected_images) != len(CLASS_NAMES):
        raise RuntimeError(
            "Could not find both required classes in the test dataset."
        )

    image_a = selected_images[1].to(device)
    image_b = selected_images[9].to(device)

    print("\nSelected real test images:")
    print("Image A: Trouser (label 1)")
    print("Image B: Ankle boot (label 9)")

    return image_a, image_b


# --------------------------------------------------
# Generate GAN candidate samples
# --------------------------------------------------

def generate_gan_candidates(generator, device, model_name):
    """
    Generate 16 images using seeded Gaussian noise.
    Save a preview grid with the sample indices.
    """

    torch.manual_seed(SEED)

    noise = torch.randn(
        GAN_PREVIEW_COUNT,
        LATENT_DIM_GAN,
        device=device
    )

    with torch.inference_mode():
        generated = generator(noise)
        generated = ((generated + 1) / 2).clamp(0, 1)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):
        image = generated[i].squeeze().cpu().numpy()

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Index {i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{model_name} - 16 Seeded Candidate Samples (Seed={SEED})",
        fontsize=12
    )

    plt.tight_layout()
    preview_path = FIGURE_DIR / (
        f"{model_name.lower().replace(' ', '_')}_interpolation_candidates.png"
    )
    plt.savefig(preview_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved GAN candidate preview: {preview_path}")

    return noise


# --------------------------------------------------
# Latent interpolation
# --------------------------------------------------

def interpolate_latents(z_a, z_b, alphas, device):
    """
    Linear interpolation between two latent vectors.

    z(alpha) = (1 - alpha) * z_A + alpha * z_B
    """

    alpha_tensor = torch.tensor(
        alphas,
        dtype=z_a.dtype,
        device=device
    ).unsqueeze(1)

    z = (
        (1 - alpha_tensor) * z_a
        + alpha_tensor * z_b
    )

    return z


def interpolate_autoencoder(model, image_a, image_b, device):
    """
    Encode two real images, interpolate their latent vectors,
    and decode the interpolated vectors.
    """

    z_a, _ = model.encode(image_a)
    z_b, _ = model.encode(image_b)

    z = interpolate_latents(
        z_a,
        z_b,
        ALPHAS,
        device
    )

    generated = model.decode(z)

    return generated.clamp(0, 1)


def interpolate_vae(model, image_a, image_b, device):
    """
    Use the posterior means (mu) as deterministic latent vectors.
    """

    mu_a, _ = model.encode(image_a)
    mu_b, _ = model.encode(image_b)

    z = interpolate_latents(
        mu_a,
        mu_b,
        ALPHAS,
        device
    )

    generated = model.decode(z)

    return generated.clamp(0, 1)


def interpolate_gan(generator, noise, pair_indices, device):
    """
    Interpolate between two seeded GAN noise vectors.
    """

    index_a, index_b = pair_indices

    if not (
        0 <= index_a < len(noise)
        and 0 <= index_b < len(noise)
    ):
        raise ValueError(
            f"GAN indices must be between 0 and {len(noise) - 1}."
        )

    z_a = noise[index_a:index_a + 1]
    z_b = noise[index_b:index_b + 1]

    z = interpolate_latents(
        z_a,
        z_b,
        ALPHAS,
        device
    )

    generated = generator(z)

    # GAN generators output images in [-1, 1].
    generated = ((generated + 1) / 2).clamp(0, 1)

    return generated


# --------------------------------------------------
# Plot final interpolation figure
# --------------------------------------------------

def save_interpolation_figure(results):
    """
    Create a 5-row x 6-column interpolation grid.
    """

    row_names = [
        "Vanilla AE",
        "Sparse AE",
        "VAE",
        "MLP GAN",
        "DCGAN",
    ]

    fig, axes = plt.subplots(
        5,
        6,
        figsize=(12, 10)
    )

    for row, model_name in enumerate(row_names):
        images = results[model_name]

        for col in range(6):
            image = images[col].squeeze().cpu().numpy()

            axes[row, col].imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            if row == 0:
                axes[row, col].set_title(
                    f"α = {ALPHAS[col]:.1f}",
                    fontsize=10
                )

        axes[row, 0].set_ylabel(
            model_name,
            rotation=0,
            labelpad=65,
            fontsize=10,
            va="center"
        )

    fig.suptitle(
        "Latent Space Interpolation Across Generative Models",
        fontsize=14,
        y=1.01
    )

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURE_DIR / "fig4_interpolation.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nFinal interpolation figure saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate latent interpolation comparisons."
    )

    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Generate GAN candidate previews without creating Figure 4."
    )

    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    print("=" * 65)
    print("LATENT SPACE INTERPOLATION")
    print("=" * 65)

    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Interpolation values: {ALPHAS}")

    # Load test data.
    _, _, test_loader = build_dataloaders()

    image_a, image_b = get_test_image_pair(
        test_loader,
        device
    )

    # --------------------------------------------------
    # Load AE-family models
    # --------------------------------------------------

    print("\nLoading autoencoder models...")

    vanilla_ae = load_checkpoint(
        Autoencoder(latent_dim=LATENT_DIM_AE),
        CHECKPOINT_DIR / "vanilla_ae_best.pth",
        device
    )

    sparse_ae = load_checkpoint(
        SparseAutoencoder(latent_dim=LATENT_DIM_AE),
        CHECKPOINT_DIR / "sparse_ae_lambda0.01_best.pth",
        device
    )

    vae = load_checkpoint(
        VAE(latent_dim=LATENT_DIM_AE),
        CHECKPOINT_DIR / "vae_best.pth",
        device
    )

    # --------------------------------------------------
    # Load GAN generators
    # --------------------------------------------------

    print("\nLoading GAN generators...")

    mlp_generator = load_checkpoint(
        MLPGenerator(latent_dim=LATENT_DIM_GAN),
        CHECKPOINT_DIR / "mlp_gan_generator.pth",
        device
    )

    dcgan_generator = load_checkpoint(
        DCGANGenerator(latent_dim=LATENT_DIM_GAN),
        CHECKPOINT_DIR / "dcgan_gan_generator.pth",
        device
    )

    # --------------------------------------------------
    # Generate GAN candidate previews
    # --------------------------------------------------

    print("\nGenerating GAN candidate samples...")

    mlp_noise = generate_gan_candidates(
        mlp_generator,
        device,
        "MLP GAN"
    )

    dcgan_noise = generate_gan_candidates(
        dcgan_generator,
        device,
        "DCGAN"
    )

    if args.preview_only:
        print("\nPreview-only mode completed.")
        print(
            "Inspect the candidate grids, update GAN_PAIRS "
            "in src/interpolation.py, then run without --preview-only."
        )
        return

    # --------------------------------------------------
    # Perform interpolation
    # --------------------------------------------------

    results = {}

    with torch.inference_mode():

        print("\nInterpolating Vanilla AE...")
        results["Vanilla AE"] = interpolate_autoencoder(
            vanilla_ae,
            image_a,
            image_b,
            device
        )

        print("Interpolating Sparse AE...")
        results["Sparse AE"] = interpolate_autoencoder(
            sparse_ae,
            image_a,
            image_b,
            device
        )

        print("Interpolating VAE...")
        results["VAE"] = interpolate_vae(
            vae,
            image_a,
            image_b,
            device
        )

        print("Interpolating MLP GAN...")
        results["MLP GAN"] = interpolate_gan(
            mlp_generator,
            mlp_noise,
            GAN_PAIRS["MLP GAN"],
            device
        )

        print("Interpolating DCGAN...")
        results["DCGAN"] = interpolate_gan(
            dcgan_generator,
            dcgan_noise,
            GAN_PAIRS["DCGAN"],
            device
        )

    # --------------------------------------------------
    # Save Figure 4
    # --------------------------------------------------

    save_interpolation_figure(results)

    print("\nGAN interpolation selections:")
    for model_name, indices in GAN_PAIRS.items():
        print(f"{model_name}: indices {indices}")

    print("\nInterpolation completed successfully.")


if __name__ == "__main__":
    main()
