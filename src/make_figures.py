import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.models.autoencoders import (
    Autoencoder,
    SparseAutoencoder,
    VAE
)
from src.models.mlp_classifier import MLPClassifier
from src.models.gan import Generator as MLPGenerator
from src.models.dcgan import Generator as DCGANGenerator
from src.models.ddpm import UNet
from src.diffusion import Diffusion
from src.utils import set_seed, get_device


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

AE_LATENT_DIM = 64
GAN_LATENT_DIM = 100
DDPM_NOISE_STEPS = 1000

CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("results")
FIGURE_DIR = Path("figures")

TRAJECTORY_STEPS = (999, 750, 500, 250, 100)


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_state(model, path, device, preferred_keys=None):
    """
    Load either a raw state_dict or a wrapped checkpoint.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device
    )

    if preferred_keys is None:
        preferred_keys = [
            "model_state_dict",
            "state_dict",
            "model",
            "generator"
        ]

    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        for key in preferred_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded: {path}")

    return model


def get_first_test_images(test_loader, count=8):
    """
    Get a fixed set of the first test images.
    """

    images, labels = next(iter(test_loader))

    if len(images) < count:
        raise RuntimeError(
            "Not enough images in the test batch."
        )

    return images[:count], labels[:count]


def save_figure(fig, filename):
    """
    Save a figure at 300 DPI.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = FIGURE_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {path}")


def image_grid_array(images, nrow):
    """
    Convert a batch of grayscale images into a NumPy grid.

    Images must have shape [N, 1, 28, 28].
    """

    images = images.detach().cpu().clamp(0, 1)

    count = images.shape[0]

    ncol = min(nrow, count)

    nrows = (
        count + ncol - 1
    ) // ncol

    grid = torch.zeros(
        1,
        nrows * 28,
        ncol * 28
    )

    for i, image in enumerate(images):

        row = i // ncol
        col = i % ncol

        grid[
            :,
            row * 28:(row + 1) * 28,
            col * 28:(col + 1) * 28
        ] = image

    return grid.squeeze(0).numpy()


# ============================================================
# CSV HELPERS
# ============================================================

def load_csv(path):
    """
    Load a CSV history file.
    """

    if not path.exists():

        print(
            f"WARNING: History file not found: {path}"
        )

        return []

    with open(
        path,
        "r",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


def numeric_column(rows, candidates):
    """
    Find a CSV column using candidate names.

    Returns:
        (column_name, values)
    """

    if not rows:
        return None, []

    available = list(rows[0].keys())

    # Exact matching.
    for candidate in candidates:

        for column in available:

            if column.lower() == candidate.lower():

                values = []

                for row in rows:

                    try:
                        values.append(
                            float(row[column])
                        )

                    except (ValueError, TypeError):
                        values.append(np.nan)

                if any(np.isfinite(values)):
                    return column, values

    # Partial matching.
    for candidate in candidates:

        for column in available:

            if candidate.lower() in column.lower():

                values = []

                for row in rows:

                    try:
                        values.append(
                            float(row[column])
                        )

                    except (ValueError, TypeError):
                        values.append(np.nan)

                if any(np.isfinite(values)):
                    return column, values

    return None, []


def get_epochs(rows):

    if not rows:
        return []

    column, values = numeric_column(
        rows,
        ["epoch"]
    )

    if column is not None:
        return values

    return list(
        range(1, len(rows) + 1)
    )


def plot_history_lines(
    ax,
    path,
    series,
    title,
    ylabel
):
    """
    Plot training history curves.

    series:
        [(label, candidate_column_names), ...]
    """

    rows = load_csv(Path(path))

    if not rows:

        ax.set_title(title)

        ax.text(
            0.5,
            0.5,
            "History file not found",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        return

    epochs = get_epochs(rows)

    plotted = False

    for label, candidates in series:

        column, values = numeric_column(
            rows,
            candidates
        )

        if column is None:

            print(
                f"WARNING: Could not find {label} "
                f"in {path}."
            )

            print(
                "Available columns:",
                list(rows[0].keys())
            )

            continue

        ax.plot(
            epochs,
            values,
            label=label,
            linewidth=1.5
        )

        plotted = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(fontsize=8)


# ============================================================
# VAE LOSS PLOT
# ============================================================

def plot_vae_history(ax, path):
    """
    Plot VAE reconstruction and KL losses
    using separate y-axes.
    """

    rows = load_csv(Path(path))

    if not rows:

        ax.set_title("VAE Training Loss Components")

        ax.text(
            0.5,
            0.5,
            "History file not found",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        return

    epochs = get_epochs(rows)

    _, recon_values = numeric_column(
        rows,
        [
            "train_recon",
            "train_recon_loss",
            "train_bce",
            "recon_loss"
        ]
    )

    _, kl_values = numeric_column(
        rows,
        [
            "train_kl",
            "train_kl_loss",
            "kl_loss"
        ]
    )

    if recon_values:

        ax.plot(
            epochs,
            recon_values,
            label="Reconstruction",
            linewidth=1.5
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()

    if kl_values:

        ax2.plot(
            epochs,
            kl_values,
            label="KL Divergence",
            linestyle="--",
            linewidth=1.5,
            color="C1"
        )

    ax2.set_ylabel(
        "KL Divergence",
        color="C1"
    )

    lines1, labels1 = ax.get_legend_handles_labels()

    lines2, labels2 = ax2.get_legend_handles_labels()

    if lines1 or lines2:

        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
            fontsize=8
        )

    ax.set_title("VAE Training Loss Components")


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models(device):

    print("\nLoading trained models...")

    vanilla_ae = load_state(
        Autoencoder(
            latent_dim=AE_LATENT_DIM
        ),
        CHECKPOINT_DIR / "vanilla_ae_best.pth",
        device
    )

    sparse_ae = load_state(
        SparseAutoencoder(
            latent_dim=AE_LATENT_DIM
        ),
        CHECKPOINT_DIR / "sparse_ae_lambda0.01_best.pth",
        device
    )

    vae = load_state(
        VAE(
            latent_dim=AE_LATENT_DIM
        ),
        CHECKPOINT_DIR / "vae_best.pth",
        device
    )

    mlp_generator = load_state(
        MLPGenerator(
            latent_dim=GAN_LATENT_DIM
        ),
        CHECKPOINT_DIR / "mlp_gan_generator.pth",
        device
    )

    dcgan_generator = load_state(
        DCGANGenerator(
            latent_dim=GAN_LATENT_DIM
        ),
        CHECKPOINT_DIR / "dcgan_gan_generator.pth",
        device
    )

    classifier = load_state(
        MLPClassifier(),
        CHECKPOINT_DIR / "mlp_classifier_best.pth",
        device
    )

    unet = load_state(
        UNet(time_dim=128),
        CHECKPOINT_DIR / "ddpm_best.pth",
        device
    )

    diffusion = Diffusion(
        noise_steps=DDPM_NOISE_STEPS,
        device=device
    )

    return {
        "vanilla_ae": vanilla_ae,
        "sparse_ae": sparse_ae,
        "vae": vae,
        "mlp_generator": mlp_generator,
        "dcgan_generator": dcgan_generator,
        "classifier": classifier,
        "unet": unet,
        "diffusion": diffusion
    }


# ============================================================
# FIGURE 1: RECONSTRUCTION GRID
# ============================================================

def make_figure1(
    models,
    images,
    device
):

    print(
        "\nGenerating Figure 1: Reconstruction grid..."
    )

    images = images.to(device)

    with torch.inference_mode():

        vanilla_recon = models[
            "vanilla_ae"
        ](images)[0]

        sparse_recon = models[
            "sparse_ae"
        ](images)[0]

        # VAE uses mu in evaluation mode.
        vae_recon = models[
            "vae"
        ](images)[0]

    rows = [
        ("Original", images),
        ("Vanilla AE", vanilla_recon),
        ("Sparse AE", sparse_recon),
        ("VAE", vae_recon)
    ]

    fig, axes = plt.subplots(
        4,
        8,
        figsize=(13, 7)
    )

    for row_index, (row_name, batch) in enumerate(rows):

        for col in range(8):

            axes[row_index, col].imshow(
                batch[col].squeeze().cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])

        axes[row_index, 0].set_ylabel(
            row_name,
            rotation=0,
            labelpad=65,
            va="center",
            fontsize=10
        )

    fig.suptitle(
        "Reconstruction Comparison",
        fontsize=14
    )

    plt.tight_layout()

    save_figure(
        fig,
        "fig1_reconstruction.png"
    )


# ============================================================
# FIGURE 2: RANDOM GENERATIVE SAMPLES
# ============================================================

def make_figure2(
    models,
    device
):

    print(
        "\nGenerating Figure 2: Random samples..."
    )

    # VAE
    torch.manual_seed(SEED)

    vae_z = torch.randn(
        16,
        AE_LATENT_DIM,
        device=device
    )

    with torch.inference_mode():

        vae_samples = models[
            "vae"
        ].decode(vae_z).clamp(0, 1)

    # MLP GAN
    torch.manual_seed(SEED)

    mlp_noise = torch.randn(
        16,
        GAN_LATENT_DIM,
        device=device
    )

    with torch.inference_mode():

        mlp_samples = models[
            "mlp_generator"
        ](mlp_noise)

        mlp_samples = (
            (mlp_samples + 1) / 2
        ).clamp(0, 1)

    # DCGAN
    torch.manual_seed(SEED)

    dcgan_noise = torch.randn(
        16,
        GAN_LATENT_DIM,
        device=device
    )

    with torch.inference_mode():

        dcgan_samples = models[
            "dcgan_generator"
        ](dcgan_noise)

        dcgan_samples = (
            (dcgan_samples + 1) / 2
        ).clamp(0, 1)

    # DDPM
    print(
        "Generating 16 DDPM samples. "
        "This may take some time..."
    )

    torch.manual_seed(SEED)

    with torch.inference_mode():

        ddpm_samples = models[
            "diffusion"
        ].sample(
            models["unet"],
            n=16
        )

    panels = [
        ("VAE", vae_samples),
        ("MLP GAN", mlp_samples),
        ("DCGAN", dcgan_samples),
        ("DDPM", ddpm_samples)
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12)
    )

    for ax, (name, samples) in zip(
        axes.flat,
        panels
    ):

        ax.imshow(
            image_grid_array(
                samples,
                nrow=4
            ),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            name,
            fontsize=12
        )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Random Samples from Generative Models",
        fontsize=15
    )

    plt.tight_layout()

    save_figure(
        fig,
        "fig2_random_samples.png"
    )


# ============================================================
# FIGURE 3: MLP REPRESENTATION LIMITATIONS
# ============================================================

class FrozenLatentDecoder(nn.Module):
    """
    Decoder trained using frozen MLP latent features.

    Architecture must match the decoder used during
    frozen decoder training.
    """

    def __init__(self):

        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.decoder(z)

        return x.view(
            -1,
            1,
            28,
            28
        )


def make_figure3(
    models,
    images,
    device
):

    print(
        "\nGenerating Figure 3: MLP limitations..."
    )

    images = images.to(device)

    frozen_decoder = load_state(
        FrozenLatentDecoder(),
        CHECKPOINT_DIR / "frozen_decoder_best.pth",
        device
    )

    classifier = models["classifier"]

    with torch.inference_mode():

        # Tahap 3: extract MLP latent vectors.
        _, z_mlp = classifier(images)

        # Tahap 2: transpose class-vector projection.
        class_vectors = torch.eye(
            10,
            device=device
        )

        class_z = torch.relu(
            class_vectors @ classifier.fc3.weight
        )

        class_h = torch.relu(
            class_z @ classifier.fc2.weight
        )

        class_projection = torch.sigmoid(
            class_h @ classifier.fc1.weight
        ).view(
            -1,
            1,
            28,
            28
        )

        # Tahap 3: transpose test latent projection.
        projected_h = torch.relu(
            z_mlp @ classifier.fc2.weight
        )

        test_projection = torch.sigmoid(
            projected_h @ classifier.fc1.weight
        ).view(
            -1,
            1,
            28,
            28
        )

        # Vanilla AE reconstruction.
        vanilla_recon = models[
            "vanilla_ae"
        ](images)[0]

        # Frozen decoder reconstruction.
        frozen_recon = frozen_decoder(z_mlp)

        # Tahap 5: Gaussian samples from MLP latent features.
        _, _, test_loader = build_dataloaders()

        all_latents = []

        for batch_images, _ in test_loader:

            batch_images = batch_images.to(device)

            _, batch_z = classifier(batch_images)

            all_latents.append(batch_z)

        all_latents = torch.cat(
            all_latents,
            dim=0
        )

        latent_mu = all_latents.mean(dim=0)

        latent_sigma = all_latents.std(
            dim=0,
            unbiased=False
        )

        torch.manual_seed(SEED)

        random_z = latent_mu + latent_sigma * torch.randn(
            64,
            64,
            device=device
        )

        gaussian_samples = frozen_decoder(random_z)

    # Prepare four panels.
    original_grid = image_grid_array(
        images,
        nrow=4
    )

    # Both grids have 5 columns and equal widths.
    class_grid = image_grid_array(
        class_projection,
        nrow=5
    )

    test_projection_grid = image_grid_array(
        test_projection,
        nrow=5
    )

    combined_projection = np.vstack([
        class_grid,
        np.ones(
            (5, class_grid.shape[1])
        ),
        test_projection_grid
    ])

    comparison_images = torch.cat([
        images,
        vanilla_recon,
        frozen_recon
    ], dim=0)

    comparison_grid = image_grid_array(
        comparison_images,
        nrow=8
    )

    gaussian_grid = image_grid_array(
        gaussian_samples,
        nrow=8
    )

    # Plot Figure 3.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11)
    )

    panel_data = [
        (
            axes[0, 0],
            original_grid,
            "Original Test Images"
        ),
        (
            axes[0, 1],
            combined_projection,
            "Wᵀ Projections\n"
            "Top: Class vectors | Bottom: Test latents"
        ),
        (
            axes[1, 0],
            comparison_grid,
            "Reconstruction Comparison\n"
            "Rows: Original | Vanilla AE | Frozen Decoder"
        ),
        (
            axes[1, 1],
            gaussian_grid,
            "Gaussian Samples from MLP Latent Distribution"
        )
    ]

    for ax, grid, title in panel_data:

        ax.imshow(
            grid,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            title,
            fontsize=10
        )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Limitations of MLP Latent Representations",
        fontsize=14
    )

    plt.tight_layout()

    save_figure(
        fig,
        "fig3_mlp_limitations.png"
    )


# ============================================================
# FIGURE 5: TRAINING CURVES AND DDPM TRAJECTORY
# ============================================================

def make_figure5(
    models,
    device
):

    print(
        "\nGenerating Figure 5: Loss curves and DDPM trajectory..."
    )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(15, 19)
    )

    # Vanilla AE
    plot_history_lines(
        axes[0, 0],
        RESULT_DIR / "vanilla_ae_history.csv",
        [
            (
                "Train MSE",
                [
                    "train_mse",
                    "train_loss",
                    "train_recon"
                ]
            ),
            (
                "Validation MSE",
                [
                    "val_mse",
                    "val_loss",
                    "val_recon"
                ]
            )
        ],
        "Vanilla AE",
        "MSE"
    )

    # Sparse AE
    plot_history_lines(
        axes[0, 1],
        RESULT_DIR / "sparse_ae_lambda0.01_history.csv",
        [
            (
                "Train MSE",
                [
                    "train_mse",
                    "train_recon",
                    "train_loss"
                ]
            ),
            (
                "Validation MSE",
                [
                    "val_mse",
                    "val_recon",
                    "val_loss"
                ]
            )
        ],
        "Sparse AE",
        "MSE"
    )

    # VAE
    plot_vae_history(
        axes[1, 0],
        RESULT_DIR / "vae_history.csv"
    )

    # MLP GAN
    plot_history_lines(
        axes[1, 1],
        RESULT_DIR / "mlp_gan_history.csv",
        [
            (
                "Discriminator",
                ["d_loss"]
            ),
            (
                "Generator",
                ["g_loss"]
            )
        ],
        "MLP GAN",
        "BCE Loss"
    )

    # DCGAN
    plot_history_lines(
        axes[2, 0],
        RESULT_DIR / "dcgan_gan_history.csv",
        [
            (
                "Discriminator",
                ["d_loss"]
            ),
            (
                "Generator",
                ["g_loss"]
            )
        ],
        "DCGAN",
        "BCE Loss"
    )

    # DDPM
    plot_history_lines(
        axes[2, 1],
        RESULT_DIR / "ddpm_history.csv",
        [
            (
                "Train Noise MSE",
                ["train_loss"]
            ),
            (
                "Validation Noise MSE",
                ["val_loss"]
            )
        ],
        "DDPM",
        "Noise Prediction MSE"
    )

    # DDPM denoising trajectory.
    print(
        "Generating DDPM denoising trajectory..."
    )

    torch.manual_seed(SEED)

    with torch.inference_mode():

        _, trajectory = models[
            "diffusion"
        ].sample(
            models["unet"],
            n=4,
            return_trajectory=True,
            trajectory_steps=TRAJECTORY_STEPS
        )

    trajectory_order = [
        999,
        750,
        500,
        250,
        100,
        0
    ]

    trajectory_images = []

    for timestep in trajectory_order:

        if timestep not in trajectory:

            raise RuntimeError(
                f"DDPM trajectory is missing timestep {timestep}. "
                f"Available steps: {list(trajectory.keys())}"
            )

        trajectory_images.append(
            trajectory[timestep]
        )

    trajectory_grid = torch.cat(
        trajectory_images,
        dim=0
    )

    trajectory_array = image_grid_array(
        trajectory_grid,
        nrow=4
    )

    axes[3, 0].imshow(
        trajectory_array,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3, 0].set_title(
        "DDPM Denoising Trajectory\n"
        "Rows: 999, 750, 500, 250, 100, 0"
    )

    axes[3, 0].set_xticks([])
    axes[3, 0].set_yticks([])

    axes[3, 0].set_ylabel(
        "4 samples per timestep",
        fontsize=9
    )

    axes[3, 1].axis("off")

    axes[3, 1].text(
        0.05,
        0.85,
        "DDPM Reverse Diffusion",
        fontsize=13,
        weight="bold"
    )

    axes[3, 1].text(
        0.05,
        0.65,
        "Each row contains 4 generated images.\n\n"
        "The rows show the denoising process from\n"
        "initial Gaussian noise to final images.\n\n"
        "Timesteps:\n"
        "999 → 750 → 500 → 250 → 100 → 0",
        fontsize=11,
        va="top"
    )

    fig.suptitle(
        "Training Losses and DDPM Denoising Trajectory",
        fontsize=15
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    save_figure(
        fig,
        "fig5_training_and_trajectory.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 65)
    print("FINAL GENERATIVE MODEL FIGURES")
    print("=" * 65)

    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # Load test dataset.
    _, _, test_loader = build_dataloaders()

    images, labels = get_first_test_images(
        test_loader,
        count=8
    )

    print("\nSelected first 8 test images.")

    print(
        "Labels:",
        labels.tolist()
    )

    # Load trained models.
    models = load_models(device)

    # Generate figures.
    make_figure1(
        models,
        images,
        device
    )

    make_figure2(
        models,
        device
    )

    make_figure3(
        models,
        images,
        device
    )

    make_figure5(
        models,
        device
    )

    # Completion.
    print("\n" + "=" * 65)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 65)

    output_files = [
        "fig1_reconstruction.png",
        "fig2_random_samples.png",
        "fig3_mlp_limitations.png",
        "fig5_training_and_trajectory.png"
    ]

    for filename in output_files:

        print(
            FIGURE_DIR / filename
        )


if __name__ == "__main__":
    main()
