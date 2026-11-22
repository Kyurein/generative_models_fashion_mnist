import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data import build_dataloaders
from src.models.autoencoders import VAE
from src.utils import get_device, set_seed



def vae_loss(reconstruction, images, mu, logvar):
    """
    Calculate BCE reconstruction loss, KL divergence, and total VAE loss.

    Both losses are summed over their feature dimensions
    and averaged over the batch.
    """
    # BCE reconstruction loss: sum over image dimensions,
    # then average over the batch.
    recon_loss = F.binary_cross_entropy(
        reconstruction,
        images,
        reduction="none"
    ).flatten(start_dim=1).sum(dim=1).mean()

    # KL divergence: sum over latent dimensions,
    # then average over the batch.
    kl_loss = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=1).mean()

    total_loss = recon_loss + kl_loss

    return recon_loss, kl_loss, total_loss


def run_epoch(model, loader, optimizer, device, training):
    """Run one training or validation epoch."""

    if training:
        model.train()
    else:
        model.eval()

    total_recon = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, _ in loader:
            images = images.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction, mu, logvar = model(images)

            recon_loss, kl_loss, loss = vae_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)

            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_loss += loss.item() * batch_size

            # Pixel-wise MSE for comparison with Vanilla and Sparse AE
            batch_mse = F.mse_loss(
                reconstruction,
                images,
                reduction="mean"
            )

            total_mse += batch_mse.item() * batch_size
            total_samples += batch_size

    return {
        "recon": total_recon / total_samples,
        "kl": total_kl / total_samples,
        "total": total_loss / total_samples,
        "mse": total_mse / total_samples
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder on Fashion-MNIST"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # Reproducibility
    set_seed(42)

    device = get_device()
    print(f"Device: {device}")

    # Load Fashion-MNIST
    train_loader, val_loader, _ = build_dataloaders()

    # Initialize model and optimizer
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Output directories
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "vae_best.pth"
    history_path = results_dir / "vae_history.csv"

    best_val_total = float("inf")
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            training=False
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = time.time() - start_time

        # Record epoch results
        row = {
            "epoch": epoch,
            "train_recon": train_metrics["recon"],
            "train_kl": train_metrics["kl"],
            "train_total": train_metrics["total"],
            "val_recon": val_metrics["recon"],
            "val_kl": val_metrics["kl"],
            "val_total": val_metrics["total"],
            "val_mse": val_metrics["mse"],
            "epoch_seconds": epoch_seconds
        }

        history.append(row)

        # Save checkpoint based on validation total loss (ELBO)
        if val_metrics["total"] < best_val_total:
            best_val_total = val_metrics["total"]

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            best_marker = " (best)"
        else:
            best_marker = ""

        # Display training progress
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Recon: {train_metrics['recon']:.4f} | "
            f"Train KL: {train_metrics['kl']:.4f} | "
            f"Train Total: {train_metrics['total']:.4f} | "
            f"Val Recon: {val_metrics['recon']:.4f} | "
            f"Val KL: {val_metrics['kl']:.4f} | "
            f"Val Total: {val_metrics['total']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | "
            f"Time: {epoch_seconds:.2f}s"
            f"{best_marker}"
        )

    # Save training history to CSV
    with history_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest validation total loss: {best_val_total:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
