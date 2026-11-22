import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.utils import set_seed, get_device
from src.models.autoencoders import Autoencoder, SparseAutoencoder


def train_one_epoch(
    model, loader, criterion, optimizer, device, sparsity_lambda
):
    model.train()

    total_mse = 0.0
    total_loss = 0.0
    total_samples = 0

    for images, _ in loader:
        images = images.to(device)

        reconstruction, z, h = model(images)

        mse = criterion(reconstruction, images)
        loss = mse + sparsity_lambda * h.abs().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_mse += mse.item() * batch_size
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_mse / total_samples, total_loss / total_samples



def validate(
    model, loader, criterion, device, sparsity_lambda,
    threshold=0.01
):
    model.eval()

    total_mse = 0.0
    total_loss = 0.0
    total_samples = 0

    inactive_units = 0
    total_hidden_units = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            reconstruction, z, h = model(images)

            mse = criterion(reconstruction, images)
            loss = mse + sparsity_lambda * h.abs().mean()

            batch_size = images.size(0)

            total_mse += mse.item() * batch_size
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            inactive_units += (h < threshold).sum().item()
            total_hidden_units += h.numel()

    val_mse = total_mse / total_samples
    val_total = total_loss / total_samples
    val_sparsity = inactive_units / total_hidden_units

    return val_mse, val_total, val_sparsity



def main():
    parser = argparse.ArgumentParser(
        description="Train Vanilla or Sparse Autoencoder"
    )

    parser.add_argument(
        "--model",
        choices=["vanilla", "sparse"],
        default="vanilla"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity_lambda", type=float, default=1e-3)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.sparsity_lambda < 0:
        parser.error("--sparsity_lambda cannot be negative")

    set_seed(42)
    device = get_device()

    print("=" * 60)
    print(f"{args.model.upper()} AUTOENCODER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    # Only Sparse AE uses the L1 penalty
    sparsity_lambda = (
        args.sparsity_lambda if args.model == "sparse" else 0.0
    )
    print(f"Sparsity lambda: {sparsity_lambda}")

    # Load datasets
    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model
    if args.model == "sparse":
        model = SparseAutoencoder(latent_dim=64).to(device)
    else:
        model = Autoencoder(latent_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use unique filenames for each lambda experiment
    if args.model == "sparse":
        lambda_tag = f"{args.sparsity_lambda:g}"
        run_name = f"sparse_ae_lambda{lambda_tag}"
    else:
        run_name = "vanilla_ae"

    checkpoint_path = checkpoint_dir / f"{run_name}_best.pth"
    history_path = results_dir / f"{run_name}_history.csv"

    best_val_mse = float("inf")
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        train_mse, train_total = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            sparsity_lambda
        )

        val_mse, val_total, val_sparsity = validate(
            model,
            val_loader,
            criterion,
            device,
            sparsity_lambda
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "train_total_loss": train_total,
            "val_mse": val_mse,
            "val_total_loss": val_total,
            "val_hidden_sparsity": val_sparsity,
            "epoch_time": epoch_time
        })

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Train Total: {train_total:.6f} | "
            f"Val Total: {val_total:.6f} | "
            f"Val Sparsity: {val_sparsity:.2%} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Select checkpoint using validation MSE, not total loss
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoint_path)
            print("  -> Best model saved!")

    # Save CSV history
    fieldnames = [
        "epoch",
        "train_mse",
        "train_total_loss",
        "val_mse",
        "val_total_loss",
        "val_hidden_sparsity",
        "epoch_time"
    ]

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining completed!")
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()


