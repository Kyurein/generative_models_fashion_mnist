import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.utils import set_seed, get_device
from src.models.autoencoders import Autoencoder


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, _ in train_loader:
        images = images.to(device)

        reconstruction, z = model(images)

        loss = criterion(reconstruction, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            reconstruction, z = model(images)

            loss = criterion(reconstruction, images)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(
        description="Train Vanilla Autoencoder on Fashion-MNIST"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("=" * 55)
    print("VANILLA AUTOENCODER TRAINING")
    print("=" * 55)
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr}")

    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    model = Autoencoder(latent_dim=64).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    criterion = nn.MSELoss()

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time
        })

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_dir / "vanilla_ae_best.pth"
            )

            print("  -> Best model saved!")

    history_path = results_dir / "vanilla_ae_history.csv"

    with open(history_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "epoch_time"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved at       : {checkpoint_dir}")
    print(f"History saved at     : {history_path}")


if __name__ == "__main__":
    main()
