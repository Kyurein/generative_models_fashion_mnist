
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.data import build_dataloaders
from src.models.mlp_classifier import MLPClassifier
from src.utils import get_device, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the MLP for one epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples
    }


def evaluate(model, loader, criterion, device):
    """Evaluate the MLP on validation or test data."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on Fashion-MNIST"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    # Reproducibility
    set_seed(42)

    device = get_device()
    print(f"Device: {device}")

    # Load Fashion-MNIST datasets
    train_loader, val_loader, test_loader = build_dataloaders()

    # Initialize model, loss function, and optimizer
    model = MLPClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Create output directories
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "mlp_classifier_best.pth"
    history_path = results_dir / "mlp_classifier_history.csv"
    metrics_path = results_dir / "mlp_classifier_metrics.json"

    best_val_accuracy = -1.0
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        # Training
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = time.time() - start_time

        # Store epoch metrics
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": epoch_seconds
        }

        history.append(row)

        # Save checkpoint based on validation accuracy
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            best_marker = " (best)"
        else:
            best_marker = ""

        # Display progress
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Time: {epoch_seconds:.2f}s"
            f"{best_marker}"
        )

    # Save training history
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

    # Load the best checkpoint before final test evaluation
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print("\nLoaded best checkpoint for test evaluation.")

    # Evaluate the test set once
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    test_accuracy = test_metrics["accuracy"]

    # Save test accuracy and related information
    final_metrics = {
        "model": "MLPClassifier",
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "epochs": args.epochs,
        "learning_rate": args.lr
    }

    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(final_metrics, json_file, indent=4)

    # Final results
    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    if test_accuracy >= 0.88:
        print("Target achieved: test accuracy >= 88%.")
    else:
        print("Target not reached: test accuracy is below 88%.")

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Test metrics: {metrics_path}")


if __name__ == "__main__":
    main()
