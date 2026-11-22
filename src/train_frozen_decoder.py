import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

from src.data import build_dataloaders
from src.metrics import evaluate_reconstruction
from src.models.mlp_classifier import MLPClassifier
from src.utils import get_device, set_seed


class FrozenLatentDecoder(nn.Module):
    """Decoder trained using frozen MLP classifier representations."""

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


class ClassifierDecoderModel(nn.Module):
    """
    Wrapper combining the frozen classifier and trained decoder.

    This wrapper allows evaluate_reconstruction() to be reused.
    """

    def __init__(self, classifier, decoder):
        super().__init__()

        self.classifier = classifier
        self.decoder = decoder

    def forward(self, images):
        with torch.no_grad():
            _, z_mlp = self.classifier(images)

        reconstruction = self.decoder(z_mlp)

        return reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Train a decoder using frozen MLP latent representations"
    )

    parser.add_argument("--epochs", type=int, default=50)
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

    # Load Fashion-MNIST
    train_loader, val_loader, test_loader = build_dataloaders()

    # Load trained MLP classifier
    classifier = MLPClassifier().to(device)

    classifier_checkpoint = (
        Path("checkpoints") / "mlp_classifier_best.pth"
    )

    if not classifier_checkpoint.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_checkpoint}"
        )

    state_dict = torch.load(
        classifier_checkpoint,
        map_location=device,
        weights_only=True
    )

    classifier.load_state_dict(state_dict)
    classifier.eval()

    # Freeze all classifier parameters
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    print(f"Loaded classifier: {classifier_checkpoint}")
    print("Classifier parameters frozen.")

    # Initialize decoder
    decoder = FrozenLatentDecoder().to(device)

    # Only decoder parameters are optimized
    optimizer = Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Output directories
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    figures_dir = Path("figures")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / "frozen_decoder_best.pth"
    )

    history_path = (
        results_dir / "frozen_decoder_history.csv"
    )

    metrics_path = (
        results_dir / "frozen_decoder_metrics.json"
    )

    best_val_loss = float("inf")
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        # Train decoder
        decoder.train()

        train_loss_sum = 0.0
        train_samples = 0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Extract features without calculating classifier gradients
            with torch.no_grad():
                _, z_mlp = classifier(images)

            reconstruction = decoder(z_mlp)

            loss = criterion(reconstruction, images)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples

        # Validation
        decoder.eval()

        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)

                _, z_mlp = classifier(images)
                reconstruction = decoder(z_mlp)

                loss = criterion(reconstruction, images)

                batch_size = images.size(0)

                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

        val_loss = val_loss_sum / val_samples

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = time.time() - start_time

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds
        })

        # Select best decoder by validation MSE
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                decoder.state_dict(),
                checkpoint_path
            )

            best_marker = " (best)"
        else:
            best_marker = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
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

    # Load best decoder checkpoint
    decoder_state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    decoder.load_state_dict(decoder_state)
    decoder.eval()

    # Combine classifier and decoder for evaluation
    reconstruction_model = ClassifierDecoderModel(
        classifier,
        decoder
    ).to(device)

    reconstruction_model.eval()

    # Evaluate test reconstruction metrics
    test_metrics = evaluate_reconstruction(
        reconstruction_model,
        test_loader,
        device
    )

    # Save example reconstructions
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)

    with torch.no_grad():
        _, z_mlp = classifier(images)
        reconstructions = decoder(z_mlp)

    comparison = torch.cat(
        [images.cpu(), reconstructions.cpu()],
        dim=0
    )

    save_image(
        comparison,
        figures_dir / "frozen_decoder_reconstructions.png",
        nrow=8
    )

    # Save evaluation metrics
    final_metrics = {
        "model": "FrozenLatentDecoder",
        "test_mse": test_metrics["test_mse"],
        "test_psnr": test_metrics["test_psnr"],
        "test_ssim": test_metrics["test_ssim"],
        "best_val_loss": best_val_loss,
        "epochs": args.epochs,
        "learning_rate": args.lr
    }

    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(final_metrics, json_file, indent=4)

    # Display final results
    print("\nTraining completed.")
    print(f"Best validation MSE: {best_val_loss:.6f}")

    print("\nTest Reconstruction Metrics:")
    print(f"Test MSE:  {test_metrics['test_mse']:.6f}")
    print(f"Test PSNR: {test_metrics['test_psnr']:.4f} dB")
    print(f"Test SSIM: {test_metrics['test_ssim']:.4f}")

    print(f"\nBest decoder: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Test metrics: {metrics_path}")
    print(
        f"Reconstruction grid: "
        f"{figures_dir / 'frozen_decoder_reconstructions.png'}"
    )


if __name__ == "__main__":
    main()
