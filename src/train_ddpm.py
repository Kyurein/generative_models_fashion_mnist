import os
import csv
import time
import argparse

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.data import build_dataloaders
from src.models.ddpm import UNet
from src.diffusion import Diffusion
from src.utils import set_seed, get_device


def train_one_epoch(model, diffusion, train_loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, _ in train_loader:
        images = images.to(device)

        # Scale images from [0, 1] to [-1, 1]
        images = images * 2.0 - 1.0

        batch_size = images.size(0)

        # Randomly select a timestep for each image
        timesteps = torch.randint(
            0,
            diffusion.noise_steps,
            (batch_size,),
            device=device
        )

        # Generate Gaussian noise
        noise = torch.randn_like(images)

        # Forward diffusion: add noise to the original images
        noisy_images = diffusion.q_sample(
            images,
            timesteps,
            noise
        )

        # Predict the added noise
        predicted_noise = model(noisy_images, timesteps)

        # Calculate noise prediction loss
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate(model, diffusion, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, _ in val_loader:
        images = images.to(device)

        # Scale images from [0, 1] to [-1, 1]
        images = images * 2.0 - 1.0

        batch_size = images.size(0)

        timesteps = torch.randint(
            0,
            diffusion.noise_steps,
            (batch_size,),
            device=device
        )

        noise = torch.randn_like(images)

        noisy_images = diffusion.q_sample(
            images,
            timesteps,
            noise
        )

        predicted_noise = model(noisy_images, timesteps)

        loss = F.mse_loss(predicted_noise, noise)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_generated_samples(model, diffusion, epoch, device):
    model.eval()

    os.makedirs("figures/ddpm_samples", exist_ok=True)
    os.makedirs("figures/ddpm_trajectory", exist_ok=True)

    # Generate 16 images and record intermediate diffusion states
    with torch.no_grad():
        generated_images, trajectory = diffusion.sample(
            model,
            n=16,
            return_trajectory=True
        )

    # Save final generated images
    sample_path = (
        f"figures/ddpm_samples/ddpm_epoch_{epoch:03d}.png"
    )

    save_image(
        generated_images.cpu(),
        sample_path,
        nrow=4
    )

    print(f"Saved generated samples: {sample_path}")

    # Save intermediate reverse diffusion states
    for timestep, images in trajectory.items():
        trajectory_path = (
            f"figures/ddpm_trajectory/"
            f"epoch_{epoch:03d}_t_{timestep:03d}.png"
        )

        save_image(
            images,
            trajectory_path,
            nrow=4
        )

    print("Saved diffusion trajectory images.")


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        },
        path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a DDPM on Fashion-MNIST"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=10
    )

    parser.add_argument(
        "--noise-steps",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    # Reproducibility
    set_seed(42)

    # Select CUDA or CPU
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures/ddpm_samples", exist_ok=True)
    os.makedirs("figures/ddpm_trajectory", exist_ok=True)

    # Load Fashion-MNIST
    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize U-Net
    model = UNet(time_dim=128).to(device)

    # Initialize diffusion process
    diffusion = Diffusion(
        noise_steps=args.noise_steps,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_loss = float("inf")
    history = []

    print("\nStarting DDPM training...")
    print(f"Epochs: {args.epochs}")
    print(f"Noise steps: {args.noise_steps}")
    print(f"Learning rate: {args.lr}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()

        # Training
        train_loss = train_one_epoch(
            model,
            diffusion,
            train_loader,
            optimizer,
            device
        )

        # Validation
        val_loss = validate(
            model,
            diffusion,
            val_loader,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - start_time

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Record training history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time
        })

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                "checkpoints/ddpm_best.pth"
            )

            print(
                f"  Best model saved "
                f"(Val Loss: {best_val_loss:.6f})"
            )

        # Save periodic checkpoints every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = (
                f"checkpoints/ddpm_epoch_{epoch:03d}.pth"
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_path
            )

        # Save generated images every N epochs
        if (
            epoch % args.sample_every == 0
            or epoch == args.epochs
        ):
            save_generated_samples(
                model,
                diffusion,
                epoch,
                device
            )

        # Save CSV history after every epoch
        history_path = "results/ddpm_history.csv"

        with open(
            history_path,
            "w",
            newline=""
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "epoch_time"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

    print("-" * 65)
    print("DDPM training completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Best checkpoint: checkpoints/ddpm_best.pth")
    print("Training history: results/ddpm_history.csv")


if __name__ == "__main__":
    main()
