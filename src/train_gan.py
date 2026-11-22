import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from src.data import build_dataloaders
from src.models.gan import Generator, Discriminator
from src.utils import set_seed, get_device


def build_models(model_name, latent_dim, device):
    if model_name == "mlp":
        generator = Generator(latent_dim=latent_dim)
        discriminator = Discriminator()

    elif model_name == "dcgan":
        # This module will be created when implementing DCGAN.
        from src.models.dcgan import (
            Generator as DCGenerator,
            Discriminator as DCDiscriminator
        )

        generator = DCGenerator(latent_dim=latent_dim)
        discriminator = DCDiscriminator()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return generator.to(device), discriminator.to(device)


def save_sample_grid(generator, fixed_z, epoch, output_dir):
    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_z)
        fake_images = (fake_images + 1) / 2
        fake_images = fake_images.clamp(0, 1)

    output_path = output_dir / f"gan_epoch_{epoch:03d}.png"

    save_image(
        fake_images,
        output_path,
        nrow=8
    )

    generator.train()
    print(f"Saved samples: {output_path}")


def save_history(history, output_path):
    if not history:
        return

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)


def train(args):
    set_seed(42)
    device = get_device()

    print(f"Device: {device}")
    print(f"GAN model: {args.model}")

    train_loader, _, _ = build_dataloaders()

    generator, discriminator = build_models(
        args.model,
        args.latent_dim,
        device
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    # Fixed noise allows us to compare generated images over time.
    fixed_z = torch.randn(
        64,
        args.latent_dim,
        device=device
    )

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    figures_dir = Path("figures")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = figures_dir / f"{args.model}_gan_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        total_real_score = 0.0
        total_fake_score = 0.0
        total_samples = 0

        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Fashion-MNIST is [0, 1].
            # Tanh generator output is [-1, 1].
            real_images = images * 2 - 1

            real_targets = torch.ones(
                batch_size, 1, device=device
            )
            fake_targets = torch.zeros(
                batch_size, 1, device=device
            )

            # ---------------------------------
            # 1. Train Discriminator
            # ---------------------------------
            optimizer_D.zero_grad()

            z = torch.randn(
                batch_size,
                args.latent_dim,
                device=device
            )

            fake_images = generator(z)

            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images.detach())

            loss_D_real = criterion(real_logits, real_targets)
            loss_D_fake = criterion(fake_logits, fake_targets)

            loss_D = loss_D_real + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            # ---------------------------------
            # 2. Train Generator
            # ---------------------------------
            optimizer_G.zero_grad()

            # Recalculate discriminator output after D update.
            generated_logits = discriminator(fake_images)

            # Generator wants discriminator to classify
            # generated images as real.
            loss_G = criterion(
                generated_logits,
                real_targets
            )

            loss_G.backward()
            optimizer_G.step()

            # ---------------------------------
            # 3. Record metrics
            # ---------------------------------
            total_d_loss += loss_D.item() * batch_size
            total_g_loss += loss_G.item() * batch_size

            total_real_score += (
                torch.sigmoid(real_logits).mean().item()
                * batch_size
            )

            total_fake_score += (
                torch.sigmoid(fake_logits).mean().item()
                * batch_size
            )

            total_samples += batch_size

        epoch_d_loss = total_d_loss / total_samples
        epoch_g_loss = total_g_loss / total_samples
        epoch_real_score = total_real_score / total_samples
        epoch_fake_score = total_fake_score / total_samples

        history_row = {
            "epoch": epoch,
            "d_loss": epoch_d_loss,
            "g_loss": epoch_g_loss,
            "d_x": epoch_real_score,
            "d_g_z": epoch_fake_score
        }

        history.append(history_row)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"G Loss: {epoch_g_loss:.4f} | "
            f"D(x): {epoch_real_score:.4f} | "
            f"D(G(z)): {epoch_fake_score:.4f}"
        )

        # Save generated images every 5 epochs.
        if epoch % 5 == 0:
            save_sample_grid(
                generator,
                fixed_z,
                epoch,
                sample_dir
            )

        # Save history after every epoch in case training stops.
        save_history(
            history,
            results_dir / f"{args.model}_gan_history.csv"
        )

    # Save a final sample grid if the last epoch
    # was not already a multiple of five.
    if args.epochs % 5 != 0:
        save_sample_grid(
            generator,
            fixed_z,
            args.epochs,
            sample_dir
        )

    # Save both networks at the end.
    torch.save(
        generator.state_dict(),
        checkpoint_dir / f"{args.model}_gan_generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        checkpoint_dir / f"{args.model}_gan_discriminator.pth"
    )

    # Also save a combined checkpoint.
    torch.save(
        {
            "model": args.model,
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict()
        },
        checkpoint_dir / f"{args.model}_gan_final.pth"
    )

    print("\nGAN training completed.")
    print("Generator checkpoint saved.")
    print("Discriminator checkpoint saved.")
    print("Training history saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP GAN or DCGAN on Fashion-MNIST"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "dcgan"],
        default="mlp"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
