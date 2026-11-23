import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_dataloaders
from src.models.autoencoders import (
    Autoencoder,
    SparseAutoencoder,
    VAE
)
from src.models.gan import (
    Generator as MLPGenerator,
    Discriminator as MLPDiscriminator
)
from src.models.dcgan import (
    Generator as DCGANGenerator,
    Discriminator as DCGANDiscriminator,
    weights_init
)
from src.models.ddpm import UNet
from src.diffusion import Diffusion
from src.utils import set_seed, get_device


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 64
GAN_LATENT_DIM = 100

GAN_LEARNING_RATE = 2e-4
GAN_BETAS = (0.5, 0.999)

GAN_TRAIN_EPOCHS = 2

WARMUP_RUNS = 5
INFERENCE_RUNS = 20

DDPM_WARMUP_RUNS = 1
DDPM_INFERENCE_RUNS = 3

NOISE_STEPS = 1000


# ============================================================
# Utility functions
# ============================================================

def synchronize(device):
    """Wait for CUDA operations to finish before timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load a model checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"Loaded: {checkpoint_path}")

    return model


def measure_latency(
    inference_function,
    device,
    warmup_runs,
    timed_runs
):
    """
    Measure average inference latency.

    Warm-up runs are excluded from the measurement.
    CUDA operations are synchronized around every timer.
    """

    # Warm-up.
    for _ in range(warmup_runs):

        inference_function()
        synchronize(device)

    elapsed_times = []

    # Actual measurements.
    for _ in range(timed_runs):

        synchronize(device)

        start_time = time.perf_counter()

        inference_function()

        synchronize(device)

        elapsed_time = time.perf_counter() - start_time

        elapsed_times.append(elapsed_time)

    average_seconds = sum(elapsed_times) / len(elapsed_times)

    average_ms = average_seconds * 1000

    return {
        "average_ms": average_ms,
        "average_seconds": average_seconds,
        "runs": timed_runs,
        "warmup_runs": warmup_runs
    }


# ============================================================
# Model loading
# ============================================================

def load_models(device):

    print("\nLoading models...")

    models = {}

    # Vanilla AE
    models["vanilla_ae"] = load_checkpoint(
        Autoencoder(latent_dim=64),
        "checkpoints/vanilla_ae_best.pth",
        device
    )

    # Sparse AE
    models["sparse_ae"] = load_checkpoint(
        SparseAutoencoder(latent_dim=64),
        "checkpoints/sparse_ae_lambda0.01_best.pth",
        device
    )

    # VAE
    models["vae"] = load_checkpoint(
        VAE(latent_dim=64),
        "checkpoints/vae_best.pth",
        device
    )

    # MLP GAN Generator
    models["mlp_gan"] = load_checkpoint(
        MLPGenerator(latent_dim=GAN_LATENT_DIM),
        "checkpoints/mlp_gan_generator.pth",
        device
    )

    # DCGAN Generator
    models["dcgan"] = load_checkpoint(
        DCGANGenerator(latent_dim=GAN_LATENT_DIM),
        "checkpoints/dcgan_gan_generator.pth",
        device
    )

    # DDPM U-Net
    models["ddpm"] = load_checkpoint(
        UNet(time_dim=128),
        "checkpoints/ddpm_best.pth",
        device
    )

    return models


# ============================================================
# Inference benchmarking
# ============================================================

def benchmark_autoencoder(model, images, device):
    """
    Measure AE / Sparse AE / VAE reconstruction latency.

    Forward pass includes encoding and decoding.
    """

    model.eval()

    def inference():
        model(images)

    return measure_latency(
        inference_function=inference,
        device=device,
        warmup_runs=WARMUP_RUNS,
        timed_runs=INFERENCE_RUNS
    )


def benchmark_gan(generator, noise, device):
    """
    Measure GAN generation latency.

    Noise is prepared outside the timer.
    Only the generator forward pass is measured.
    """

    generator.eval()

    def inference():
        generator(noise)

    return measure_latency(
        inference_function=inference,
        device=device,
        warmup_runs=WARMUP_RUNS,
        timed_runs=INFERENCE_RUNS
    )


def benchmark_ddpm(model, diffusion, device):
    """
    Measure DDPM generation latency.

    Each inference starts from Gaussian noise and performs
    all 1000 reverse diffusion steps.
    """

    model.eval()

    def inference():
        diffusion.sample(
            model,
            n=BATCH_SIZE
        )

    return measure_latency(
        inference_function=inference,
        device=device,
        warmup_runs=DDPM_WARMUP_RUNS,
        timed_runs=DDPM_INFERENCE_RUNS
    )


def benchmark_all_inference(models, test_loader, device):

    print("\n" + "=" * 65)
    print("INFERENCE LATENCY BENCHMARK")
    print("=" * 65)

    # Use a fixed batch of 64 real test images.
    images, _ = next(iter(test_loader))

    images = images[:BATCH_SIZE].to(device)

    if images.shape[0] != BATCH_SIZE:
        raise ValueError(
            f"Expected {BATCH_SIZE} test images, "
            f"got {images.shape[0]}."
        )

    # Fixed Gaussian noise for GANs.
    set_seed(SEED)

    gan_noise = torch.randn(
        BATCH_SIZE,
        GAN_LATENT_DIM,
        device=device
    )

    results = {}

    with torch.inference_mode():

        # ----------------------------------------------------
        # Vanilla AE
        # ----------------------------------------------------

        print("\nBenchmarking Vanilla AE...")

        results["vanilla_ae"] = benchmark_autoencoder(
            models["vanilla_ae"],
            images,
            device
        )

        # ----------------------------------------------------
        # Sparse AE
        # ----------------------------------------------------

        print("\nBenchmarking Sparse AE...")

        results["sparse_ae"] = benchmark_autoencoder(
            models["sparse_ae"],
            images,
            device
        )

        # ----------------------------------------------------
        # VAE
        # ----------------------------------------------------

        print("\nBenchmarking VAE...")

        results["vae"] = benchmark_autoencoder(
            models["vae"],
            images,
            device
        )

        # ----------------------------------------------------
        # MLP GAN
        # ----------------------------------------------------

        print("\nBenchmarking MLP GAN...")

        results["mlp_gan"] = benchmark_gan(
            models["mlp_gan"],
            gan_noise,
            device
        )

        # ----------------------------------------------------
        # DCGAN
        # ----------------------------------------------------

        print("\nBenchmarking DCGAN...")

        results["dcgan"] = benchmark_gan(
            models["dcgan"],
            gan_noise,
            device
        )

        # ----------------------------------------------------
        # DDPM
        # ----------------------------------------------------

        print("\nBenchmarking DDPM (1000 reverse steps)...")

        diffusion = Diffusion(
            noise_steps=NOISE_STEPS,
            device=device
        )

        results["ddpm"] = benchmark_ddpm(
            models["ddpm"],
            diffusion,
            device
        )

    return results


# ============================================================
# GAN training time benchmark
# ============================================================

def build_gan_training_models(model_name, device):
    """Initialize fresh GAN models for training-time measurement."""

    if model_name == "mlp_gan":

        generator = MLPGenerator(
            latent_dim=GAN_LATENT_DIM
        )

        discriminator = MLPDiscriminator()

    elif model_name == "dcgan":

        generator = DCGANGenerator(
            latent_dim=GAN_LATENT_DIM
        )

        discriminator = DCGANDiscriminator()

        # Use the same DCGAN initialization as training.
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    else:
        raise ValueError(
            f"Unknown GAN model: {model_name}"
        )

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    return generator, discriminator


def train_one_gan_epoch(
    generator,
    discriminator,
    train_loader,
    optimizer_g,
    optimizer_d,
    criterion,
    device
):
    """Train one GAN epoch."""

    generator.train()
    discriminator.train()

    total_batches = len(train_loader)

    for batch_index, (images, _) in enumerate(train_loader):

        images = images.to(device)

        # Convert real images from [0, 1] to [-1, 1].
        real_images = images * 2 - 1

        current_batch = real_images.size(0)

        real_labels = torch.ones(
            current_batch,
            1,
            device=device
        )

        fake_labels = torch.zeros(
            current_batch,
            1,
            device=device
        )

        # ----------------------------------------------------
        # Train Discriminator
        # ----------------------------------------------------

        optimizer_d.zero_grad(set_to_none=True)

        # Real images.
        real_output = discriminator(real_images)

        real_loss = criterion(
            real_output,
            real_labels
        )

        # Fake images.
        noise = torch.randn(
            current_batch,
            GAN_LATENT_DIM,
            device=device
        )

        fake_images = generator(noise)

        fake_output = discriminator(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_output,
            fake_labels
        )

        d_loss = real_loss + fake_loss

        d_loss.backward()

        optimizer_d.step()

        # ----------------------------------------------------
        # Train Generator
        # ----------------------------------------------------

        optimizer_g.zero_grad(set_to_none=True)

        noise = torch.randn(
            current_batch,
            GAN_LATENT_DIM,
            device=device
        )

        fake_images = generator(noise)

        fake_output = discriminator(fake_images)

        g_loss = criterion(
            fake_output,
            real_labels
        )

        g_loss.backward()

        optimizer_g.step()

    return


def benchmark_gan_training(model_name, train_loader, device):
    """
    Measure training time per epoch for a GAN.

    Uses fresh model initialization and measures two epochs.
    """

    print("\n" + "-" * 65)
    print(f"Training time benchmark: {model_name}")
    print("-" * 65)

    # Reproducible initialization and training.
    set_seed(SEED)

    generator, discriminator = build_gan_training_models(
        model_name,
        device
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=GAN_LEARNING_RATE,
        betas=GAN_BETAS
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=GAN_LEARNING_RATE,
        betas=GAN_BETAS
    )

    criterion = nn.BCEWithLogitsLoss()

    epoch_times = []

    for epoch in range(1, GAN_TRAIN_EPOCHS + 1):

        synchronize(device)

        start_time = time.perf_counter()

        train_one_gan_epoch(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            criterion=criterion,
            device=device
        )

        synchronize(device)

        elapsed_time = time.perf_counter() - start_time

        epoch_times.append(elapsed_time)

        print(
            f"{model_name} | "
            f"Epoch {epoch}/{GAN_TRAIN_EPOCHS} | "
            f"Time: {elapsed_time:.2f} seconds"
        )

    average_epoch_time = sum(epoch_times) / len(epoch_times)

    print(
        f"Average {model_name} training time: "
        f"{average_epoch_time:.2f} seconds/epoch"
    )

    return {
        "epoch_times_seconds": epoch_times,
        "average_seconds_per_epoch": average_epoch_time,
        "measured_epochs": GAN_TRAIN_EPOCHS
    }


def benchmark_gan_training_times(train_loader, device):

    print("\n" + "=" * 65)
    print("GAN TRAINING TIME BENCHMARK")
    print("=" * 65)

    results = {}

    for model_name in ["mlp_gan", "dcgan"]:

        results[model_name] = benchmark_gan_training(
            model_name,
            train_loader,
            device
        )

    return results


# ============================================================
# Save results
# ============================================================

def save_benchmark_results(
    inference_results,
    training_results,
    device
):

    results_dir = Path("results")

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    benchmark_data = {
        "device": str(device),
        "inference_batch_size": BATCH_SIZE,
        "inference": inference_results,
        "gan_training": training_results
    }

    json_path = results_dir / "benchmark.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            benchmark_data,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # CSV output
    # --------------------------------------------------------

    csv_path = results_dir / "benchmark.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model",
            "inference_batch_size",
            "inference_latency_ms",
            "inference_latency_seconds",
            "inference_runs",
            "training_time_per_epoch_seconds"
        ])

        model_names = [
            "vanilla_ae",
            "sparse_ae",
            "vae",
            "mlp_gan",
            "dcgan",
            "ddpm"
        ]

        for model_name in model_names:

            inference = inference_results[model_name]

            training_time = ""

            if model_name in training_results:

                training_time = training_results[
                    model_name
                ]["average_seconds_per_epoch"]

            writer.writerow([
                model_name,
                BATCH_SIZE,
                round(inference["average_ms"], 4),
                round(inference["average_seconds"], 6),
                inference["runs"],
                (
                    round(training_time, 4)
                    if training_time != ""
                    else ""
                )
            ])

    print("\nBenchmark results saved:")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("GENERATIVE MODEL PERFORMANCE BENCHMARK")
    print("=" * 65)

    set_seed(SEED)

    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Inference batch size: {BATCH_SIZE}")
    print(f"GAN training epochs: {GAN_TRAIN_EPOCHS}")

    # Load dataset.
    train_loader, _, test_loader = build_dataloaders()

    # Load trained checkpoints for inference.
    models = load_models(device)

    # --------------------------------------------------------
    # Part 1: Inference latency
    # --------------------------------------------------------

    inference_results = benchmark_all_inference(
        models,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Part 2: GAN training time
    # --------------------------------------------------------

    training_results = benchmark_gan_training_times(
        train_loader,
        device
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_benchmark_results(
        inference_results,
        training_results,
        device
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 65)

    print("\nInference latency (batch size 64):")

    for model_name, result in inference_results.items():

        print(
            f"{model_name:15s}: "
            f"{result['average_ms']:.3f} ms "
            f"({result['average_seconds']:.4f} seconds)"
        )

    print("\nGAN training time:")

    for model_name, result in training_results.items():

        print(
            f"{model_name:15s}: "
            f"{result['average_seconds_per_epoch']:.2f} "
            f"seconds/epoch"
        )

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
