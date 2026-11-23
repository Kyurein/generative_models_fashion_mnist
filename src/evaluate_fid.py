import argparse
import json
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.data import build_dataloaders
from src.models.autoencoders import Autoencoder, SparseAutoencoder, VAE
from src.models.gan import Generator as MLPGenerator
from src.models.dcgan import Generator as DCGANGenerator
from src.models.ddpm import UNet
from src.diffusion import Diffusion
from src.utils import set_seed, get_device


LATENT_DIM = 64
GAN_LATENT_DIM = 100
SEED = 42

MODEL_NAMES = [
    "vanilla_ae",
    "sparse_ae",
    "vae",
    "mlp_gan",
    "dcgan",
    "ddpm",
]


def load_checkpoint(model, checkpoint_path, device):
    """
    Load either a raw state_dict or a training checkpoint dictionary.
    """

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

    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


def prepare_fid_images(images):
    """
    Convert grayscale [0, 1] images into RGB uint8 images [0, 255].

    TorchMetrics FID with normalize=False expects uint8 input.
    """

    images = images.clamp(0, 1)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images = (images * 255).round().to(torch.uint8)

    return images


def convert_gan_images(images):
    """Convert GAN output from [-1, 1] to [0, 1]."""

    return ((images + 1) / 2).clamp(0, 1)


def collect_real_images(test_loader, num_samples):
    """Collect a fixed set of real test images."""

    collected_images = []
    collected_count = 0

    for images, _ in test_loader:

        remaining = num_samples - collected_count

        if remaining <= 0:
            break

        batch = images[:remaining].cpu()

        collected_images.append(batch)
        collected_count += len(batch)

        if collected_count >= num_samples:
            break

    if collected_count < num_samples:
        raise ValueError(
            f"Requested {num_samples} real images, "
            f"but only collected {collected_count}."
        )

    return torch.cat(collected_images, dim=0)


def collect_ae_latents(model, train_loader, device):
    """
    Collect encoder latents from the training set.

    Used only for optional Gaussian-fit latent sampling.
    Real latent vectors are not directly decoded for FID.
    """

    all_latents = []

    model.eval()

    with torch.no_grad():

        for images, _ in train_loader:

            images = images.to(device)

            z, _ = model.encode(images)

            all_latents.append(z.cpu())

    return torch.cat(all_latents, dim=0)


def fit_gaussian_latent_distribution(latent_vectors):
    """
    Fit a diagonal Gaussian N(mu, sigma^2) to the latent vectors.
    """

    latent_mu = latent_vectors.mean(dim=0)

    latent_sigma = latent_vectors.std(
        dim=0,
        unbiased=False
    )

    latent_sigma = latent_sigma.clamp(min=1e-6)

    return latent_mu, latent_sigma


def generate_images(
    model_name,
    model,
    num_samples,
    batch_size,
    device,
    latent_distribution=None,
    diffusion=None
):
    """
    Generate images using the selected generative model.
    """

    generated_images = []

    model.eval()

    with torch.no_grad():

        for start in range(0, num_samples, batch_size):

            current_batch = min(
                batch_size,
                num_samples - start
            )

            # Display generation progress.
            print(
                f"\r{model_name}: "
                f"{start + current_batch}/{num_samples}",
                end="",
                flush=True
            )

            # Vanilla AE / Sparse AE:
            # Sample new vectors from a fitted Gaussian distribution.
            if model_name in ["vanilla_ae", "sparse_ae"]:

                if latent_distribution is None:
                    raise ValueError(
                        "AE Gaussian latent distribution was not provided."
                    )

                latent_mu, latent_sigma = latent_distribution

                # Important: move CPU latent statistics to the model device.
                latent_mu = latent_mu.to(device)
                latent_sigma = latent_sigma.to(device)

                z = (
                    latent_mu.unsqueeze(0)
                    + latent_sigma.unsqueeze(0)
                    * torch.randn(
                        current_batch,
                        LATENT_DIM,
                        device=device
                    )
                )

                images = model.decode(z)

            # VAE:
            # Sample from the standard normal prior.
            elif model_name == "vae":

                z = torch.randn(
                    current_batch,
                    LATENT_DIM,
                    device=device
                )

                images = model.decode(z)

            # MLP GAN / DCGAN:
            # Generate from random Gaussian noise.
            elif model_name in ["mlp_gan", "dcgan"]:

                z = torch.randn(
                    current_batch,
                    GAN_LATENT_DIM,
                    device=device
                )

                images = model(z)

                images = convert_gan_images(images)

            # DDPM:
            # Start from Gaussian noise and reverse the diffusion process.
            elif model_name == "ddpm":

                if diffusion is None:
                    raise ValueError(
                        "Diffusion object was not provided."
                    )

                images = diffusion.sample(
                    model,
                    n=current_batch
                )

            else:
                raise ValueError(
                    f"Unknown model name: {model_name}"
                )

            generated_images.append(images.cpu())

    # Move to a new line after the progress output.
    print()

    return torch.cat(generated_images, dim=0)


def calculate_fid(real_images, generated_images, device):
    """
    Calculate FID using the same real reference images for every model.
    """

    fid_metric = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    ).to(device)

    metric_batch_size = 128

    # Extract real image features.
    print("Calculating real image Inception features...")

    for start in range(0, len(real_images), metric_batch_size):

        batch = real_images[
            start:start + metric_batch_size
        ].to(device)

        batch = prepare_fid_images(batch)

        fid_metric.update(
            batch,
            real=True
        )

    # Extract generated image features.
    print("Calculating generated image Inception features...")

    for start in range(
        0,
        len(generated_images),
        metric_batch_size
    ):

        batch = generated_images[
            start:start + metric_batch_size
        ].to(device)

        batch = prepare_fid_images(batch)

        fid_metric.update(
            batch,
            real=False
        )

    fid_score = fid_metric.compute().item()

    return fid_score


def load_existing_results(results_path):
    """
    Load existing results so separate evaluation runs can be merged.
    """

    if results_path.exists():

        try:

            with open(
                results_path,
                "r",
                encoding="utf-8"
            ) as file:

                results = json.load(file)

            if "scores" not in results:
                results["scores"] = {}

            return results

        except (json.JSONDecodeError, OSError):

            print(
                "Warning: Existing FID results could not be read."
            )

            print("Creating a new results file.")

    return {
        "dataset": "Fashion-MNIST",
        "real_reference": "test",
        "fid_feature": 2048,
        "scores": {}
    }


def evaluate_model(
    model_name,
    model,
    checkpoint_path,
    real_images,
    train_loader,
    num_samples,
    batch_size,
    device,
    use_ae_gaussian=False
):
    """
    Load, generate, and evaluate one model.
    """

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    latent_distribution = None
    diffusion = None

    # Optional Gaussian-fit latent sampling for AEs.
    if model_name in ["vanilla_ae", "sparse_ae"]:

        if not use_ae_gaussian:

            print(
                "Skipping FID: N/A (Latent Voids)."
            )

            return "N/A (Latent Voids)"

        print(
            "Collecting training-set latent vectors..."
        )

        latent_vectors = collect_ae_latents(
            model,
            train_loader,
            device
        )

        latent_distribution = fit_gaussian_latent_distribution(
            latent_vectors
        )

        print(
            "Fitting diagonal Gaussian latent distribution completed."
        )

    # Initialize DDPM diffusion schedule.
    elif model_name == "ddpm":

        diffusion = Diffusion(
            noise_steps=1000,
            device=device
        )

    # Reseed before generation.
    # Each model receives the same reproducible random seed,
    # regardless of the models evaluated before it.
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    generated_images = generate_images(
        model_name=model_name,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        latent_distribution=latent_distribution,
        diffusion=diffusion
    )

    print(
        f"Generated {len(generated_images)} images "
        f"for {model_name}."
    )

    fid_score = calculate_fid(
        real_images,
        generated_images,
        device
    )

    print(
        f"\n{model_name} FID: {fid_score:.4f}"
    )

    return fid_score


def save_results(results, results_path):
    """Save current FID results to JSON."""

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate generative models using FID."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=7000,
        help="Number of real and generated images."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for image generation."
    )

    parser.add_argument(
        "--skip",
        nargs="*",
        choices=MODEL_NAMES,
        default=[],
        help="Models to skip."
    )

    parser.add_argument(
        "--ae-gaussian",
        action="store_true",
        help=(
            "Evaluate AEs using new samples from a fitted "
            "diagonal Gaussian latent distribution."
        )
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    # Reproducibility.
    set_seed(SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    # Prepare output directory.
    results_dir = Path("results")
    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = results_dir / "fid_scores.json"

    # Load Fashion-MNIST data.
    train_loader, _, test_loader = build_dataloaders()

    # Collect the same real test images for every model.
    print("\nCollecting real reference images...")

    real_images = collect_real_images(
        test_loader,
        args.num_samples
    )

    print(
        f"Collected {len(real_images)} real test images."
    )

    # Load previous results to merge separate runs.
    results = load_existing_results(results_path)

    results["dataset"] = "Fashion-MNIST"
    results["real_reference"] = "test"
    results["real_samples"] = args.num_samples
    results["generated_samples_per_model"] = args.num_samples
    results["fid_feature"] = 2048

    # Model configurations.
    model_configs = {

        "vanilla_ae": {
            "class": Autoencoder,
            "checkpoint": "checkpoints/vanilla_ae_best.pth"
        },

        "sparse_ae": {
            "class": SparseAutoencoder,
            "checkpoint": "checkpoints/sparse_ae_lambda0.01_best.pth"
        },

        "vae": {
            "class": VAE,
            "checkpoint": "checkpoints/vae_best.pth"
        },

        "mlp_gan": {
            "class": MLPGenerator,
            "checkpoint": "checkpoints/mlp_gan_generator.pth"
        },

        "dcgan": {
            "class": DCGANGenerator,
            "checkpoint": "checkpoints/dcgan_gan_generator.pth"
        },

        "ddpm": {
            "class": UNet,
            "checkpoint": "checkpoints/ddpm_best.pth"
        }
    }

    # Evaluate models one by one.
    for model_name, config in model_configs.items():

        if model_name in args.skip:

            print(f"\nSkipping {model_name}.")
            continue

        checkpoint_path = Path(
            config["checkpoint"]
        )

        if not checkpoint_path.exists():

            print(
                f"\nWarning: Checkpoint not found for "
                f"{model_name}: {checkpoint_path}"
            )

            print("Skipping this model.")
            continue

        model = config["class"]()

        try:

            score = evaluate_model(
                model_name=model_name,
                model=model,
                checkpoint_path=checkpoint_path,
                real_images=real_images,
                train_loader=train_loader,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=device,
                use_ae_gaussian=args.ae_gaussian
            )

            results["scores"][model_name] = score

        except Exception as error:

            print(
                f"\nError evaluating {model_name}: {error}"
            )

            raise

        # Save after every completed model.
        save_results(
            results,
            results_path
        )

        print(
            f"Updated results: {results_path}"
        )

    # Display final scores.
    print("\n" + "=" * 60)
    print("FINAL FID RESULTS")
    print("=" * 60)

    for model_name, score in results["scores"].items():

        if isinstance(score, (int, float)):

            print(
                f"{model_name:15s}: {score:.4f}"
            )

        else:

            print(
                f"{model_name:15s}: {score}"
            )

    print("\nFID evaluation completed.")

    print(
        f"Results saved to: {results_path}"
    )


if __name__ == "__main__":
    main()
