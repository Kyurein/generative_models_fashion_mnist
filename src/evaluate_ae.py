import argparse
import json
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.models.autoencoders import (
    Autoencoder,
    SparseAutoencoder,
    VAE
)
from src.utils import get_device, set_seed
from src.metrics import (
    evaluate_reconstruction,
    evaluate_hidden_sparsity
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Autoencoder and VAE models"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["vanilla", "sparse", "vae"],
        required=True
    )

    parser.add_argument(
        "--sparsity_lambda",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print(f"Device: {device}")

    # Load test dataset
    _, _, test_loader = build_dataloaders()

    # Initialize model and determine checkpoint name
    if args.model == "vanilla":
        model = Autoencoder(latent_dim=64)
        run_name = "vanilla_ae"

    elif args.model == "sparse":
        model = SparseAutoencoder(latent_dim=64)

        lambda_tag = f"{args.sparsity_lambda:g}"
        run_name = f"sparse_ae_lambda{lambda_tag}"

    else:
        model = VAE(latent_dim=64)
        run_name = "vae"

    model = model.to(device)

    checkpoint_path = (
        Path("checkpoints") / f"{run_name}_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load trained model
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    # Evaluate reconstruction metrics
    metrics = evaluate_reconstruction(
        model,
        test_loader,
        device
    )

    # Hidden sparsity is only applicable to AE models
    if args.model != "vae":
        hidden_sparsity = evaluate_hidden_sparsity(
            model,
            test_loader,
            device
        )

        metrics["hidden_sparsity"] = hidden_sparsity

    # Add model information
    metrics["model"] = args.model

    if args.model == "sparse":
        metrics["sparsity_lambda"] = args.sparsity_lambda

    # Save evaluation results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{run_name}_metrics.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\nEvaluation Results:")

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.6f}")
        else:
            print(f"{metric_name}: {value}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
