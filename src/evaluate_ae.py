import argparse
import json
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.utils import get_device
from src.models.autoencoders import Autoencoder, SparseAutoencoder
from src.metrics import (
    evaluate_reconstruction,
    evaluate_hidden_sparsity
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Autoencoder on Fashion-MNIST test set"
    )

    parser.add_argument(
        "--model",
        choices=["vanilla", "sparse"],
        default="vanilla"
    )

    parser.add_argument(
        "--sparsity_lambda",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    if args.sparsity_lambda < 0:
        parser.error("--sparsity_lambda cannot be negative")

    device = get_device()
    print(f"Evaluation device: {device}")

    # Load test dataset
    _, _, test_loader = build_dataloaders()
    print(f"Test batches: {len(test_loader)}")

    # Initialize selected model and checkpoint name
    if args.model == "sparse":
        model = SparseAutoencoder(latent_dim=64).to(device)
        lambda_tag = f"{args.sparsity_lambda:g}"
        run_name = f"sparse_ae_lambda{lambda_tag}"
    else:
        model = Autoencoder(latent_dim=64).to(device)
        run_name = "vanilla_ae"

    checkpoint_path = Path("checkpoints") / f"{run_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load trained weights
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
        model=model,
        test_loader=test_loader,
        device=device
    )

    # Evaluate hidden-layer sparsity
    hidden_sparsity = evaluate_hidden_sparsity(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=0.01
    )

    metrics["hidden_sparsity"] = hidden_sparsity
    metrics["sparsity_threshold"] = 0.01
    metrics["model"] = args.model

    if args.model == "sparse":
        metrics["sparsity_lambda"] = args.sparsity_lambda

    # Save metrics
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"{run_name}_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n" + "=" * 50)
    print(f"{args.model.upper()} AUTOENCODER TEST RESULTS")
    print("=" * 50)
    print(f"Test MSE: {metrics['test_mse']:.6f}")
    print(f"Test PSNR: {metrics['test_psnr']:.4f} dB")
    print(f"Test SSIM: {metrics['test_ssim']:.4f}")
    print(f"Hidden sparsity: {hidden_sparsity:.2%}")

    if args.model == "sparse":
        print(f"Sparsity lambda: {args.sparsity_lambda:g}")

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()

