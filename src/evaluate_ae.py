import json
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.utils import get_device
from src.models.autoencoders import Autoencoder
from src.metrics import evaluate_reconstruction


def main():
    # 1. Setup device
    device = get_device()
    print(f"Evaluation device: {device}")

    # 2. Load dataset
    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"Test batches: {len(test_loader)}")

    # 3. Initialize Autoencoder
    model = Autoencoder(latent_dim=64).to(device)

    # 4. Load best trained checkpoint
    checkpoint_path = Path("checkpoints/vanilla_ae_best.pth")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Vanilla Autoencoder checkpoint loaded successfully.")

    # 5. Evaluate reconstruction performance
    metrics = evaluate_reconstruction(
        model=model,
        test_loader=test_loader,
        device=device
    )

    # 6. Save evaluation results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "vanilla_ae_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # 7. Display results
    print("\n" + "=" * 50)
    print("VANILLA AUTOENCODER - TEST RESULTS")
    print("=" * 50)

    print(f"Test MSE  : {metrics['test_mse']:.6f}")
    print(f"Test PSNR : {metrics['test_psnr']:.4f} dB")
    print(f"Test SSIM : {metrics['test_ssim']:.4f}")

    print("=" * 50)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
