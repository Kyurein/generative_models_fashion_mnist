import torch

from src.models.ddpm import UNet


def main():
    batch_size = 8
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet().to(device)

    # Random noisy images
    x_t = torch.randn(
        batch_size, 1, 28, 28,
        device=device
    )

    # Different timestep for each image
    t = torch.randint(
        0, 1000,
        (batch_size,),
        device=device
    )

    model.eval()

    with torch.no_grad():
        predicted_noise = model(x_t, t)

    print("Input image shape:", x_t.shape)
    print("Timestep shape:", t.shape)
    print("Predicted noise shape:", predicted_noise.shape)
    print("Device:", device)

    assert predicted_noise.shape == (
        batch_size, 1, 28, 28
    )

    print("\nDDPM U-Net shape test passed!")


if __name__ == "__main__":
    main()
