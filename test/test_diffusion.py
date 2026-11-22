import torch

from src.diffusion import Diffusion
from src.models.ddpm import UNet


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use a smaller schedule for a quick sampling test
    diffusion = Diffusion(
        noise_steps=10,
        device=device
    )

    batch_size = 8

    # Original images scaled to [-1, 1]
    x0 = torch.rand(
        batch_size, 1, 28, 28,
        device=device
    ) * 2 - 1

    # Random timesteps
    t = torch.randint(
        0, diffusion.noise_steps,
        (batch_size,),
        device=device
    )

    noise = torch.randn_like(x0)

    # Forward diffusion
    x_t = diffusion.q_sample(x0, t, noise)

    print("Original image shape:", x0.shape)
    print("Timestep shape:", t.shape)
    print("Noisy image shape:", x_t.shape)

    assert x_t.shape == x0.shape

    # Reverse diffusion
    model = UNet().to(device)
    model.eval()

    generated, trajectory = diffusion.sample(
        model,
        n=8,
        return_trajectory=True,
        trajectory_steps=(9, 5, 1)
    )

    print("Generated image shape:", generated.shape)
    print("Generated image range:",
          generated.min().item(),
          "to",
          generated.max().item())

    print("Trajectory steps:", sorted(trajectory.keys(), reverse=True))

    assert generated.shape == (8, 1, 28, 28)
    assert generated.min() >= 0
    assert generated.max() <= 1

    print("\nDDPM diffusion test passed!")


if __name__ == "__main__":
    main()
