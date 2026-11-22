import torch

from src.models.dcgan import (
    Generator,
    Discriminator,
    weights_init
)


def main():
    batch_size = 8
    latent_dim = 100

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    z = torch.randn(batch_size, latent_dim, device=device)

    with torch.no_grad():
        fake_images = generator(z)
        predictions = discriminator(fake_images)

    print("Input noise shape:", z.shape)
    print("Generated image shape:", fake_images.shape)
    print("Discriminator output shape:", predictions.shape)

    print(
        "Generated image range:",
        fake_images.min().item(),
        "to",
        fake_images.max().item()
    )

    assert fake_images.shape == (8, 1, 28, 28)
    assert predictions.shape == (8, 1)

    print("\nDCGAN shape test passed!")


if __name__ == "__main__":
    main()
