import torch

from src.models.gan import Generator, Discriminator


def main():
    batch_size = 8
    latent_dim = 100

    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()

    # Generate random latent vectors
    z = torch.randn(batch_size, latent_dim)

    # Generate fake images
    fake_images = generator(z)

    # Discriminate generated images
    logits = discriminator(fake_images)

    print("Input noise shape:", z.shape)
    print("Generated image shape:", fake_images.shape)
    print("Discriminator output shape:", logits.shape)

    print(
        "Generated image range:",
        fake_images.min().item(),
        "to",
        fake_images.max().item()
    )

    assert fake_images.shape == (batch_size, 1, 28, 28)
    assert logits.shape == (batch_size, 1)
    assert fake_images.min().item() >= -1.0
    assert fake_images.max().item() <= 1.0

    print("\nGAN shape test passed!")


if __name__ == "__main__":
    main()
