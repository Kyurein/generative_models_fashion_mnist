import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history
history = pd.read_csv("results/mlp_gan_history.csv")

# Create output directory
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot Generator and Discriminator losses
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["d_loss"],
    label="Discriminator Loss"
)

plt.plot(
    history["epoch"],
    history["g_loss"],
    label="Generator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Vanilla MLP GAN Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_dir / "mlp_gan_loss.png",
    dpi=300
)

plt.show()
plt.close()

print("Training loss plot saved to figures/mlp_gan_loss.png")