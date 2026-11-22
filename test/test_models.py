import torch
from src.models.autoencoders import VAE

model = VAE(latent_dim=64)

x = torch.rand(8, 1, 28, 28)

# Training mode
model.train()
reconstruction, mu, logvar = model(x)

print("Reconstruction:", reconstruction.shape)
print("Mu:", mu.shape)
print("Logvar:", logvar.shape)

# Evaluation mode
model.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model(x)

print("Evaluation reconstruction:", reconstruction.shape)