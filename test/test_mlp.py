import torch
from src.models.mlp_classifier import MLPClassifier

model = MLPClassifier()

# Batch of 8 Fashion-MNIST images
x = torch.rand(8, 1, 28, 28)

logits, z_mlp = model(x)

print("Logits shape:", logits.shape)
print("Z_MLP shape:", z_mlp.shape)

# Check layer weights
print("fc1 weight:", model.fc1.weight.shape)
print("fc2 weight:", model.fc2.weight.shape)
print("fc3 weight:", model.fc3.weight.shape)
