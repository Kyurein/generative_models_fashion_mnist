"""Fashion-MNIST loading with the assignment's 80:10:10 split."""
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from src.utils import set_seed, get_device

SEED = 42
BATCH_SIZE = 128
DATA_DIR = Path("data")


def build_dataloaders(batch_size: int = BATCH_SIZE, seed: int = SEED):
    """Combine the official 60k/10k files, then split all 70k as 80/10/10.

    This follows the assignment's explicit 80:10:10 requirement. The resulting
    test set is a newly randomized holdout, not the original torchvision test
    partition.
    """
    set_seed(seed)
    transform = transforms.ToTensor()  # [1, 28, 28], pixel range [0, 1]

    train_part = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_part = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    full_dataset = ConcatDataset([train_part, test_part])

    total = len(full_dataset)  # 70,000
    train_size = int(0.8 * total)  # 56,000
    val_size = int(0.1 * total)   # 7,000
    test_size = total - train_size - val_size  # 7,000

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Separate generators keep training shuffle reproducible.
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=train_generator
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def main():
    set_seed(SEED)
    train_loader, val_loader, test_loader = build_dataloaders()
    device = get_device()

    images, labels = next(iter(train_loader))
    print(f"Device: {device}")
    print(f"Split sizes: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")
    print(f"Pixel range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    print("Class IDs in first batch:", sorted(set(labels.tolist())))


if __name__ == "__main__":
    main()
