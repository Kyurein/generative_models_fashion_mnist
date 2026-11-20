"""Shared utilities for reproducible experiments."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is prioritized over maximum performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
