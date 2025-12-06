"""Utility functions for reproducibility."""

import numpy as np
import torch

def seed():
    """
    Set random seeds for reproducibility across numpy and PyTorch.

    Sets a fixed seed (1) for numpy and PyTorch random number generators
    to ensure reproducible results across multiple runs.
    """
    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)