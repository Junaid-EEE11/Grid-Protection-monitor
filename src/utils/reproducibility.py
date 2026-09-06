"""Deterministic reproducibility utilities ensuring exact seed setting across libraries."""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Set random seeds across Python random, NumPy, and PyTorch for scientific reproducibility.

    Args:
        seed: Integer seed value.
        deterministic_torch: If True, configures cuDNN / PyTorch for deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
