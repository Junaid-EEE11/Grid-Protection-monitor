"""Dataset loaders for PyTorch Geometric graphs and Scikit-Learn tabular models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.features.normalizer import FeatureNormalizer
from src.utils.logging_utils import get_logger

logger = get_logger("dataset_loader")


class DistributionDataset(Dataset):
    """PyTorch Dataset wrapping a list of PyTorch Geometric Data objects with on-the-fly normalization."""

    def __init__(self, data_list: List[Data], normalizer: Optional[FeatureNormalizer] = None):
        self.data_list = data_list
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        item = self.data_list[idx].clone()
        if self.normalizer is not None and self.normalizer.fitted:
            item.x = torch.as_tensor(self.normalizer.transform(item.x), dtype=torch.float)
        return item


def create_dataloaders(
    data_list: List[Data],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 32,
    normalize: bool = True,
    num_workers: int = 0,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader, Optional[FeatureNormalizer]]:
    """Construct leakage-safe training, validation, and test PyG DataLoaders.

    Args:
        data_list: Full list of PyG Data instances.
        train_indices: Array of integer training indices.
        val_indices: Array of integer validation indices.
        test_indices: Array of integer test indices.
        batch_size: Batch size for training and inference.
        normalize: If True, fits FeatureNormalizer strictly on training data.
        num_workers: PyTorch worker count.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, fitted_normalizer).
    """
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    normalizer = None
    if normalize and len(train_data) > 0:
        # Fit normalizer strictly on training node features
        train_x_concat = torch.cat([d.x for d in train_data], dim=0)
        normalizer = FeatureNormalizer().fit(train_x_concat)

    train_ds = DistributionDataset(train_data, normalizer=normalizer)
    val_ds = DistributionDataset(val_data, normalizer=normalizer)
    test_ds = DistributionDataset(test_data, normalizer=normalizer)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, normalizer
