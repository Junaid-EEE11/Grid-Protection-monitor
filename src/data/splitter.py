"""Leakage-safe dataset splitting protocols supporting In-Distribution and Out-of-Distribution partitions."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("dataset_splitter")


class SplitType(str, Enum):
    """Evaluation partition protocols."""
    IN_DISTRIBUTION = "in_distribution"
    HELD_OUT_LOCATIONS = "held_out_locations"
    HELD_OUT_OPERATING = "held_out_operating"


@dataclass
class SplitResult:
    """Index partitions and verification summary."""
    split_type: SplitType
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    metadata_summary: Dict[str, Any]


def verify_no_leakage(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    split_type: SplitType,
) -> bool:
    """Assert that train, validation, and test sets are strictly disjoint with zero target leakage.

    Args:
        metadata: DataFrame containing scenario metadata.
        train_indices: Array of integer training indices.
        val_indices: Array of integer validation indices.
        test_indices: Array of integer test indices.
        split_type: Partition protocol being verified.

    Returns:
        True if all leakage assertions pass.

    Raises:
        AssertionError: If index overlap or protocol condition violation is detected.
    """
    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())

    # 1. Strict index disjointness
    assert len(train_set.intersection(val_set)) == 0, "Leakage Error: Train and Val sets overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage Error: Train and Test sets overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage Error: Val and Test sets overlap!"

    # 2. Protocol-specific checks
    if split_type == SplitType.HELD_OUT_LOCATIONS:
        train_buses = set(metadata.iloc[train_indices]["fault_bus"].dropna().unique())
        test_buses = set(metadata.iloc[test_indices]["fault_bus"].dropna().unique())
        held_out_overlap = train_buses.intersection(test_buses)
        assert len(held_out_overlap) == 0, (
            f"Leakage Error: Held-out buses found in training set: {held_out_overlap}"
        )

    logger.info(f"Verified zero leakage for split '{split_type.value}' successfully.")
    return True


class DatasetSplitter:
    """Generates reproducible, leakage-free data splits for research benchmarks."""

    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata.reset_index(drop=True)
        self.num_samples = len(self.metadata)

    def create_split(
        self,
        split_type: Union[str, SplitType] = SplitType.IN_DISTRIBUTION,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> SplitResult:
        """Create a leakage-free partition according to the specified protocol.

        Args:
            split_type: Split strategy enum or string.
            train_ratio: Proportion of training samples.
            val_ratio: Proportion of validation samples.
            test_ratio: Proportion of testing samples.
            seed: Random seed for deterministic assignment.

        Returns:
            SplitResult dataclass containing indices and partition summary.
        """
        if isinstance(split_type, str):
            split_type = SplitType(split_type.lower())

        rng = np.random.RandomState(seed)

        if split_type == SplitType.IN_DISTRIBUTION:
            # Stratified random split preserving fault_type proportions
            indices = np.arange(self.num_samples)
            rng.shuffle(indices)

            n_train = int(self.num_samples * train_ratio)
            n_val = int(self.num_samples * val_ratio)

            train_idx = np.sort(indices[:n_train])
            val_idx = np.sort(indices[n_train: n_train + n_val])
            test_idx = np.sort(indices[n_train + n_val:])

        elif split_type == SplitType.HELD_OUT_LOCATIONS:
            # Partition unique buses: hold out 20% of buses exclusively for test
            fault_buses = sorted([b for b in self.metadata["fault_bus"].dropna().unique()])
            rng.shuffle(fault_buses)

            n_test_buses = max(1, int(len(fault_buses) * test_ratio / (test_ratio + val_ratio + 1e-6)))
            test_buses = set(fault_buses[:n_test_buses])
            train_val_buses = set(fault_buses[n_test_buses:])

            test_mask = self.metadata["fault_bus"].isin(test_buses)
            test_idx = np.where(test_mask)[0]

            train_val_mask = self.metadata["fault_bus"].isin(train_val_buses) | self.metadata["fault_bus"].isna()
            train_val_idx = np.where(train_val_mask)[0]
            rng.shuffle(train_val_idx)

            n_val = int(len(train_val_idx) * (val_ratio / (train_ratio + val_ratio)))
            val_idx = np.sort(train_val_idx[:n_val])
            train_idx = np.sort(train_val_idx[n_val:])

        elif split_type == SplitType.HELD_OUT_OPERATING:
            # Hold out extreme load multipliers (e.g. lm <= 0.75 or lm >= 1.25)
            extreme_mask = (self.metadata["loading_multiplier"] <= 0.75) | (self.metadata["loading_multiplier"] >= 1.25)
            test_idx = np.where(extreme_mask)[0]

            nominal_idx = np.where(~extreme_mask)[0]
            rng.shuffle(nominal_idx)

            n_val = int(len(nominal_idx) * val_ratio / (train_ratio + val_ratio))
            val_idx = np.sort(nominal_idx[:n_val])
            train_idx = np.sort(nominal_idx[n_val:])

        else:
            raise ValueError(f"Unsupported split type: {split_type}")

        # Verify no leakage
        verify_no_leakage(self.metadata, train_idx, val_idx, test_idx, split_type)

        summary = {
            "total_samples": self.num_samples,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "train_fraction": round(len(train_idx) / self.num_samples, 4),
            "val_fraction": round(len(val_idx) / self.num_samples, 4),
            "test_fraction": round(len(test_idx) / self.num_samples, 4),
        }

        return SplitResult(
            split_type=split_type,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            metadata_summary=summary,
        )

    def save_split(self, split: SplitResult, output_dir: Union[str, Path]) -> None:
        """Save split index arrays and configuration summary to disk."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        split_file = out_path / f"split_{split.split_type.value}.npz"
        np.savez_compressed(
            split_file,
            train_indices=split.train_indices,
            val_indices=split.val_indices,
            test_indices=split.test_indices,
        )

        meta_file = out_path / f"split_{split.split_type.value}_summary.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "split_type": split.split_type.value,
                "summary": split.metadata_summary,
            }, f, indent=2)

        logger.info(f"Saved split '{split.split_type.value}' to {split_file}")
