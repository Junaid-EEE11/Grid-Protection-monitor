"""Unit tests verifying leakage-safe splitting and zero data overlap."""

import numpy as np
import pandas as pd
import pytest

from src.data.splitter import DatasetSplitter, SplitType, verify_no_leakage


@pytest.fixture
def sample_metadata():
    rows = []
    buses = [f"bus_{i}" for i in range(20)]
    for i in range(100):
        is_f = i % 2 == 1
        f_bus = buses[i % len(buses)] if is_f else None
        lm = 0.7 if i < 20 else (1.3 if i > 80 else 1.0)
        rows.append({
            "scenario_id": f"sc_{i}",
            "is_fault": is_f,
            "fault_type": "SLG" if is_f else "NORMAL",
            "fault_bus": f_bus,
            "loading_multiplier": lm,
            "fault_resistance_ohm": 1.0 if is_f else 0.0,
        })
    return pd.DataFrame(rows)


def test_in_distribution_split(sample_metadata):
    splitter = DatasetSplitter(sample_metadata)
    res = splitter.create_split(SplitType.IN_DISTRIBUTION, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(res.train_indices) == 70
    assert len(res.val_indices) == 15
    assert len(res.test_indices) == 15
    assert verify_no_leakage(sample_metadata, res.train_indices, res.val_indices, res.test_indices, SplitType.IN_DISTRIBUTION)


def test_held_out_locations_split(sample_metadata):
    splitter = DatasetSplitter(sample_metadata)
    res = splitter.create_split(SplitType.HELD_OUT_LOCATIONS, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_buses = set(sample_metadata.iloc[res.train_indices]["fault_bus"].dropna().unique())
    test_buses = set(sample_metadata.iloc[res.test_indices]["fault_bus"].dropna().unique())

    assert len(train_buses.intersection(test_buses)) == 0
    assert verify_no_leakage(sample_metadata, res.train_indices, res.val_indices, res.test_indices, SplitType.HELD_OUT_LOCATIONS)


def test_held_out_operating_split(sample_metadata):
    splitter = DatasetSplitter(sample_metadata)
    res = splitter.create_split(SplitType.HELD_OUT_OPERATING, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(res.test_indices) > 0
    assert verify_no_leakage(sample_metadata, res.train_indices, res.val_indices, res.test_indices, SplitType.HELD_OUT_OPERATING)
