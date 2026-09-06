"""Baseline training and inference pipeline for tabular ML and MLP models."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.normalizer import FeatureNormalizer
from src.models.baselines import (
    ThresholdBaseline,
    TabularLogisticBaseline,
    TabularRandomForestBaseline,
    TabularMLP,
)
from src.models.heads import MultiTaskOutput
from src.utils.logging_utils import get_logger

logger = get_logger("train_baselines")


def train_mlp_model(
    mlp: TabularMLP,
    X_train: np.ndarray,
    y_det_train: np.ndarray,
    y_type_train: np.ndarray,
    y_loc_train: np.ndarray,
    X_val: np.ndarray,
    y_det_val: np.ndarray,
    y_type_val: np.ndarray,
    y_loc_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TabularMLP:
    """Train the TabularMLP baseline model."""
    dev = torch.device(device)
    mlp.to(dev)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float),
        torch.tensor(y_det_train, dtype=torch.long),
        torch.tensor(y_type_train, dtype=torch.long),
        torch.tensor(y_loc_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    det_crit = nn.CrossEntropyLoss()
    type_crit = nn.CrossEntropyLoss()
    loc_crit = nn.CrossEntropyLoss()

    mlp.train()
    for epoch in range(1, epochs + 1):
        for bx, b_det, b_type, b_loc in train_loader:
            bx, b_det, b_type, b_loc = bx.to(dev), b_det.to(dev), b_type.to(dev), b_loc.to(dev)
            optimizer.zero_grad()
            out = mlp(bx)

            loss_det = det_crit(out.det_logits, b_det)
            loss_type = type_crit(out.type_logits, b_type)
            f_mask = b_det == 1
            if f_mask.any():
                loss_loc = loc_crit(out.loc_logits[f_mask], b_loc[f_mask])
            else:
                loss_loc = torch.tensor(0.0, device=dev)

            total_loss = loss_det + loss_type + 1.5 * loss_loc
            total_loss.backward()
            optimizer.step()

    return mlp


def train_and_eval_baselines(
    X_matrix: np.ndarray,
    metadata_df: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    num_nodes: int = 123,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Train all non-graph baselines and generate predictions on the test set.

    Args:
        X_matrix: Array of shape (num_samples, num_features).
        metadata_df: DataFrame with labels: is_fault, fault_type_idx, fault_bus_idx.
        train_indices: Training sample indices.
        val_indices: Validation sample indices.
        test_indices: Test sample indices.
        num_nodes: Number of bus nodes.
        seed: Random seed.

    Returns:
        Dictionary mapping model_name -> dict of test predictions and probabilities.
    """
    y_det = metadata_df["is_fault"].astype(int).values
    y_type = metadata_df["fault_type_idx"].values
    y_loc = metadata_df["fault_bus_idx"].values

    # Leakage-safe normalization strictly on train split
    normalizer = FeatureNormalizer().fit(X_matrix[train_indices])
    X_train = normalizer.transform(X_matrix[train_indices])
    X_val = normalizer.transform(X_matrix[val_indices])
    X_test = normalizer.transform(X_matrix[test_indices])

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # 1. Physics Threshold Baseline (uses un-normalized voltages)
    logger.info("Evaluating Physics Threshold baseline...")
    raw_v_train = X_matrix[train_indices, :num_nodes]
    raw_v_test = X_matrix[test_indices, :num_nodes]
    thresh_model = ThresholdBaseline(voltage_threshold_pu=0.90)
    pred_det_thresh = thresh_model.predict(raw_v_test)
    results["threshold"] = {
        "pred_det": pred_det_thresh,
        "pred_type": np.zeros_like(pred_det_thresh),
        "pred_loc": np.zeros_like(pred_det_thresh),
        "prob_det": np.column_stack([1.0 - pred_det_thresh, pred_det_thresh]),
    }

    # 2. Logistic Regression Baseline
    logger.info("Training Tabular Logistic Regression baseline...")
    log_model = TabularLogisticBaseline(random_state=seed)
    log_model.fit(
        X_train,
        y_det[train_indices],
        y_type[train_indices],
        y_loc[train_indices],
    )
    p_det_log, p_type_log, p_loc_log = log_model.predict(X_test)
    prob_det_log, prob_type_log, _ = log_model.predict_proba(X_test)
    results["logistic_regression"] = {
        "pred_det": p_det_log,
        "pred_type": p_type_log,
        "pred_loc": p_loc_log,
        "prob_det": prob_det_log,
        "prob_type": prob_type_log,
    }

    # 3. Random Forest Baseline
    logger.info("Training Tabular Random Forest baseline...")
    rf_model = TabularRandomForestBaseline(n_estimators=100, max_depth=12, random_state=seed)
    rf_model.fit(
        X_train,
        y_det[train_indices],
        y_type[train_indices],
        y_loc[train_indices],
    )
    p_det_rf, p_type_rf, p_loc_rf = rf_model.predict(X_test)
    prob_det_rf, prob_type_rf, _ = rf_model.predict_proba(X_test)
    results["random_forest"] = {
        "pred_det": p_det_rf,
        "pred_type": p_type_rf,
        "pred_loc": p_loc_rf,
        "prob_det": prob_det_rf,
        "prob_type": prob_type_rf,
    }

    # 4. Tabular MLP Baseline
    logger.info("Training Tabular MLP baseline...")
    in_feats = X_train.shape[1]
    mlp = TabularMLP(in_features=in_feats, num_nodes=num_nodes, hidden_dims=[256, 128])
    mlp = train_mlp_model(
        mlp,
        X_train,
        y_det[train_indices],
        y_type[train_indices],
        y_loc[train_indices],
        X_val,
        y_det[val_indices],
        y_type[val_indices],
        y_loc[val_indices],
        epochs=40,
        lr=1e-3,
    )
    mlp.eval()
    with torch.no_grad():
        out = mlp(torch.tensor(X_test, dtype=torch.float))
        prob_det_mlp = torch.softmax(out.det_logits, dim=-1).cpu().numpy()
        prob_type_mlp = torch.softmax(out.type_logits, dim=-1).cpu().numpy()
        p_det_mlp = out.det_logits.argmax(dim=-1).cpu().numpy()
        p_type_mlp = out.type_logits.argmax(dim=-1).cpu().numpy()
        p_loc_mlp = out.loc_logits.argmax(dim=-1).cpu().numpy()

    results["mlp"] = {
        "pred_det": p_det_mlp,
        "pred_type": p_type_mlp,
        "pred_loc": p_loc_mlp,
        "prob_det": prob_det_mlp,
        "prob_type": prob_type_mlp,
    }

    return results
