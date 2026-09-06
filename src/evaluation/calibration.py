"""Model calibration assessment (ECE, MCE, Reliability Curves) and Temperature Scaling post-processing."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReliabilityDiagramData:
    """Binned confidence and empirical accuracy values for reliability diagrams."""
    bin_centers: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    ece: float
    mce: float


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, ReliabilityDiagramData]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: True binary or integer multiclass labels.
        y_prob: Predicted probability array of shape (N,) or (N, C).
        n_bins: Number of equal-width probability bins.

    Returns:
        Tuple of (ece, mce, ReliabilityDiagramData).
    """
    trues = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    if probs.ndim == 2 and probs.shape[1] > 1:
        # Multiclass: take max predicted probability and predicted class
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        accuracies = (preds == trues).astype(float)
    else:
        # Binary: probability of positive class
        if probs.ndim == 2:
            probs = probs[:, 1]
        preds = (probs >= 0.5).astype(int)
        confidences = np.where(preds == 1, probs, 1.0 - probs)
        accuracies = (preds == trues).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    mce = 0.0
    n_samples = len(trues)

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        center = 0.5 * (low + high)
        bin_centers.append(center)
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)

            diff = abs(bin_acc - bin_conf)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
        else:
            bin_accs.append(center)
            bin_confs.append(center)

    diag_data = ReliabilityDiagramData(
        bin_centers=bin_centers,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
        ece=float(ece),
        mce=float(mce),
    )

    return float(ece), float(mce), diag_data


class TemperatureScaler(nn.Module):
    """Calibrates model logits post-training via single-parameter temperature scaling."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned scalar temperature parameter."""
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def fit(self, val_logits: Union[torch.Tensor, np.ndarray], val_labels: Union[torch.Tensor, np.ndarray]) -> "TemperatureScaler":
        """Optimize temperature parameter to minimize cross-entropy loss on validation data."""
        if not isinstance(val_logits, torch.Tensor):
            val_logits = torch.tensor(val_logits, dtype=torch.float)
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels, dtype=torch.long)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self
