"""Non-graph baseline models: Overcurrent/Undervoltage thresholding, Logistic Regression, Random Forests, and MLP."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from src.models.heads import MultiTaskOutput, DetectionHead, ClassificationHead, LocalizationHead


class ThresholdBaseline:
    """Physics-rule heuristic baseline: triggers fault if minimum voltage sag exceeds threshold."""

    def __init__(self, voltage_threshold_pu: float = 0.90, vuf_threshold: float = 0.05):
        self.voltage_threshold_pu = voltage_threshold_pu
        self.vuf_threshold = vuf_threshold

    def predict(self, raw_v_pu_matrix: np.ndarray) -> np.ndarray:
        """Predict binary fault detection from node voltage matrix.

        Args:
            raw_v_pu_matrix: Array of shape (num_samples, num_features).

        Returns:
            Binary predictions array of shape (num_samples,).
        """
        min_voltages = np.min(raw_v_pu_matrix, axis=1)
        return (min_voltages < self.voltage_threshold_pu).astype(int)


class TabularLogisticBaseline:
    """Multi-task Scikit-Learn Logistic Regression baseline."""

    def __init__(self, max_iter: int = 500, random_state: int = 42):
        self.det_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.type_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.loc_model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, X: np.ndarray, y_det: np.ndarray, y_type: np.ndarray, y_loc: np.ndarray) -> "TabularLogisticBaseline":
        self.det_model.fit(X, y_det)
        self.type_model.fit(X, y_type)
        fault_mask = y_loc >= 0
        if np.any(fault_mask):
            self.loc_model.fit(X[fault_mask], y_loc[fault_mask])
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_det = self.det_model.predict(X)
        pred_type = self.type_model.predict(X)
        pred_loc = self.loc_model.predict(X) if hasattr(self.loc_model, "classes_") else np.zeros(len(X), dtype=int)
        return pred_det, pred_type, pred_loc

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob_det = self.det_model.predict_proba(X)
        prob_type = self.type_model.predict_proba(X)
        prob_loc = self.loc_model.predict_proba(X) if hasattr(self.loc_model, "classes_") else np.zeros((len(X), 2))
        return prob_det, prob_type, prob_loc


class TabularRandomForestBaseline:
    """Multi-task Scikit-Learn Random Forest baseline."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42):
        self.det_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.type_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.loc_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X: np.ndarray, y_det: np.ndarray, y_type: np.ndarray, y_loc: np.ndarray) -> "TabularRandomForestBaseline":
        self.det_model.fit(X, y_det)
        self.type_model.fit(X, y_type)
        fault_mask = y_loc >= 0
        if np.any(fault_mask):
            self.loc_model.fit(X[fault_mask], y_loc[fault_mask])
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_det = self.det_model.predict(X)
        pred_type = self.type_model.predict(X)
        pred_loc = self.loc_model.predict(X) if hasattr(self.loc_model, "classes_") else np.zeros(len(X), dtype=int)
        return pred_det, pred_type, pred_loc

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob_det = self.det_model.predict_proba(X)
        prob_type = self.type_model.predict_proba(X)
        prob_loc = self.loc_model.predict_proba(X) if hasattr(self.loc_model, "classes_") else np.zeros((len(X), 2))
        return prob_det, prob_type, prob_loc


class TabularMLP(nn.Module):
    """Topology-agnostic Multi-Layer Perceptron deep learning baseline."""

    def __init__(
        self,
        in_features: int,
        num_nodes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.det_head = DetectionHead(curr_dim, hidden_dim=64, dropout=dropout)
        self.type_head = ClassificationHead(curr_dim, num_classes=5, hidden_dim=64, dropout=dropout)
        self.loc_head = nn.Linear(curr_dim, num_nodes)

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        """Forward pass taking flattened measurement tensor of shape (batch_size, in_features)."""
        emb = self.backbone(x)
        det_logits = self.det_head(emb)
        type_logits = self.type_head(emb)
        loc_logits = self.loc_head(emb)
        return MultiTaskOutput(det_logits=det_logits, type_logits=type_logits, loc_logits=loc_logits)
