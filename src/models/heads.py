"""Multi-task prediction heads for fault detection, fault type classification, and node localization."""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiTaskOutput:
    """Unified multi-task model output container."""
    det_logits: torch.Tensor     # Shape: (batch_size, 2)
    type_logits: torch.Tensor    # Shape: (batch_size, num_fault_types)
    loc_logits: torch.Tensor     # Shape: (batch_size, num_nodes) or (total_nodes, 1)


class DetectionHead(nn.Module):
    """Binary classification head for fault detection (Normal vs Faulted)."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """Multiclass classification head for fault type (Normal, SLG, LL, LLG, 3P)."""

    def __init__(self, in_features: int, num_classes: int = 5, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalizationHead(nn.Module):
    """Node-level scoring head producing a probability distribution over feeder buses."""

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute scalar score for each node in the graph.

        Args:
            node_embeddings: Tensor of shape (total_nodes_in_batch, in_features).

        Returns:
            Tensor of shape (total_nodes_in_batch, 1).
        """
        return self.net(node_embeddings)
