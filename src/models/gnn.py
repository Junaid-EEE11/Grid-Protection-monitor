"""Topology-aware Physics-Guided Multi-Task Graph Neural Network architecture."""

from enum import Enum
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

from src.models.heads import MultiTaskOutput, DetectionHead, ClassificationHead, LocalizationHead


class GNNBackboneType(str, Enum):
    """Supported graph convolution layer architectures."""
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"


class PhysicsGuidedGNN(nn.Module):
    """Multi-task Graph Neural Network processing unbalanced distribution grid topologies."""

    def __init__(
        self,
        in_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_nodes_per_graph: int = 123,
        num_fault_types: int = 5,
        backbone: Union[str, GNNBackboneType] = GNNBackboneType.GCN,
        dropout: float = 0.2,
        use_physics_features: bool = True,
        raw_feature_dim: int = 9,
    ):
        super().__init__()
        self.use_physics_features = use_physics_features
        self.raw_feature_dim = raw_feature_dim
        actual_in_dim = in_node_features if use_physics_features else raw_feature_dim
        self.num_nodes_per_graph = num_nodes_per_graph

        if isinstance(backbone, str):
            backbone = GNNBackboneType(backbone.lower())
        self.backbone_type = backbone

        self.input_proj = nn.Linear(actual_in_dim, hidden_dim)

        # Graph convolution message-passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if backbone == GNNBackboneType.GCN:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif backbone == GNNBackboneType.GAT:
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
            elif backbone == GNNBackboneType.SAGE:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Multi-task heads
        # Global graph representation: concatenation of mean and max pooled node vectors
        global_dim = hidden_dim * 2
        self.det_head = DetectionHead(global_dim, hidden_dim=64, dropout=dropout)
        self.type_head = ClassificationHead(global_dim, num_classes=num_fault_types, hidden_dim=64, dropout=dropout)
        self.loc_head = LocalizationHead(hidden_dim, hidden_dim=64, dropout=dropout)

    def forward(self, data: Union[Data, Batch]) -> MultiTaskOutput:
        """Execute multi-task forward pass across graph batch.

        Args:
            data: PyTorch Geometric Data or Batch instance.

        Returns:
            MultiTaskOutput dataclass with det_logits, type_logits, and loc_logits.
        """
        x = data.x
        if not self.use_physics_features:
            x = x[:, : self.raw_feature_dim]

        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial projection
        h = self.input_proj(x)
        h = self.act(h)

        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_in

        # 1. Global pooling for detection and fault classification
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=-1)

        det_logits = self.det_head(h_global)
        type_logits = self.type_head(h_global)

        # 2. Node-level scoring for fault localization
        node_scores = self.loc_head(h).squeeze(-1)  # Shape: (total_nodes_in_batch,)

        # Reshape node scores to (batch_size, num_nodes)
        batch_size = int(batch.max().item()) + 1
        num_nodes_total = x.size(0)
        nodes_per_g = num_nodes_total // batch_size
        loc_logits = node_scores.view(batch_size, nodes_per_g)

        return MultiTaskOutput(
            det_logits=det_logits,
            type_logits=type_logits,
            loc_logits=loc_logits,
        )
