"""Sensor coverage masking schemes modeling partial observability in distribution grids."""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np

from src.topology.graph_builder import FeederGraph


class SensorCoverage(str, Enum):
    """Standardized sensor observability regimes."""
    DENSE = "dense"        # 100% bus monitoring
    MODERATE = "moderate"  # ~50% bus monitoring
    SPARSE = "sparse"      # ~20% bus monitoring
    CRITICAL = "critical"  # Substation and feeder boundary heads (~10%)


class SensorMaskGenerator:
    """Generates realistic sensor availability masks based on utility deployment strategies."""

    def __init__(self, feeder_graph: FeederGraph):
        self.feeder_graph = feeder_graph
        self.num_nodes = feeder_graph.num_nodes
        self.bus_to_idx = feeder_graph.bus_to_idx
        self.idx_to_bus = feeder_graph.idx_to_bus
        self.G = feeder_graph.networkx_graph

    def get_critical_sensors(self) -> List[int]:
        """Identify critical grid locations: substation bus, degree >= 3 branch points, and dead-ends."""
        critical_indices = set()

        # 1. Substation node (index 0 or node containing 'sub' / 'source' / '150')
        for bus, idx in self.bus_to_idx.items():
            if any(k in bus for k in ["source", "150", "sub", "650"]):
                critical_indices.add(idx)

        # 2. Key branch points (degree >= 3)
        for node, degree in self.G.degree():
            if degree >= 3 and node in self.bus_to_idx:
                critical_indices.add(self.bus_to_idx[node])

        # 3. Feeder boundary dead-ends (degree == 1)
        for node, degree in self.G.degree():
            if degree == 1 and node in self.bus_to_idx:
                critical_indices.add(self.bus_to_idx[node])

        return sorted(list(critical_indices))

    def generate_mask(
        self,
        coverage: Union[str, SensorCoverage] = SensorCoverage.SPARSE,
        custom_fraction: Optional[float] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate a binary sensor presence mask of shape (num_nodes,).

        Args:
            coverage: Predefined coverage regime (dense, moderate, sparse, critical).
            custom_fraction: Optional explicit float fraction between 0.0 and 1.0.
            seed: Random seed for deterministic sensor placement.

        Returns:
            Binary boolean numpy array of shape (num_nodes,).
        """
        if isinstance(coverage, str):
            coverage = SensorCoverage(coverage.lower())

        mask = np.zeros(self.num_nodes, dtype=bool)

        if coverage == SensorCoverage.DENSE:
            mask.fill(True)
            return mask

        # Target fraction
        if custom_fraction is not None:
            target_frac = max(0.01, min(1.0, float(custom_fraction)))
        elif coverage == SensorCoverage.MODERATE:
            target_frac = 0.50
        elif coverage == SensorCoverage.SPARSE:
            target_frac = 0.20
        elif coverage == SensorCoverage.CRITICAL:
            target_frac = 0.10
        else:
            target_frac = 0.20

        target_count = max(1, int(round(self.num_nodes * target_frac)))

        # Always include critical nodes first
        critical_nodes = self.get_critical_sensors()
        selected_nodes = set(critical_nodes[:target_count])

        # Fill remaining sensor budget randomly with fixed seed
        if len(selected_nodes) < target_count:
            remaining_nodes = [i for i in range(self.num_nodes) if i not in selected_nodes]
            rng = np.random.RandomState(seed)
            needed = target_count - len(selected_nodes)
            sampled = rng.choice(remaining_nodes, size=min(needed, len(remaining_nodes)), replace=False)
            selected_nodes.update(sampled)

        for idx in selected_nodes:
            mask[idx] = True

        return mask
