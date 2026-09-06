"""Graph builder transforming OpenDSS electrical circuits into NetworkX and PyTorch Geometric topologies."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.simulation.feeder_loader import FeederLoader
from src.utils.logging_utils import get_logger

logger = get_logger("graph_builder")


@dataclass
class FeederGraph:
    """Container for distribution graph representations across formats."""
    feeder_name: str
    networkx_graph: nx.Graph
    bus_to_idx: Dict[str, int]
    idx_to_bus: Dict[int, str]
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    bus_coords: Dict[str, Tuple[float, float]]
    num_nodes: int
    num_edges: int


class DistributionGraphBuilder:
    """Constructs topological and electrical graphs from distribution feeder definitions."""

    def __init__(self, master_dss_path: Union[str, Path]):
        self.loader = FeederLoader(master_dss_path)
        self.feeder_name = self.loader.master_dss_path.stem

    def build_graph(self, permute_edges: bool = False, seed: int = 42) -> FeederGraph:
        """Extract feeder topology and return a unified FeederGraph instance.

        Args:
            permute_edges: If True, randomly rewires graph edges (ablation baseline).
            seed: Random seed for edge permutation.

        Returns:
            FeederGraph dataclass.
        """
        self.loader.compile()
        summary = self.loader.get_summary()
        branches = self.loader.get_branches()
        coords = self.loader.get_bus_coordinates()

        # Deterministic node ordering
        all_buses = sorted(list(set(summary.bus_names)))
        bus_to_idx = {bus: idx for idx, bus in enumerate(all_buses)}
        idx_to_bus = {idx: bus for bus, idx in bus_to_idx.items()}
        num_nodes = len(all_buses)

        nx_graph = nx.Graph()
        for bus in all_buses:
            coord = coords.get(bus, (0.0, 0.0))
            nx_graph.add_node(bus, x=coord[0], y=coord[1], index=bus_to_idx[bus])

        src_indices = []
        dst_indices = []
        edge_attrs = []

        for branch in branches:
            b1 = branch["bus1"].lower()
            b2 = branch["bus2"].lower()
            if b1 in bus_to_idx and b2 in bus_to_idx:
                u, v = bus_to_idx[b1], bus_to_idx[b2]
                length = float(branch.get("length_kft", 0.1))
                r1 = float(branch.get("r1", 0.1))
                x1 = float(branch.get("x1", 0.1))
                phases = float(branch.get("phases", 3))
                is_sw = 1.0 if branch.get("type") == "switch" else 0.0

                attr = [length, r1, x1, phases, is_sw]

                # Bidirectional edges for undirected graph
                src_indices.extend([u, v])
                dst_indices.extend([v, u])
                edge_attrs.extend([attr, attr])
                nx_graph.add_edge(b1, b2, length=length, r1=r1, x1=x1, is_switch=is_sw)

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if permute_edges:
            # Ablation: randomly shuffle edge connectivity while maintaining node set
            rng = np.random.RandomState(seed)
            num_edges = edge_index.size(1)
            perm_dst = rng.permutation(edge_index[1].numpy())
            edge_index = torch.stack([edge_index[0], torch.from_numpy(perm_dst)], dim=0)

        return FeederGraph(
            feeder_name=self.feeder_name,
            networkx_graph=nx_graph,
            bus_to_idx=bus_to_idx,
            idx_to_bus=idx_to_bus,
            edge_index=edge_index,
            edge_attr=edge_attr,
            bus_coords=coords,
            num_nodes=num_nodes,
            num_edges=nx_graph.number_of_edges(),
        )

    def to_pyg_data(
        self,
        node_features_dict: Dict[str, np.ndarray],
        feeder_graph: FeederGraph,
        is_fault: bool,
        fault_type_idx: int,
        fault_bus_idx: int,
        sensor_mask: Optional[np.ndarray] = None,
    ) -> Data:
        """Assemble a PyTorch Geometric Data object for a single simulation scenario.

        Args:
            node_features_dict: Dict of bus_name -> node feature vector.
            feeder_graph: FeederGraph instance.
            is_fault: Binary detection label.
            fault_type_idx: Integer class index for fault category.
            fault_bus_idx: Integer node index of fault location (-1 if normal).
            sensor_mask: Optional binary mask (num_nodes,) indicating sensor availability.

        Returns:
            torch_geometric.data.Data instance.
        """
        num_nodes = feeder_graph.num_nodes
        sample_bus = feeder_graph.idx_to_bus[0]
        feat_dim = len(node_features_dict.get(sample_bus, np.zeros(20)))

        x_arr = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        for idx in range(num_nodes):
            bus = feeder_graph.idx_to_bus[idx]
            if bus in node_features_dict:
                x_arr[idx] = node_features_dict[bus]

        if sensor_mask is not None:
            # Mask out unobserved nodes by zeroing their feature vectors or attaching mask channel
            mask_arr = np.asarray(sensor_mask, dtype=np.float32).reshape(-1, 1)
            x_arr = x_arr * mask_arr

        x_tensor = torch.tensor(x_arr, dtype=torch.float)
        y_det = torch.tensor([1 if is_fault else 0], dtype=torch.long)
        y_type = torch.tensor([fault_type_idx], dtype=torch.long)
        y_loc = torch.tensor([fault_bus_idx], dtype=torch.long)

        data = Data(
            x=x_tensor,
            edge_index=feeder_graph.edge_index,
            edge_attr=feeder_graph.edge_attr,
            y_det=y_det,
            y_type=y_type,
            y_loc=y_loc,
            num_nodes=num_nodes,
        )

        if sensor_mask is not None:
            data.sensor_mask = torch.tensor(sensor_mask, dtype=torch.bool)

        return data
