"""Graph-distance computation and topological localization error metrics."""

from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np

from src.topology.graph_builder import FeederGraph


class GraphDistanceCalculator:
    """Pre-computes topological hop and physical shortest-path distance matrices on feeder graphs."""

    def __init__(self, feeder_graph: FeederGraph):
        self.feeder_graph = feeder_graph
        self.num_nodes = feeder_graph.num_nodes
        self.hop_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.phys_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self._compute_distance_matrices()

    def _compute_distance_matrices(self) -> None:
        """Compute all-pairs topological and physical shortest path matrices using NetworkX."""
        G = self.feeder_graph.networkx_graph
        bus_to_idx = self.feeder_graph.bus_to_idx

        # Initialize with infinity for disconnected components if any
        self.hop_matrix.fill(float("inf"))
        self.phys_matrix.fill(float("inf"))

        # Set diagonal to 0
        np.fill_diagonal(self.hop_matrix, 0.0)
        np.fill_diagonal(self.phys_matrix, 0.0)

        # Topological hop distances
        hop_lengths = dict(nx.all_pairs_shortest_path_length(G))
        for src_bus, targets in hop_lengths.items():
            if src_bus in bus_to_idx:
                u = bus_to_idx[src_bus]
                for dst_bus, d in targets.items():
                    if dst_bus in bus_to_idx:
                        v = bus_to_idx[dst_bus]
                        self.hop_matrix[u, v] = float(d)

        # Physical / length-weighted distances (in kft)
        phys_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
        for src_bus, targets in phys_lengths.items():
            if src_bus in bus_to_idx:
                u = bus_to_idx[src_bus]
                for dst_bus, d in targets.items():
                    if dst_bus in bus_to_idx:
                        v = bus_to_idx[dst_bus]
                        self.phys_matrix[u, v] = float(d)

    def get_hop_distance(self, u_idx: int, v_idx: int) -> float:
        """Get topological shortest path in number of hops between two bus indices."""
        if 0 <= u_idx < self.num_nodes and 0 <= v_idx < self.num_nodes:
            return float(self.hop_matrix[u_idx, v_idx])
        return float("inf")

    def get_physical_distance_kft(self, u_idx: int, v_idx: int) -> float:
        """Get physical shortest path distance in kft between two bus indices."""
        if 0 <= u_idx < self.num_nodes and 0 <= v_idx < self.num_nodes:
            return float(self.phys_matrix[u_idx, v_idx])
        return float("inf")

    def evaluate_localization_errors(
        self, y_pred_indices: Union[List[int], np.ndarray], y_true_indices: Union[List[int], np.ndarray]
    ) -> Dict[str, Any]:
        """Compute comprehensive graph-distance localization error metrics.

        Args:
            y_pred_indices: Predicted bus indices.
            y_true_indices: True faulted bus indices.

        Returns:
            Dictionary containing exact accuracy, hop distance statistics, and CDF percentiles.
        """
        preds = np.asarray(y_pred_indices, dtype=int)
        trues = np.asarray(y_true_indices, dtype=int)

        # Filter out normal (non-faulted) cases where true index is -1
        fault_mask = trues >= 0
        if not np.any(fault_mask):
            return {
                "exact_accuracy": 1.0,
                "mean_hop_error": 0.0,
                "median_hop_error": 0.0,
                "max_hop_error": 0.0,
                "within_1_hop_acc": 1.0,
                "within_2_hop_acc": 1.0,
            }

        valid_preds = preds[fault_mask]
        valid_trues = trues[fault_mask]

        hop_errors = []
        phys_errors = []

        for p, t in zip(valid_preds, valid_trues):
            h_err = self.get_hop_distance(p, t)
            p_err = self.get_physical_distance_kft(p, t)
            # Replace unreachable infinity with max diameter
            h_err = min(h_err, 50.0)
            p_err = min(p_err, 500.0)
            hop_errors.append(h_err)
            phys_errors.append(p_err)

        hop_errors_arr = np.array(hop_errors)
        exact_match = (hop_errors_arr == 0).astype(float)
        within_1 = (hop_errors_arr <= 1).astype(float)
        within_2 = (hop_errors_arr <= 2).astype(float)
        within_3 = (hop_errors_arr <= 3).astype(float)

        return {
            "num_fault_samples": int(len(valid_trues)),
            "exact_accuracy": float(np.mean(exact_match)),
            "within_1_hop_acc": float(np.mean(within_1)),
            "within_2_hop_acc": float(np.mean(within_2)),
            "within_3_hop_acc": float(np.mean(within_3)),
            "mean_hop_error": float(np.mean(hop_errors_arr)),
            "median_hop_error": float(np.median(hop_errors_arr)),
            "std_hop_error": float(np.std(hop_errors_arr)),
            "max_hop_error": float(np.max(hop_errors_arr)),
            "mean_phys_error_kft": float(np.mean(phys_errors)),
            "hop_errors": hop_errors_arr.tolist(),
        }
