"""Physics-derived electrical feature extractor for nodes and edges in distribution networks."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.features.sequence_components import compute_sequence_components


class PhysicsFeatureExtractor:
    """Computes physically grounded electrical indicators from raw nodal and branch phasor measurements."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def extract_node_features(
        self,
        bus_measurements: Dict[str, Dict[str, Any]],
        pre_fault_bus_measurements: Optional[Dict[str, Dict[str, Any]]] = None,
        include_physics: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute feature vectors for every bus node.

        Args:
            bus_measurements: Post-fault bus dictionary with v_pu, v_ang, phase_mask.
            pre_fault_bus_measurements: Optional pre-fault baseline measurements.
            include_physics: If True, augments raw features with sequence components and unbalance factors.

        Returns:
            Dictionary mapping bus_name -> 1D numpy feature vector.
        """
        node_features: Dict[str, np.ndarray] = {}

        for bus, bdata in bus_measurements.items():
            v_pu = np.array(bdata["v_pu"], dtype=float)
            v_ang = np.array(bdata["v_ang"], dtype=float)
            p_mask = np.array(bdata["phase_mask"], dtype=float)

            # 1. Raw features: [v_pu (3), v_ang_rad (3), phase_mask (3)] -> 9 dims
            v_ang_rad = np.radians(v_ang)
            raw_feats = np.concatenate([v_pu, v_ang_rad, p_mask])

            if not include_physics:
                node_features[bus] = raw_feats
                continue

            # 2. Sequence components: [V0, V1, V2]
            v0, v1, v2 = compute_sequence_components(v_pu, v_ang, p_mask)

            # 3. Physical unbalance factors
            vuf = v2 / (v1 + self.eps)  # Voltage unbalance factor
            zuf = v0 / (v1 + self.eps)  # Zero-sequence unbalance factor

            # 4. Voltage sag indicator
            present_v = [v_pu[i] for i in range(3) if p_mask[i] == 1]
            min_v = min(present_v) if present_v else 1.0
            sag = max(0.0, 1.0 - min_v)

            # 5. Differential pre/post quantities if pre-fault data is provided
            if pre_fault_bus_measurements and bus in pre_fault_bus_measurements:
                pre_bdata = pre_fault_bus_measurements[bus]
                pre_v_pu = np.array(pre_bdata["v_pu"], dtype=float)
                pre_v_ang = np.array(pre_bdata["v_ang"], dtype=float)
                pre_v0, pre_v1, pre_v2 = compute_sequence_components(
                    pre_v_pu, pre_v_ang, pre_bdata["phase_mask"]
                )
                delta_v = np.abs(v_pu - pre_v_pu)
                delta_v1 = abs(v1 - pre_v1)
                delta_v0 = abs(v0 - pre_v0)
            else:
                delta_v = np.zeros(3, dtype=float)
                delta_v1 = 0.0
                delta_v0 = 0.0

            physics_feats = np.array([
                v0, v1, v2,
                vuf, zuf, sag,
                delta_v[0], delta_v[1], delta_v[2],
                delta_v1, delta_v0,
            ], dtype=float)

            full_feats = np.concatenate([raw_feats, physics_feats])
            node_features[bus] = full_feats

        return node_features

    def extract_edge_features(
        self,
        branches: List[Dict[str, Any]],
        branch_measurements: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """Compute edge attribute vectors for distribution lines/switches.

        Args:
            branches: List of branch records from feeder loader.
            branch_measurements: Optional current measurements for lines.

        Returns:
            Dictionary mapping (bus1, bus2) -> 1D numpy feature vector.
        """
        edge_features: Dict[Tuple[str, str], np.ndarray] = {}

        for branch in branches:
            bus1 = branch["bus1"].lower()
            bus2 = branch["bus2"].lower()
            length = float(branch.get("length_kft", 0.1))
            r1 = float(branch.get("r1", 0.1))
            x1 = float(branch.get("x1", 0.1))
            phases = float(branch.get("phases", 3))
            is_switch = 1.0 if branch.get("type") == "switch" else 0.0

            line_name = branch["name"].lower()
            if branch_measurements and line_name in branch_measurements:
                i_mag = branch_measurements[line_name]["i_mag"]
                max_i = max(i_mag) if i_mag else 0.0
            else:
                max_i = 0.0

            feat = np.array([length, r1, x1, phases, is_switch, max_i], dtype=float)
            edge_features[(bus1, bus2)] = feat
            # Symmetric edge for undirected graph representation
            edge_features[(bus2, bus1)] = feat

        return edge_features
