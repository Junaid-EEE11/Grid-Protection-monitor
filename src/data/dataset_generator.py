"""Dataset generation engine generating labeled scenarios, extracting physics features, and formatting outputs."""

import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.measurement_mask import SensorMaskGenerator
from src.features.physics_engine import PhysicsFeatureExtractor
from src.simulation.engine import PowerSystemSimulator
from src.simulation.fault_models import FaultScenario, FaultType
from src.topology.graph_builder import DistributionGraphBuilder, FeederGraph
from src.utils.logging_utils import get_logger

logger = get_logger("dataset_generator")

FAULT_TYPE_TO_IDX = {
    FaultType.NORMAL.value: 0,
    FaultType.SLG.value: 1,
    FaultType.LL.value: 2,
    FaultType.LLG.value: 3,
    FaultType.THREE_PHASE.value: 4,
}
IDX_TO_FAULT_TYPE = {v: k for k, v in FAULT_TYPE_TO_IDX.items()}


@dataclass
class SimulationDataset:
    """In-memory dataset container storing metadata, graph representations, and tabular features."""
    feeder_name: str
    metadata: pd.DataFrame
    graph_data_list: List[Data]
    tabular_features: np.ndarray
    feeder_graph: FeederGraph


class DatasetGenerator:
    """Orchestrates OpenDSS simulation runs across parameter sweeps to build benchmark datasets."""

    def __init__(
        self,
        master_dss_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.master_dss_path = Path(master_dss_path).resolve()
        self.config = config or {}
        self.simulator = PowerSystemSimulator(self.master_dss_path)
        self.graph_builder = DistributionGraphBuilder(self.master_dss_path)
        self.feeder_graph = self.graph_builder.build_graph()
        self.physics_extractor = PhysicsFeatureExtractor()
        self.mask_generator = SensorMaskGenerator(self.feeder_graph)

    def generate_scenario_list(
        self,
        num_normal: int = 50,
        load_multipliers: Optional[List[float]] = None,
        fault_resistances: Optional[List[float]] = None,
        der_levels_kw: Optional[List[float]] = None,
        target_buses: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[FaultScenario]:
        """Construct the complete list of parameter combinations to simulate."""
        rng = np.random.RandomState(seed)
        scenarios: List[FaultScenario] = []
        feeder_name = self.feeder_graph.feeder_name

        loads = load_multipliers or [0.7, 0.85, 1.0, 1.15, 1.3]
        rfs = fault_resistances or [0.01, 0.5, 2.0, 10.0, 25.0, 50.0]
        ders = der_levels_kw or [0.0, 250.0, 500.0]
        buses = target_buses or [b for b in self.feeder_graph.bus_to_idx.keys() if "source" not in b]

        # 1. Normal scenarios (varying load and DER)
        for i in range(num_normal):
            lm = float(rng.choice(loads))
            der = float(rng.choice(ders))
            der_bus = str(rng.choice(buses)) if der > 0 else None
            sc = FaultScenario.create_normal(
                scenario_id=f"norm_{i:04d}",
                feeder_name=feeder_name,
                loading_multiplier=lm,
                der_penetration_kw=der,
                der_bus=der_bus,
                seed=seed + i,
            )
            scenarios.append(sc)

        # 2. Fault scenarios
        fault_idx = 0
        fault_configs = [
            (FaultType.SLG, [["A"], ["B"], ["C"]]),
            (FaultType.LL, [["A", "B"], ["B", "C"], ["C", "A"]]),
            (FaultType.LLG, [["A", "B"], ["B", "C"], ["C", "A"]]),
            (FaultType.THREE_PHASE, [["A", "B", "C"]]),
        ]

        for bus in buses:
            for ft, phase_combos in fault_configs:
                for phases in phase_combos:
                    for rf in rfs:
                        lm = float(rng.choice(loads))
                        der = float(rng.choice(ders))
                        der_bus = str(rng.choice(buses)) if der > 0 else None

                        sc = FaultScenario.create_fault(
                            scenario_id=f"flt_{fault_idx:05d}",
                            feeder_name=feeder_name,
                            fault_type=ft,
                            fault_bus=bus,
                            faulted_phases=phases,
                            fault_resistance_ohm=rf,
                            loading_multiplier=lm,
                            der_penetration_kw=der,
                            der_bus=der_bus,
                            seed=seed + fault_idx + 1000,
                        )
                        scenarios.append(sc)
                        fault_idx += 1

        return scenarios

    def run_generation(
        self,
        scenarios: List[FaultScenario],
        sensor_coverage: str = "dense",
        custom_mask: Optional[np.ndarray] = None,
        max_scenarios: Optional[int] = None,
    ) -> SimulationDataset:
        """Simulate all scenarios, extract features, and construct graph / tabular structures."""
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]

        metadata_rows = []
        graph_data_list: List[Data] = []
        tabular_rows = []

        sensor_mask = (
            custom_mask
            if custom_mask is not None
            else self.mask_generator.generate_mask(sensor_coverage)
        )

        total = len(scenarios)
        logger.info(f"Starting simulation of {total} scenarios for {self.feeder_graph.feeder_name}...")

        converged_count = 0
        for i, sc in enumerate(scenarios):
            sim_res = self.simulator.execute_scenario(sc)
            if not sim_res["success"]:
                continue

            converged_count += 1
            post_meas = sim_res["post_fault"]["buses"]
            pre_meas = sim_res["pre_fault"]["buses"]

            # Physics feature extraction
            node_feats = self.physics_extractor.extract_node_features(
                bus_measurements=post_meas,
                pre_fault_bus_measurements=pre_meas,
                include_physics=True,
            )

            # Target labels
            is_fault = sc.is_fault
            type_idx = FAULT_TYPE_TO_IDX[sc.fault_type.value]
            loc_idx = self.feeder_graph.bus_to_idx.get(sc.fault_bus, -1) if sc.fault_bus else -1

            # Assemble PyG Data object
            pyg_data = self.graph_builder.to_pyg_data(
                node_features_dict=node_feats,
                feeder_graph=self.feeder_graph,
                is_fault=is_fault,
                fault_type_idx=type_idx,
                fault_bus_idx=loc_idx,
                sensor_mask=sensor_mask,
            )
            pyg_data.scenario_id = sc.scenario_id
            graph_data_list.append(pyg_data)

            # Assemble flat tabular feature vector for baseline ML models
            # (Concatenate observed node features into a 1D vector)
            flat_feat = pyg_data.x.numpy().flatten()
            tabular_rows.append(flat_feat)

            meta_entry = sc.to_dict()
            meta_entry["fault_type_idx"] = type_idx
            meta_entry["fault_bus_idx"] = loc_idx
            metadata_rows.append(meta_entry)

        meta_df = pd.DataFrame(metadata_rows)
        tabular_matrix = np.array(tabular_rows, dtype=np.float32)

        logger.info(
            f"Completed simulation: {converged_count}/{total} converged scenarios "
            f"({converged_count/max(1, total)*100:.1f}%)."
        )

        return SimulationDataset(
            feeder_name=self.feeder_graph.feeder_name,
            metadata=meta_df,
            graph_data_list=graph_data_list,
            tabular_features=tabular_matrix,
            feeder_graph=self.feeder_graph,
        )

    def save_dataset(self, dataset: SimulationDataset, output_dir: Union[str, Path]) -> None:
        """Persist generated dataset and metadata to disk."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Metadata
        meta_csv = out_path / "metadata.csv"
        dataset.metadata.to_csv(meta_csv, index=False)

        # 2. Tabular features
        tab_npz = out_path / "tabular_features.npz"
        np.savez_compressed(tab_npz, features=dataset.tabular_features)

        # 3. PyG graph list
        graph_pt = out_path / "graph_data.pt"
        torch.save(dataset.graph_data_list, graph_pt)

        # 4. Feeder topology dictionary
        topo_json = out_path / "feeder_topology.json"
        with open(topo_json, "w", encoding="utf-8") as f:
            json.dump({
                "feeder_name": dataset.feeder_name,
                "num_nodes": dataset.feeder_graph.num_nodes,
                "num_edges": dataset.feeder_graph.num_edges,
                "bus_to_idx": dataset.feeder_graph.bus_to_idx,
                "idx_to_bus": dataset.feeder_graph.idx_to_bus,
                "bus_coords": dataset.feeder_graph.bus_coords,
            }, f, indent=2)

        logger.info(f"Saved dataset successfully to {out_path}")
