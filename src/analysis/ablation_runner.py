"""Ablation experiment runner evaluating architectural and feature components."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.data.dataset_loader import create_dataloaders
from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization
from src.models.gnn import PhysicsGuidedGNN, GNNBackboneType
from src.topology.graph_distances import GraphDistanceCalculator
from src.training.trainer import MultiTaskTrainer, TrainConfig
from src.utils.logging_utils import get_logger

logger = get_logger("ablation_runner")


class AblationRunner:
    """Orchestrates scientific ablation studies comparing physics-guided, topology-only, and permuted models."""

    def __init__(self, feeder_graph: Any, graph_dist_calc: GraphDistanceCalculator):
        self.feeder_graph = feeder_graph
        self.graph_dist_calc = graph_dist_calc

    def run_all_ablations(
        self,
        data_list: List[Any],
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        metadata_df: pd.DataFrame,
        epochs: int = 30,
        device: str = "cpu",
    ) -> Dict[str, Dict[str, float]]:
        """Run standard suite of model ablations.

        Ablations:
        1. Full Proposed (Physics-Guided GNN + Message Passing)
        2. Topology-Only GNN (Raw electrical features only, ablated sequence components)
        3. Permuted Graph GNN (Physics features + randomly rewired graph connectivity)

        Returns:
            Dictionary mapping ablation_name -> evaluation metrics.
        """
        results: Dict[str, Dict[str, float]] = {}

        train_loader, val_loader, test_loader, normalizer = create_dataloaders(
            data_list, train_indices, val_indices, test_indices, batch_size=32, normalize=True
        )

        in_dim = data_list[0].x.size(1)
        num_nodes = self.feeder_graph.num_nodes

        # 1. Full Physics-Guided GNN
        logger.info("Running Ablation 1: Full Physics-Guided GNN...")
        m1 = PhysicsGuidedGNN(
            in_node_features=in_dim,
            hidden_dim=128,
            num_layers=3,
            num_nodes_per_graph=num_nodes,
            use_physics_features=True,
        )
        t1 = MultiTaskTrainer(m1, config=TrainConfig(epochs=epochs, device=device))
        t1.train(train_loader, val_loader)
        eval1 = self._eval_model(m1, test_loader, metadata_df.iloc[test_indices])
        results["full_physics_gnn"] = eval1

        # 2. Topology-Only GNN (Ablate physics sequence features)
        logger.info("Running Ablation 2: Topology-Only GNN (Raw Phasors)...")
        m2 = PhysicsGuidedGNN(
            in_node_features=in_dim,
            hidden_dim=128,
            num_layers=3,
            num_nodes_per_graph=num_nodes,
            use_physics_features=False,
            raw_feature_dim=9,
        )
        t2 = MultiTaskTrainer(m2, config=TrainConfig(epochs=epochs, device=device))
        t2.train(train_loader, val_loader)
        eval2 = self._eval_model(m2, test_loader, metadata_df.iloc[test_indices])
        results["topology_only_gnn"] = eval2

        # 3. Permuted Topology GNN
        logger.info("Running Ablation 3: Permuted Graph GNN...")
        perm_data_list = [d.clone() for d in data_list]
        rng = np.random.RandomState(42)
        for d in perm_data_list:
            perm_dst = torch.from_numpy(rng.permutation(d.edge_index[1].numpy()))
            d.edge_index = torch.stack([d.edge_index[0], perm_dst], dim=0)

        tr_p, v_p, te_p, _ = create_dataloaders(
            perm_data_list, train_indices, val_indices, test_indices, batch_size=32, normalize=True
        )
        m3 = PhysicsGuidedGNN(
            in_node_features=in_dim,
            hidden_dim=128,
            num_layers=3,
            num_nodes_per_graph=num_nodes,
            use_physics_features=True,
        )
        t3 = MultiTaskTrainer(m3, config=TrainConfig(epochs=epochs, device=device))
        t3.train(tr_p, v_p)
        eval3 = self._eval_model(m3, te_p, metadata_df.iloc[test_indices])
        results["permuted_graph_gnn"] = eval3

        return results

    def _eval_model(
        self, model: torch.nn.Module, test_loader: Any, test_meta: pd.DataFrame
    ) -> Dict[str, float]:
        """Internal helper evaluating test performance."""
        model.eval()
        p_det, p_type, p_loc = [], [], []
        prob_det, prob_type, prob_loc = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                p_det.extend(out.det_logits.argmax(dim=-1).cpu().numpy())
                p_type.extend(out.type_logits.argmax(dim=-1).cpu().numpy())
                p_loc.extend(out.loc_logits.argmax(dim=-1).cpu().numpy())

                prob_det.extend(torch.softmax(out.det_logits, dim=-1).cpu().numpy())
                prob_type.extend(torch.softmax(out.type_logits, dim=-1).cpu().numpy())
                prob_loc.extend(torch.softmax(out.loc_logits, dim=-1).cpu().numpy())

        y_true_det = test_meta["is_fault"].astype(int).values
        y_true_type = test_meta["fault_type_idx"].values
        y_true_loc = test_meta["fault_bus_idx"].values

        det_m = evaluate_detection(y_true_det, p_det, np.array(prob_det))
        type_m = evaluate_classification(y_true_type, p_type)
        loc_m = evaluate_localization(p_loc, y_true_loc, self.graph_dist_calc, np.array(prob_loc))

        return {
            "det_f1": det_m["f1"],
            "det_auroc": det_m["auroc"],
            "type_macro_f1": type_m["macro_f1"],
            "loc_exact_acc": loc_m["exact_accuracy"],
            "loc_mean_hop_err": loc_m["mean_hop_error"],
            "loc_within_1_hop": loc_m["within_1_hop_acc"],
        }
