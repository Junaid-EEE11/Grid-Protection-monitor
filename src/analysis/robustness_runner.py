"""Robustness benchmark runner evaluating noise sensitivity, sensor loss, and sparsity."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.data.dataset_loader import DistributionDataset
from src.data.measurement_mask import SensorMaskGenerator
from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization
from src.topology.graph_distances import GraphDistanceCalculator
from src.utils.logging_utils import get_logger
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = get_logger("robustness_runner")


class RobustnessRunner:
    """Evaluates trained models under observational degradation: measurement noise and sensor dropouts."""

    def __init__(self, feeder_graph: Any, graph_dist_calc: GraphDistanceCalculator):
        self.feeder_graph = feeder_graph
        self.graph_dist_calc = graph_dist_calc
        self.mask_gen = SensorMaskGenerator(feeder_graph)

    def evaluate_noise_robustness(
        self,
        model: torch.nn.Module,
        test_data_list: List[Any],
        test_metadata: pd.DataFrame,
        noise_levels: List[float] = [0.0, 0.01, 0.02, 0.05, 0.10],
        seed: int = 42,
    ) -> Dict[float, Dict[str, float]]:
        """Evaluate model performance across varying Gaussian measurement noise levels."""
        results: Dict[float, Dict[str, float]] = {}
        model.eval()
        rng = np.random.RandomState(seed)

        for sigma in noise_levels:
            corrupted_list = []
            for d in test_data_list:
                dc = d.clone()
                if sigma > 0:
                    noise = torch.tensor(rng.normal(0.0, sigma, size=dc.x.shape), dtype=torch.float)
                    dc.x = dc.x + noise
                corrupted_list.append(dc)

            loader = PyGDataLoader(DistributionDataset(corrupted_list), batch_size=32, shuffle=False)
            res = self._eval_loader(model, loader, test_metadata)
            results[float(sigma)] = res
            logger.info(f"Noise sigma={sigma:.2f} -> Det F1: {res['det_f1']:.3f}, Loc Hop Err: {res['loc_mean_hop_err']:.2f}")

        return results

    def evaluate_sensor_sparsity_robustness(
        self,
        model: torch.nn.Module,
        test_data_list: List[Any],
        test_metadata: pd.DataFrame,
        fractions: List[float] = [1.0, 0.50, 0.30, 0.20, 0.10],
        seed: int = 42,
    ) -> Dict[float, Dict[str, float]]:
        """Evaluate model performance across degrading sensor monitoring fractions."""
        results: Dict[float, Dict[str, float]] = {}
        model.eval()

        for frac in fractions:
            mask = self.mask_gen.generate_mask(custom_fraction=frac, seed=seed)
            mask_t = torch.tensor(mask, dtype=torch.float).unsqueeze(-1)

            masked_list = []
            for d in test_data_list:
                dc = d.clone()
                dc.x = dc.x * mask_t
                masked_list.append(dc)

            loader = PyGDataLoader(DistributionDataset(masked_list), batch_size=32, shuffle=False)
            res = self._eval_loader(model, loader, test_metadata)
            results[float(frac)] = res
            logger.info(f"Sensor Coverage {frac*100:.0f}% -> Det F1: {res['det_f1']:.3f}, Loc Hop Err: {res['loc_mean_hop_err']:.2f}")

        return results

    def _eval_loader(
        self, model: torch.nn.Module, loader: Any, test_meta: pd.DataFrame
    ) -> Dict[str, float]:
        p_det, p_type, p_loc = [], [], []
        prob_det = []

        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                p_det.extend(out.det_logits.argmax(dim=-1).cpu().numpy())
                p_type.extend(out.type_logits.argmax(dim=-1).cpu().numpy())
                p_loc.extend(out.loc_logits.argmax(dim=-1).cpu().numpy())
                prob_det.extend(torch.softmax(out.det_logits, dim=-1).cpu().numpy())

        y_true_det = test_meta["is_fault"].astype(int).values
        y_true_type = test_meta["fault_type_idx"].values
        y_true_loc = test_meta["fault_bus_idx"].values

        det_m = evaluate_detection(y_true_det, p_det, np.array(prob_det))
        type_m = evaluate_classification(y_true_type, p_type)
        loc_m = evaluate_localization(p_loc, y_true_loc, self.graph_dist_calc)

        return {
            "det_f1": det_m["f1"],
            "det_auroc": det_m["auroc"],
            "type_macro_f1": type_m["macro_f1"],
            "loc_exact_acc": loc_m["exact_accuracy"],
            "loc_mean_hop_err": loc_m["mean_hop_error"],
            "loc_within_1_hop": loc_m["within_1_hop_acc"],
        }
