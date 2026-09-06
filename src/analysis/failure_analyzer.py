"""Automated failure diagnosis module investigating misclassifications and spatial localization errors."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.topology.graph_distances import GraphDistanceCalculator


@dataclass
class FailureReport:
    """Detailed failure analysis findings."""
    total_samples: int
    num_detection_errors: int
    num_classification_errors: int
    num_severe_localization_errors: int
    errors_by_fault_type: Dict[str, int]
    errors_by_rf_regime: Dict[str, float]
    hardest_scenarios: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureAnalyzer:
    """Diagnoses model mistakes across physical conditions, impedance regimes, and sensor proximity."""

    def __init__(self, graph_dist_calc: Optional[GraphDistanceCalculator] = None):
        self.graph_dist_calc = graph_dist_calc

    def analyze(
        self,
        test_metadata: pd.DataFrame,
        y_pred_det: np.ndarray,
        y_pred_type: np.ndarray,
        y_pred_loc: np.ndarray,
        prob_det: Optional[np.ndarray] = None,
    ) -> FailureReport:
        """Run granular scenario-level failure attribution.

        Args:
            test_metadata: Test scenario metadata DataFrame.
            y_pred_det: Predicted detection labels.
            y_pred_type: Predicted type labels.
            y_pred_loc: Predicted location indices.
            prob_det: Predicted detection probabilities.

        Returns:
            FailureReport dataclass.
        """
        meta = test_metadata.reset_index(drop=True)
        n = len(meta)

        y_true_det = meta["is_fault"].astype(int).values
        y_true_type = meta["fault_type_idx"].values
        y_true_loc = meta["fault_bus_idx"].values

        det_err = (y_pred_det != y_true_det)
        type_err = (y_pred_type != y_true_type)

        loc_errs = []
        for i in range(n):
            if y_true_det[i] == 1:
                if self.graph_dist_calc:
                    h_err = self.graph_dist_calc.get_hop_distance(y_pred_loc[i], y_true_loc[i])
                else:
                    h_err = 0.0 if y_pred_loc[i] == y_true_loc[i] else 1.0
                loc_errs.append(h_err)
            else:
                loc_errs.append(0.0)

        loc_errs_arr = np.array(loc_errs)
        severe_loc_err = (loc_errs_arr >= 3).astype(int)

        # Breakdown by fault type
        ft_errors: Dict[str, int] = {}
        for ft in meta["fault_type"].unique():
            ft_mask = (meta["fault_type"] == ft).values
            ft_errors[str(ft)] = int(np.sum(type_err[ft_mask]))

        # Breakdown by fault resistance (low <= 1 ohm, med 1-10 ohm, high > 10 ohm)
        rf_regimes = {}
        fault_mask = (y_true_det == 1)
        if np.any(fault_mask):
            rfs = meta.loc[fault_mask, "fault_resistance_ohm"].values
            loc_err_faults = loc_errs_arr[fault_mask]

            low_rf = rfs <= 1.0
            med_rf = (rfs > 1.0) & (rfs <= 10.0)
            high_rf = rfs > 10.0

            rf_regimes["low_rf_le_1_mean_hop_err"] = float(np.mean(loc_err_faults[low_rf])) if np.any(low_rf) else 0.0
            rf_regimes["med_rf_1_10_mean_hop_err"] = float(np.mean(loc_err_faults[med_rf])) if np.any(med_rf) else 0.0
            rf_regimes["high_rf_gt_10_mean_hop_err"] = float(np.mean(loc_err_faults[high_rf])) if np.any(high_rf) else 0.0

        # Hardest failure scenarios
        hard_idx = np.where(det_err | (loc_errs_arr >= 2))[0]
        hard_scenarios = []
        for idx in hard_idx[:10]:
            row = meta.iloc[idx].to_dict()
            row["pred_det"] = int(y_pred_det[idx])
            row["pred_type"] = int(y_pred_type[idx])
            row["pred_loc"] = int(y_pred_loc[idx])
            row["hop_error"] = float(loc_errs_arr[idx])
            hard_scenarios.append(row)

        return FailureReport(
            total_samples=n,
            num_detection_errors=int(np.sum(det_err)),
            num_classification_errors=int(np.sum(type_err)),
            num_severe_localization_errors=int(np.sum(severe_loc_err)),
            errors_by_fault_type=ft_errors,
            errors_by_rf_regime=rf_regimes,
            hardest_scenarios=hard_scenarios,
        )
