"""Comprehensive metric computation across detection, classification, and localization tasks."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.topology.graph_distances import GraphDistanceCalculator


@dataclass
class FullEvaluationReport:
    """Unified evaluation metrics across all prediction tasks."""
    model_name: str
    dataset_name: str
    detection_metrics: Dict[str, float]
    classification_metrics: Dict[str, float]
    localization_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_detection(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute binary detection metrics.

    Args:
        y_true: Ground-truth binary labels (0=Normal, 1=Faulted).
        y_pred: Predicted binary labels.
        y_prob: Optional predicted positive-class probability.

    Returns:
        Dictionary containing Accuracy, Precision, Recall, F1, AUROC, AUPRC, FPR@95%TPR, Brier Score.
    """
    trues = np.asarray(y_true, dtype=int)
    preds = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(trues, preds))
    prec = float(precision_score(trues, preds, zero_division=0))
    rec = float(recall_score(trues, preds, zero_division=0))
    f1 = float(f1_score(trues, preds, zero_division=0))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": 0.5,
        "auprc": 0.5,
        "fpr_at_95_tpr": 1.0,
        "brier_score": 0.25,
    }

    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float)
        if probs.ndim == 2:
            probs = probs[:, 1]  # Positive class probability

        if len(np.unique(trues)) > 1:
            try:
                metrics["auroc"] = float(roc_auc_score(trues, probs))
                metrics["auprc"] = float(average_precision_score(trues, probs))
                metrics["brier_score"] = float(brier_score_loss(trues, probs))

                # FPR at 95% TPR
                fpr, tpr, _ = roc_curve(trues, probs)
                idx_95 = np.where(tpr >= 0.95)[0]
                metrics["fpr_at_95_tpr"] = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0
            except Exception:
                pass

    return metrics


def evaluate_classification(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    num_classes: int = 5,
) -> Dict[str, float]:
    """Compute multiclass fault type classification metrics.

    Args:
        y_true: Ground-truth integer class indices (0=Normal, 1=SLG, 2=LL, 3=LLG, 4=3P).
        y_pred: Predicted class indices.
        num_classes: Total number of fault classes.

    Returns:
        Dictionary containing Macro F1, Weighted F1, Balanced Accuracy, and Per-Class F1.
    """
    trues = np.asarray(y_true, dtype=int)
    preds = np.asarray(y_pred, dtype=int)

    macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(trues, preds, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(trues, preds))

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
    }

    # Per-class F1
    per_class_f1s = f1_score(trues, preds, average=None, zero_division=0, labels=list(range(num_classes)))
    for c_idx, f1_val in enumerate(per_class_f1s):
        metrics[f"class_{c_idx}_f1"] = float(f1_val)

    return metrics


def evaluate_localization(
    y_pred_loc: Union[List[int], np.ndarray],
    y_true_loc: Union[List[int], np.ndarray],
    graph_dist_calc: Optional[GraphDistanceCalculator] = None,
    loc_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute exact node accuracy, Top-3 accuracy, and topological hop error metrics.

    Args:
        y_pred_loc: Predicted bus indices.
        y_true_loc: True faulted bus indices (-1 for normal cases).
        graph_dist_calc: GraphDistanceCalculator instance.
        loc_probs: Optional probability array of shape (num_samples, num_nodes).

    Returns:
        Dictionary containing exact accuracy, top-k accuracy, and hop distance statistics.
    """
    preds = np.asarray(y_pred_loc, dtype=int)
    trues = np.asarray(y_true_loc, dtype=int)

    fault_mask = trues >= 0
    if not np.any(fault_mask):
        return {
            "exact_accuracy": 1.0,
            "top_3_accuracy": 1.0,
            "top_5_accuracy": 1.0,
            "mean_hop_error": 0.0,
            "median_hop_error": 0.0,
            "within_1_hop_acc": 1.0,
            "within_2_hop_acc": 1.0,
        }

    valid_preds = preds[fault_mask]
    valid_trues = trues[fault_mask]

    exact_acc = float(np.mean(valid_preds == valid_trues))

    top_3_acc = exact_acc
    top_5_acc = exact_acc
    if loc_probs is not None:
        valid_probs = loc_probs[fault_mask]
        top3_preds = np.argsort(valid_probs, axis=1)[:, -3:]
        top5_preds = np.argsort(valid_probs, axis=1)[:, -5:]
        top_3_acc = float(np.mean([t in top3_preds[i] for i, t in enumerate(valid_trues)]))
        top_5_acc = float(np.mean([t in top5_preds[i] for i, t in enumerate(valid_trues)]))

    metrics = {
        "exact_accuracy": exact_acc,
        "top_3_accuracy": top_3_acc,
        "top_5_accuracy": top_5_acc,
        "mean_hop_error": 0.0,
        "median_hop_error": 0.0,
        "within_1_hop_acc": exact_acc,
        "within_2_hop_acc": exact_acc,
    }

    if graph_dist_calc is not None:
        topo_res = graph_dist_calc.evaluate_localization_errors(valid_preds, valid_trues)
        metrics.update({
            "mean_hop_error": topo_res["mean_hop_error"],
            "median_hop_error": topo_res["median_hop_error"],
            "within_1_hop_acc": topo_res["within_1_hop_acc"],
            "within_2_hop_acc": topo_res["within_2_hop_acc"],
            "within_3_hop_acc": topo_res["within_3_hop_acc"],
            "max_hop_error": topo_res["max_hop_error"],
        })

    return metrics
