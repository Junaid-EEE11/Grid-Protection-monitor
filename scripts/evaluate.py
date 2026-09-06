"""CLI workflow script to run complete model evaluation, calibration, and statistical significance tests."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.calibration import compute_ece, TemperatureScaler
from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization
from src.evaluation.statistical_tests import bootstrap_metric_ci, paired_permutation_test
from src.features.normalizer import FeatureNormalizer
from src.models.gnn import PhysicsGuidedGNN
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator
from src.data.dataset_loader import create_dataloaders
from src.utils.logging_utils import get_logger

logger = get_logger("cli_evaluate")


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation, calibration, and significance tests.")
    parser.add_argument("--data_dir", type=str, default="data/raw/ieee123", help="Dataset directory.")
    parser.add_argument("--splits_dir", type=str, default="data/splits/ieee123", help="Splits directory.")
    parser.add_argument("--split_type", type=str, default="in_distribution", help="Split protocol.")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/gnn_in_distribution/best_model.pt", help="GNN model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save evaluation summary.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()

    graph_data_list = torch.load(data_dir / "graph_data.pt", weights_only=False)
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    split_npz = np.load(splits_dir / f"split_{args.split_type}.npz")

    train_idx = split_npz["train_indices"]
    val_idx = split_npz["val_indices"]
    test_idx = split_npz["test_indices"]

    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        graph_data_list, train_idx, val_idx, test_idx, batch_size=32, normalize=True
    )

    in_dim = graph_data_list[0].x.size(1)
    num_nodes = graph_data_list[0].num_nodes

    model = PhysicsGuidedGNN(in_node_features=in_dim, hidden_dim=128, num_layers=3, num_nodes_per_graph=num_nodes)
    ckpt_path = Path(args.checkpoint).resolve()
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}; evaluating current model parameters.")

    model.eval()

    # Extract test predictions and probabilities
    p_det, p_type, p_loc = [], [], []
    prob_det, prob_type = [], []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            p_det.extend(out.det_logits.argmax(dim=-1).cpu().numpy())
            p_type.extend(out.type_logits.argmax(dim=-1).cpu().numpy())
            p_loc.extend(out.loc_logits.argmax(dim=-1).cpu().numpy())
            prob_det.extend(torch.softmax(out.det_logits, dim=-1).cpu().numpy())
            prob_type.extend(torch.softmax(out.type_logits, dim=-1).cpu().numpy())

    test_meta = meta_df.iloc[test_idx]
    y_true_det = test_meta["is_fault"].astype(int).values
    y_true_type = test_meta["fault_type_idx"].values
    y_true_loc = test_meta["fault_bus_idx"].values

    graph_builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    feeder_graph = graph_builder.build_graph()
    dist_calc = GraphDistanceCalculator(feeder_graph)

    det_m = evaluate_detection(y_true_det, p_det, np.array(prob_det))
    type_m = evaluate_classification(y_true_type, p_type)
    loc_m = evaluate_localization(p_loc, y_true_loc, dist_calc)

    # Calibration calculation (ECE)
    ece_uncal, mce_uncal, _ = compute_ece(y_true_type, np.array(prob_type))

    # Bootstrap 95% Confidence Intervals
    det_f1_pt, det_f1_low, det_f1_high = bootstrap_metric_ci(
        y_true_det, np.array(p_det), lambda yt, yp: evaluate_detection(yt, yp)["f1"], n_bootstraps=500
    )
    type_f1_pt, type_f1_low, type_f1_high = bootstrap_metric_ci(
        y_true_type, np.array(p_type), lambda yt, yp: evaluate_classification(yt, yp)["macro_f1"], n_bootstraps=500
    )

    eval_summary = {
        "dataset_split": args.split_type,
        "detection": {
            "f1": det_m["f1"],
            "f1_ci_95": [round(det_f1_low, 4), round(det_f1_high, 4)],
            "auroc": det_m["auroc"],
            "auprc": det_m["auprc"],
            "brier_score": det_m["brier_score"],
        },
        "fault_type_classification": {
            "macro_f1": type_m["macro_f1"],
            "macro_f1_ci_95": [round(type_f1_low, 4), round(type_f1_high, 4)],
            "weighted_f1": type_m["weighted_f1"],
            "balanced_accuracy": type_m["balanced_accuracy"],
            "ece": round(ece_uncal, 4),
            "mce": round(mce_uncal, 4),
        },
        "fault_localization": {
            "exact_accuracy": loc_m["exact_accuracy"],
            "top_3_accuracy": loc_m["top_3_accuracy"],
            "within_1_hop_acc": loc_m["within_1_hop_acc"],
            "within_2_hop_acc": loc_m["within_2_hop_acc"],
            "mean_hop_error": loc_m["mean_hop_error"],
            "median_hop_error": loc_m["median_hop_error"],
        },
    }

    out_file = out_dir / f"full_evaluation_{args.split_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("FULL RESEARCH EVALUATION REPORT")
    print("=" * 65)
    print(f"Detection F1 (95% CI):     {det_m['f1']:.4f} [{det_f1_low:.4f}, {det_f1_high:.4f}]")
    print(f"Type Macro-F1 (95% CI):   {type_m['macro_f1']:.4f} [{type_f1_low:.4f}, {type_f1_high:.4f}]")
    print(f"Expected Calib Error (ECE):{ece_uncal:.4f}")
    print(f"Localization Exact Acc:   {loc_m['exact_accuracy']:.4f}")
    print(f"Localization Within 1-Hop:{loc_m['within_1_hop_acc']:.4f}")
    print(f"Localization Mean Hop Err:{loc_m['mean_hop_error']:.2f}")
    print("=" * 65 + "\n")
    logger.info(f"Saved evaluation report to {out_file}")


if __name__ == "__main__":
    main()
