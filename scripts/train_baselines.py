"""CLI workflow script to train and evaluate all baseline ML models."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator
from src.training.train_baselines import train_and_eval_baselines
from src.utils.logging_utils import get_logger

logger = get_logger("cli_train_baselines")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate non-graph baseline models.")
    parser.add_argument("--data_dir", type=str, default="data/raw/ieee123", help="Directory containing dataset files.")
    parser.add_argument("--splits_dir", type=str, default="data/splits/ieee123", help="Directory containing split npz.")
    parser.add_argument("--split_type", type=str, default="in_distribution", help="Split protocol to use.")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save baseline results.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(data_dir / "metadata.csv")
    tab_features = np.load(data_dir / "tabular_features.npz")["features"]
    split_npz = np.load(splits_dir / f"split_{args.split_type}.npz")

    train_idx = split_npz["train_indices"]
    val_idx = split_npz["val_indices"]
    test_idx = split_npz["test_indices"]

    # Load graph topology for distance evaluations
    graph_builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    feeder_graph = graph_builder.build_graph()
    dist_calc = GraphDistanceCalculator(feeder_graph)

    logger.info(f"Training baselines on {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples...")
    raw_results = train_and_eval_baselines(
        X_matrix=tab_features,
        metadata_df=meta_df,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        num_nodes=feeder_graph.num_nodes,
    )

    test_meta = meta_df.iloc[test_idx]
    y_true_det = test_meta["is_fault"].astype(int).values
    y_true_type = test_meta["fault_type_idx"].values
    y_true_loc = test_meta["fault_bus_idx"].values

    metrics_summary = {}
    print("\n" + "=" * 65)
    print(f"BASELINE BENCHMARK RESULTS ({args.split_type.upper()})")
    print("=" * 65)

    for model_name, res in raw_results.items():
        det_m = evaluate_detection(y_true_det, res["pred_det"], res.get("prob_det"))
        type_m = evaluate_classification(y_true_type, res["pred_type"])
        loc_m = evaluate_localization(res["pred_loc"], y_true_loc, dist_calc, res.get("prob_loc"))

        model_summary = {
            "det_f1": det_m["f1"],
            "det_auroc": det_m["auroc"],
            "det_brier": det_m["brier_score"],
            "type_macro_f1": type_m["macro_f1"],
            "type_weighted_f1": type_m["weighted_f1"],
            "loc_exact_acc": loc_m["exact_accuracy"],
            "loc_top3_acc": loc_m["top_3_accuracy"],
            "loc_mean_hop_err": loc_m["mean_hop_error"],
            "loc_within_1_hop": loc_m["within_1_hop_acc"],
        }
        metrics_summary[model_name] = model_summary

        print(f"Model: {model_name:<20} | Det F1: {model_summary['det_f1']:.3f} | Type F1: {model_summary['type_macro_f1']:.3f} | Loc Acc: {model_summary['loc_exact_acc']:.3f} | Hop Err: {model_summary['loc_mean_hop_err']:.2f}")

    print("=" * 65 + "\n")

    out_file = out_dir / f"baseline_metrics_{args.split_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved baseline metrics to {out_file}")


if __name__ == "__main__":
    main()
