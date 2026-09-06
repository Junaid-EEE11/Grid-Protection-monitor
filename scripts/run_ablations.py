"""CLI workflow script to run scientific ablation studies."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.ablation_runner import AblationRunner
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator
from src.utils.logging_utils import get_logger

logger = get_logger("cli_ablations")


def main():
    parser = argparse.ArgumentParser(description="Run controlled ablation studies.")
    parser.add_argument("--data_dir", type=str, default="data/raw/ieee123", help="Dataset directory.")
    parser.add_argument("--splits_dir", type=str, default="data/splits/ieee123", help="Splits directory.")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per ablation model.")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save ablation metrics.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_data_list = torch.load(data_dir / "graph_data.pt", weights_only=False)
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    split_npz = np.load(splits_dir / "split_in_distribution.npz")

    train_idx = split_npz["train_indices"]
    val_idx = split_npz["val_indices"]
    test_idx = split_npz["test_indices"]

    graph_builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    feeder_graph = graph_builder.build_graph()
    dist_calc = GraphDistanceCalculator(feeder_graph)

    runner = AblationRunner(feeder_graph, dist_calc)
    logger.info("Executing ablation matrix...")
    results = runner.run_all_ablations(
        data_list=graph_data_list,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        metadata_df=meta_df,
        epochs=args.epochs,
    )

    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    for model_name, m in results.items():
        print(f"{model_name:<25} | Det F1: {m['det_f1']:.3f} | Type F1: {m['type_macro_f1']:.3f} | Loc 1-Hop: {m['loc_within_1_hop']:.3f} | Hop Err: {m['loc_mean_hop_err']:.2f}")
    print("=" * 65 + "\n")

    out_file = out_dir / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ablation results to {out_file}")


if __name__ == "__main__":
    main()
