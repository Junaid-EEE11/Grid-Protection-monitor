"""CLI workflow script to run noise and sensor sparsity robustness stress tests."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.robustness_runner import RobustnessRunner
from src.models.gnn import PhysicsGuidedGNN
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator
from src.utils.logging_utils import get_logger

logger = get_logger("cli_robustness")


def main():
    parser = argparse.ArgumentParser(description="Run robustness benchmarks against noise and sensor loss.")
    parser.add_argument("--data_dir", type=str, default="data/raw/ieee123", help="Dataset directory.")
    parser.add_argument("--splits_dir", type=str, default="data/splits/ieee123", help="Splits directory.")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/gnn_in_distribution/best_model.pt", help="GNN model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save robustness metrics.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_data_list = torch.load(data_dir / "graph_data.pt", weights_only=False)
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    split_npz = np.load(splits_dir / "split_in_distribution.npz")
    test_idx = split_npz["test_indices"]

    test_data_list = [graph_data_list[i] for i in test_idx]
    test_meta = meta_df.iloc[test_idx]

    graph_builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    feeder_graph = graph_builder.build_graph()
    dist_calc = GraphDistanceCalculator(feeder_graph)

    in_dim = graph_data_list[0].x.size(1)
    num_nodes = graph_data_list[0].num_nodes
    model = PhysicsGuidedGNN(in_node_features=in_dim, hidden_dim=128, num_layers=3, num_nodes_per_graph=num_nodes)
    ckpt_path = Path(args.checkpoint).resolve()
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    runner = RobustnessRunner(feeder_graph, dist_calc)

    print("\n" + "=" * 65)
    print("RUNNING MEASUREMENT NOISE ROBUSTNESS SWEEP")
    print("=" * 65)
    noise_results = runner.evaluate_noise_robustness(
        model, test_data_list, test_meta, noise_levels=[0.0, 0.01, 0.02, 0.05, 0.10]
    )

    print("\n" + "=" * 65)
    print("RUNNING SENSOR SPARSITY ROBUSTNESS SWEEP")
    print("=" * 65)
    sparsity_results = runner.evaluate_sensor_sparsity_robustness(
        model, test_data_list, test_meta, fractions=[1.0, 0.50, 0.30, 0.20, 0.10]
    )
    print("=" * 65 + "\n")

    robustness_data = {
        "noise_sweep": {str(k): v for k, v in noise_results.items()},
        "sparsity_sweep": {str(k): v for k, v in sparsity_results.items()},
    }

    out_file = out_dir / "robustness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(robustness_data, f, indent=2)
    logger.info(f"Saved robustness results to {out_file}")


if __name__ == "__main__":
    main()
