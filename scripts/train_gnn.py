"""CLI workflow script to train the PhysicsGuidedGNN model."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import create_dataloaders
from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization
from src.models.gnn import PhysicsGuidedGNN, GNNBackboneType
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator
from src.training.trainer import MultiTaskTrainer, TrainConfig
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("cli_train_gnn")


def main():
    parser = argparse.ArgumentParser(description="Train Physics-Guided GNN for multi-task grid fault diagnosis.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML.")
    parser.add_argument("--data_dir", type=str, default="data/raw/ieee123", help="Directory containing dataset files.")
    parser.add_argument("--splits_dir", type=str, default="data/splits/ieee123", help="Directory containing split npz.")
    parser.add_argument("--split_type", type=str, default="in_distribution", help="Split protocol to use.")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints", help="Directory to save checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and split indices
    logger.info("Loading graph dataset and partition indices...")
    graph_data_list = torch.load(data_dir / "graph_data.pt", weights_only=False)
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    split_npz = np.load(splits_dir / f"split_{args.split_type}.npz")

    train_idx = split_npz["train_indices"]
    val_idx = split_npz["val_indices"]
    test_idx = split_npz["test_indices"]

    # 2. Build loaders
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        graph_data_list,
        train_idx,
        val_idx,
        test_idx,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        normalize=True,
    )

    # 3. Instantiate model
    in_dim = graph_data_list[0].x.size(1)
    num_nodes = graph_data_list[0].num_nodes
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})

    model = PhysicsGuidedGNN(
        in_node_features=in_dim,
        hidden_dim=m_cfg.get("hidden_dim", 128),
        num_layers=m_cfg.get("num_layers", 3),
        num_nodes_per_graph=num_nodes,
        num_fault_types=m_cfg.get("num_fault_types", 5),
        backbone=m_cfg.get("backbone", "gcn"),
        dropout=m_cfg.get("dropout", 0.20),
        use_physics_features=True,
    )

    train_config = TrainConfig(
        lr=t_cfg.get("lr", 1e-3),
        weight_decay=t_cfg.get("weight_decay", 1e-4),
        epochs=t_cfg.get("epochs", 35),
        patience=t_cfg.get("patience", 8),
        lambda_det=t_cfg.get("lambda_det", 1.0),
        lambda_type=t_cfg.get("lambda_type", 1.0),
        lambda_loc=t_cfg.get("lambda_loc", 1.5),
    )

    save_subdir = out_dir / f"gnn_{args.split_type}"
    save_subdir.mkdir(parents=True, exist_ok=True)

    trainer = MultiTaskTrainer(model, config=train_config, save_dir=save_subdir)
    logger.info("Starting Physics-Guided GNN training...")
    history = trainer.train(train_loader, val_loader)

    # Save normalizer stats alongside model checkpoint
    if normalizer is not None:
        np.savez_compressed(
            save_subdir / "normalizer.npz",
            mean=normalizer.mean,
            std=normalizer.std,
        )

    # 4. Final test evaluation
    graph_builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    feeder_graph = graph_builder.build_graph()
    dist_calc = GraphDistanceCalculator(feeder_graph)

    test_meta = meta_df.iloc[test_idx]
    y_true_det = test_meta["is_fault"].astype(int).values
    y_true_type = test_meta["fault_type_idx"].values
    y_true_loc = test_meta["fault_bus_idx"].values

    p_det, p_type, p_loc = [], [], []
    prob_det, prob_type, prob_loc = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            p_det.extend(out.det_logits.argmax(dim=-1).cpu().numpy())
            p_type.extend(out.type_logits.argmax(dim=-1).cpu().numpy())
            p_loc.extend(out.loc_logits.argmax(dim=-1).cpu().numpy())

            prob_det.extend(torch.softmax(out.det_logits, dim=-1).cpu().numpy())
            prob_type.extend(torch.softmax(out.type_logits, dim=-1).cpu().numpy())
            prob_loc.extend(torch.softmax(out.loc_logits, dim=-1).cpu().numpy())

    det_m = evaluate_detection(y_true_det, p_det, np.array(prob_det))
    type_m = evaluate_classification(y_true_type, p_type)
    loc_m = evaluate_localization(p_loc, y_true_loc, dist_calc, np.array(prob_loc))

    metrics = {
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

    print("\n" + "=" * 65)
    print(f"PHYSICS-GUIDED GNN TEST EVALUATION ({args.split_type.upper()})")
    print("=" * 65)
    print(f"Detection F1:             {metrics['det_f1']:.4f}")
    print(f"Detection AUROC:          {metrics['det_auroc']:.4f}")
    print(f"Fault Type Macro-F1:      {metrics['type_macro_f1']:.4f}")
    print(f"Localization Exact Acc:   {metrics['loc_exact_acc']:.4f}")
    print(f"Localization Top-3 Acc:   {metrics['loc_top3_acc']:.4f}")
    print(f"Localization Mean Hop Err:{metrics['loc_mean_hop_err']:.2f}")
    print(f"Localization Within 1-Hop:{metrics['loc_within_1_hop']:.4f}")
    print("=" * 65 + "\n")

    metrics_file = Path("results/metrics") / f"gnn_metrics_{args.split_type}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
