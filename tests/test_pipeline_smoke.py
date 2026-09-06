"""End-to-end smoke test verifying data generation, splitting, training, and evaluation pipeline."""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch

from src.data.dataset_generator import DatasetGenerator
from src.data.dataset_loader import create_dataloaders
from src.data.splitter import DatasetSplitter, SplitType
from src.models.gnn import PhysicsGuidedGNN
from src.training.trainer import MultiTaskTrainer, TrainConfig


@pytest.mark.smoke
def test_full_pipeline_smoke():
    """Run an end-to-end smoke test across all pipeline stages with a minimal scenario slice."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Initialize DatasetGenerator
        generator = DatasetGenerator(
            master_dss_path="feeders/ieee123/IEEE123Master.dss"
        )

        # 2. Generate 12 scenarios
        scenarios = generator.generate_scenario_list(
            num_normal=4,
            load_multipliers=[1.0],
            fault_resistances=[1.0],
            target_buses=["1", "2", "3", "4"],
        )[:12]

        dataset = generator.run_generation(scenarios, max_scenarios=12)
        assert len(dataset.graph_data_list) > 0

        # Save dataset
        data_out = temp_dir / "data"
        generator.save_dataset(dataset, data_out)
        assert (data_out / "metadata.csv").exists()

        # 3. Create Split
        splitter = DatasetSplitter(dataset.metadata)
        split = splitter.create_split(SplitType.IN_DISTRIBUTION, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert len(split.train_indices) > 0

        # 4. Load Dataloaders
        train_loader, val_loader, test_loader, normalizer = create_dataloaders(
            dataset.graph_data_list,
            split.train_indices,
            split.val_indices,
            split.test_indices,
            batch_size=4,
            normalize=True,
        )

        # 5. Train GNN for 2 epochs
        in_dim = dataset.graph_data_list[0].x.size(1)
        num_nodes = dataset.feeder_graph.num_nodes

        model = PhysicsGuidedGNN(
            in_node_features=in_dim,
            hidden_dim=32,
            num_layers=2,
            num_nodes_per_graph=num_nodes,
        )
        trainer = MultiTaskTrainer(
            model,
            config=TrainConfig(epochs=2, device="cpu"),
            save_dir=temp_dir / "checkpoints",
        )
        history = trainer.train(train_loader, val_loader)
        assert len(history.train_losses) == 2

        # 6. Evaluation
        eval_metrics = trainer.evaluate(test_loader)
        assert "det_acc" in eval_metrics
        assert "loc_acc" in eval_metrics

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
