"""Unit tests for multi-task model heads, baselines, and GNN forward pass."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.baselines import TabularLogisticBaseline, TabularMLP, TabularRandomForestBaseline, ThresholdBaseline
from src.models.gnn import PhysicsGuidedGNN


def test_threshold_baseline():
    model = ThresholdBaseline(voltage_threshold_pu=0.90)
    # 2 samples, 3 buses
    X = np.array([
        [0.95, 0.98, 1.01],
        [0.85, 0.98, 1.01],
    ])
    preds = model.predict(X)
    assert np.array_equal(preds, [0, 1])


def test_tabular_baselines_fit_predict():
    X = np.random.randn(30, 10)
    y_det = np.random.randint(0, 2, size=30)
    y_type = np.random.randint(0, 5, size=30)
    y_loc = np.random.randint(0, 10, size=30)

    log_m = TabularLogisticBaseline()
    log_m.fit(X, y_det, y_type, y_loc)
    p_det, p_type, p_loc = log_m.predict(X)
    assert len(p_det) == 30

    rf_m = TabularRandomForestBaseline(n_estimators=10)
    rf_m.fit(X, y_det, y_type, y_loc)
    p_det, p_type, p_loc = rf_m.predict(X)
    assert len(p_det) == 30


def test_tabular_mlp_forward():
    mlp = TabularMLP(in_features=20, num_nodes=10)
    x = torch.randn(4, 20)
    out = mlp(x)
    assert out.det_logits.shape == (4, 2)
    assert out.type_logits.shape == (4, 5)
    assert out.loc_logits.shape == (4, 10)


def test_physics_guided_gnn_forward():
    # 2 graphs in batch, each with 5 nodes and 20 features
    num_nodes = 5
    in_dim = 20
    d1 = Data(x=torch.randn(num_nodes, in_dim), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    d2 = Data(x=torch.randn(num_nodes, in_dim), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    batch = Batch.from_data_list([d1, d2])

    model = PhysicsGuidedGNN(
        in_node_features=in_dim, hidden_dim=32, num_layers=2, num_nodes_per_graph=num_nodes
    )
    out = model(batch)

    assert out.det_logits.shape == (2, 2)
    assert out.type_logits.shape == (2, 5)
    assert out.loc_logits.shape == (2, 5)
