"""Unit tests for graph extraction, topology connectivity, and distance calculations."""

import pytest
from src.topology.graph_builder import DistributionGraphBuilder
from src.topology.graph_distances import GraphDistanceCalculator


@pytest.fixture
def feeder_graph():
    builder = DistributionGraphBuilder("feeders/ieee123/IEEE123Master.dss")
    return builder.build_graph()


def test_feeder_graph_structure(feeder_graph):
    """Verify IEEE 123-bus graph attributes."""
    assert feeder_graph.num_nodes == 123
    assert feeder_graph.num_edges > 100
    assert feeder_graph.edge_index.size(0) == 2
    assert feeder_graph.edge_attr.size(1) == 5


def test_graph_distance_calculator(feeder_graph):
    """Verify hop and physical distance calculations."""
    calc = GraphDistanceCalculator(feeder_graph)
    assert calc.hop_matrix.shape == (123, 123)
    assert calc.get_hop_distance(0, 0) == 0.0

    # Adjacent buses should have 1 hop
    u = feeder_graph.bus_to_idx["1"]
    v = feeder_graph.bus_to_idx["2"]
    assert calc.get_hop_distance(u, v) == 1.0


def test_localization_error_evaluation(feeder_graph):
    """Verify localization metrics computation."""
    calc = GraphDistanceCalculator(feeder_graph)
    y_pred = [0, 1, 2]
    y_true = [0, 1, 3]

    res = calc.evaluate_localization_errors(y_pred, y_true)
    assert res["num_fault_samples"] == 3
    assert res["exact_accuracy"] == pytest.approx(2.0 / 3.0)
    assert "mean_hop_error" in res
