"""Unit tests for sequence component transformations and physics feature extraction."""

import math
import numpy as np
import pytest

from src.features.normalizer import FeatureNormalizer
from src.features.physics_engine import PhysicsFeatureExtractor
from src.features.sequence_components import calculate_symmetrical_phasors, compute_sequence_components


def test_symmetrical_phasors_balanced():
    """Test that a balanced 3-phase set yields only positive sequence (V1) and zero V0, V2."""
    va = 1.0 + 0.0j
    vb = -0.5 - 0.8660254j  # 1.0 /_ -120 deg
    vc = -0.5 + 0.8660254j  # 1.0 /_ +120 deg

    v0, v1, v2 = calculate_symmetrical_phasors(va, vb, vc)

    assert abs(v0) == pytest.approx(0.0, abs=1e-4)
    assert abs(v1) == pytest.approx(1.0, abs=1e-4)
    assert abs(v2) == pytest.approx(0.0, abs=1e-4)


def test_symmetrical_phasors_unbalanced_slg():
    """Test that an SLG fault (Phase A collapsed to 0) induces non-zero zero and negative sequence."""
    va = 0.0 + 0.0j
    vb = -0.5 - 0.8660254j
    vc = -0.5 + 0.8660254j

    v0, v1, v2 = calculate_symmetrical_phasors(va, vb, vc)

    assert abs(v0) > 0.30
    assert abs(v1) > 0.60
    assert abs(v2) > 0.30


def test_compute_sequence_components_wrapper():
    """Test sequence component helper with magnitudes and angles in degrees."""
    mags = [1.0, 1.0, 1.0]
    angs = [0.0, -120.0, 120.0]

    v0, v1, v2 = compute_sequence_components(mags, angs, [1, 1, 1])
    assert v0 == pytest.approx(0.0, abs=1e-4)
    assert v1 == pytest.approx(1.0, abs=1e-4)
    assert v2 == pytest.approx(0.0, abs=1e-4)


def test_physics_feature_extractor():
    """Test physics feature extraction across buses."""
    bus_meas = {
        "bus1": {
            "v_pu": [1.0, 0.98, 1.02],
            "v_ang": [0.0, -120.0, 120.0],
            "phase_mask": [1, 1, 1],
        }
    }
    extractor = PhysicsFeatureExtractor()
    node_feats = extractor.extract_node_features(bus_meas)
    assert "bus1" in node_feats
    assert node_feats["bus1"].ndim == 1
    assert len(node_feats["bus1"]) == 20


def test_feature_normalizer_no_leakage():
    """Test that FeatureNormalizer fits strictly on train and transforms val/test."""
    X_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    X_test = np.array([[20.0, 200.0]])

    normalizer = FeatureNormalizer()
    assert not normalizer.fitted

    X_train_norm = normalizer.fit_transform(X_train)
    assert normalizer.fitted
    assert np.allclose(np.mean(X_train_norm, axis=0), [0.0, 0.0])

    X_test_norm = normalizer.transform(X_test)
    assert np.allclose(X_test_norm, [[0.0, 0.0]])
