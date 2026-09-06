"""Unit tests for evaluation metrics, calibration scoring, and statistical bootstrap."""

import numpy as np
import pytest

from src.evaluation.calibration import compute_ece, TemperatureScaler
from src.evaluation.metrics import evaluate_classification, evaluate_detection, evaluate_localization
from src.evaluation.statistical_tests import bootstrap_metric_ci, paired_permutation_test


def test_detection_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_prob = [0.1, 0.9, 0.4, 0.2, 0.85]

    m = evaluate_detection(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(0.80)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(2.0 / 3.0)
    assert 0.0 <= m["auroc"] <= 1.0


def test_classification_metrics():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]

    m = evaluate_classification(y_true, y_pred, num_classes=5)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)


def test_calibration_ece():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    ece, mce, diag = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(diag.bin_centers) == 5


def test_bootstrap_ci():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])

    pt, low, high = bootstrap_metric_ci(
        y_true, y_pred, lambda yt, yp: evaluate_detection(yt, yp)["f1"], n_bootstraps=100
    )
    assert low <= pt <= high


def test_paired_permutation_test():
    err_a = np.array([0.1, 0.2, 0.1, 0.3])
    err_b = np.array([0.5, 0.6, 0.4, 0.7])

    diff, p_val = paired_permutation_test(err_a, err_b, n_permutations=200)
    assert diff < 0.0
    assert 0.0 <= p_val <= 1.0
