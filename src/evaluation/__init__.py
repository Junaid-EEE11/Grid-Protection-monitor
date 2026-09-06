"""Evaluation metrics, calibration assessment, and statistical significance testing."""

from src.evaluation.metrics import evaluate_detection, evaluate_classification, evaluate_localization, FullEvaluationReport
from src.evaluation.calibration import compute_ece, TemperatureScaler, ReliabilityDiagramData
from src.evaluation.statistical_tests import bootstrap_metric_ci, paired_permutation_test

__all__ = [
    "evaluate_detection",
    "evaluate_classification",
    "evaluate_localization",
    "FullEvaluationReport",
    "compute_ece",
    "TemperatureScaler",
    "ReliabilityDiagramData",
    "bootstrap_metric_ci",
    "paired_permutation_test",
]
