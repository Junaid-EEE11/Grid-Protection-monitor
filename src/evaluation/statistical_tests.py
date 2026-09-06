"""Statistical significance testing and non-parametric bootstrap confidence interval estimation."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for an evaluation metric.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        metric_fn: Function metric_fn(y_true_sample, y_pred_sample) -> float.
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Desired confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    trues = np.asarray(y_true)
    preds = np.asarray(y_pred)
    n_samples = len(trues)

    point_estimate = float(metric_fn(trues, preds))

    rng = np.random.RandomState(seed)
    bootstrap_estimates = []

    for _ in range(n_bootstraps):
        sample_indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_trues = trues[sample_indices]
        sample_preds = preds[sample_indices]
        try:
            val = float(metric_fn(sample_trues, sample_preds))
            bootstrap_estimates.append(val)
        except Exception:
            continue

    if not bootstrap_estimates:
        return point_estimate, point_estimate, point_estimate

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_estimates, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrap_estimates, 100.0 * (1.0 - alpha)))

    return point_estimate, ci_lower, ci_upper


def paired_permutation_test(
    metric_errors_model_a: np.ndarray,
    metric_errors_model_b: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute two-sided paired permutation test for difference in model performance.

    Args:
        metric_errors_model_a: Sample-level metric errors for Model A (e.g. hop error or 0/1 indicator).
        metric_errors_model_b: Sample-level metric errors for Model B.
        n_permutations: Number of random sign permutations.
        seed: Random seed.

    Returns:
        Tuple of (observed_difference, p_value).
    """
    diffs = np.asarray(metric_errors_model_a) - np.asarray(metric_errors_model_b)
    obs_diff = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    perm_diffs = []

    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diffs.append(float(np.mean(diffs * signs)))

    perm_diffs_arr = np.array(perm_diffs)
    p_val = float(np.mean(np.abs(perm_diffs_arr) >= np.abs(obs_diff)))

    return obs_diff, p_val
