"""Feature normalizer ensuring zero target leakage between dataset splits."""

from typing import Optional, Union
import numpy as np
import torch


class FeatureNormalizer:
    """Standardizes feature matrices using parameters fitted strictly on training data."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "FeatureNormalizer":
        """Fit normalization mean and standard deviation from training feature array.

        Args:
            X: Array of shape (N, num_features) or (N, num_nodes, num_features).

        Returns:
            Fitted FeatureNormalizer instance.
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.ndim == 3:
            # Reshape (N, num_nodes, num_features) to (N * num_nodes, num_features)
            X_np = X_np.reshape(-1, X_np.shape[-1])

        self.mean = np.nanmean(X_np, axis=0)
        self.std = np.nanstd(X_np, axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply standardization to new data using fitted parameters.

        Args:
            X: Data array or tensor to normalize.

        Returns:
            Normalized array or tensor with same shape and type.
        """
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer must be fitted on training data before calling transform.")

        if isinstance(X, torch.Tensor):
            device = X.device
            mean_t = torch.as_tensor(self.mean, dtype=X.dtype, device=device)
            std_t = torch.as_tensor(self.std, dtype=X.dtype, device=device)
            return (X - mean_t) / std_t
        else:
            X_np = np.asarray(X)
            return (X_np - self.mean) / self.std

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Fit on data and return transformed output."""
        return self.fit(X).transform(X)
