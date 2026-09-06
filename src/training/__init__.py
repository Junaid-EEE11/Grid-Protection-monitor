"""Training loops and multi-task optimization for GNNs and baseline models."""

from src.training.trainer import MultiTaskTrainer, TrainConfig, TrainHistory
from src.training.train_baselines import train_and_eval_baselines

__all__ = [
    "MultiTaskTrainer",
    "TrainConfig",
    "TrainHistory",
    "train_and_eval_baselines",
]
