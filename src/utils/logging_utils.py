"""Structured logging and experiment tracking utilities."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_logger(name: str = "distribution_fault_research", level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with standard formatting.

    Args:
        name: Name of the logger.
        level: Logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ExperimentTracker:
    """Local JSON/CSV experiment tracking to record parameters, seeds, and metrics reproducibly."""

    def __init__(self, experiment_dir: Union[str, Path]):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.experiment_dir / "experiment_runs.jsonl"

    def log_run(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        seed: int,
        metrics: Dict[str, Any],
        model_name: str,
        dataset_name: str,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record a single experiment run entry with metadata.

        Args:
            experiment_name: Identifier for the experiment.
            config: Full experiment configuration.
            seed: Random seed used.
            metrics: Dictionary of evaluated metrics.
            model_name: Model identifier.
            dataset_name: Dataset / split identifier.
            artifacts: Optional paths to saved models or figures.

        Returns:
            Dictionary record of the logged run.
        """
        run_record = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts or {},
        }

        with open(self.runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        return run_record
