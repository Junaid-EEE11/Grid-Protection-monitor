"""Configuration management module supporting YAML loading, validation, and hierarchical merging."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file from disk.

    Args:
        config_path: File system path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file cannot be parsed as valid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {path}: {exc}") from exc

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries with override precedence.

    Args:
        base_config: Base dictionary.
        override_config: Dictionary containing overrides.

    Returns:
        A new recursively merged dictionary.
    """
    merged = dict(base_config)
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
