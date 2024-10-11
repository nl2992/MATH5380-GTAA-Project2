"""Config loader – reads YAML files into BacktestConfig instances."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gtaa.models import BacktestConfig


def load_config(path: str | Path) -> BacktestConfig:
    """Load a YAML config file and return a BacktestConfig."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return BacktestConfig(**{k: v for k, v in raw.items() if hasattr(BacktestConfig, k)})


def config_from_dict(d: dict) -> BacktestConfig:
    """Build a BacktestConfig from a plain dict (e.g. from notebook widgets)."""
    return BacktestConfig(**{k: v for k, v in d.items() if hasattr(BacktestConfig, k)})
