"""Simple file-based cache for downloaded data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_exists(path: Path) -> bool:
    return Path(path).exists()
