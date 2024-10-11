"""Return series cleaning utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(df: pd.DataFrame, n_sigma: float = 3.0) -> pd.DataFrame:
    """Winsorize each column to ±n_sigma cross-sectional standard deviations."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    lo = (mean - n_sigma * std).values[:, None]
    hi = (mean + n_sigma * std).values[:, None]
    return df.clip(lower=lo, upper=hi, axis=0)


def drop_short_series(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Drop columns with fewer than min_obs non-NaN values."""
    return df.loc[:, df.notna().sum() >= min_obs]


def fill_forward_limited(df: pd.DataFrame, limit: int = 1) -> pd.DataFrame:
    """Forward-fill up to `limit` consecutive NaNs."""
    return df.ffill(limit=limit)


def audit_missing(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return a table of per-column missing value counts in [start, end]."""
    window = df.loc[start:end]
    total = len(window)
    missing = window.isna().sum()
    pct = (missing / total * 100).round(2)
    return pd.DataFrame({"total_obs": total, "missing": missing, "pct_missing": pct})
