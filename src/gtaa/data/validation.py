"""Data validation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_month_end_index(df: pd.DataFrame, name: str = "") -> list[str]:
    """Return a list of warning strings if index is not all month-end."""
    issues = []
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append(f"{name}: index is not DatetimeIndex")
        return issues
    not_month_end = df.index[df.index != df.index + pd.offsets.MonthEnd(0)]
    if len(not_month_end) > 0:
        issues.append(f"{name}: {len(not_month_end)} dates are not month-end")
    return issues


def check_no_lookahead(weights_dates: pd.DatetimeIndex, returns_dates: pd.DatetimeIndex) -> bool:
    """Return True if every weight date strictly precedes its paired return date."""
    if len(weights_dates) != len(returns_dates):
        return False
    deltas = (returns_dates - weights_dates).days
    return bool((deltas > 0).all())


def check_weights_zero_sum(weights: pd.Series, tol: float = 1e-6) -> bool:
    """Return True if weights sum to approximately zero (long/short portfolio)."""
    return abs(weights.sum()) < tol


def check_portfolio_vol(weights: pd.Series, cov: pd.DataFrame, target: float, tol: float = 0.001) -> bool:
    """Return True if portfolio vol is within tol of target."""
    w = weights.reindex(cov.index).fillna(0.0).values
    vol = float(np.sqrt(w @ cov.values @ w))
    return abs(vol - target) < tol
