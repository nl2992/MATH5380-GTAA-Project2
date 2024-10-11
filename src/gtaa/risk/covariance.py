"""Rolling covariance estimation.

Project 2 spec:
- 36-month trailing window (never fewer in Project 2 mode).
- Population covariance (ddof=0).
- Annualize by multiplying by 12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_covariance(
    returns: pd.DataFrame,
    asof_date,
    lookback: int = 36,
    annualization: int = 12,
    min_lookback: int = 36,
) -> pd.DataFrame:
    """Compute annualized population covariance from a trailing window.

    Args:
        returns: Full monthly return history (date × asset).
        asof_date: Use data up to and including this date.
        lookback: Number of monthly observations in the window.
        annualization: Multiply covariance matrix by this factor (12 for annual).
        min_lookback: Raise if fewer than this many observations are available.

    Returns:
        Annualized covariance matrix (asset × asset).
    """
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < min_lookback:
        raise ValueError(
            f"Insufficient history: need {min_lookback} months, have {len(window)}"
        )
    return window.cov(ddof=0) * annualization


def rolling_covariance_partial(
    returns: pd.DataFrame,
    asof_date,
    lookback: int = 36,
    annualization: int = 12,
) -> pd.DataFrame:
    """Compute population covariance using only assets with a complete window.

    Unlike rolling_covariance, this function does not raise on partial data.
    Instead it silently drops any column that has at least one NaN in the
    lookback window, computing the covariance on the surviving universe.

    Used by scale_weight_panel to allow the active universe to grow over time
    as ETF price histories extend.

    Args:
        returns:      Full monthly return history (date × asset).
        asof_date:    Use data up to and including this date.
        lookback:     Number of monthly observations.
        annualization: Multiply covariance by this factor.

    Convention A (used throughout): returns through month-end t are known at t.
        Weights formed at t include return t in the covariance estimate.
        Weights at t earn returns at t+1 (enforced in portfolio/returns.py).

        So: window = returns.loc[:asof_date].tail(lookback)
            includes the return AT asof_date (month t), NOT t+1.  <- no look-ahead.

    Returns:
        Annualized population covariance matrix (surviving_assets × surviving_assets).
        Empty DataFrame if fewer than 2 valid assets remain.

    Raises:
        ValueError: If the window itself has fewer than lookback rows (not enough
                    total history regardless of NaN columns).
    """
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < lookback:
        raise ValueError(
            f"Insufficient total history: need {lookback} months, have {len(window)}"
        )
    # Keep only columns with zero NaN in this window
    valid_cols = window.columns[window.notna().all()].tolist()
    if len(valid_cols) < 2:
        return pd.DataFrame()
    return window[valid_cols].cov(ddof=0) * annualization


def shrunk_covariance(
    returns: pd.DataFrame,
    asof_date,
    lookback: int = 36,
    annualization: int = 12,
    shrinkage: float = 0.0,
) -> pd.DataFrame:
    """Ledoit-Wolf-style linear shrinkage toward diagonal.

    shrinkage=0 → no shrinkage (pure sample), shrinkage=1 → pure diagonal.
    HW_GTAA_2 uses shrinkage=0 (population cov); this is provided for labs.
    """
    cov = rolling_covariance(returns, asof_date, lookback, annualization)
    if shrinkage == 0.0:
        return cov
    diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
    return (1 - shrinkage) * cov + shrinkage * diag
