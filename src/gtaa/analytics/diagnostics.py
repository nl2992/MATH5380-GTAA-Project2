"""Diagnostic helpers for strategy inspection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.risk.covariance import rolling_covariance
from gtaa.risk.scaling import portfolio_vol


def factor_vol_timeseries(
    scaled_weights: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 36,
    annualization: int = 12,
) -> pd.Series:
    """Compute ex-ante factor volatility at each rebalance date."""
    vols = {}
    for date in scaled_weights.index:
        try:
            cov = rolling_covariance(returns, date, lookback, annualization, lookback)
            vols[date] = portfolio_vol(scaled_weights.loc[date], cov)
        except ValueError:
            pass
    return pd.Series(vols, name="ex_ante_vol")


def correlation_over_time(returns_a: pd.Series, returns_b: pd.Series, window: int = 36) -> pd.Series:
    """Rolling correlation between two return series."""
    return returns_a.rolling(window).corr(returns_b)
