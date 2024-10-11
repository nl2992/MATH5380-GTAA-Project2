"""Low-volatility factor: long low-realized-vol assets, short high-vol.

Signal: trailing 36-month annualized volatility.
Direction: Low vol → positive weight (ascending=False in rank).

Per Project 2 spec: use 36-month trailing volatility to match the
covariance window length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


def realized_vol_signal(returns: pd.DataFrame, asof_date, lookback: int = 36) -> pd.Series:
    """Annualized realized volatility using a trailing window.

    Args:
        returns: Monthly return DataFrame.
        asof_date: Use data up to and including this date.
        lookback: Number of months.

    Returns:
        Cross-sectional annualized volatility per asset.
    """
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < lookback:
        return pd.Series(dtype=float)
    return window.std(ddof=0) * np.sqrt(12)


class LowVolFactor(BaseFactor):
    """Low-realized-volatility factor (36-month trailing vol)."""

    spec = FactorSpec(
        name="low_vol",
        signal_type="risk",
        ascending_rank=False,
        target_vol=0.01,
    )

    def __init__(self, lookback: int = 36):
        self.lookback = lookback

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        signal = realized_vol_signal(data.returns, date, self.lookback)
        if signal.empty:
            return pd.Series(0.0, index=data.returns.columns)
        # ascending=False → lowest vol gets highest rank → positive weight
        return standardized_rank_weights(signal, ascending=False)
