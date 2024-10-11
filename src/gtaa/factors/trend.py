"""Trend / time-series momentum factor.

Signal: 12-month total return (price momentum without skip for multi-asset,
or 12-1 for country equities per convention).
Direction: Positive trend → long (ascending=True in rank).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


def trend_signal(returns: pd.DataFrame, asof_date, lookback: int = 12) -> pd.Series:
    """Cumulative return over a trailing window as trend signal."""
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < lookback:
        return pd.Series(dtype=float)
    return (1 + window).prod() - 1


class TrendFactor(BaseFactor):
    """Cross-sectional 12-month trend / momentum factor."""

    spec = FactorSpec(
        name="trend",
        signal_type="momentum",
        ascending_rank=True,
        target_vol=0.01,
    )

    def __init__(self, lookback: int = 12):
        self.lookback = lookback

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        signal = trend_signal(data.returns, date, self.lookback)
        if signal.empty:
            return pd.Series(0.0, index=data.returns.columns)
        return standardized_rank_weights(signal, ascending=True)
