"""Drawdown-resilience factor: long assets with shallower max drawdowns.

Signal: maximum drawdown over a trailing window (negative number).
Direction: Less negative (shallower) max drawdown → positive weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


def max_drawdown_signal(returns: pd.DataFrame, asof_date, lookback: int = 36) -> pd.Series:
    """Maximum drawdown (negative) for each asset over a trailing window."""
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < lookback:
        return pd.Series(dtype=float)

    def col_mdd(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        hwm = cum.cummax()
        dd = cum / hwm - 1
        return float(dd.min())

    return window.apply(col_mdd)


class DrawdownResilienceFactor(BaseFactor):
    """Long shallow-drawdown, short deep-drawdown assets."""

    spec = FactorSpec(
        name="drawdown_resilience",
        signal_type="risk",
        ascending_rank=False,
        target_vol=0.01,
    )

    def __init__(self, lookback: int = 36):
        self.lookback = lookback

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        signal = max_drawdown_signal(data.returns, date, self.lookback)
        if signal.empty:
            return pd.Series(0.0, index=data.returns.columns)
        # Max drawdown is negative; ascending=False → least negative = highest rank = positive weight
        return standardized_rank_weights(signal, ascending=False)
