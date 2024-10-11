"""Downside-volatility factor: long low-downside-risk assets.

Signal: semi-deviation (volatility of negative returns only), annualized.
Direction: Low downside vol → positive weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


def downside_vol_signal(returns: pd.DataFrame, asof_date, lookback: int = 36) -> pd.Series:
    """Annualized downside (semi-)deviation over a trailing window."""
    window = returns.loc[:asof_date].tail(lookback)
    if len(window) < lookback:
        return pd.Series(dtype=float)
    neg = window.clip(upper=0.0)
    return neg.std(ddof=0) * np.sqrt(12)


class DownsideVolFactor(BaseFactor):
    """Downside semi-deviation factor."""

    spec = FactorSpec(
        name="downside_vol",
        signal_type="risk",
        ascending_rank=False,
        target_vol=0.01,
    )

    def __init__(self, lookback: int = 36):
        self.lookback = lookback

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        signal = downside_vol_signal(data.returns, date, self.lookback)
        if signal.empty:
            return pd.Series(0.0, index=data.returns.columns)
        return standardized_rank_weights(signal, ascending=False)
