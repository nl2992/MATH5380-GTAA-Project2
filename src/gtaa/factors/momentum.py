"""LEGACY: This module was used for HW1/HW2 regime replication only.

For the Project 2 GTAA strategy, use:
  from gtaa.factors.signals import momentum_12_1_from_returns, momentum_12_1_from_prices

The MomentumFactor class below is preserved for backward-compatible test coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


def momentum_signal(returns: pd.DataFrame, asof_date, lookback: int = 12, skip: int = 1) -> pd.Series:
    """LEGACY wrapper. Delegates to momentum_12_1_from_returns for Project 2 strategy.
    For HW replication only. Returns the signal at asof_date for the given history."""
    from gtaa.factors.signals import momentum_12_1_from_returns
    signal_panel = momentum_12_1_from_returns(returns.loc[:asof_date])
    if asof_date in signal_panel.index:
        return signal_panel.loc[asof_date].dropna()
    return pd.Series(dtype=float)


class MomentumFactor(BaseFactor):
    """12-1 price momentum factor for country equities.

    LEGACY / HW use only. For the Project 2 GTAA strategy use
    gtaa.factors.signals.momentum_12_1_from_returns directly.
    """

    spec = FactorSpec(
        name="momentum",
        signal_type="momentum",
        ascending_rank=True,
        target_vol=0.01,
    )

    def __init__(self, lookback: int = 12, skip: int = 1):
        self.lookback = lookback
        self.skip = skip

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        signal = momentum_signal(data.returns, date, self.lookback, self.skip)
        if signal.empty:
            return pd.Series(0.0, index=data.returns.columns)
        # High momentum → positive weight
        return standardized_rank_weights(signal, ascending=True)
