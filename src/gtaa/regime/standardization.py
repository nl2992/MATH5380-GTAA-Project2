"""Expanding-window z-score standardization for regime variables.

HW_GTAA_1 uses a leave-one-out expanding window: z_t is computed using only
data from t-1 and earlier (prior history only). The current observation is NOT
included in its own mean/std, matching the Excel reference implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_zscore(series: pd.Series, min_periods: int = 2) -> pd.Series:
    """Leave-one-out expanding z-score: z_t = (x_t - mean(x_1..t-1)) / std(x_1..t-1, ddof=1).

    Matches HW1 Excel formula: =(x_t - AVERAGE(prior)) / STDEV(prior), where STDEV uses ddof=1.
    Mean and std are computed from prior history only (shift(1) before applying).
    Returns NaN until min_periods prior observations are available.
    """
    mean = series.expanding(min_periods=min_periods).mean().shift(1)
    std = series.expanding(min_periods=min_periods).std(ddof=1).shift(1)
    z = (series - mean) / std
    return z


def expanding_zscore_df(df: pd.DataFrame, min_periods: int = 2) -> pd.DataFrame:
    """Apply expanding_zscore column-wise."""
    return df.apply(expanding_zscore, min_periods=min_periods)


def cpi_yoy(cpi: pd.Series) -> pd.Series:
    """Compute CPI year-over-year change: (CPI_t / CPI_{t-12}) - 1."""
    return cpi / cpi.shift(12) - 1


def lei_mom_change(lei: pd.Series) -> pd.Series:
    """Month-over-month change in LEI level."""
    return lei.diff()
