"""Date alignment utilities.

Critical invariant: weights formed at date t are multiplied by returns at t+1.
Any violation is look-ahead bias.
"""
from __future__ import annotations

import pandas as pd


def align_weights_to_next_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift weights forward by one period so w_t earns r_{t+1}.

    Returns:
        (aligned_weights, aligned_returns) with a common index.
        The first return period is dropped (no prior weights).
        The last weight date is dropped (no subsequent returns).
    """
    common_assets = weights.columns.intersection(returns.columns)
    w = weights[common_assets]
    r = returns[common_assets]

    # w at t → pair with r at t+1
    # After shifting, row at date d holds w from date d-1.
    # Drop the first row (no prior weight) then restrict r to the same dates.
    w_shifted = w.shift(1).dropna(how="all")
    common_dates = w_shifted.index.intersection(r.index)
    return w_shifted.loc[common_dates], r.loc[common_dates]


def next_month_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return dates shifted forward by one month-end."""
    return dates + pd.offsets.MonthEnd(1)


def portfolio_return_series(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute monthly portfolio returns given weight and return DataFrames.

    weights.index must contain weight-formation dates.
    Portfolio return at t is (w_{t-1} · r_t).

    This is the only correct way to form portfolio returns; do not bypass this.
    """
    w_aligned, r_aligned = align_weights_to_next_returns(weights, returns)
    port_ret = (w_aligned * r_aligned).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret
