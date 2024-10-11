"""Signal computation for the GTAA strategy.

Momentum from returns: 12-1 cumulative, skipping t-1 (equity, FI)
Momentum from prices:  P_{t-2}/P_{t-13} - 1 (commodity, FX)
Carry signals:         pass-through from pre-computed signal sheets

All signals use only data available through t-1 (no look-ahead).
"""
from __future__ import annotations

import pandas as pd


def momentum_12_1_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Conservative 12-1 momentum signal computed from return series.

    At signal date t, the cumulative return uses months t-12 through t-2
    (11 monthly returns), skipping the most recent month t-1 to avoid
    short-term reversal contamination.

    Formula:
        MOM_{i,t} = Π_{j=2}^{12} (1 + r_{i,t-j}) - 1

    Implementation:
        gross.shift(2) at time t gives r_{t-2}
        rolling(11) window ending at t-2 covers t-12 ... t-2  ✓

    Args:
        returns: Monthly return DataFrame (date × asset). Index must be sorted.

    Returns:
        Signal DataFrame of same shape. NaN where insufficient history.
    """
    returns = returns.sort_index()
    gross = 1.0 + returns
    signal = (
        gross
        .shift(2)
        .rolling(11, min_periods=11)
        .apply(lambda x: x.prod() - 1.0, raw=True)
    )
    return signal


def momentum_12_1_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Conservative 12-1 momentum signal computed from price series.

    Price-based equivalent of the return-based signal:
        MOM_{i,t} = P_{i,t-2} / P_{i,t-13} - 1

    This spans the same 11 monthly return periods as the return-based version.

    Args:
        prices: Monthly adjusted-close price DataFrame (date × asset).

    Returns:
        Signal DataFrame of same shape. NaN where insufficient history.
    """
    prices = prices.sort_index()
    return prices.shift(2) / prices.shift(13) - 1.0


def fixed_income_carry_signal(fi_carry: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for the pre-computed fixed-income carry signal panel.

    The fi_carry_signals sheet already contains the appropriate yield/carry
    proxy for each FI ETF. Higher values correspond to higher carry, so
    rank_standardized_weights should be called with high_is_good=True.

    Args:
        fi_carry: DataFrame of carry signals (date × FI ETF ticker).

    Returns:
        Copy of the input (no transformation needed at this stage).
    """
    return fi_carry.copy()


def fx_carry_signal(fx_carry: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for the FX carry signal (foreign 3M rate minus USD 3M rate).

    Higher = higher carry. Use with high_is_good=True.
    """
    return fx_carry.copy()
