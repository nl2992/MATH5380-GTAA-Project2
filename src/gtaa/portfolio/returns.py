"""Portfolio return calculation with correct t+1 alignment.

Core no-look-ahead rule:
    Weights formed at month-end t are applied to the return EARNED
    during month t+1 (i.e., from t to t+1).

Implementation uses returns.shift(-1) to align next-month returns
with the weight date. The last weight row is always dropped because
there is no subsequent return to earn.
"""
from __future__ import annotations

import pandas as pd


def portfolio_returns_from_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute strategy returns: weights at t earn returns at t+1.

    Args:
        weights: Scaled weight DataFrame (date × asset).
                 These are the weights held from date t through t+1.
        returns: Monthly return DataFrame (date × asset).
                 return at date t = return earned during month t.

    Returns:
        Monthly portfolio return Series indexed by the weight date t.
        The last date of weights is always NaN and is dropped.
        Name: "portfolio_return".
    """
    weights = weights.sort_index()
    # Restrict returns to the same date range as weights so that the
    # element-wise multiplication does not inject spurious zero-returns
    # for dates where weights have no rows (e.g. pre-overlap period).
    returns = (
        returns.sort_index()
        .reindex(index=weights.index, columns=weights.columns)
    )

    # next_returns.loc[t] = return earned at t+1
    # We extend the returns one period forward so shift(-1) can capture t+1.
    # Because we've already reindexed to weights.index, the shift operates
    # only within that window.
    next_returns = returns.shift(-1)

    # Weighted sum across assets
    port_rets = (weights * next_returns).sum(axis=1)

    # Drop the last date (no subsequent return exists)
    port_rets = port_rets.iloc[:-1].dropna()
    port_rets.name = "portfolio_return"

    return port_rets
