"""Turnover calculation using drifted weights.

Two drift conventions:
- Derivative-overlay (HW2 default): drifted_i = w_i * (1 + r_i)
  Portfolio notional is fixed; each asset drifts independently.
- Funded portfolio (GTAA strategy): drifted_i = w_i * (1 + r_i) / (1 + R)
  Total portfolio value grows by R; weights are fractions of a growing NAV.

Two function interfaces:
  compute_turnover_series  -- HW2-style, takes a weight panel + returns
  compute_turnover         -- GTAA-style, funded-portfolio convention
"""
from __future__ import annotations

import pandas as pd


def drift_weights(
    prev_weights: pd.Series,
    asset_returns: pd.Series,
    funded_portfolio: bool = False,
) -> pd.Series:
    """Compute how weights drift after earning one period of returns.

    Args:
        prev_weights: Weights at the end of the previous rebalance.
        asset_returns: Realized returns for the period (same asset universe).
        funded_portfolio: If True, normalize by (1 + portfolio_return).
            Default False uses derivative-overlay (HW2 / Project 2 spec).

    Returns:
        Post-drift weights after the return period.
    """
    asset_returns = asset_returns.reindex(prev_weights.index).fillna(0.0)
    drifted = prev_weights * (1 + asset_returns)
    if funded_portfolio:
        portfolio_return = float((prev_weights * asset_returns).sum())
        drifted = drifted / (1 + portfolio_return)
    return drifted


def turnover(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    asset_returns: pd.Series,
    funded_portfolio: bool = False,
) -> float:
    """One-way turnover at a rebalance date.

    Args:
        prev_weights: Weights before the period's returns.
        target_weights: Desired weights at current rebalance date.
        asset_returns: Returns earned since previous rebalance (causes drift).
        funded_portfolio: Passed through to drift_weights.

    Returns:
        Sum of absolute weight changes (one-way).
    """
    drifted = drift_weights(prev_weights, asset_returns, funded_portfolio)
    target = target_weights.reindex(drifted.index).fillna(0.0)
    return float((target - drifted).abs().sum())


def compute_turnover_series(
    final_weights: pd.DataFrame,
    returns: pd.DataFrame,
    funded_portfolio: bool = False,
) -> pd.Series:
    """Compute turnover at each rebalance date (HW2-style interface).

    Args:
        final_weights: Scaled portfolio weights (date x asset).
        returns: Monthly asset returns (date x asset).
        funded_portfolio: If True use funded-portfolio drift (ETF strategies).
            Default False uses derivative-overlay (HW2 / Project 2).

    Returns:
        pd.Series of one-way turnover indexed by rebalance date.
    """
    dates = final_weights.index
    turnover_vals = {}

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_w = final_weights.loc[prev_date]
        curr_w = final_weights.loc[curr_date]

        try:
            ret = returns.loc[curr_date]
        except KeyError:
            continue

        turnover_vals[curr_date] = turnover(prev_w, curr_w, ret, funded_portfolio)

    return pd.Series(turnover_vals, name="turnover")


def compute_turnover(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """One-way turnover using the funded-portfolio drift convention.

    At each rebalance t:
        drifted_i = prev_w_i * (1 + r_{i,t}) / (1 + R_t)
        where R_t = sum_i prev_w_i * r_{i,t}  (portfolio return)
        turnover_t = sum_i |target_w_i - drifted_i|

    Args:
        weights: Scaled final portfolio weights (date x asset).
        returns: Monthly asset returns (date x asset), same universe.

    Returns:
        One-way turnover Series indexed by rebalance date.
    """
    weights = weights.sort_index()
    returns = returns.sort_index().reindex(columns=weights.columns)

    to_vals: dict = {}
    dates = weights.index.intersection(returns.index)

    for i in range(1, len(dates)):
        prev_dt = dates[i - 1]
        curr_dt = dates[i]

        prev_w  = weights.loc[prev_dt].fillna(0.0)
        target_w = weights.loc[curr_dt].fillna(0.0)
        asset_ret = returns.loc[curr_dt].fillna(0.0)

        port_ret = float((prev_w * asset_ret).sum())
        denom = 1.0 + port_ret

        if abs(denom) < 1e-12:
            continue

        drifted_w = prev_w * (1.0 + asset_ret) / denom
        to_vals[curr_dt] = float((target_w - drifted_w).abs().sum())

    return pd.Series(to_vals, name="turnover")
