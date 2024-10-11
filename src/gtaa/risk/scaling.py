"""Volatility scaling – scale portfolio weights to a target annualized vol."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.risk.covariance import rolling_covariance_partial


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Annualized ex-ante portfolio volatility.

    Args:
        weights: Asset weights (may include assets not in cov; they are zeroed).
        cov: Annualized covariance matrix.

    Returns:
        Scalar annualized volatility.
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    var = float(w @ cov.values @ w)
    return float(np.sqrt(max(var, 0.0)))


def scale_to_target_vol(
    weights: pd.Series,
    cov: pd.DataFrame,
    target_vol: float = 0.01,
) -> pd.Series:
    """Scale weights so the portfolio has exactly target_vol ex-ante volatility.

    Raises:
        ValueError: If current volatility is zero or NaN (cannot scale).
    """
    vol = portfolio_vol(weights, cov)
    if vol <= 0 or np.isnan(vol):
        raise ValueError(
            f"Cannot scale: portfolio volatility is {vol}. "
            "Check that weights and covariance are non-degenerate."
        )
    return weights * (target_vol / vol)


def scale_weight_panel(
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.01,
    lookback: int = 36,
    annualization: int = 12,
) -> pd.DataFrame:
    """Scale a panel of raw FMP weights to target_vol at every date.

    At each date t:
      1. Identify assets with a complete lookback window of returns (no NaN).
      2. Compute population covariance (ddof=0) on that partial universe.
      3. Scale the weight vector so its ex-ante vol equals target_vol.

    Assets that lack sufficient return history at a given date are zeroed out.
    This allows the active universe to grow over time as ETF histories lengthen.

    Args:
        raw_weights:  Raw FMP weights (date × asset). Sum should be ~0 per row.
        returns:      Monthly return history (date × asset).
        target_vol:   Annualized volatility target (e.g. 0.01 = 1%).
        lookback:     Number of monthly return observations in the covariance window.
        annualization: Factor to annualize monthly covariance (12 for annual).

    Returns:
        Scaled weight DataFrame (same shape as raw_weights), with dates that
        have insufficient history silently skipped (zero row).
    """
    raw_weights = raw_weights.sort_index()
    returns = returns.sort_index()

    scaled = pd.DataFrame(
        0.0, index=raw_weights.index, columns=raw_weights.columns, dtype=float
    )

    for dt in raw_weights.index:
        try:
            # Covariance on assets with complete lookback (handles partial universe)
            cov = rolling_covariance_partial(
                returns, dt, lookback=lookback, annualization=annualization
            )
        except ValueError:
            continue

        if cov.empty:
            continue

        # Restrict weights to the assets included in the covariance
        w = raw_weights.loc[dt].reindex(cov.index).fillna(0.0)

        vol = portfolio_vol(w, cov)
        if vol <= 0.0 or np.isnan(vol):
            continue

        scaled_w = w * (target_vol / vol)
        scaled.loc[dt, cov.index] = scaled_w.values

    return scaled
