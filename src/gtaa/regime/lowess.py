"""Zero-order LOWESS (locally weighted scatter smoother) forecast.

HW_GTAA_1: Given kernel weights w_t for each historical date t,
the conditional expected return for asset j is the weighted mean of
future (next-month) returns:

    E[r_{j,t+1} | current regime] = Σ_t w_t * r_{j,t+1}

This is a zero-order (constant) regression, i.e. weighted average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.regime.distances import euclidean_distances_to_reference, normalized_distances
from gtaa.regime.kernels import polynomial_kernel, normalize_weights
from gtaa.regime.standardization import expanding_zscore_df


def zero_order_lowess_forecast(
    kernel_weights: pd.Series,
    next_returns: pd.Series,
) -> float:
    """Weighted mean of next-period returns using kernel weights.

    Args:
        kernel_weights: Normalized kernel weights for each historical date.
        next_returns: Forward (t+1) return series aligned to the same dates.

    Returns:
        Scalar conditional expected return.
    """
    aligned = kernel_weights.align(next_returns, join="inner")
    w, r = aligned
    w = w.fillna(0.0)
    r = r.fillna(0.0)
    return float((w * r).sum())


def compute_regime_forecasts(
    standardized_macro: pd.DataFrame,
    asset_returns: pd.DataFrame,
    asof_date,
    kernel_degree: int = 3,
) -> pd.Series:
    """Compute conditional return forecasts for all assets at asof_date.

    Pipeline:
        1. Euclidean distances of all historical regimes to current regime.
        2. Normalize distances to [0,1].
        3. Apply polynomial kernel to get weights.
        4. Normalize kernel weights to sum to 1.
        5. Weighted mean of next-period returns.

    Args:
        standardized_macro: Expanding-z-scored macro DataFrame.
        asset_returns: Monthly asset returns (date × asset).
        asof_date: The date we are forecasting FROM (weights formed here).
        kernel_degree: Polynomial kernel degree.

    Returns:
        pd.Series of conditional expected returns per asset.
    """
    distances = euclidean_distances_to_reference(standardized_macro, asof_date)
    norm_d = normalized_distances(distances)
    raw_weights = polynomial_kernel(norm_d, degree=kernel_degree)
    weights = normalize_weights(raw_weights)

    # For each asset, compute weighted sum of NEXT period returns
    forecasts = {}
    for asset in asset_returns.columns:
        ret = asset_returns[asset]
        # next-period returns: shift back by 1 so weight at t pairs with r_{t+1}
        next_ret = ret.shift(-1)
        f = zero_order_lowess_forecast(weights, next_ret)
        forecasts[asset] = f

    return pd.Series(forecasts, name=str(asof_date))
