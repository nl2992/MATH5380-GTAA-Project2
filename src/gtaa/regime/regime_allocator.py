"""Dynamic factor allocation using macro-regime kernel forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.regime.lowess import compute_regime_forecasts
from gtaa.regime.standardization import expanding_zscore_df


def dynamic_factor_weights(
    factor_returns: pd.DataFrame,
    macro: pd.DataFrame,
    asof_date,
    kernel_degree: int = 3,
    fallback_equal_weight: bool = True,
) -> pd.Series:
    """Compute regime-weighted factor allocations at asof_date.

    Args:
        factor_returns: Monthly factor return series (date × factor).
        macro: Raw macro DataFrame (date × variable). Will be z-scored expanding.
        asof_date: Current evaluation date.
        kernel_degree: Polynomial kernel degree.
        fallback_equal_weight: If all forecasts are non-positive, use equal weights.

    Returns:
        pd.Series of factor allocation weights (summing to 1.0).
    """
    std_macro = expanding_zscore_df(macro)
    forecasts = compute_regime_forecasts(
        std_macro, factor_returns, asof_date, kernel_degree
    )

    if fallback_equal_weight and (forecasts <= 0).all():
        n = len(forecasts)
        return pd.Series(1.0 / n, index=forecasts.index)

    # Positive forecasts only; renormalize
    pos = forecasts.clip(lower=0.0)
    total = pos.sum()
    if total == 0:
        n = len(pos)
        return pd.Series(1.0 / n, index=pos.index)
    return pos / total
