"""Final risk table generation: covariance, volatilities, correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.risk.covariance import rolling_covariance_partial


def final_risk_tables(
    returns: pd.DataFrame,
    final_date,
    lookback: int = 36,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Compute covariance, individual volatilities, and correlation at final_date.

    Uses rolling_covariance_partial so assets with incomplete lookback windows
    are excluded gracefully.

    Args:
        returns:    Full return history (date × asset).
        final_date: Compute covariance as of this date.
        lookback:   Window length in months (default 36).

    Returns:
        cov:   Annualized population covariance matrix (valid_assets × valid_assets).
        vols:  Annualized volatility per asset (Series).
        corr:  Correlation matrix derived from cov.
    """
    cov = rolling_covariance_partial(returns, final_date, lookback=lookback)

    if cov.empty:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=float), empty

    vols = pd.Series(
        np.sqrt(np.diag(cov.values)),
        index=cov.index,
        name="Annualized Volatility",
    )

    # Correlation = cov / (vol_i × vol_j)
    corr = cov.div(vols, axis=0).div(vols, axis=1)
    corr = corr.clip(-1.0, 1.0)

    return cov, vols, corr
