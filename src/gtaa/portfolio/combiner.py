"""Factor combination and final portfolio scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.models import BacktestConfig
from gtaa.risk.covariance import rolling_covariance
from gtaa.risk.scaling import scale_to_target_vol


def combine_fmps_equal_weight(scaled_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average scaled FMP weights equally across factors.

    Args:
        scaled_weights: Dict mapping factor name → scaled weight DataFrame.

    Returns:
        Combined weight DataFrame (date × asset).
    """
    all_dfs = list(scaled_weights.values())
    common_index = all_dfs[0].index
    for df in all_dfs[1:]:
        common_index = common_index.intersection(df.index)

    combined = sum(df.loc[common_index] for df in all_dfs) / len(all_dfs)
    combined.index.name = "date"
    return combined


def combine_fmps_custom(
    scaled_weights: dict[str, pd.DataFrame],
    factor_weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted combination of FMPs using user-specified factor weights.

    factor_weights values must sum to 1.0.
    """
    total = sum(factor_weights.values())
    common_index = list(scaled_weights.values())[0].index
    for df in scaled_weights.values():
        common_index = common_index.intersection(df.index)

    combined = sum(
        df.loc[common_index] * (factor_weights.get(name, 0.0) / total)
        for name, df in scaled_weights.items()
    )
    return combined


def scale_combined_portfolio(
    combined_weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Apply 1% vol scaling to the combined (multi-factor) portfolio.

    Each row in combined_weights is scaled using the 36-month covariance
    estimated at that date.
    """
    final_rows = {}
    for date in combined_weights.index:
        raw = combined_weights.loc[date]
        try:
            cov = rolling_covariance(
                returns,
                date,
                lookback=config.covariance_lookback_months,
                annualization=config.covariance_annualization,
                min_lookback=config.covariance_lookback_months,
            )
            scaled = scale_to_target_vol(raw, cov, config.portfolio_target_vol)
        except ValueError:
            continue
        final_rows[date] = scaled

    result = pd.DataFrame(final_rows).T
    result.index.name = "date"
    return result
