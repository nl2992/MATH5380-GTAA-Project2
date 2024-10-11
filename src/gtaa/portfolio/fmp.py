"""Factor-Mimicking Portfolio (FMP) construction and combination.

For each factor and each rebalance date:
1. Compute raw factor weights (cross-sectional, sum ~0).
2. Estimate rolling 36-month covariance.
3. Scale weights to 1% target annualized volatility.
"""
from __future__ import annotations

import pandas as pd

from gtaa.factors.base import BaseFactor
from gtaa.models import BacktestConfig, DataBundle
from gtaa.risk.covariance import rolling_covariance
from gtaa.risk.scaling import scale_to_target_vol


def build_fmp_weights(
    factor: BaseFactor,
    data: DataBundle,
    rebalance_dates: pd.DatetimeIndex,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw and 1%-scaled FMP weights for all rebalance dates.

    Args:
        factor: A BaseFactor instance with compute_raw_weights method.
        data: DataBundle with aligned returns and signals.
        rebalance_dates: Monthly rebalance dates to compute weights on.
        config: BacktestConfig controlling covariance window and vol target.

    Returns:
        (raw_weights_df, scaled_weights_df) each indexed by rebalance date.
    """
    raw_rows = {}
    scaled_rows = {}

    for date in rebalance_dates:
        try:
            raw = factor.compute_raw_weights(data, date)
            cov = rolling_covariance(
                data.returns,
                date,
                lookback=config.covariance_lookback_months,
                annualization=config.covariance_annualization,
                min_lookback=config.covariance_lookback_months,
            )
            scaled = scale_to_target_vol(raw, cov, target_vol=config.factor_target_vol)
        except (ValueError, KeyError, IndexError):
            continue

        raw_rows[date] = raw
        scaled_rows[date] = scaled

    raw_df = pd.DataFrame(raw_rows).T
    scaled_df = pd.DataFrame(scaled_rows).T

    raw_df.index.name = "date"
    scaled_df.index.name = "date"

    return raw_df, scaled_df


def combine_weight_panels(
    weight_panels: dict[str, pd.DataFrame],
    allocation_weights: dict[str, float],
) -> pd.DataFrame:
    """Combine multiple scaled FMP weight panels into a single portfolio.

    Only dates where ALL sleeves simultaneously have active weights are
    included (intersection of date indices). This ensures the combined
    portfolio is a true multi-sleeve portfolio throughout its history —
    no partial-sleeve periods are included.

    Assets are the union of all sleeve asset universes; a sleeve's assets
    that are not present in another sleeve default to zero on shared dates.

    Args:
        weight_panels:     Dict of sleeve_name → scaled weight DataFrame (date × asset).
        allocation_weights: Dict of sleeve_name → scalar weight (e.g. 1/3 each).

    Returns:
        Combined weight DataFrame (intersection_dates × all_assets), sorted by date.
    """
    # Intersection: only keep dates where every sleeve is simultaneously live
    date_sets = [set(w.index) for w in weight_panels.values()]
    all_dates = sorted(set.intersection(*date_sets))
    all_assets = sorted(set().union(*[set(w.columns) for w in weight_panels.values()]))

    combo = pd.DataFrame(0.0, index=pd.to_datetime(all_dates), columns=all_assets)
    combo = combo.sort_index()
    combo.index.name = "date"

    intersection_idx = pd.to_datetime(all_dates)
    for name, panel in weight_panels.items():
        alloc = allocation_weights.get(name, 0.0)
        panel = panel.sort_index().reindex(intersection_idx).fillna(0.0)
        combo.loc[panel.index, panel.columns] += alloc * panel

    return combo
