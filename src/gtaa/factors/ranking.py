"""Cross-sectional rank-standardized weight construction.

Converts a signal DataFrame (date × asset) into raw FMP weights:
  1. For each date, rank assets by signal value.
  2. Demean the ranks.
  3. Divide by the cross-sectional standard deviation (ddof=0).

Result: long/short weights that sum to zero across the active universe.

Note on ddof convention:
    ddof=0 here (population std over cross-section of ranks).
    This differs from the HW1/HW2 rank standardization which used ddof=1
    to match Excel STDEV. The GTAA strategy uses the population convention
    per the assignment specification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_standardized_weights(
    signals: pd.DataFrame,
    high_is_good: bool = True,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Convert a signal panel into cross-sectional rank-standardized FMP weights.

    For each date:
      1. Drop assets with NaN signal.
      2. Rank: ascending if high_is_good, descending otherwise.
      3. Demean and standardize by cross-sectional std (ddof=0).

    Args:
        signals:      Signal DataFrame (date × asset).
        high_is_good: True → high signal value → positive weight (momentum, carry).
                      False → low signal value → positive weight (e.g. value P/E).
        min_assets:   Minimum non-NaN assets required; skip date if below.

    Returns:
        Raw FMP weight DataFrame (same shape as signals). Rows with insufficient
        data are zero-filled.
    """
    signals = signals.sort_index()
    weights = pd.DataFrame(
        0.0, index=signals.index, columns=signals.columns, dtype=float
    )

    for dt, row in signals.iterrows():
        x = row.dropna()

        if len(x) < min_assets:
            continue

        ranks = x.rank(ascending=high_is_good, method="average")

        denom = float(ranks.std(ddof=0))
        if denom == 0.0 or np.isnan(denom):
            continue

        w = (ranks - ranks.mean()) / denom
        weights.loc[dt, w.index] = w.values

    return weights
