"""Base class and rank-standardization utility shared by all factors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from gtaa.models import DataBundle, FactorSpec


def standardized_rank_weights(signal: pd.Series, ascending: bool = True) -> pd.Series:
    """Cross-sectional rank-based weights, demeaned and scaled.

    Ranks are computed across all assets at a single date.
    The resulting weights sum to approximately zero (long/short).

    ascending=True: highest-ranked asset gets the most positive weight.
    ascending=False: lowest-ranked asset (smallest signal value) gets the most
        positive weight (use for P/E: low P/E → positive weight → value long).

    Args:
        signal: Cross-section of signals at one date.
        ascending: If True, high signal → high rank → positive weight.

    Returns:
        Demeaned, rank-scaled weights.
    """
    clean = signal.dropna()
    if len(clean) < 2:
        return pd.Series(0.0, index=signal.index)

    ranks = clean.rank(ascending=ascending, method="average")
    weights = (ranks - ranks.mean()) / ranks.std(ddof=1)  # ddof=1: matches Excel STDEV (sample); covariance uses ddof=0 separately

    return weights.reindex(signal.index).fillna(0.0)


class BaseFactor(ABC):
    """All factors must implement this interface."""

    spec: FactorSpec

    @abstractmethod
    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        """Return raw (unscaled) factor weights at month-end date.

        Weights must:
        - Use only information available on or before `date`.
        - Sum to approximately zero.
        - Be non-NaN for all assets in the active universe.
        """
