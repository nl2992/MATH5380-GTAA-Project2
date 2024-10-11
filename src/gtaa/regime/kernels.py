"""Kernel weight functions for regime-distance weighting.

HW_GTAA_1 uses a polynomial (tricube) kernel. Nearby regimes receive high
weight; dissimilar regimes receive low weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def polynomial_kernel(normalized_distances: pd.Series, degree: int = 3) -> pd.Series:
    """Tricube-style polynomial kernel: K(u) = (1 - u^degree)^degree.

    Args:
        normalized_distances: Values in [0, 1].  u=0 → weight=1, u=1 → weight=0.
        degree: Polynomial degree.  HW1 uses degree=3 (tricube).

    Returns:
        Kernel weights, non-negative, not yet normalized to sum to 1.
    """
    u = normalized_distances.clip(0.0, 1.0)
    weights = (1 - u ** degree) ** degree
    return weights


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize weights so they sum to 1."""
    total = weights.sum()
    if total == 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total
