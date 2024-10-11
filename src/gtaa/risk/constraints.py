"""Optional portfolio constraints (gross exposure cap, net target)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_gross_exposure_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Scale weights down so gross exposure (|w|.sum()) <= cap."""
    gross = weights.abs().sum()
    if gross <= cap or gross == 0:
        return weights
    return weights * (cap / gross)


def apply_net_exposure_target(weights: pd.Series, target: float = 0.0) -> pd.Series:
    """Shift all weights by a constant so net exposure equals target.

    This preserves relative differences between weights.
    net_shift = target - current_net
    """
    current_net = weights.sum()
    shift = (target - current_net) / len(weights)
    return weights + shift
