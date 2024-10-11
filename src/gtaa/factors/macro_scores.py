"""Macro-regime score factor.

Uses LOWESS conditional return forecasts to produce factor-like weights.
This is the "macro overlay" factor used in Expansion 2.
"""
from __future__ import annotations

import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec
from gtaa.regime.lowess import compute_regime_forecasts
from gtaa.regime.standardization import expanding_zscore_df


class MacroScoreFactor(BaseFactor):
    """Factor weights derived from macro-regime conditional return forecasts."""

    spec = FactorSpec(
        name="macro_score",
        signal_type="macro",
        ascending_rank=True,
        target_vol=0.01,
    )

    def __init__(self, kernel_degree: int = 3):
        self.kernel_degree = kernel_degree

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        if data.macro is None:
            raise ValueError("MacroScoreFactor requires DataBundle.macro")

        std_macro = expanding_zscore_df(data.macro.loc[:date])
        forecasts = compute_regime_forecasts(
            std_macro, data.returns.loc[:date], date, self.kernel_degree
        )
        return standardized_rank_weights(forecasts, ascending=True)
