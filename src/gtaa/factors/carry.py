"""Carry factor proxy.

For country equities: dividend yield (1/PE as a rough carry proxy).
For multi-asset: bond yield level (TLT vs SHY slope) or dividend yield.

Since the project data contains P/E, we use earnings yield (1/PE) as
a carry-like signal for country equities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


class CarryFactor(BaseFactor):
    """Earnings-yield carry proxy (1/PE) for country equities."""

    spec = FactorSpec(
        name="carry",
        signal_type="carry",
        ascending_rank=True,
        target_vol=0.01,
    )

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        if data.valuations is None:
            raise ValueError("CarryFactor requires DataBundle.valuations (P/E ratios)")

        pe = data.valuations.loc[:date].iloc[-1]
        pe = pe.replace(0, float("nan"))
        earnings_yield = 1.0 / pe  # high earnings yield → attractive carry

        return standardized_rank_weights(earnings_yield, ascending=True)
