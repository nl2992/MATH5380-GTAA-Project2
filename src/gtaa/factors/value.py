"""Value factor: long low-P/E countries, short high-P/E countries.

Signal: P/E ratio at date t (from .valuations in DataBundle).
Direction: Low P/E → positive weight (ascending=False in rank_weights, or
           equivalently rank ascending=True on 1/PE).

We rank P/E ascending=False (rank=1 is highest P/E → gets negative weight)
so that low-P/E assets end up with positive weights.  Equivalently:

    raw = -standardized_rank_weights(pe, ascending=True)

Both produce identical results. We use ascending=False directly for clarity.
"""
from __future__ import annotations

import pandas as pd

from gtaa.factors.base import BaseFactor, standardized_rank_weights
from gtaa.models import DataBundle, FactorSpec


class ValuePEFactor(BaseFactor):
    """Country equity value factor using P/E ratios."""

    spec = FactorSpec(
        name="value_pe",
        signal_type="valuation",
        ascending_rank=False,
        target_vol=0.01,
    )

    def compute_raw_weights(self, data: DataBundle, date) -> pd.Series:
        if data.valuations is None:
            raise ValueError("DataBundle.valuations is required for ValuePEFactor")

        pe_history = data.valuations.loc[:date]
        if len(pe_history) == 0:
            raise ValueError(f"No P/E data available on or before {date}")

        pe = pe_history.iloc[-1]
        pe = pe.replace(0, float("nan"))  # 0 P/E is invalid

        # ascending=False → highest P/E gets rank 1 → negative weight after demeaning
        # so low P/E gets positive weight (value long)
        return standardized_rank_weights(pe, ascending=False)
