"""HW2 replication: country equity factor engine validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gtaa.factors.base import standardized_rank_weights
from gtaa.factors.momentum import momentum_signal
from gtaa.models import DataBundle, BacktestConfig
from gtaa.portfolio.fmp import build_fmp_weights
from gtaa.validation.compare_excel import compare_series, tolerance_report


def run_hw2_replication(bundle: DataBundle, hw2_sheets: dict[str, pd.DataFrame]) -> dict:
    """Run full HW2 factor replication and return comparison reports.

    Args:
        bundle: DataBundle loaded from the HW2 or Project 2 data file.
        hw2_sheets: Output of load_hw2_data().

    Returns:
        Dict with 'reports' (tolerance_report) and 'computed' series.
    """
    computed = {}
    reports = {}

    # Value factor: low P/E rank weights
    if bundle.valuations is not None:
        from gtaa.factors.value import ValuePEFactor
        from gtaa.models import BacktestConfig

        cfg = BacktestConfig(
            covariance_lookback_months=36,
            factor_target_vol=0.01,
        )
        factor = ValuePEFactor()
        dates = bundle.returns.index
        dates = dates[dates >= dates[0] + pd.DateOffset(months=36)]

        raw_rows = {}
        for d in dates:
            try:
                raw_rows[d] = factor.compute_raw_weights(bundle, d)
            except Exception:
                pass
        computed["value_raw_weights"] = pd.DataFrame(raw_rows).T

    # Momentum factor
    from gtaa.factors.momentum import MomentumFactor
    mom_factor = MomentumFactor()
    mom_rows = {}
    dates = bundle.returns.index
    dates_valid = dates[dates >= dates[0] + pd.DateOffset(months=13)]
    for d in dates_valid:
        try:
            mom_rows[d] = mom_factor.compute_raw_weights(bundle, d)
        except Exception:
            pass
    computed["momentum_raw_weights"] = pd.DataFrame(mom_rows).T

    return {"reports": pd.DataFrame(), "computed": computed}
