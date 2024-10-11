"""Priority 1: raw factor weights sum to approximately zero."""
import numpy as np
import pandas as pd
import pytest

from gtaa.factors.base import standardized_rank_weights
from gtaa.factors.low_vol import LowVolFactor
from gtaa.factors.value import ValuePEFactor
from gtaa.models import DataBundle


def _make_bundle(n=60, n_assets=10, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n, freq="ME")
    assets = [f"C{i}" for i in range(n_assets)]
    ret = pd.DataFrame(rng.normal(0.005, 0.05, (n, n_assets)), index=dates, columns=assets)
    pe = pd.DataFrame(rng.uniform(10, 30, (n, n_assets)), index=dates, columns=assets)
    return DataBundle(returns=ret, valuations=pe)


def test_standardized_rank_weights_sum_zero():
    """Rank weights should sum to zero (within floating-point tolerance)."""
    signal = pd.Series([10, 20, 5, 15, 8, 25], index=[f"A{i}" for i in range(6)])
    w = standardized_rank_weights(signal)
    assert abs(w.sum()) < 1e-10


def test_value_factor_weights_sum_zero():
    """ValuePEFactor raw weights must sum approximately to zero."""
    bundle = _make_bundle()
    factor = ValuePEFactor()
    date = bundle.returns.index[-1]
    raw = factor.compute_raw_weights(bundle, date)
    assert abs(raw.sum()) < 1e-10


def test_low_vol_factor_weights_sum_zero():
    """LowVolFactor raw weights must sum approximately to zero."""
    bundle = _make_bundle()
    factor = LowVolFactor(lookback=36)
    date = bundle.returns.index[-1]
    raw = factor.compute_raw_weights(bundle, date)
    assert abs(raw.sum()) < 1e-10


def test_value_factor_low_pe_positive_weight():
    """The lowest P/E asset must have positive weight in the value factor."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2005-01-31", periods=60, freq="ME")
    assets = ["cheap", "medium", "expensive"]
    pe = pd.DataFrame(
        {"cheap": [10.0] * 60, "medium": [15.0] * 60, "expensive": [25.0] * 60},
        index=dates,
    )
    ret = pd.DataFrame(rng.normal(0, 0.05, (60, 3)), index=dates, columns=assets)
    bundle = DataBundle(returns=ret, valuations=pe)
    factor = ValuePEFactor()
    raw = factor.compute_raw_weights(bundle, dates[-1])
    assert raw["cheap"] > 0, "Low-PE asset should have positive weight"
    assert raw["expensive"] < 0, "High-PE asset should have negative weight"
