"""Priority 2: HW2 factor engine unit tests."""
import numpy as np
import pandas as pd
import pytest

from gtaa.factors.low_vol import LowVolFactor
from gtaa.factors.momentum import MomentumFactor, momentum_signal
from gtaa.factors.value import ValuePEFactor
from gtaa.models import DataBundle, BacktestConfig
from gtaa.portfolio.fmp import build_fmp_weights


def _make_bundle(n=60, n_assets=10, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n, freq="ME")
    assets = [f"C{i}" for i in range(n_assets)]
    ret = pd.DataFrame(rng.normal(0.005, 0.05, (n, n_assets)), index=dates, columns=assets)
    pe = pd.DataFrame(rng.uniform(8, 30, (n, n_assets)), index=dates, columns=assets)
    return DataBundle(returns=ret, valuations=pe)


def test_fmp_raw_weights_sum_zero_every_date():
    """Value FMP raw weights must sum to zero at every rebalance date."""
    bundle = _make_bundle()
    cfg = BacktestConfig(covariance_lookback_months=36, factor_target_vol=0.01)
    raw_df, _ = build_fmp_weights(
        ValuePEFactor(), bundle, bundle.returns.index[36:], cfg
    )
    row_sums = raw_df.sum(axis=1)
    assert (row_sums.abs() < 1e-10).all()


def test_fmp_scaled_vol_near_target():
    """Every scaled FMP weight row must yield ~1% ex-ante vol."""
    from gtaa.risk.covariance import rolling_covariance
    from gtaa.risk.scaling import portfolio_vol

    bundle = _make_bundle()
    cfg = BacktestConfig(covariance_lookback_months=36, factor_target_vol=0.01)
    _, scaled_df = build_fmp_weights(
        LowVolFactor(), bundle, bundle.returns.index[36:], cfg
    )
    for date in scaled_df.index:
        cov = rolling_covariance(bundle.returns, date, 36, 12, 36)
        vol = portfolio_vol(scaled_df.loc[date], cov)
        assert abs(vol - 0.01) < 5e-4, f"Vol at {date}: {vol:.6f} != 0.01"


def test_momentum_12_1_signal_uses_correct_window():
    """12-1 momentum should use months t-12 to t-1 (skip=1)."""
    n = 24
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    assets = ["A"]
    # Constant +1% for first 12 months, then -1%
    ret_vals = [0.01] * 12 + [-0.01] * 12
    ret = pd.DataFrame(ret_vals, index=dates, columns=assets)

    asof = dates[23]  # last date
    signal = momentum_signal(ret, asof, lookback=12, skip=1)

    # Window is months 11..22 (0-indexed), all -1% → negative cumulative return
    assert float(signal["A"]) < 0
