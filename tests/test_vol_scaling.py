"""Priority 1: volatility scaling correctness."""
import numpy as np
import pandas as pd
import pytest

from gtaa.risk.covariance import rolling_covariance
from gtaa.risk.scaling import portfolio_vol, scale_to_target_vol


def _make_cov(n_assets=4, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=60, freq="ME")
    assets = [f"X{i}" for i in range(n_assets)]
    ret = pd.DataFrame(rng.normal(0, 0.05, (60, n_assets)), index=dates, columns=assets)
    return rolling_covariance(ret, ret.index[-1], lookback=36), assets


def test_scaled_portfolio_hits_target_vol():
    """After scaling, ex-ante vol should equal target."""
    cov, assets = _make_cov()
    raw = pd.Series([0.3, -0.2, 0.1, -0.2], index=assets)
    scaled = scale_to_target_vol(raw, cov, target_vol=0.01)
    actual_vol = portfolio_vol(scaled, cov)
    assert abs(actual_vol - 0.01) < 1e-10


def test_scaling_raises_on_zero_weights():
    """Zero-weight portfolio should raise, not return NaN silently."""
    cov, assets = _make_cov()
    zero_w = pd.Series(0.0, index=assets)
    with pytest.raises(ValueError):
        scale_to_target_vol(zero_w, cov, target_vol=0.01)


def test_scaling_is_linear():
    """Doubling weights then scaling should give same result as scaling original."""
    cov, assets = _make_cov()
    raw = pd.Series([0.3, -0.2, 0.1, -0.2], index=assets)
    scaled1 = scale_to_target_vol(raw, cov, target_vol=0.01)
    scaled2 = scale_to_target_vol(raw * 2, cov, target_vol=0.01)
    pd.testing.assert_series_equal(scaled1.round(12), scaled2.round(12))
