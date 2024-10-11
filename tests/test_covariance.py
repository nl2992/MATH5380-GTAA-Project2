"""Priority 1: covariance window uses exactly the specified lookback."""
import numpy as np
import pandas as pd
import pytest

from gtaa.risk.covariance import rolling_covariance


def _make_returns(n=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n, freq="ME")
    assets = ["A", "B", "C", "D"]
    return pd.DataFrame(rng.normal(0.005, 0.05, (n, len(assets))), index=dates, columns=assets)


def test_covariance_uses_exactly_lookback_months():
    """The covariance must use exactly `lookback` observations."""
    ret = _make_returns(n=60)
    asof = ret.index[47]  # 48th date (0-indexed)
    cov = rolling_covariance(ret, asof, lookback=36)

    # Manually compute cov of last 36 rows before asof
    window = ret.loc[:asof].tail(36)
    expected = window.cov(ddof=0) * 12
    pd.testing.assert_frame_equal(cov.round(10), expected.round(10))


def test_covariance_raises_if_insufficient_history():
    """Must raise ValueError if fewer than min_lookback observations available."""
    ret = _make_returns(n=30)
    asof = ret.index[-1]
    with pytest.raises(ValueError, match="Insufficient"):
        rolling_covariance(ret, asof, lookback=36, min_lookback=36)


def test_covariance_positive_semidefinite():
    """Covariance matrix eigenvalues must be non-negative."""
    ret = _make_returns(n=60)
    asof = ret.index[-1]
    cov = rolling_covariance(ret, asof, lookback=36)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert (eigenvalues >= -1e-10).all(), f"Non-PSD matrix: min eigenvalue = {eigenvalues.min()}"


def test_covariance_is_annualized():
    """Covariance should equal monthly cov * 12."""
    ret = _make_returns(n=60)
    asof = ret.index[-1]
    window = ret.tail(36)
    monthly = window.cov(ddof=0)
    annual = rolling_covariance(ret, asof, lookback=36, annualization=12)
    pd.testing.assert_frame_equal((monthly * 12).round(12), annual.round(12))
