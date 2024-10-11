"""Priority 2: performance statistic correctness."""
import numpy as np
import pandas as pd
import pytest

from gtaa.analytics.performance import (
    annualized_arithmetic_return,
    annualized_volatility,
    avg_drawdown,
    drawdowns,
    growth_of_one,
    information_ratio,
    max_drawdown,
)


def _flat_returns(monthly_r=0.01, n=60):
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.Series(monthly_r, index=dates)


def test_growth_of_one_monotone_positive_returns():
    """Growth of $1 must be monotonically increasing for positive returns."""
    ret = _flat_returns(0.005)
    g = growth_of_one(ret)
    assert (g.diff().dropna() > 0).all()


def test_annualized_return_matches_manual():
    """12 × mean monthly return should equal annualized return."""
    ret = _flat_returns(0.01)
    expected = 12 * 0.01
    assert abs(annualized_arithmetic_return(ret) - expected) < 1e-12


def test_annualized_vol_matches_manual():
    """Annualized vol = monthly std(ddof=0) × sqrt(12)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    ret = pd.Series(rng.normal(0.005, 0.04, 60), index=dates)
    expected = ret.std(ddof=0) * np.sqrt(12)
    assert abs(annualized_volatility(ret) - expected) < 1e-12


def test_information_ratio_positive_for_positive_returns():
    """IR should be positive when mean return > 0."""
    ret = _flat_returns(0.005)
    assert information_ratio(ret) > 0


def test_drawdowns_non_positive():
    """All drawdown values must be <= 0."""
    rng = np.random.default_rng(10)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    ret = pd.Series(rng.normal(0, 0.05, 60), index=dates)
    dd = drawdowns(ret)
    assert (dd <= 0 + 1e-12).all()


def test_max_drawdown_is_minimum_of_drawdowns():
    """Max drawdown = min(drawdown series)."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    ret = pd.Series(rng.normal(0, 0.05, 60), index=dates)
    assert abs(max_drawdown(ret) - drawdowns(ret).min()) < 1e-12


def test_zero_returns_zero_drawdown():
    """Constant zero returns → zero drawdown."""
    ret = _flat_returns(0.0)
    dd = drawdowns(ret)
    assert (dd.abs() < 1e-12).all()
