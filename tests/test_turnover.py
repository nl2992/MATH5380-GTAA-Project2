"""Priority 1: turnover uses drifted weights, not raw weight difference."""
import numpy as np
import pandas as pd
import pytest

from gtaa.portfolio.turnover import drift_weights, turnover


def test_drift_weights_derivative_overlay():
    """Default (derivative-overlay): drifted[i] = prev[i] * (1 + r[i]) — no normalization.

    This matches HW2 Excel formula Z5 = B4*(1+N5) exactly.
    """
    prev = pd.Series({"A": 0.6, "B": 0.4})
    ret  = pd.Series({"A": 0.10, "B": -0.05})
    drifted = drift_weights(prev, ret)

    assert abs(drifted["A"] - 0.6 * 1.10) < 1e-10
    assert abs(drifted["B"] - 0.4 * 0.95) < 1e-10


def test_drift_weights_funded_portfolio():
    """funded_portfolio=True: drifted[i] = prev[i]*(1+r[i]) / (1+R).

    For a fully-invested long-only portfolio (sum=1), this preserves sum=1.
    """
    prev = pd.Series({"A": 0.6, "B": 0.4})
    ret  = pd.Series({"A": 0.10, "B": -0.05})
    drifted = drift_weights(prev, ret, funded_portfolio=True)

    R = float((prev * ret).sum())
    assert abs(drifted["A"] - 0.6 * 1.10 / (1 + R)) < 1e-10
    assert abs(drifted["B"] - 0.4 * 0.95 / (1 + R)) < 1e-10
    # Long-only, sum=1 → drifted still sums to 1
    assert abs(drifted.sum() - 1.0) < 1e-10


def test_turnover_uses_drift_not_raw_diff():
    """Turnover must reflect the drift, not just |target - prev|."""
    prev   = pd.Series({"A": 0.5, "B": -0.5})
    target = pd.Series({"A": 0.5, "B": -0.5})   # same target
    ret    = pd.Series({"A": 0.10, "B": -0.10})  # assets earn different returns

    naive  = float((target - prev).abs().sum())
    actual = turnover(prev, target, ret)

    assert actual > naive, f"Turnover with drift ({actual:.4f}) should exceed naive ({naive:.4f})"


def test_zero_turnover_with_zero_returns():
    """If all returns are zero, weights don't drift, and rebalancing to same target = 0 turnover."""
    prev   = pd.Series({"A": 0.3, "B": -0.3, "C": 0.2, "D": -0.2})
    target = prev.copy()
    ret    = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
    assert abs(turnover(prev, target, ret)) < 1e-10


def test_turnover_nonnegative():
    """Turnover must always be non-negative."""
    rng    = np.random.default_rng(55)
    assets = [f"X{i}" for i in range(6)]
    prev   = pd.Series(rng.normal(0, 0.1, 6), index=assets)
    target = pd.Series(rng.normal(0, 0.1, 6), index=assets)
    ret    = pd.Series(rng.normal(0.01, 0.03, 6), index=assets)
    assert turnover(prev, target, ret) >= 0
