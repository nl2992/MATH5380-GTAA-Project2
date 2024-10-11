"""Verify that covariance estimation uses exactly the declared lookback window."""
import numpy as np
import pandas as pd
import pytest
from gtaa.risk.covariance import rolling_covariance, rolling_covariance_partial


def make_monthly_dates(n, start="2000-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


class TestCovarianceWindow:
    def test_exact_lookback_rows_used(self):
        """rolling_covariance at asof_date=dates[38] with lookback=36 uses dates[3:39]."""
        dates = make_monthly_dates(42)
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.normal(0, 0.02, (42, 3)), index=dates, columns=["A","B","C"])
        asof = dates[38]  # index 38
        cov = rolling_covariance(returns, asof, lookback=36)
        # Expected window: dates[3:39] (36 rows: indices 3..38 inclusive)
        expected_window = returns.iloc[3:39]
        expected_cov = expected_window.cov(ddof=0) * 12
        pd.testing.assert_frame_equal(cov, expected_cov, check_exact=False, atol=1e-12)

    def test_future_data_excluded(self):
        """Mutating return at t+1 must not change covariance formed at t."""
        dates = make_monthly_dates(42)
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.normal(0, 0.02, (42, 3)), index=dates, columns=["A","B","C"])
        asof = dates[38]
        cov_before = rolling_covariance(returns, asof, lookback=36)
        returns_perturbed = returns.copy()
        returns_perturbed.loc[dates[39:], :] = 999.0  # perturb t+1 and beyond
        cov_after = rolling_covariance(returns_perturbed, asof, lookback=36)
        pd.testing.assert_frame_equal(cov_before, cov_after, check_exact=False, atol=1e-12)

    def test_partial_universe_drops_nan_columns(self):
        """rolling_covariance_partial drops assets with NaN in the window."""
        dates = make_monthly_dates(40)
        returns = pd.DataFrame(
            np.random.default_rng(2).normal(0, 0.02, (40, 3)),
            index=dates, columns=["A","B","C"]
        )
        # Make column C have NaN for the last 10 rows (within the 36-month window)
        returns.loc[dates[-10:], "C"] = np.nan
        asof = dates[-1]
        cov = rolling_covariance_partial(returns, asof, lookback=36)
        assert "C" not in cov.columns, "Column C with NaN should be dropped"
        assert "A" in cov.columns and "B" in cov.columns

    def test_sufficient_history_raises_if_not_enough(self):
        """rolling_covariance raises ValueError if fewer than lookback rows exist."""
        dates = make_monthly_dates(20)
        returns = pd.DataFrame(np.zeros((20, 2)), index=dates, columns=["A","B"])
        with pytest.raises(ValueError, match="Insufficient"):
            rolling_covariance(returns, dates[-1], lookback=36)
