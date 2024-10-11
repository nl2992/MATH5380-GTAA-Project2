"""Verify that portfolio returns are booked one month after weights are set."""
import numpy as np
import pandas as pd
import pytest
from gtaa.portfolio.returns import portfolio_returns_from_weights


def make_monthly_dates(n, start="2000-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


class TestPortfolioReturnAlignment:
    def test_weight_at_t_earns_return_at_t_plus_1(self):
        """Weight set at t must earn the return at t+1, not t."""
        dates = make_monthly_dates(5)
        # Single asset, weight = 1.0 at date[1], zero elsewhere
        weights = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
        # Return at date[2] = 0.05, all others = 0
        returns = pd.DataFrame({"A": [0.0, 0.0, 0.05, 0.0, 0.0]}, index=dates)
        port_ret = portfolio_returns_from_weights(weights, returns)
        # Weight at dates[1] earns return at dates[2] = 0.05
        assert dates[1] in port_ret.index, "dates[1] not in result"
        assert abs(port_ret.loc[dates[1]] - 0.05) < 1e-12, \
            f"Expected 0.05, got {port_ret.loc[dates[1]]}"

    def test_return_at_t_not_earned_by_weight_at_t(self):
        """Return at date t must NOT be earned by the weight set at the same date t."""
        dates = make_monthly_dates(5)
        weights = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
        # Set return ONLY at dates[1] (same date as the weight) to a large value
        returns = pd.DataFrame({"A": [0.0, 999.0, 0.0, 0.0, 0.0]}, index=dates)
        port_ret = portfolio_returns_from_weights(weights, returns)
        if dates[1] in port_ret.index:
            assert abs(port_ret.loc[dates[1]] - 999.0) > 1.0, \
                "Same-month return was booked — look-ahead contamination"
            # The weight at dates[1] earns returns[dates[2]] = 0.0
            assert abs(port_ret.loc[dates[1]]) < 1e-12, \
                f"Expected 0, got {port_ret.loc[dates[1]]}"

    def test_no_return_when_no_future_data(self):
        """Last weight date has no future return to earn — must be dropped."""
        dates = make_monthly_dates(3)
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        port_ret = portfolio_returns_from_weights(weights, returns)
        assert dates[-1] not in port_ret.index, \
            "Last weight date should not appear in result (no next-month return)"
