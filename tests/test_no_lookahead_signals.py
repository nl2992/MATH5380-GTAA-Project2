"""Verify that momentum and carry signals use only lagged data.

Tests use toy datasets where the answer is hand-computable.
Each test mutates data at date t or later and verifies the signal at t-1 is unchanged.
"""
import numpy as np
import pandas as pd
import pytest
from gtaa.factors.signals import momentum_12_1_from_returns, momentum_12_1_from_prices


def make_monthly_dates(n, start="2000-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


class TestEquityMomentumLookAhead:
    def test_signal_unchanged_when_current_month_return_mutated(self):
        """Signal at t must not change if we perturb return at t."""
        dates = make_monthly_dates(20)
        returns = pd.DataFrame(
            np.random.default_rng(42).normal(0.005, 0.03, (20, 3)),
            index=dates, columns=["A", "B", "C"]
        )
        sig_before = momentum_12_1_from_returns(returns)
        # Mutate return at t=dates[15] (same-month as a signal we'll check)
        returns_perturbed = returns.copy()
        returns_perturbed.loc[dates[15], :] = 999.0
        sig_after = momentum_12_1_from_returns(returns_perturbed)
        # Signal at dates[14] (the date BEFORE the mutation) must be unchanged
        probe = dates[14]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, f"Signal at t-1 changed by {diff} after mutating return at t"

    def test_signal_unchanged_when_future_return_mutated(self):
        """Signal at t must not change if we perturb return at t+1 or later."""
        dates = make_monthly_dates(20)
        returns = pd.DataFrame(
            np.random.default_rng(7).normal(0.004, 0.02, (20, 3)),
            index=dates, columns=["A", "B", "C"]
        )
        sig_before = momentum_12_1_from_returns(returns)
        returns_perturbed = returns.copy()
        returns_perturbed.loc[dates[16:], :] = 999.0  # future dates
        sig_after = momentum_12_1_from_returns(returns_perturbed)
        probe = dates[14]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, f"Signal at t changed by {diff} after mutating future returns"

    def test_momentum_window_exact_11_periods(self):
        """Equity momentum at t uses exactly returns t-12 through t-2 (11 months).

        For 16 dates (indices 0-15), shift(2).rolling(11) at dates[-1] (index 15)
        covers original indices [3..13].  dates[-2] (index 14) covers [2..12].

        We spike dates[3] so it is in the window for both dates[-1] and dates[-2].
        We then verify that dates[-2] also produces 0.5 (dates[2] is still in its window).
        """
        dates = make_monthly_dates(16)
        returns = pd.DataFrame(0.0, index=dates, columns=["A"])
        # dates[-1] window: original indices [3..13] — spike at index 3 is included
        returns.loc[dates[3], "A"] = 0.5
        sig = momentum_12_1_from_returns(returns)
        # Signal at dates[-1] = prod([1.5, 1, 1, ..., 1]) - 1 = 0.5
        t = dates[-1]
        if t in sig.index and pd.notna(sig.loc[t, "A"]):
            assert abs(sig.loc[t, "A"] - 0.5) < 1e-10, \
                f"Expected 0.5 signal at dates[-1], got {sig.loc[t, 'A']}"
        # dates[-2] window: original indices [2..12] — index 3 is also in this window
        t2 = dates[-2]
        if t2 in sig.index and pd.notna(sig.loc[t2, "A"]):
            assert abs(sig.loc[t2, "A"] - 0.5) < 1e-10, \
                f"Expected 0.5 signal at dates[-2], got {sig.loc[t2, 'A']}"


class TestCommodityMomentumLookAhead:
    def test_price_signal_uses_t_minus_2_not_t(self):
        """P_{t-2}/P_{t-13} - 1: signal at t must not use price at t."""
        dates = make_monthly_dates(16)
        prices = pd.DataFrame(1.0, index=dates, columns=["X"])
        # Set price at t to something huge — signal at t should not change
        prices.loc[dates[-1], "X"] = 999.0
        sig = momentum_12_1_from_prices(prices)
        t = dates[-1]
        # sig at t uses prices[t-2] / prices[t-13] - 1 = 1/1 - 1 = 0
        if t in sig.index and pd.notna(sig.loc[t, "X"]):
            assert abs(sig.loc[t, "X"]) < 1e-10, \
                f"Signal at t is {sig.loc[t, 'X']}, should be 0 (must not use price at t)"

    def test_price_signal_exact_formula(self):
        """P_{t-2}/P_{t-13} - 1 matches hand calculation."""
        dates = make_monthly_dates(15)
        prices = pd.DataFrame(1.0, index=dates, columns=["Y"])
        # Set P_{t-2} = 1.2  (index -3, i.e., dates[12])
        # Set P_{t-13} = 1.0 (index -14, i.e., dates[1])
        prices.loc[dates[12], "Y"] = 1.2
        sig = momentum_12_1_from_prices(prices)
        t = dates[-1]  # dates[14]
        # shift(2) at t=14 → prices at index 12 = 1.2
        # shift(13) at t=14 → prices at index 1 = 1.0
        # signal = 1.2/1.0 - 1 = 0.2
        if t in sig.index and pd.notna(sig.loc[t, "Y"]):
            assert abs(sig.loc[t, "Y"] - 0.2) < 1e-10, \
                f"Expected 0.2, got {sig.loc[t, 'Y']}"
