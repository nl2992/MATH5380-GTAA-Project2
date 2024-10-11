"""FI momentum signal: no-look-ahead and formula correctness tests.

FI momentum reuses momentum_12_1_from_returns.
These tests verify that:
  - Mutating FI returns at t and later does not change the signal at t.
  - The signal uses the 11-period cumulative return from t-12 to t-2.
"""
import numpy as np
import pandas as pd

from gtaa.factors.signals import momentum_12_1_from_returns


def make_monthly_dates(n: int, start: str = "2003-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


FI_TICKERS = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT", "VTIP", "TIP", "LQD", "HYG"]


class TestFIMomentumNoLookahead:
    def test_signal_unchanged_when_current_return_mutated(self):
        """FI momentum signal at t must not change if r_t is perturbed."""
        dates = make_monthly_dates(25)
        returns = pd.DataFrame(
            np.random.default_rng(301).normal(0.002, 0.01, (25, len(FI_TICKERS))),
            index=dates,
            columns=FI_TICKERS,
        )
        sig_before = momentum_12_1_from_returns(returns)
        returns_perturbed = returns.copy()
        returns_perturbed.loc[dates[18], :] = 999.0
        sig_after = momentum_12_1_from_returns(returns_perturbed)

        probe = dates[17]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, (
                f"FI momentum at t changed by {diff} after mutating r_t"
            )

    def test_signal_unchanged_when_future_returns_mutated(self):
        """FI momentum signal at t must not change if future returns are perturbed."""
        dates = make_monthly_dates(25)
        returns = pd.DataFrame(
            np.random.default_rng(402).normal(0.001, 0.015, (25, len(FI_TICKERS))),
            index=dates,
            columns=FI_TICKERS,
        )
        sig_before = momentum_12_1_from_returns(returns)
        returns_perturbed = returns.copy()
        returns_perturbed.loc[dates[19:], :] = 999.0
        sig_after = momentum_12_1_from_returns(returns_perturbed)

        probe = dates[17]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, (
                f"FI momentum at t changed by {diff} after mutating future returns"
            )

    def test_fi_momentum_uses_11_return_periods(self):
        """FI momentum uses gross returns from t-12 through t-2 (11 periods)."""
        dates = make_monthly_dates(16)
        returns = pd.DataFrame(0.0, index=dates, columns=["SHV"])
        # Spike at index 4 → in the t=-1 window starting at index 3
        returns.loc[dates[4], "SHV"] = 0.3
        sig = momentum_12_1_from_returns(returns)
        t = dates[-1]  # index 15
        # shift(2) at t=15 covers original index 13 back 11 steps → index 3..13
        # spike at index 4 is in this window → product ≠ 0
        if t in sig.index and pd.notna(sig.loc[t, "SHV"]):
            assert abs(sig.loc[t, "SHV"] - 0.3) < 1e-10, (
                f"Expected 0.3, got {sig.loc[t, 'SHV']}"
            )

    def test_fi_momentum_not_affected_by_most_recent_return(self):
        """FI momentum must not use return at t-1 (skip convention)."""
        dates = make_monthly_dates(16)
        returns = pd.DataFrame(0.0, index=dates, columns=["TLT"])
        # Spike only at t-1 (index 14); signal at t=15 should still be 0
        returns.loc[dates[14], "TLT"] = 0.5
        sig = momentum_12_1_from_returns(returns)
        t = dates[-1]
        if t in sig.index and pd.notna(sig.loc[t, "TLT"]):
            assert abs(sig.loc[t, "TLT"]) < 1e-10, (
                f"FI momentum uses t-1 return (got {sig.loc[t, 'TLT']}), "
                "should skip it"
            )
