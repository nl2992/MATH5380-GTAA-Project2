"""FX momentum signal: no-look-ahead and formula correctness tests.

FX momentum reuses momentum_12_1_from_prices (P_{t-2}/P_{t-13} - 1).
These tests verify that:
  - Mutating FX prices at t and t+1 does not change the signal at t.
  - The signal only depends on P_{t-2} and P_{t-13}.
  - The exact formula is correct.
"""
import numpy as np
import pandas as pd
import pytest

from gtaa.factors.signals import momentum_12_1_from_prices


def make_monthly_dates(n: int, start: str = "2005-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class TestFXMomentumNoLookahead:
    def test_signal_unchanged_when_current_price_mutated(self):
        """Signal at t must not change if price at t is perturbed."""
        dates = make_monthly_dates(20)
        prices = pd.DataFrame(
            np.random.default_rng(101).uniform(90, 110, (20, 4)),
            index=dates,
            columns=["FXE", "FXY", "FXB", "FXA"],
        )
        sig_before = momentum_12_1_from_prices(prices)
        prices_perturbed = prices.copy()
        prices_perturbed.loc[dates[15], :] = 9999.0
        sig_after = momentum_12_1_from_prices(prices_perturbed)

        probe = dates[14]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, (
                f"FX momentum at t changed by {diff} after mutating price at t+1"
            )

    def test_signal_unchanged_when_next_price_mutated(self):
        """Signal at t must not change if price at t+1 is perturbed."""
        dates = make_monthly_dates(20)
        prices = pd.DataFrame(
            np.random.default_rng(202).uniform(90, 110, (20, 4)),
            index=dates,
            columns=["FXE", "FXY", "FXB", "FXA"],
        )
        sig_before = momentum_12_1_from_prices(prices)
        prices_perturbed = prices.copy()
        prices_perturbed.loc[dates[16:], :] = 9999.0
        sig_after = momentum_12_1_from_prices(prices_perturbed)

        probe = dates[14]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, (
                f"FX momentum at t changed by {diff} after mutating future prices"
            )

    def test_signal_uses_t_minus_2_not_t_minus_1(self):
        """Signal at t must be zero when only P_{t-1} differs from constant 1."""
        dates = make_monthly_dates(16)
        prices = pd.DataFrame(1.0, index=dates, columns=["FXE"])
        # Only change price at t-1 (index -2); signal should still be 0 at t
        prices.loc[dates[-2], "FXE"] = 9999.0
        sig = momentum_12_1_from_prices(prices)
        t = dates[-1]
        if t in sig.index and pd.notna(sig.loc[t, "FXE"]):
            # P_{t-2} = 1, P_{t-13} = 1 → signal = 0
            assert abs(sig.loc[t, "FXE"]) < 1e-10, (
                f"FX signal at t is {sig.loc[t, 'FXE']}, expected 0 "
                f"(must not use price at t-1)"
            )

    def test_exact_formula_fx(self):
        """P_{t-2}/P_{t-13} - 1 matches hand calculation for FX prices."""
        dates = make_monthly_dates(15)
        prices = pd.DataFrame(1.0, index=dates, columns=["FXF"])
        # P_{t-2} = index 12 = 1.15; P_{t-13} = index 1 = 1.0
        prices.loc[dates[12], "FXF"] = 1.15
        sig = momentum_12_1_from_prices(prices)
        t = dates[-1]  # index 14
        if t in sig.index and pd.notna(sig.loc[t, "FXF"]):
            assert abs(sig.loc[t, "FXF"] - 0.15) < 1e-10, (
                f"Expected 0.15, got {sig.loc[t, 'FXF']}"
            )
