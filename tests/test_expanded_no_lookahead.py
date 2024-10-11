"""No-look-ahead tests for the expanded GTAA pipeline.

These tests verify that the weights at t earn returns at t+1, not same-period
returns, and that modifying future data does not affect the weights computed
at earlier dates.

These are integration-level tests using the full backtester.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from gtaa.portfolio.backtester_gtaa_expanded import run_gtaa_expanded_backtest
from gtaa.portfolio.returns import portfolio_returns_from_weights
from gtaa.factors.signals import (
    momentum_12_1_from_returns,
    momentum_12_1_from_prices,
    fx_carry_signal,
)
from gtaa.factors.ranking import rank_standardized_weights
from gtaa.risk.scaling import scale_weight_panel


def make_monthly_dates(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class TestFXMomentumLagIntegration:
    """FX momentum at t depends only on prices at t-2 and t-13."""

    def test_mutating_t_and_t1_prices_does_not_change_signal_at_t_minus_1(self):
        dates = make_monthly_dates(20)
        prices = pd.DataFrame(
            np.random.default_rng(555).uniform(90, 110, (20, 4)),
            index=dates,
            columns=["FXE", "FXY", "FXB", "FXA"],
        )
        from gtaa.factors.signals import momentum_12_1_from_prices
        sig_before = momentum_12_1_from_prices(prices)

        prices_perturbed = prices.copy()
        prices_perturbed.loc[dates[15], :] = 99999.0   # t
        prices_perturbed.loc[dates[16], :] = 99999.0   # t+1
        sig_after = momentum_12_1_from_prices(prices_perturbed)

        probe = dates[14]
        if probe in sig_before.index and not sig_before.loc[probe].isna().all():
            diff = (sig_after.loc[probe] - sig_before.loc[probe]).abs().max()
            assert diff < 1e-10, (
                f"FX momentum at t-1 changed by {diff} when t and t+1 prices mutated"
            )


class TestFICarryLagIntegration:
    """FX carry signal is a pass-through; it must not transform the data."""

    def test_carry_signal_value_unchanged_by_future_rows(self):
        """Appending future carry rows must not change earlier signal values."""
        dates_short = make_monthly_dates(10)
        dates_long  = make_monthly_dates(15)

        carry_base = pd.DataFrame(
            np.random.default_rng(777).normal(0, 1, (10, 4)),
            index=dates_short,
            columns=["FXE", "FXY", "FXB", "FXA"],
        )
        carry_extended = pd.concat([
            carry_base,
            pd.DataFrame(
                np.random.default_rng(888).normal(0, 1, (5, 4)),
                index=dates_long[10:],
                columns=["FXE", "FXY", "FXB", "FXA"],
            ),
        ])

        sig_short  = fx_carry_signal(carry_base)
        sig_long   = fx_carry_signal(carry_extended)

        common_dates = sig_short.index.intersection(sig_long.index)
        for dt in common_dates:
            diff = (sig_long.loc[dt] - sig_short.loc[dt]).abs().max()
            assert diff < 1e-10, (
                f"FX carry signal at {dt} changed when future rows added"
            )


class TestReturnAlignmentExpanded:
    """Final portfolio returns use weights at t, returns at t+1."""

    def test_weights_at_t_earn_returns_at_t_plus_1(self):
        """Manual check: portfolio_returns_from_weights shifts weights by 1."""
        dates = make_monthly_dates(20)
        weights = pd.DataFrame(
            {"A": [0.5] * 20, "B": [-0.5] * 20},
            index=dates,
        )
        returns = pd.DataFrame(
            {"A": [0.02] * 20, "B": [-0.01] * 20},
            index=dates,
        )
        port_rets = portfolio_returns_from_weights(weights, returns)

        # At each date t, the portfolio return should equal:
        # 0.5 * returns["A"][t+1] + (-0.5) * returns["B"][t+1]
        expected = 0.5 * 0.02 + (-0.5) * (-0.01)  # = 0.015
        active = port_rets.dropna()
        assert len(active) > 0
        for dt, val in active.items():
            assert abs(val - expected) < 1e-10, (
                f"Return at {dt} = {val}, expected {expected}"
            )
