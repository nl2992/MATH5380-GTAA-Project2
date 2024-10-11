"""FX carry signal: pass-through and rank-direction tests.

The fx_carry_signal() function is a pass-through. These tests verify:
  - The output equals the input (no transformation applied).
  - rank_standardized_weights with high_is_good=True assigns positive weight
    to the highest carry currency and negative weight to the lowest.
"""
import numpy as np
import pandas as pd

from gtaa.factors.signals import fx_carry_signal
from gtaa.factors.ranking import rank_standardized_weights


def make_monthly_dates(n: int, start: str = "2008-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class TestFXCarrySignalPassthrough:
    def test_output_equals_input(self):
        """fx_carry_signal must return a copy equal to the input."""
        dates = make_monthly_dates(12)
        data = pd.DataFrame(
            {"FXE": [1.5, 1.4, 1.6] + [1.5] * 9,
             "FXY": [-0.5, -0.3, -0.7] + [-0.5] * 9},
            index=dates,
        )
        result = fx_carry_signal(data)
        pd.testing.assert_frame_equal(result, data)

    def test_returns_copy_not_view(self):
        """Mutating the returned DataFrame must not affect the original."""
        dates = make_monthly_dates(5)
        data = pd.DataFrame({"FXE": [1.0] * 5, "FXY": [2.0] * 5}, index=dates)
        result = fx_carry_signal(data)
        result.iloc[0, 0] = 999.0
        assert data.iloc[0, 0] == 1.0


class TestFXCarryRankDirection:
    def test_high_carry_gets_positive_weight(self):
        """With high_is_good=True, highest carry → highest positive weight."""
        dates = make_monthly_dates(3)
        carry = pd.DataFrame(
            {"FXA": [4.5, 4.3, 4.4],
             "FXE": [2.0, 2.1, 2.0],
             "FXF": [-0.5, -0.3, -0.4],
             "FXY": [-1.0, -0.9, -1.1]},
            index=dates,
        )
        signal = fx_carry_signal(carry)
        weights = rank_standardized_weights(signal, high_is_good=True)

        for dt in dates:
            row = weights.loc[dt]
            assert row["FXA"] > 0, f"FXA (highest carry) must have positive weight at {dt}"
            assert row["FXY"] < 0, f"FXY (lowest carry) must have negative weight at {dt}"

    def test_weights_sum_to_zero(self):
        """Cross-sectional FX carry weights must sum to zero (zero-sum FMP)."""
        dates = make_monthly_dates(6)
        carry = pd.DataFrame(
            np.random.default_rng(77).normal(0, 1, (6, 6)),
            index=dates,
            columns=["FXE", "FXY", "FXB", "FXA", "FXC", "FXF"],
        )
        weights = rank_standardized_weights(fx_carry_signal(carry), high_is_good=True)
        row_sums = weights.sum(axis=1)
        assert (row_sums.abs() < 1e-10).all(), "FX carry FMP weights must sum to zero"
