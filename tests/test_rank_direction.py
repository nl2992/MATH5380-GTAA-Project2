"""Verify ranking convention: high signal → positive weight, low signal → negative weight."""
import numpy as np
import pandas as pd
import pytest
from gtaa.factors.ranking import rank_standardized_weights


def make_monthly_dates(n, start="2010-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


class TestRankDirection:
    def test_highest_signal_gets_positive_weight(self):
        """Asset with highest signal must get the most positive weight."""
        dates = make_monthly_dates(1)
        signal = pd.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]}, index=dates)
        weights = rank_standardized_weights(signal, high_is_good=True)
        assert weights.loc[dates[0], "C"] > 0, "Highest signal must get positive weight"
        assert weights.loc[dates[0], "A"] < 0, "Lowest signal must get negative weight"

    def test_lowest_signal_gets_negative_weight(self):
        dates = make_monthly_dates(1)
        signal = pd.DataFrame({"X": [5.0], "Y": [3.0], "Z": [1.0]}, index=dates)
        weights = rank_standardized_weights(signal, high_is_good=True)
        assert weights.loc[dates[0], "X"] > 0
        assert weights.loc[dates[0], "Z"] < 0

    def test_raw_weights_sum_to_zero_every_row(self):
        """Every date's weights must sum to zero (long/short FMP structure)."""
        dates = make_monthly_dates(5)
        rng = np.random.default_rng(99)
        signal = pd.DataFrame(rng.normal(0, 1, (5, 4)), index=dates, columns=list("ABCD"))
        weights = rank_standardized_weights(signal, high_is_good=True)
        row_sums = weights.sum(axis=1)
        assert (row_sums.abs() < 1e-10).all(), \
            f"Row sums not zero: {row_sums.values}"

    def test_high_is_good_false_inverts_direction(self):
        """high_is_good=False: lowest signal must get positive weight."""
        dates = make_monthly_dates(1)
        signal = pd.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]}, index=dates)
        weights = rank_standardized_weights(signal, high_is_good=False)
        assert weights.loc[dates[0], "A"] > 0, "Lowest signal should get positive weight when high_is_good=False"
        assert weights.loc[dates[0], "C"] < 0

    def test_single_row_nan_skipped(self):
        """Rows that are all NaN produce no weights."""
        dates = make_monthly_dates(2)
        signal = pd.DataFrame(
            {"A": [float("nan"), 1.0], "B": [float("nan"), -1.0]},
            index=dates
        )
        weights = rank_standardized_weights(signal, high_is_good=True)
        # First row — all NaN — should produce zero or NaN weights
        if dates[0] in weights.index:
            assert weights.loc[dates[0]].abs().sum() < 1e-10, \
                "All-NaN row should produce zero weights"
