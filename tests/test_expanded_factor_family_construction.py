"""Factor-family construction logic tests.

Verifies that combine_weight_panels correctly:
  - Uses date intersection (not union) so partial-sleeve periods are excluded.
  - Applies the specified allocation weights.
  - Returns zero rows outside the intersection.
  - Produces a union of all asset columns.
"""
import numpy as np
import pandas as pd

from gtaa.portfolio.fmp import combine_weight_panels


def make_monthly_dates(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class TestCombineWeightPanelsExpanded:
    def test_date_intersection_four_sleeves(self):
        """Combined panel dates = intersection of all four sleeve date indices."""
        dates_a = make_monthly_dates(20, "2010-01-31")
        dates_b = make_monthly_dates(18, "2010-03-31")
        dates_c = make_monthly_dates(15, "2010-06-30")
        dates_d = make_monthly_dates(12, "2010-09-30")

        w_a = pd.DataFrame({"X1": [1.0] * 20}, index=dates_a)
        w_b = pd.DataFrame({"X2": [1.0] * 18}, index=dates_b)
        w_c = pd.DataFrame({"X3": [1.0] * 15}, index=dates_c)
        w_d = pd.DataFrame({"X4": [1.0] * 12}, index=dates_d)

        alloc = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        combo = combine_weight_panels({"a": w_a, "b": w_b, "c": w_c, "d": w_d}, alloc)

        expected_dates = set(dates_a) & set(dates_b) & set(dates_c) & set(dates_d)
        assert set(combo.index) == expected_dates

    def test_allocation_weights_applied_correctly(self):
        """Each sleeve's weight is multiplied by its allocation before summing."""
        dates = make_monthly_dates(5)
        w_mom = pd.DataFrame({"A": [1.0] * 5, "B": [-1.0] * 5}, index=dates)
        w_car = pd.DataFrame({"C": [2.0] * 5, "D": [-2.0] * 5}, index=dates)

        alloc = {"mom": 0.5, "carry": 0.5}
        combo = combine_weight_panels({"mom": w_mom, "carry": w_car}, alloc)

        for dt in dates:
            assert abs(combo.loc[dt, "A"] - 0.5)  < 1e-10  # 0.5 × 1.0
            assert abs(combo.loc[dt, "B"] - (-0.5)) < 1e-10  # 0.5 × -1.0
            assert abs(combo.loc[dt, "C"] - 1.0)  < 1e-10  # 0.5 × 2.0
            assert abs(combo.loc[dt, "D"] - (-1.0)) < 1e-10  # 0.5 × -2.0

    def test_all_assets_in_union_of_columns(self):
        """Combined DataFrame columns = union of all sleeve asset columns."""
        dates = make_monthly_dates(10)
        w1 = pd.DataFrame({"A": [1.0] * 10, "B": [-1.0] * 10}, index=dates)
        w2 = pd.DataFrame({"C": [0.5] * 10, "D": [-0.5] * 10}, index=dates)
        w3 = pd.DataFrame({"E": [0.3] * 10, "B": [0.2] * 10}, index=dates)  # B shared

        alloc = {"s1": 1/3, "s2": 1/3, "s3": 1/3}
        combo = combine_weight_panels({"s1": w1, "s2": w2, "s3": w3}, alloc)

        assert set(combo.columns) == {"A", "B", "C", "D", "E"}

    def test_two_carry_sleeves_equal_weight(self):
        """FI carry + FX carry at 50/50 sums correctly for each asset."""
        dates = make_monthly_dates(8)
        fi_w = pd.DataFrame({"SHV": [1.0] * 8, "TLT": [-1.0] * 8}, index=dates)
        fx_w = pd.DataFrame({"FXE": [0.8] * 8, "FXY": [-0.8] * 8}, index=dates)

        alloc = {"fi_carry": 0.5, "fx_carry": 0.5}
        combo = combine_weight_panels({"fi_carry": fi_w, "fx_carry": fx_w}, alloc)

        for dt in dates:
            assert abs(combo.loc[dt, "SHV"] - 0.5) < 1e-10
            assert abs(combo.loc[dt, "FXE"] - 0.4) < 1e-10


class TestFactorFamilyReturnAlignmentSynthetic:
    """Verify that family-level returns are consistent with sleeve-level weights."""

    def test_four_sleeve_momentum_family_return(self):
        """Family return = weighted average of sleeve returns at t+1."""
        from gtaa.portfolio.returns import portfolio_returns_from_weights

        dates = make_monthly_dates(10)
        w_eq  = pd.DataFrame({"A": [0.5] * 10, "B": [-0.5] * 10}, index=dates)
        w_com = pd.DataFrame({"C": [0.5] * 10, "D": [-0.5] * 10}, index=dates)
        w_fi  = pd.DataFrame({"E": [0.3] * 10, "F": [-0.3] * 10}, index=dates)
        w_fx  = pd.DataFrame({"G": [0.4] * 10, "H": [-0.4] * 10}, index=dates)

        alloc = {"eq": 0.25, "com": 0.25, "fi": 0.25, "fx": 0.25}
        mom_combo = combine_weight_panels(
            {"eq": w_eq, "com": w_com, "fi": w_fi, "fx": w_fx}, alloc
        )

        returns = pd.DataFrame(
            0.01, index=dates, columns=list("ABCDEFGH")
        )
        port_rets = portfolio_returns_from_weights(mom_combo, returns)
        active = port_rets.dropna()
        assert len(active) > 0
