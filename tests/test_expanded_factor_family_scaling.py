"""Factor-family and final-portfolio volatility-targeting tests.

After combining sleeves into factor families and combining families into
the final portfolio, the ex-ante annualized volatility of each stage must
equal 1% (within numerical tolerance).

These tests are constructed with synthetic data so they are fast and
deterministic.
"""
import numpy as np
import pandas as pd

from gtaa.risk.covariance import rolling_covariance_partial
from gtaa.risk.scaling import portfolio_vol, scale_weight_panel
from gtaa.factors.ranking import rank_standardized_weights
from gtaa.portfolio.fmp import combine_weight_panels


def _make_returns(n_dates: int, n_assets: int, seed: int = 42, prefix: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_dates, freq="ME")
    assets = [f"{prefix}{i}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(0.005, 0.04, (n_dates, n_assets)), index=dates, columns=assets
    )


def _make_signal(returns: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = pd.DataFrame(
        rng.normal(0, 1, returns.shape), index=returns.index, columns=returns.columns
    )
    return signal


LOOKBACK = 36
TARGET = 0.01
TOL = 1e-6


class TestSleeveScaling:
    def test_single_sleeve_vol_equals_target(self):
        """Each sleeve scaled with scale_weight_panel hits 1% at every active date."""
        rets = _make_returns(60, 6, seed=1)
        sig = _make_signal(rets, seed=2)
        raw = rank_standardized_weights(sig, high_is_good=True)
        scaled = scale_weight_panel(raw, rets, TARGET, LOOKBACK)

        active_dates = scaled.index[scaled.abs().sum(axis=1) > 0]
        assert len(active_dates) > 0, "No active dates after scaling"

        for dt in active_dates:
            cov = rolling_covariance_partial(rets, dt, lookback=LOOKBACK, annualization=12)
            if cov.empty:
                continue
            w = scaled.loc[dt].reindex(cov.index).fillna(0.0)
            vol = portfolio_vol(w, cov)
            assert abs(vol - TARGET) < TOL, (
                f"Sleeve vol at {dt}: {vol:.6f}, expected {TARGET}"
            )


class TestFactorFamilyScaling:
    def test_momentum_family_vol_equals_target(self):
        """Cross-Asset Momentum family (4 sleeves) hits 1% after family rescaling."""
        rets_eq  = _make_returns(60, 5, seed=10, prefix="EQ")
        rets_com = _make_returns(60, 4, seed=11, prefix="COM")
        rets_fi  = _make_returns(60, 6, seed=12, prefix="FI")
        rets_fx  = _make_returns(60, 4, seed=13, prefix="FX")

        all_rets = pd.concat([rets_eq, rets_com, rets_fi, rets_fx], axis=1)

        def _sleeve(rets, seed):
            sig = _make_signal(rets, seed)
            raw = rank_standardized_weights(sig, high_is_good=True)
            return scale_weight_panel(raw, rets, TARGET, LOOKBACK)

        s_eq  = _sleeve(rets_eq,  seed=20)
        s_com = _sleeve(rets_com, seed=21)
        s_fi  = _sleeve(rets_fi,  seed=22)
        s_fx  = _sleeve(rets_fx,  seed=23)

        alloc = {"eq": 0.25, "com": 0.25, "fi": 0.25, "fx": 0.25}
        mom_pre = combine_weight_panels(
            {"eq": s_eq, "com": s_com, "fi": s_fi, "fx": s_fx}, alloc
        )
        mom_family = scale_weight_panel(
            mom_pre, all_rets.reindex(columns=mom_pre.columns), TARGET, LOOKBACK
        )

        active = mom_family.index[mom_family.abs().sum(axis=1) > 0]
        assert len(active) > 0, "No active dates for momentum family"

        for dt in active:
            cov = rolling_covariance_partial(
                all_rets.reindex(columns=mom_family.columns),
                dt, lookback=LOOKBACK, annualization=12,
            )
            if cov.empty:
                continue
            w = mom_family.loc[dt].reindex(cov.index).fillna(0.0)
            vol = portfolio_vol(w, cov)
            assert abs(vol - TARGET) < TOL, (
                f"Momentum family vol at {dt}: {vol:.6f}, expected {TARGET}"
            )

    def test_final_portfolio_vol_equals_target(self):
        """Final GTAA (after combining two families and rescaling) hits 1%."""
        rets_eq   = _make_returns(60, 5, seed=30, prefix="EQ")
        rets_fi   = _make_returns(60, 5, seed=31, prefix="FI")
        rets_fi2  = _make_returns(60, 5, seed=33, prefix="FI2")
        rets_fx   = _make_returns(60, 3, seed=32, prefix="FX")
        all_rets  = pd.concat([rets_eq, rets_fi, rets_fi2, rets_fx], axis=1)

        def _sleeve(rets, seed):
            sig = _make_signal(rets, seed)
            raw = rank_standardized_weights(sig, high_is_good=True)
            return scale_weight_panel(raw, rets, TARGET, LOOKBACK)

        s_mom1 = _sleeve(rets_eq,  seed=40)
        s_mom2 = _sleeve(rets_fi,  seed=41)
        s_car1 = _sleeve(rets_fi2, seed=42)
        s_car2 = _sleeve(rets_fx,  seed=43)

        mom_pre  = combine_weight_panels({"m1": s_mom1, "m2": s_mom2}, {"m1": 0.5, "m2": 0.5})
        mom_fam  = scale_weight_panel(mom_pre, all_rets.reindex(columns=mom_pre.columns), TARGET, LOOKBACK)

        carr_pre = combine_weight_panels({"c1": s_car1, "c2": s_car2}, {"c1": 0.5, "c2": 0.5})
        carr_fam = scale_weight_panel(carr_pre, all_rets.reindex(columns=carr_pre.columns), TARGET, LOOKBACK)

        combo_pre = combine_weight_panels({"mom": mom_fam, "carry": carr_fam}, {"mom": 0.5, "carry": 0.5})
        final     = scale_weight_panel(combo_pre, all_rets.reindex(columns=combo_pre.columns), TARGET, LOOKBACK)

        active = final.index[final.abs().sum(axis=1) > 0]
        assert len(active) > 0, "No active dates for final portfolio"

        for dt in active:
            cov = rolling_covariance_partial(
                all_rets.reindex(columns=final.columns),
                dt, lookback=LOOKBACK, annualization=12,
            )
            if cov.empty:
                continue
            w = final.loc[dt].reindex(cov.index).fillna(0.0)
            vol = portfolio_vol(w, cov)
            assert abs(vol - TARGET) < TOL, (
                f"Final GTAA vol at {dt}: {vol:.6f}, expected {TARGET}"
            )
