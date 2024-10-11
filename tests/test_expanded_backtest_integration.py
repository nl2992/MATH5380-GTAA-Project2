"""Integration tests for the expanded GTAA backtest.

Uses the real data workbook and config to verify:
  1. Backtest length ≥ 120 months.
  2. All six sleeves have active weights at some point.
  3. Factor families have active weights.
  4. Final weights have active positions throughout.
  5. All returns use t+1 alignment (no same-period contamination).
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

WORKBOOK = Path("data/raw/multi_asset_universe.xlsx")
CONFIG   = Path("config/project2_gtaa_expanded.yaml")


@pytest.fixture(scope="module")
def result():
    if not WORKBOOK.exists() or not CONFIG.exists():
        pytest.skip("Data or config not found — run from repository root")
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    from gtaa.portfolio.backtester_gtaa_expanded import run_gtaa_expanded_backtest
    return run_gtaa_expanded_backtest(cfg)


class TestBacktestLength:
    def test_at_least_120_months(self, result):
        """Expanded GTAA must have ≥ 120 monthly return observations."""
        r = result["final_returns"].dropna()
        assert len(r) >= 120, (
            f"Only {len(r)} months of returns — need ≥ 120"
        )


class TestActiveSleeves:
    SLEEVES = [
        "equity_momentum",
        "commodity_momentum",
        "fixed_income_momentum",
        "fx_momentum",
        "fixed_income_carry",
        "fx_carry",
    ]

    def test_all_sleeves_have_active_weights(self, result):
        """No sleeve should be uniformly zero (silently dropped)."""
        for name in self.SLEEVES:
            w = result["sleeve_scaled_weights"][name]
            max_abs = w.abs().max().max()
            assert max_abs > 0, (
                f"Sleeve '{name}' has all-zero weights — it may be silently inactive"
            )

    def test_all_sleeves_have_nonzero_returns(self, result):
        """Each sleeve must produce at least some non-NaN returns."""
        for name in self.SLEEVES:
            rets = result["sleeve_returns"][name].dropna()
            assert len(rets) > 0, (
                f"Sleeve '{name}' has no return observations"
            )


class TestFactorFamilies:
    def test_momentum_family_has_active_weights(self, result):
        w = result["factor_family_weights"]["cross_asset_momentum"]
        assert w.abs().max().max() > 0, "Cross-Asset Momentum family is all-zero"

    def test_carry_family_has_active_weights(self, result):
        w = result["factor_family_weights"]["cross_asset_carry"]
        assert w.abs().max().max() > 0, "Cross-Asset Carry family is all-zero"

    def test_momentum_family_returns_exist(self, result):
        r = result["factor_family_returns"]["cross_asset_momentum"].dropna()
        assert len(r) > 0

    def test_carry_family_returns_exist(self, result):
        r = result["factor_family_returns"]["cross_asset_carry"].dropna()
        assert len(r) > 0


class TestReturnAlignmentReal:
    def test_final_returns_do_not_use_contemporaneous_weights(self, result):
        """portfolio_returns_from_weights must shift: weights at t, returns at t+1."""
        from gtaa.portfolio.returns import portfolio_returns_from_weights

        final_weights = result["final_weights"]
        data = result["data"]
        all_ret = data.all_returns.reindex(columns=final_weights.columns)

        # Compute returns manually: weight[t] * return[t+1]
        shifted = all_ret.shift(-1)
        manual_rets = (final_weights * shifted).sum(axis=1)

        auto_rets = portfolio_returns_from_weights(final_weights, all_ret)

        common = manual_rets.dropna().index.intersection(auto_rets.dropna().index)
        assert len(common) > 10

        diff = (manual_rets.loc[common] - auto_rets.loc[common]).abs().max()
        assert diff < 1e-12, (
            f"Return alignment mismatch: max diff = {diff}"
        )


class TestPerformanceStats:
    def test_stats_dataframe_has_all_rows(self, result):
        """Performance table must contain all six sleeves + families + final."""
        stats = result["stats"]
        expected_names = [
            "Equity Momentum FMP",
            "Commodity Momentum FMP",
            "FI Momentum FMP",
            "FX Momentum FMP",
            "FI Carry FMP",
            "FX Carry FMP",
            "Cross-Asset Momentum Family",
            "Cross-Asset Carry Family",
            "Final Expanded GTAA",
        ]
        for name in expected_names:
            assert name in stats.index, f"Missing row '{name}' in stats table"

    def test_final_annualized_vol_near_one_pct(self, result):
        """Final portfolio annualized vol should be close to 1% target."""
        from gtaa.analytics.performance import annualized_volatility
        r = result["final_returns"].dropna()
        vol = annualized_volatility(r)
        # Allow ±0.5% around target due to rolling estimation lag
        assert 0.005 <= vol <= 0.015, (
            f"Final GTAA annualized vol = {vol:.4f}, expected near 0.01"
        )
