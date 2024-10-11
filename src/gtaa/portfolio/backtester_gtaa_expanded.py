"""Six-sleeve expanded GTAA backtest — two factor families.

Cross-Asset Momentum: equity + commodity + FI + FX momentum (25% each)
Cross-Asset Carry:    FI carry + FX carry (50% each)

Each sleeve → 1% vol. Families → 1% vol. Final portfolio → 1% vol.
Weights at t earn returns at t+1 throughout.
"""
from __future__ import annotations

import pandas as pd

from gtaa.io.excel_loader import load_gtaa_expanded_data
from gtaa.factors.signals import (
    momentum_12_1_from_returns,
    momentum_12_1_from_prices,
    fixed_income_carry_signal,
    fx_carry_signal,
)
from gtaa.factors.ranking import rank_standardized_weights
from gtaa.risk.scaling import scale_weight_panel
from gtaa.portfolio.fmp import combine_weight_panels
from gtaa.portfolio.returns import portfolio_returns_from_weights
from gtaa.portfolio.turnover import compute_turnover
from gtaa.analytics.performance import (
    annualized_arithmetic_return,
    annualized_volatility,
    information_ratio,
    avg_drawdown,
    max_drawdown,
)
from gtaa.analytics.risk_tables import final_risk_tables


def _perf_row(name: str, rets: pd.Series, to: pd.Series | None = None) -> dict:
    return {
        "Name":                 name,
        "Ann Return":           annualized_arithmetic_return(rets),
        "Ann Vol":              annualized_volatility(rets),
        "Information Ratio":    information_ratio(rets),
        "Avg Drawdown":         avg_drawdown(rets),
        "Max Drawdown":         max_drawdown(rets),
        "Avg Monthly Turnover": float(to.mean())      if to is not None and len(to) > 0 else float("nan"),
        "Ann Turnover":         float(to.mean()) * 12 if to is not None and len(to) > 0 else float("nan"),
    }


def run_gtaa_expanded_backtest(config: dict) -> dict:
    workbook   = config["data"]["workbook"]
    lookback   = config["risk"]["covariance_lookback_months"]
    sleeve_vol = config["risk"].get("sleeve_target_vol", 0.01)
    family_vol = config["risk"].get("factor_family_target_vol", 0.01)
    final_vol  = config["risk"]["final_target_vol"]
    ff_cfg     = config["factor_families"]
    alloc      = config["allocation"]["base_case"]

    data = load_gtaa_expanded_data(workbook)

    # ── signals ────────────────────────────────────────────────────────────────
    eq_mom_sig   = momentum_12_1_from_returns(data.equity_returns)
    com_mom_sig  = momentum_12_1_from_prices(data.commodity_prices)
    fi_mom_sig   = momentum_12_1_from_returns(data.fi_returns)
    fx_mom_sig   = momentum_12_1_from_prices(data.fx_prices)
    fi_carry_sig = fixed_income_carry_signal(data.fi_carry)
    fx_carry_sig = fx_carry_signal(data.fx_carry)

    signals = {
        "equity_momentum":       eq_mom_sig,
        "commodity_momentum":    com_mom_sig,
        "fixed_income_momentum": fi_mom_sig,
        "fx_momentum":           fx_mom_sig,
        "fixed_income_carry":    fi_carry_sig,
        "fx_carry":              fx_carry_sig,
    }

    # ── raw rank-standardized weights ─────────────────────────────────────────
    eq_mom_raw   = rank_standardized_weights(eq_mom_sig,   high_is_good=True)
    com_mom_raw  = rank_standardized_weights(com_mom_sig,  high_is_good=True)
    fi_mom_raw   = rank_standardized_weights(fi_mom_sig,   high_is_good=True)
    fx_mom_raw   = rank_standardized_weights(fx_mom_sig,   high_is_good=True)
    fi_carry_raw = rank_standardized_weights(fi_carry_sig, high_is_good=True)
    fx_carry_raw = rank_standardized_weights(fx_carry_sig, high_is_good=True)

    raw_weights = {
        "equity_momentum":       eq_mom_raw,
        "commodity_momentum":    com_mom_raw,
        "fixed_income_momentum": fi_mom_raw,
        "fx_momentum":           fx_mom_raw,
        "fixed_income_carry":    fi_carry_raw,
        "fx_carry":              fx_carry_raw,
    }

    # ── sleeve-level vol scaling (1%) ─────────────────────────────────────────
    eq_mom_sc   = scale_weight_panel(eq_mom_raw,   data.equity_returns,    sleeve_vol, lookback)
    com_mom_sc  = scale_weight_panel(com_mom_raw,  data.commodity_returns, sleeve_vol, lookback)
    fi_mom_sc   = scale_weight_panel(fi_mom_raw,   data.fi_returns,        sleeve_vol, lookback)
    fx_mom_sc   = scale_weight_panel(fx_mom_raw,   data.fx_returns,        sleeve_vol, lookback)
    fi_carry_sc = scale_weight_panel(fi_carry_raw, data.fi_returns,        sleeve_vol, lookback)
    fx_carry_sc = scale_weight_panel(fx_carry_raw, data.fx_returns,        sleeve_vol, lookback)

    sleeve_scaled_weights = {
        "equity_momentum":       eq_mom_sc,
        "commodity_momentum":    com_mom_sc,
        "fixed_income_momentum": fi_mom_sc,
        "fx_momentum":           fx_mom_sc,
        "fixed_income_carry":    fi_carry_sc,
        "fx_carry":              fx_carry_sc,
    }

    # ── Cross-Asset Momentum family (1%) ──────────────────────────────────────
    mom_pre = combine_weight_panels(
        {"equity_momentum": eq_mom_sc, "commodity_momentum": com_mom_sc,
         "fixed_income_momentum": fi_mom_sc, "fx_momentum": fx_mom_sc},
        ff_cfg["cross_asset_momentum"],
    )
    mom_family = scale_weight_panel(
        mom_pre, data.all_returns.reindex(columns=mom_pre.columns), family_vol, lookback
    )

    # ── Cross-Asset Carry family (1%) ─────────────────────────────────────────
    carry_pre = combine_weight_panels(
        {"fixed_income_carry": fi_carry_sc, "fx_carry": fx_carry_sc},
        ff_cfg["cross_asset_carry"],
    )
    carry_family = scale_weight_panel(
        carry_pre, data.all_returns.reindex(columns=carry_pre.columns), family_vol, lookback
    )

    factor_family_weights = {
        "cross_asset_momentum": mom_family,
        "cross_asset_carry":    carry_family,
    }

    # ── final portfolio (1%) ───────────────────────────────────────────────────
    combo_pre = combine_weight_panels(
        {"cross_asset_momentum": mom_family, "cross_asset_carry": carry_family}, alloc
    )
    final_weights = scale_weight_panel(
        combo_pre, data.all_returns.reindex(columns=combo_pre.columns), final_vol, lookback
    )

    # ── returns: weights at t → returns at t+1 ────────────────────────────────
    sleeve_returns = {
        name: portfolio_returns_from_weights(w, data.all_returns.reindex(columns=w.columns))
        for name, w in sleeve_scaled_weights.items()
    }
    factor_family_returns = {
        name: portfolio_returns_from_weights(w, data.all_returns.reindex(columns=w.columns))
        for name, w in factor_family_weights.items()
    }
    final_returns = portfolio_returns_from_weights(
        final_weights, data.all_returns.reindex(columns=final_weights.columns)
    )
    final_turnover = compute_turnover(
        final_weights, data.all_returns.reindex(columns=final_weights.columns)
    )

    # ── performance ───────────────────────────────────────────────────────────
    stats = pd.DataFrame([
        _perf_row("Equity Momentum FMP",        sleeve_returns["equity_momentum"]),
        _perf_row("Commodity Momentum FMP",      sleeve_returns["commodity_momentum"]),
        _perf_row("FI Momentum FMP",             sleeve_returns["fixed_income_momentum"]),
        _perf_row("FX Momentum FMP",             sleeve_returns["fx_momentum"]),
        _perf_row("FI Carry FMP",                sleeve_returns["fixed_income_carry"]),
        _perf_row("FX Carry FMP",                sleeve_returns["fx_carry"]),
        _perf_row("Cross-Asset Momentum Family", factor_family_returns["cross_asset_momentum"]),
        _perf_row("Cross-Asset Carry Family",    factor_family_returns["cross_asset_carry"]),
        _perf_row("Final Expanded GTAA",         final_returns, final_turnover),
    ]).set_index("Name")

    final_date = final_weights.index[-1]
    final_cov, final_vols, final_corr = final_risk_tables(
        data.all_returns.reindex(columns=final_weights.columns), final_date, lookback=lookback
    )

    return {
        "data":                   data,
        "signals":                signals,
        "raw_weights":            raw_weights,
        "sleeve_scaled_weights":  sleeve_scaled_weights,
        "factor_family_weights":  factor_family_weights,
        "final_weights":          final_weights,
        "sleeve_returns":         sleeve_returns,
        "factor_family_returns":  factor_family_returns,
        "final_returns":          final_returns,
        "final_turnover":         final_turnover,
        "stats":                  stats,
        "final_cov":              final_cov,
        "final_vols":             final_vols,
        "final_corr":             final_corr,
    }
