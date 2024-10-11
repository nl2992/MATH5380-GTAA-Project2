"""Main backtest runner for the three-sleeve GTAA Momentum + Carry strategy.

Pipeline:
  1. Load data from multi_asset_universe.xlsx
  2. Compute signals  (momentum from returns/prices; carry from signal sheet)
  3. Rank-standardize → raw FMP weights (zero-sum per row)
  4. Scale each FMP sleeve to 1% ex-ante annualized vol (36-month cov, ddof=0)
  5. Equal-weight combine the three scaled sleeves
  6. Rescale combined portfolio to 1% vol (full cross-asset covariance)
  7. Apply weights at t, earn returns at t+1
  8. Compute turnover (funded-portfolio convention)
  9. Compute performance statistics and risk tables

Usage:
    from gtaa.portfolio.backtester_gtaa_mom_carry import run_gtaa_mom_carry_backtest
    import yaml

    with open("config/project2_gtaa_mom_carry.yaml") as f:
        cfg = yaml.safe_load(f)

    result = run_gtaa_mom_carry_backtest(cfg)
"""
from __future__ import annotations

import pandas as pd

from gtaa.io.excel_loader import load_gtaa_mom_carry_data
from gtaa.factors.signals import (
    momentum_12_1_from_returns,
    momentum_12_1_from_prices,
    fixed_income_carry_signal,
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
        "Name":                       name,
        "Ann Return":                 annualized_arithmetic_return(rets),
        "Ann Vol":                    annualized_volatility(rets),
        "Information Ratio":          information_ratio(rets),
        "Avg Drawdown":               avg_drawdown(rets),
        "Max Drawdown":               max_drawdown(rets),
        "Avg Monthly Turnover":       float(to.mean())  if to is not None and len(to) > 0 else float("nan"),
        "Ann Turnover":               float(to.mean()) * 12 if to is not None and len(to) > 0 else float("nan"),
    }


def run_gtaa_mom_carry_backtest(config: dict) -> dict:
    """Run the full GTAA Momentum + Carry backtest.

    Args:
        config: Parsed YAML config dict (see config/project2_gtaa_mom_carry.yaml).

    Returns:
        Dict with keys:
          data                  — GTAAMomCarryData
          equity_signal         — momentum signal DataFrame
          commodity_signal      — momentum signal DataFrame
          fi_signal             — carry signal DataFrame
          equity_raw_weights    — raw FMP weights (zero-sum)
          commodity_raw_weights
          fi_raw_weights
          equity_scaled_weights — 1%-vol scaled
          commodity_scaled_weights
          fi_scaled_weights
          combo_weights         — equal-weighted before final scaling
          final_weights         — final 1%-vol scaled combined portfolio
          equity_fmp_returns    — monthly FMP returns (t weights → t+1 returns)
          commodity_fmp_returns
          fi_fmp_returns
          final_returns         — final GTAA portfolio returns
          final_turnover        — one-way turnover series
          stats                 — performance statistics DataFrame
          final_cov             — last 36-month covariance matrix
          final_vols            — last-date annualized volatilities
          final_corr            — last-date correlation matrix
    """
    workbook  = config["data"]["workbook"]
    lookback  = config["risk"]["covariance_lookback_months"]
    fmp_vol   = config["risk"]["fmp_target_vol"]
    final_vol = config["risk"]["final_target_vol"]
    alloc     = config["allocation"]["base_case"]

    # ── 1. Load data ───────────────────────────────────────────────────────────
    data = load_gtaa_mom_carry_data(workbook)

    # ── 2. Signals ─────────────────────────────────────────────────────────────
    equity_signal    = momentum_12_1_from_returns(data.equity_returns)
    commodity_signal = momentum_12_1_from_prices(data.commodity_prices)
    fi_signal        = fixed_income_carry_signal(data.fi_carry)

    # ── 3. Raw rank-standardized weights (high_is_good=True for all sleeves) ──
    equity_raw    = rank_standardized_weights(equity_signal,    high_is_good=True)
    commodity_raw = rank_standardized_weights(commodity_signal, high_is_good=True)
    fi_raw        = rank_standardized_weights(fi_signal,        high_is_good=True)

    # ── 4. Scale each FMP sleeve to fmp_vol (1%) ──────────────────────────────
    equity_scaled    = scale_weight_panel(equity_raw,    data.equity_returns,    fmp_vol, lookback)
    commodity_scaled = scale_weight_panel(commodity_raw, data.commodity_returns, fmp_vol, lookback)
    fi_scaled        = scale_weight_panel(fi_raw,        data.fi_returns,        fmp_vol, lookback)

    # ── 5. Equal-weight combination ────────────────────────────────────────────
    combo_weights = combine_weight_panels(
        {
            "equity_momentum":    equity_scaled,
            "commodity_momentum": commodity_scaled,
            "fixed_income_carry": fi_scaled,
        },
        alloc,
    )

    # ── 6. Rescale combined portfolio to final_vol (1%) ───────────────────────
    final_weights = scale_weight_panel(
        combo_weights,
        data.all_returns.reindex(columns=combo_weights.columns),
        final_vol,
        lookback,
    )

    # ── 7. Returns: weights at t → returns at t+1 ─────────────────────────────
    equity_fmp_returns    = portfolio_returns_from_weights(equity_scaled,    data.equity_returns)
    commodity_fmp_returns = portfolio_returns_from_weights(commodity_scaled, data.commodity_returns)
    fi_fmp_returns        = portfolio_returns_from_weights(fi_scaled,        data.fi_returns)
    final_returns         = portfolio_returns_from_weights(
        final_weights,
        data.all_returns.reindex(columns=final_weights.columns),
    )

    # ── 8. Turnover (funded-portfolio convention) ──────────────────────────────
    final_turnover = compute_turnover(
        final_weights,
        data.all_returns.reindex(columns=final_weights.columns),
    )

    # ── 9. Performance statistics ──────────────────────────────────────────────
    stats = pd.DataFrame([
        _perf_row("Equity Momentum FMP",    equity_fmp_returns),
        _perf_row("Commodity Momentum FMP", commodity_fmp_returns),
        _perf_row("Fixed-Income Carry FMP", fi_fmp_returns),
        _perf_row("Final GTAA Portfolio",   final_returns, final_turnover),
    ]).set_index("Name")

    # ── 10. Final risk tables ──────────────────────────────────────────────────
    final_date = final_weights.index[-1]
    all_ret_aligned = data.all_returns.reindex(columns=final_weights.columns)
    final_cov, final_vols, final_corr = final_risk_tables(
        all_ret_aligned, final_date, lookback=lookback
    )

    return {
        "data":                   data,
        "equity_signal":          equity_signal,
        "commodity_signal":       commodity_signal,
        "fi_signal":              fi_signal,
        "equity_raw_weights":     equity_raw,
        "commodity_raw_weights":  commodity_raw,
        "fi_raw_weights":         fi_raw,
        "equity_scaled_weights":  equity_scaled,
        "commodity_scaled_weights": commodity_scaled,
        "fi_scaled_weights":      fi_scaled,
        "combo_weights":          combo_weights,
        "final_weights":          final_weights,
        "equity_fmp_returns":     equity_fmp_returns,
        "commodity_fmp_returns":  commodity_fmp_returns,
        "fi_fmp_returns":         fi_fmp_returns,
        "final_returns":          final_returns,
        "final_turnover":         final_turnover,
        "stats":                  stats,
        "final_cov":              final_cov,
        "final_vols":             final_vols,
        "final_corr":             final_corr,
    }
